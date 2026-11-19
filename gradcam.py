"""Grad-CAM on the last convolution, plus a gallery of official-test mistakes.

The class score is the logit, so the backward pass stops before SoftMax.
Maps are the ReLU output after the last convolution, shape (1, 32, 16, 16).
Default checkpoint is ablation variant A, seed 42: no augmentation, no BatchNorm.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from load_dataset import NUM_CATEGORIES, classes
from model import (
    SEED,
    build_layers,
    load_batch_norm,
    load_norm_stats,
    numpy_to_tensor,
    seed_everything,
    set_training,
    standardize,
    tensor_to_numpy,
)
from test import load_test_data

import cpp


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "A" / "seed42"
OUTPUT_DIR = ROOT / "results" / "gradcam"
FAILURES_PATH = ROOT / "results" / "failures.md"
GALLERY_SIZE = 24
PAIR_LIMIT = 6
CELL = 128


def cam_bounds(layers):
    """End indexes of the map stack and of the logit stack.

    The map stack includes the ReLU that follows the last convolution.
    The logit stack stops on the last Dense, before SoftMax.
    """
    conv_indexes = [index for index, layer in enumerate(layers) if isinstance(layer, cpp.Conv2D)]
    if not conv_indexes:
        raise ValueError("Grad-CAM needs a convolution")
    map_end = conv_indexes[-1] + 1
    while map_end < len(layers) and isinstance(layers[map_end], cpp.BatchNorm2D):
        map_end += 1
    if map_end < len(layers) and isinstance(layers[map_end], cpp.ReLU):
        map_end += 1
    logit_end = len(layers) - 1 if isinstance(layers[-1], cpp.SoftMax) else len(layers)
    if map_end >= logit_end:
        raise ValueError("No classifier remains after the last convolution")
    return map_end, logit_end


def gradcam(layers, image, class_id):
    """Class-logit Grad-CAM for one standardized CHW image.

    Returns a float heatmap at the convolution's spatial size. Positive values
    are the regions that raised this class logit.
    """
    map_end, logit_end = cam_bounds(layers)
    activation = numpy_to_tensor(image[None].astype(np.float32))
    for layer in layers[:map_end]:
        activation = layer.forward(activation)
    maps = tensor_to_numpy(activation)

    hidden = activation
    for layer in layers[map_end:logit_end]:
        hidden = layer.forward(hidden)

    grad = np.zeros((1, NUM_CATEGORIES), dtype=np.float32)
    grad[0, class_id] = 1.0
    gradient = numpy_to_tensor(grad)
    for layer in reversed(layers[map_end:logit_end]):
        gradient = layer.backward(gradient)
    grad_maps = tensor_to_numpy(gradient)

    weights = grad_maps[0].mean(axis=(1, 2))
    heatmap = np.tensordot(weights, maps[0], axes=(0, 0))
    return np.maximum(heatmap, 0.0)


def predict_test(net, images):
    """Predicted class and max softmax for each standardized NCHW image."""
    predicted = np.empty(len(images), dtype=np.int32)
    confidence = np.empty(len(images), dtype=np.float32)
    for start in range(0, len(images), 64):
        if (start // 64) % 40 == 0:
            print(f"Scoring batch {start // 64 + 1}/{(len(images) + 63) // 64}")
        batch = images[start:start + 64]
        probabilities = tensor_to_numpy(net.forward(numpy_to_tensor(batch)))
        predicted[start:start + len(batch)] = np.argmax(probabilities, axis=1)
        confidence[start:start + len(batch)] = probabilities.max(axis=1)
    return predicted, confidence


def select_examples(y_true, y_pred, confidence):
    """Low-confidence mistakes, with room reserved for 7→8 and 27→29."""
    chosen = []

    def take(mask, limit):
        indexes = np.flatnonzero(mask)
        indexes = indexes[np.argsort(confidence[indexes])]
        for index in indexes[:limit]:
            index = int(index)
            if index not in chosen:
                chosen.append(index)

    take((y_true == 7) & (y_pred == 8), PAIR_LIMIT)
    take((y_true == 27) & (y_pred == 29), PAIR_LIMIT)
    mistakes = np.flatnonzero(y_true != y_pred)
    mistakes = mistakes[np.argsort(confidence[mistakes])]
    for index in mistakes:
        if len(chosen) >= GALLERY_SIZE:
            break
        index = int(index)
        if index not in chosen:
            chosen.append(index)
    return chosen[:GALLERY_SIZE]


def colorize(heatmap, size):
    heat = heatmap.astype(np.float32)
    peak = float(heat.max())
    if peak > 0:
        heat = heat / peak
    heat_u8 = (heat * 255).astype(np.uint8)
    heat_u8 = cv2.resize(heat_u8, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)


def panel(image_chw, heatmap, caption):
    """Original and overlay side by side, with a one-line caption."""
    rgb = np.clip(np.moveaxis(image_chw, 0, -1) * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (CELL, CELL), interpolation=cv2.INTER_NEAREST)
    color = colorize(heatmap, CELL)
    blended = cv2.addWeighted(bgr, 0.55, color, 0.45, 0)

    body = np.concatenate([bgr, blended], axis=1)
    header = np.zeros((28, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, caption, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([header, body], axis=0)


def save_confusion(confusion, path):
    peak = confusion.max()
    if peak == 0:
        scaled = np.zeros(confusion.shape, dtype=np.uint8)
    else:
        scaled = (confusion.astype(np.float64) / peak * 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
    color = cv2.resize(color, (confusion.shape[1] * 8, confusion.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), color)


def write_failures(path, directory, y_true, y_pred, confidence, chosen, filenames):
    try:
        shown = Path(directory).resolve().relative_to(ROOT)
    except ValueError:
        shown = directory
    confusion_7_8 = int(np.sum((y_true == 7) & (y_pred == 8)))
    confusion_27_29 = int(np.sum((y_true == 27) & (y_pred == 29)))
    mistakes = int(np.sum(y_true != y_pred))
    lines = [
        "# Test mistakes",
        "",
        f"Checkpoint `{shown.as_posix()}`. Dropout and BatchNorm are in eval mode.",
        "Normalization is the training-set mean and standard deviation saved beside the checkpoint.",
        "Grad-CAM is the ReLU after the last convolution, from the gradient of the predicted logit.",
        "",
        f"Mistakes: {mistakes} / {len(y_true)}.",
        f"Speed limit 100 read as 120 (7→8): {confusion_7_8}.",
        f"Pedestrians read as bicycles crossing (27→29): {confusion_27_29}.",
        "",
        "Classes 7 and 8 are the same round speed-limit sign. At 32×32 the digits collapse, so the heatmap sits on the sign body rather than on the number.",
        "Classes 27 and 29 are both triangular warnings with a dark figure. The resolution removes the difference between a person and a bicycle.",
        "",
        "| file | true | predicted | confidence |",
        "| --- | --- | --- | ---: |",
    ]
    for filename, index in zip(filenames, chosen):
        true_id = int(y_true[index])
        pred_id = int(y_pred[index])
        lines.append(
            f"| `{filename}` | {true_id} {classes[true_id]} | {pred_id} {classes[pred_id]} | {confidence[index]:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_model(directory):
    config_path = directory / "config.json"
    use_bn = False
    if config_path.is_file():
        use_bn = bool(json.loads(config_path.read_text(encoding="utf-8")).get("use_bn", False))
    seed_everything(SEED)
    layers, dropout, batch_norms = build_layers(use_bn)
    net = cpp.Network()
    for layer in layers:
        net.addLayer(layer)
    cpp.Serialization.loadNetwork(net, str(directory / "best_model.bin"))
    if batch_norms:
        load_batch_norm(batch_norms, directory / "bn.npz")
    set_training(dropout, batch_norms, False)
    return net, layers


def main():
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    net, layers = load_model(directory)
    mean, std = load_norm_stats(directory / "norm.npz")

    images, y_true = load_test_data()
    standardized = standardize(images, mean, std)
    y_pred, confidence = predict_test(net, standardized)

    confusion = np.zeros((NUM_CATEGORIES, NUM_CATEGORIES), dtype=np.int32)
    for true_id, pred_id in zip(y_true, y_pred):
        confusion[true_id, pred_id] += 1

    chosen = select_examples(y_true, y_pred, confidence)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion(confusion, OUTPUT_DIR / "confusion.png")

    panels = []
    filenames = []
    for rank, index in enumerate(chosen):
        heatmap = gradcam(layers, standardized[index], int(y_pred[index]))
        caption = f"t{int(y_true[index]):02d} p{int(y_pred[index]):02d} {confidence[index]:.2f}"
        image = panel(images[index], heatmap, caption)
        filename = f"{rank:02d}_true{int(y_true[index]):02d}_pred{int(y_pred[index]):02d}.png"
        cv2.imwrite(str(OUTPUT_DIR / filename), image)
        panels.append(image)
        filenames.append(filename)
        print(f"{filename} heatmap max {heatmap.max():.4f}")

    columns = 4
    rows = []
    for start in range(0, len(panels), columns):
        row = panels[start:start + columns]
        if len(row) < columns:
            pad = np.zeros_like(row[0])
            row.extend([pad] * (columns - len(row)))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(OUTPUT_DIR / "gallery.png"), np.concatenate(rows, axis=0))

    write_failures(FAILURES_PATH, directory, y_true, y_pred, confidence, chosen, filenames)
    print(f"Wrote {OUTPUT_DIR}")
    print(f"Wrote {FAILURES_PATH}")


if __name__ == "__main__":
    main()
