"""
Official GTSRB evaluation.

Writes accuracy, macro/weighted precision-recall-F1, a per-class table, the
confusion matrix, and the most common confusions. The reported numbers use
training-set normalization. A second pass repeats the old test-set
normalization so the leakage can be measured; that pass is not the headline result.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from load_dataset import NUM_CATEGORIES, classes
from model import (
    SEED,
    create_network,
    evaluate_with_confusion,
    load_batch_norm,
    load_norm_stats,
    numpy_to_tensor,
    seed_everything,
    set_training,
    standardize,
)
from test import load_test_data

import cpp


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def scores_from_confusion(confusion):
    """Precision, recall, and F1 from a matrix whose rows are true classes."""
    confusion = np.asarray(confusion)
    tp = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    total = support.sum()

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    weights = support / total
    return {
        "test_images": int(total),
        "correct_predictions": int(tp.sum()),
        "accuracy": float(tp.sum() / total),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float(np.sum(precision * weights)),
        "weighted_recall": float(np.sum(recall * weights)),
        "weighted_f1": float(np.sum(f1 * weights)),
        "per_class": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(np.int64),
        },
    }


def top_confusions(confusion, k=15):
    pairs = []
    for true_id in range(confusion.shape[0]):
        for pred_id in range(confusion.shape[1]):
            if true_id == pred_id:
                continue
            count = int(confusion[true_id, pred_id])
            if count > 0:
                pairs.append((count, true_id, pred_id))
    pairs.sort(reverse=True)
    return pairs[:k]


def save_heatmap(confusion, path):
    peak = confusion.max()
    if peak == 0:
        scaled = np.zeros(confusion.shape, dtype=np.uint8)
    else:
        scaled = (confusion.astype(np.float64) / peak * 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
    color = cv2.resize(color, (confusion.shape[1] * 10, confusion.shape[0] * 10), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), color)


def write_per_class(path, per_class):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "name", "support", "precision", "recall", "f1"])
        for class_id in range(NUM_CATEGORIES):
            writer.writerow([
                class_id,
                classes[class_id],
                int(per_class["support"][class_id]),
                f"{per_class['precision'][class_id]:.6f}",
                f"{per_class['recall'][class_id]:.6f}",
                f"{per_class['f1'][class_id]:.6f}",
            ])


def write_confusions(path, pairs):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["count", "true_id", "true_name", "pred_id", "pred_name"])
        for count, true_id, pred_id in pairs:
            writer.writerow([count, true_id, classes[true_id], pred_id, classes[pred_id]])


def public_metrics(scores):
    return {key: value for key, value in scores.items() if key != "per_class"}


def cpu_logits(net, batch):
    """Run the host stack and stop before SoftMax, matching forwardDevice."""
    current = numpy_to_tensor(batch)
    for index in range(net.numLayers()):
        layer = net.getLayer(index)
        if isinstance(layer, cpp.SoftMax):
            break
        current = layer.forward(current)
    return current.to_numpy()


def check_first_batch(net, images):
    """One real batch must match the host logits before the full test runs."""
    count = min(64, len(images))
    batch = np.ascontiguousarray(images[:count], dtype=np.float32)
    reference = cpu_logits(net, batch)
    got, _ = cpp.forward_device(net, batch)
    gap = float(np.max(np.abs(got - reference)))
    limit = 1e-4 * max(1.0, float(np.max(np.abs(reference))))
    if gap > limit:
        raise SystemExit(f"CUDA logits differ from the CPU stack by {gap:.3e}, limit {limit:.3e}")


def confusion_on_device(net, images, labels):
    """Official test. The timer is upload, kernels, and the logit download."""
    confusion = np.zeros((NUM_CATEGORIES, NUM_CATEGORIES), dtype=int)
    total_ms = 0.0
    for start in range(0, len(images), 64):
        batch = np.ascontiguousarray(images[start:start + 64], dtype=np.float32)
        logits, milliseconds = cpp.forward_device(net, batch)
        total_ms += float(milliseconds)
        predicted = np.argmax(logits, axis=1)
        for truth, guess in zip(labels[start:start + 64], predicted):
            confusion[int(truth), int(guess)] += 1
    return confusion, total_ms


def load_checkpoint(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    use_bn = bool(config["use_bn"])
    net, _, _, dropout, batch_norms = create_network(use_bn=use_bn)
    cpp.Serialization.loadNetwork(net, str(directory / "best_model.bin"))
    if batch_norms:
        load_batch_norm(batch_norms, directory / "bn.npz")
    set_training(dropout, batch_norms, False)
    mean, std = load_norm_stats(directory / "norm.npz")
    return net, mean, std, use_bn


def write_metrics(path, scores, extra):
    metrics = public_metrics(scores)
    metrics["protocol"] = "training-set per-channel mean/std"
    metrics.update(extra)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    accuracy = metrics["accuracy"] * 100
    print(f"Train-stat accuracy: {accuracy:.4f}% ({metrics['correct_predictions']} / {metrics['test_images']})")
    print(f"Macro F1: {metrics['macro_f1'] * 100:.4f}%")
    return metrics


def evaluate_default():
    seed_everything(SEED)
    net, _, _, dropout, batch_norms = create_network()
    cpp.Serialization.loadNetwork(net, "best_model.bin")
    load_batch_norm(batch_norms)
    set_training(dropout, batch_norms, False)

    X_test, y_test = load_test_data()
    mean_test = np.mean(X_test, axis=(0, 2, 3), keepdims=True)
    std_test = np.std(X_test, axis=(0, 2, 3), keepdims=True)

    mean, std = load_norm_stats()
    X_train_stat = standardize(X_test, mean, std)
    X_test_stat = standardize(X_test, mean_test, std_test)

    confusion = evaluate_with_confusion(net, X_train_stat, y_test, NUM_CATEGORIES)
    leaky = evaluate_with_confusion(net, X_test_stat, y_test, NUM_CATEGORIES)

    scores = scores_from_confusion(confusion)
    leaky_scores = scores_from_confusion(leaky)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_per_class(RESULTS_DIR / "per_class.csv", scores["per_class"])
    np.savetxt(RESULTS_DIR / "confusion.csv", confusion, fmt="%d", delimiter=",")
    save_heatmap(confusion, RESULTS_DIR / "confusion.png")
    write_confusions(RESULTS_DIR / "top_confusions.csv", top_confusions(confusion))

    metrics = public_metrics(scores)
    metrics["protocol"] = "training-set per-channel mean/std"
    metrics["test_set_normalization"] = public_metrics(leaky_scores)
    (RESULTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    accuracy = metrics["accuracy"] * 100
    leaky_accuracy = metrics["test_set_normalization"]["accuracy"] * 100
    print(f"Train-stat accuracy: {accuracy:.4f}% ({metrics['correct_predictions']} / {metrics['test_images']})")
    print(f"Macro F1: {metrics['macro_f1'] * 100:.4f}%")
    print(f"Test-stat accuracy (old protocol): {leaky_accuracy:.4f}%")
    print(f"Wrote {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Official GTSRB evaluation.")
    parser.add_argument("--checkpoint", type=Path, help="Directory with config.json, best_model.bin, and norm.npz.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.checkpoint is None:
        if args.device == "cuda":
            raise SystemExit("CUDA eval needs --checkpoint, and that checkpoint must not use BatchNorm.")
        evaluate_default()
        return

    seed_everything(SEED)
    net, mean, std, use_bn = load_checkpoint(args.checkpoint)
    if args.device == "cuda":
        if use_bn:
            raise SystemExit("CUDA eval is model A only. This checkpoint uses BatchNorm.")
        if not cpp.cuda_available():
            raise SystemExit("This build has no CUDA. Configure with nvcc and rebuild the cpp module.")

    images, labels = load_test_data()
    images = standardize(images, mean, std)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu":
        confusion = evaluate_with_confusion(net, images, labels, NUM_CATEGORIES)
        scores = scores_from_confusion(confusion)
        path = RESULTS_DIR / "test_metrics_cpu.json"
        write_metrics(path, scores, {"device": "cpu", "checkpoint": str(args.checkpoint)})
        print(f"Wrote {path}")
        return

    check_first_batch(net, images)
    confusion, total_ms = confusion_on_device(net, images, labels)
    scores = scores_from_confusion(confusion)
    images_per_second = scores["test_images"] / (total_ms / 1000.0)
    path = RESULTS_DIR / "test_metrics_cuda.json"
    write_metrics(path, scores, {
        "device": "cuda",
        "checkpoint": str(args.checkpoint),
        "milliseconds": total_ms,
        "images_per_second": images_per_second,
    })
    print(f"Device time: {total_ms:.3f} ms, {images_per_second:.3f} img/s")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
