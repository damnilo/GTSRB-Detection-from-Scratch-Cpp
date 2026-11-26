"""Brightness, contrast, blur, and calibration for one saved checkpoint.

Pixels stay in [0, 1] while a condition is applied, then the checkpoint's
training-set mean and standard deviation standardize the image. The clean
official test is one row of the same pass. This file does not train.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from eval import load_checkpoint
from model import numpy_to_tensor, standardize, tensor_to_numpy
from test import load_test_data


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "A" / "seed42"
RESULTS_PATH = ROOT / "results" / "robustness_a_seed42.json"
ECE_BINS = 15
CLEAN_CORRECT = 12033

CONDITIONS = (
    {"name": "clean", "kind": "clean"},
    {"name": "brightness -0.2", "kind": "brightness", "beta": -0.2},
    {"name": "brightness -0.1", "kind": "brightness", "beta": -0.1},
    {"name": "brightness +0.1", "kind": "brightness", "beta": 0.1},
    {"name": "brightness +0.2", "kind": "brightness", "beta": 0.2},
    {"name": "contrast 0.6", "kind": "contrast", "alpha": 0.6},
    {"name": "contrast 0.8", "kind": "contrast", "alpha": 0.8},
    {"name": "contrast 1.2", "kind": "contrast", "alpha": 1.2},
    {"name": "contrast 1.4", "kind": "contrast", "alpha": 1.4},
    {"name": "blur 3x3", "kind": "blur", "ksize": 3, "sigma": 0.0},
    {"name": "blur 5x5 sigma 1", "kind": "blur", "ksize": 5, "sigma": 1.0},
)


def apply_condition(images, condition):
    """Return a new NCHW array in [0, 1]. images is not modified."""
    kind = condition["kind"]
    if kind == "clean":
        return images
    if kind == "brightness":
        return np.clip(images + condition["beta"], 0.0, 1.0)
    if kind == "contrast":
        return np.clip((images - 0.5) * condition["alpha"] + 0.5, 0.0, 1.0)
    blurred = np.empty_like(images)
    ksize = condition["ksize"]
    sigma = condition["sigma"]
    for index, chw in enumerate(images):
        hwc = np.moveaxis(chw, 0, -1)
        hwc = cv2.GaussianBlur(hwc, (ksize, ksize), sigma)
        blurred[index] = np.moveaxis(hwc, -1, 0)
    return blurred


def predict(net, images):
    """Predicted class and the softmax probability of that class."""
    predicted = []
    confidence = []
    for start in range(0, len(images), 64):
        batch = np.ascontiguousarray(images[start:start + 64], dtype=np.float32)
        probabilities = tensor_to_numpy(net.forward(numpy_to_tensor(batch)))
        choice = np.argmax(probabilities, axis=1)
        predicted.append(choice)
        confidence.append(probabilities[np.arange(len(choice)), choice])
    return np.concatenate(predicted), np.concatenate(confidence)


def expected_calibration_error(correct, confidence, bins):
    """15-bin ECE. Each bin's weight is its share of the test set."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(correct)
    error = 0.0
    rows = []
    for index in range(bins):
        low = edges[index]
        high = edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        error += (count / total) * abs(accuracy - mean_confidence)
        rows.append({
            "low": float(low),
            "high": float(high),
            "count": count,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
        })
    return float(error), rows


def score_condition(net, images, labels, mean, std, condition):
    pixels = apply_condition(images, condition)
    predicted, confidence = predict(net, standardize(pixels, mean, std))
    correct = predicted == labels
    class_27 = labels == 27
    support_27 = int(class_27.sum())
    seven_to_eight = int(np.sum((labels == 7) & (predicted == 8)))
    ece, bins = expected_calibration_error(correct, confidence, ECE_BINS)
    row = {
        "name": condition["name"],
        "kind": condition["kind"],
        "test_images": int(len(labels)),
        "correct_predictions": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "class_27_support": support_27,
        "class_27_correct": int(np.sum(correct & class_27)),
        "class_27_recall": float(correct[class_27].mean()) if support_27 else 0.0,
        "confusions_7_to_8": seven_to_eight,
        "mean_confidence": float(confidence.mean()),
        "ece": ece,
        "ece_bins": bins,
    }
    for key in ("beta", "alpha", "ksize", "sigma"):
        if key in condition:
            row[key] = condition[key]
    return row


def main():
    net, mean, std, _use_bn = load_checkpoint(CHECKPOINT)
    images, labels = load_test_data()
    labels = np.asarray(labels)
    rows = []
    for condition in CONDITIONS:
        print(f"Scoring {condition['name']}")
        row = score_condition(net, images, labels, mean, std, condition)
        print(
            f"  accuracy {row['accuracy'] * 100:.2f}% "
            f"({row['correct_predictions']} / {row['test_images']}), "
            f"ECE {row['ece'] * 100:.2f}%, 7→8 {row['confusions_7_to_8']}"
        )
        rows.append(row)

    clean = rows[0]["correct_predictions"]
    if clean != CLEAN_CORRECT:
        raise SystemExit(
            f"Clean checkpoint scored {clean} / {rows[0]['test_images']}, "
            f"expected {CLEAN_CORRECT}."
        )

    payload = {
        "checkpoint": "checkpoints/A/seed42",
        "protocol": (
            "Perturb pixels in [0, 1], clip to [0, 1], then apply this "
            "checkpoint's training-set per-channel mean and std. "
            "Brightness adds beta. Contrast is (pixel - 0.5) * alpha + 0.5. "
            "Blur is cv2.GaussianBlur; sigma 0 uses the kernel size. "
            "ECE uses 15 bins on the softmax probability of the predicted class."
        ),
        "ece_bins": ECE_BINS,
        "conditions": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
