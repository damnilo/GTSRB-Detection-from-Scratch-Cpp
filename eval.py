"""
Official GTSRB evaluation.

Writes accuracy, macro/weighted precision-recall-F1, a per-class table, the
confusion matrix, and the most common confusions. The reported numbers use
training-set normalization. A second pass repeats the old test-set
normalization so the leakage can be measured; that pass is not the headline result.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from load_dataset import NUM_CATEGORIES, classes
from model import SEED, create_network, evaluate_with_confusion, load_norm_stats, seed_everything, standardize
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


def main():
    seed_everything(SEED)
    net, _, _, dropout = create_network()
    cpp.Serialization.loadNetwork(net, "best_model.bin")
    dropout.setTraining(False)

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


if __name__ == "__main__":
    main()
