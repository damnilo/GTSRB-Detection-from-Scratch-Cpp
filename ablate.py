"""Train the accuracy variants and summarize official-test metrics across seeds.

Variants share the training-set normalization protocol. Validation and test
images are never augmented. The reported number for a variant is the mean
across seeds, with the sample standard deviation.

    python ablate.py
    python ablate.py D
    python ablate.py D 42

A finished run is skipped when its metrics.json is already present.
"""

import json
import sys
from pathlib import Path

import numpy as np

from eval import scores_from_confusion
from load_dataset import NUM_CATEGORIES
from model import CHECKPOINT_DIR, create_network, evaluate_with_confusion, load_batch_norm, load_norm_stats, set_training, standardize, train
from test import load_test_data

import cpp


VARIANTS = {
    "A": {"use_aug": False, "use_bn": False, "optimizer_name": "sgd"},
    "B": {"use_aug": True, "use_bn": False, "optimizer_name": "sgd"},
    "C": {"use_aug": False, "use_bn": True, "optimizer_name": "sgd"},
    "D": {"use_aug": True, "use_bn": True, "optimizer_name": "sgd"},
    "E": {"use_aug": True, "use_bn": True, "optimizer_name": "adamw"},
}
SEEDS = (42, 43, 44, 45, 46)
SUMMARY_KEYS = ("accuracy", "macro_f1", "class_27_recall", "confusions_7_to_8")
ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "ablation.json"


def run_dir(variant, seed):
    return CHECKPOINT_DIR / variant / f"seed{seed}"


def evaluate_run(directory, use_bn):
    """Official test with this run's training-set mean and std."""
    net, _, _, dropout, batch_norms = create_network(use_bn=use_bn)
    cpp.Serialization.loadNetwork(net, str(directory / "best_model.bin"))
    load_batch_norm(batch_norms, directory / "bn.npz")
    set_training(dropout, batch_norms, False)

    images, labels = load_test_data()
    mean, std = load_norm_stats(directory / "norm.npz")
    confusion = evaluate_with_confusion(net, standardize(images, mean, std), labels, NUM_CATEGORIES)
    scores = scores_from_confusion(confusion)
    return {
        "accuracy": scores["accuracy"],
        "macro_f1": scores["macro_f1"],
        "class_27_recall": float(scores["per_class"]["recall"][27]),
        "confusions_7_to_8": int(confusion[7, 8]),
    }


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return float(array.mean()), std


def summarize(runs):
    by_variant = {}
    for run in runs:
        by_variant.setdefault(run["variant"], []).append(run)
    summary = {}
    for variant, rows in by_variant.items():
        summary[variant] = {
            "seeds": [row["seed"] for row in rows],
            "use_aug": VARIANTS[variant]["use_aug"],
            "use_bn": VARIANTS[variant]["use_bn"],
            "optimizer": VARIANTS[variant]["optimizer_name"],
        }
        for key in SUMMARY_KEYS:
            mean, std = mean_std([row[key] for row in rows])
            summary[variant][key] = {"mean": mean, "std": std}
    return summary


def format_pair(stats, scale):
    return f"{stats['mean'] * scale:.{2 if scale == 100 else 1}f} ± {stats['std'] * scale:.{2 if scale == 100 else 1}f}"


def print_summary(summary):
    print()
    print(f"{'variant':<8} {'accuracy %':>16} {'macro F1 %':>16} {'recall 27 %':>16} {'7→8':>14}")
    for variant in VARIANTS:
        if variant not in summary:
            continue
        row = summary[variant]
        print(
            f"{variant:<8} "
            f"{format_pair(row['accuracy'], 100):>16} "
            f"{format_pair(row['macro_f1'], 100):>16} "
            f"{format_pair(row['class_27_recall'], 100):>16} "
            f"{format_pair(row['confusions_7_to_8'], 1):>14}"
        )


def merge_runs(runs):
    """Keep metrics from other invocations, and replace rows this invocation produced."""
    existing = []
    if RESULTS_PATH.is_file():
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8")).get("runs", [])
    by_key = {(row["variant"], row["seed"]): row for row in existing}
    for row in runs:
        by_key[(row["variant"], row["seed"])] = row
    order = [(variant, seed) for variant in VARIANTS for seed in SEEDS]
    return [by_key[key] for key in order if key in by_key]


def selected_runs(argv):
    variants = list(VARIANTS)
    seeds = list(SEEDS)
    if len(argv) >= 2:
        variant = argv[1].upper()
        if variant not in VARIANTS:
            raise SystemExit(f"Unknown variant {argv[1]}. Choose one of {', '.join(VARIANTS)}.")
        variants = [variant]
    if len(argv) >= 3:
        seeds = [int(argv[2])]
    return [(variant, seed) for variant in variants for seed in seeds]


def main():
    runs = []
    for variant, seed in selected_runs(sys.argv):
        directory = run_dir(variant, seed)
        metrics_path = directory / "metrics.json"
        flags = VARIANTS[variant]
        if metrics_path.is_file():
            print(f"Skipping {variant} seed {seed}; {metrics_path} already exists.")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(
                f"Training {variant} seed {seed}: aug={flags['use_aug']} "
                f"bn={flags['use_bn']} optimizer={flags['optimizer_name']}"
            )
            train(seed=seed, checkpoint_dir=directory, **flags)
            metrics = evaluate_run(directory, flags["use_bn"])
            metrics.update({"variant": variant, "seed": seed, **flags})
            directory.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            print(
                f"{variant} seed {seed}: accuracy {metrics['accuracy'] * 100:.2f}%, "
                f"macro F1 {metrics['macro_f1'] * 100:.2f}%, "
                f"class 27 recall {metrics['class_27_recall'] * 100:.2f}%, "
                f"7→8 {metrics['confusions_7_to_8']}"
            )
        metrics.setdefault("variant", variant)
        metrics.setdefault("seed", seed)
        runs.append(metrics)

    combined = merge_runs(runs)
    summary = summarize(combined)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"runs": combined, "summary": summary}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
