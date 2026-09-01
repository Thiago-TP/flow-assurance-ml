"""Stage 3 — Metrics and confusion matrix from out-of-fold predictions.

Consumes the OOF prediction table written by the training stage, so no model
is refit here. Writes the metrics JSON (global, per fold, per class) and the
row-normalized confusion matrix.

Usage
-----
    uv run scripts/03_evaluate.py [--model {rf,xgb}] [--task {prediction,detection}]
                                  [--filter {gaussian,statistical,none}]

Outputs (tag = <model>_<task>_<filter>)
---------------------------------------
    results/metrics/<tag>_metrics.json
    results/figures/<tag>_confusion_matrix.png
"""

import json
import sys

import pandas as pd
from flowml.cli import run_parser, run_tag
from flowml.config import FAULT_CLASSES, FIGURES_DIR, METRICS_DIR, WINDOW_CLASSES
from flowml.evaluation import (
    global_metrics,
    per_class_metrics,
    per_fold_metrics,
    plot_confusion_matrix,
)


def main() -> None:
    """Parse arguments, compute metrics from the OOF table, and save outputs."""
    args = run_parser(__doc__.splitlines()[0]).parse_args()
    tag = run_tag(args.model, args.task, args.filter_type)
    label_map = FAULT_CLASSES if args.task == "prediction" else WINDOW_CLASSES

    oof_path = METRICS_DIR / f"{tag}_oof.parquet"
    if not oof_path.exists():
        sys.exit(
            f"{oof_path} not found. Train first:\n"
            f"  uv run scripts/02_train.py --model {args.model} "
            f"--task {args.task} --filter {args.filter_type}"
        )
    oof = pd.read_parquet(oof_path)
    y_true, y_pred = oof["y_true"].to_numpy(), oof["y_pred"].to_numpy()

    metrics = {
        "tag": tag,
        "n_windows": len(oof),
        "global": global_metrics(y_true, y_pred),
        "per_fold_f1_macro": per_fold_metrics(oof),
        "per_class": per_class_metrics(y_true, y_pred, label_map),
    }

    print(f"Evaluation — {tag}")
    for name, value in metrics["global"].items():
        print(f"  {name:<12}: {value:.4f}")
    print(
        f"  per-fold F1 : {metrics['per_fold_f1_macro']['mean']:.4f} "
        f"± {metrics['per_fold_f1_macro']['std']:.4f}"
    )
    print("\n  Per class (F1 / support):")
    for cls, row in metrics["per_class"].items():
        print(f"    {cls:>3} {row['name']:<32} {row['f1']:.3f} / {row['support']:,}")

    metrics_path = METRICS_DIR / f"{tag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {metrics_path}")

    plot_confusion_matrix(
        y_true,
        y_pred,
        label_map,
        title=f"Confusion matrix — {tag} (row-normalized)",
        out_path=FIGURES_DIR / f"{tag}_confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
