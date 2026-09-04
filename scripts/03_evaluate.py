"""Stage 3 — Metrics and confusion matrix from held-out predictions.

Consumes the evaluation table written by stage 2 (test predictions of the
holdout split, or outer-fold predictions of the nested CV), so no model is
refit here. Writes the metrics JSON (global, per fold, per class) and the
row-normalized confusion matrix.

With ``--class-grouping hydrate`` the stored predictions are collapsed onto
Normal / Other Problem / Hydrate before scoring, which measures how well a
model trained on the full class set serves the coarser triage question.
``--class-grouping custom`` applies the user-defined ``CUSTOM_CLASS_GROUPING``
from ``config.py`` instead.

Usage
-----
    uv run scripts/03_evaluate.py [--model {rf,xgb}] [--task {prediction,detection}]
                                  [--class-grouping {none,hydrate,custom}]
                                  [--eval {holdout,nested}]
                                  [--cv-group {instance_id,well_id}] [--no-normalization]

Outputs (tag = <model>_<task>_<norm>, suffixed with _<class-grouping> when grouping)
--------------------------------------------------------------------------------
    results/metrics/<tag>_metrics.json
    results/figures/<tag>_confusion_matrix.png
"""

import json
import sys

import pandas as pd

from flowml.cli import add_class_grouping_arg, run_parser, run_tag
from flowml.config import (
    FAULT_CLASSES,
    FIGURES_DIR,
    METRICS_DIR,
    WINDOW_CLASSES,
)
from flowml.evaluation import (
    global_metrics,
    per_class_metrics,
    per_fold_metrics,
    plot_confusion_matrix,
)
from flowml.train_val_test import group_labels, grouping_label_map


def main() -> None:
    """Parse arguments, compute metrics from the OOF table, and save outputs."""
    parser = run_parser(__doc__.splitlines()[0])
    add_class_grouping_arg(parser)
    args = parser.parse_args()

    source_tag = run_tag(args.model, args.task, not args.no_normalization, args.cv_group, args.eval)
    eval_path = METRICS_DIR / f"{source_tag}_eval.parquet"
    if not eval_path.exists():
        sys.exit(
            f"{eval_path} not found. Train first:\n"
            f"  uv run scripts/02_train_val_test.py --model {args.model} "
            f"--task {args.task}"
        )
    preds = pd.read_parquet(eval_path)

    if args.class_grouping == "none":
        tag = source_tag
        label_map = FAULT_CLASSES if args.task == "prediction" else WINDOW_CLASSES
    else:
        tag = f"{source_tag}_{args.class_grouping}"
        label_map = grouping_label_map(args.class_grouping)
        preds["y_true"] = group_labels(preds["y_true"].to_numpy(), args.class_grouping)
        preds["y_pred"] = group_labels(preds["y_pred"].to_numpy(), args.class_grouping)

    y_true, y_pred = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()

    metrics = {
        "tag": tag,
        "class_grouping": args.class_grouping,
        "evaluation": args.eval,
        "n_windows": len(preds),
        "global": global_metrics(y_true, y_pred),
        "per_fold_f1_macro": per_fold_metrics(preds),
        "per_class": per_class_metrics(y_true, y_pred, label_map),
    }

    print(f"Evaluation — {tag}")
    for name, value in metrics["global"].items():
        print(f"  {name:<12}: {value:.4f}")
    print(
        f"  per-fold F1 : {metrics['per_fold_f1_macro']['mean']:.4f} "
        f"± {metrics['per_fold_f1_macro']['std']:.4f}"
    )
    print("\n  Per class (precision / recall / F1 / support):")
    for cls, row in metrics["per_class"].items():
        print(
            f"    {cls:>3} {row['name']:<32} {row['precision']:.3f} / "
            f"{row['recall']:.3f} / {row['f1']:.3f} / {row['support']:,}"
        )

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
