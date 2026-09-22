"""Stage 3 — Metrics and confusion matrix from held-out predictions.

Consumes the evaluation table written by stage 2 (test predictions of the
holdout split, or out-of-fold predictions of the nested / leave-one-out
protocols), so no model is refit here. Writes the metrics JSON (global, per
fold, per class, per group) and the row-normalized confusion matrix.

With ``--class-grouping hydrate`` the run is judged on the coarser triage
question — Normal / Other Problem / Hydrate — and two strategies are reported
side by side against the same grouped truth, exactly as stage 5 does for its
trees:

- ``native`` — the predictions of the ensemble *trained* on the grouped
  labels (``02_train_val_test.py --class-grouping hydrate``).
- ``collapse`` — the predictions of the standard run, mapped onto the three
  groups afterwards. Answers "how well does the existing model already serve
  the triage question?"

Both runs share their splits — those are built from the dataset's own classes
whatever label set is modeled — so the two columns describe the same held-out
windows. Whichever run is missing is reported as such and the other is still
scored. ``--class-grouping custom`` works the same way on the user-defined
``CUSTOM_CLASS_GROUPING`` from ``config.py``.

When the evaluation table has few groups — always with wells as groups — every
group is also scored on its own (``evaluation.per_group_metrics``), so a well
predicted entirely wrong shows up even when the pooled number hides it; under
``--eval leave-one-out`` that sheet is the point of the protocol.

Usage
-----
    uv run scripts/03_evaluate.py [--model {rf,xgb,dt}] [--task {prediction,detection}]
                                  [--class-grouping {standard,hydrate,custom}]
                                  [--eval {holdout,nested,leave-one-out}]
                                  [--cv-group {instance_id,well_id}] [--normalization {none,instance,normal-operation-values}]
                                  [--frozen-sensors {keep,flag,drop}]
                                  [--allow-overlap]

The predictions are read from — and the metrics written into — the run
directory stage 2 wrote them to: the one ``main.py`` names in
``FLOWML_RUN_DIR``, the one ``--run`` points at, or otherwise the newest run
holding this configuration's evaluation table (see the ``runs`` module).

Outputs, inside the run directory (tag = the stage-2 tag of the run,
``_<class-grouping>`` included; confusion matrices are strategy-suffixed when
a grouping reports both)
--------------------------------------------------------------------------------
    metrics/<tag>_metrics.json
    figures/<tag>[_<strategy>]_confusion_matrix.png
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from flowml.cli import (
    add_class_grouping_arg,
    add_frozen_sensors_arg,
    add_normalization_arg,
    add_run_arg,
    run_parser,
    run_tag,
)
from flowml.config import (
    FAULT_CLASSES,
    WINDOW_CLASSES,
)
from flowml.evaluation import (
    global_metrics,
    per_class_metrics,
    per_fold_metrics,
    per_group_metrics,
    plot_confusion_matrix,
)
from flowml.runs import append_index, record_stage, resolve_run
from flowml.train_val_test import group_labels, grouping_label_map


def load_predictions(metrics_dir: Path, tag: str, collapse: str | None) -> pd.DataFrame | None:
    """Read one run's held-out predictions, optionally grouping them.

    Parameters
    ----------
    metrics_dir : Path
        The run's ``metrics`` directory.
    tag : str
        Stage-2 tag of the run to read.
    collapse : str | None
        Class grouping to map the stored labels onto, or ``None`` to take
        them as they are (the run already modeled that label set).

    Returns
    -------
    pd.DataFrame | None
        The evaluation table, or ``None`` when the run has not been trained.
    """
    path = metrics_dir / f"{tag}_eval.parquet"
    if not path.exists():
        return None
    preds = pd.read_parquet(path)
    if collapse is not None:
        preds["y_true"] = group_labels(preds["y_true"].to_numpy(), collapse)
        preds["y_pred"] = group_labels(preds["y_pred"].to_numpy(), collapse)
    return preds


def score(preds: pd.DataFrame, label_map: dict[int, str]) -> dict:
    """Compute every metric block of one strategy's predictions."""
    y_true, y_pred = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()
    block = {
        "n_windows": len(preds),
        "n_groups": int(preds["group"].nunique()),
        "global": global_metrics(y_true, y_pred),
        "per_fold_f1_macro": per_fold_metrics(preds),
        "per_class": per_class_metrics(y_true, y_pred, label_map),
    }
    per_group = per_group_metrics(preds)
    if per_group:
        block["per_group"] = per_group
    return block


def report(name: str, block: dict, label_map: dict[int, str], cv_group: str) -> None:
    """Print one strategy's metrics."""
    print(f"\n  Strategy '{name}' ({block['n_windows']:,} windows):")
    for metric, value in block["global"].items():
        print(f"    {metric:<12}: {value:.4f}")
    folds = block["per_fold_f1_macro"]
    print(f"    per-fold F1 : {folds['mean']:.4f} ± {folds['std']:.4f} ({len(folds) - 2} folds)")
    print("\n    Per class (precision / recall / F1 / support):")
    for cls, row in block["per_class"].items():
        print(
            f"      {cls:>3} {row['name']:<32} {row['precision']:.3f} / "
            f"{row['recall']:.3f} / {row['f1']:.3f} / {row['support']:,}"
        )
    if "per_group" in block:
        print(f"\n    Per group ({cv_group}; F1-macro / accuracy / windows / true classes):")
        for group, row in block["per_group"].items():
            classes = ", ".join(label_map.get(c, str(c)) for c in row["classes"])
            print(
                f"      {group:>6} {row['f1_macro']:.3f} / {row['accuracy']:.3f} / "
                f"{row['n_windows']:>7,} / {classes}"
            )


def main() -> None:
    """Parse arguments, compute metrics per strategy, and write the outputs."""
    parser = run_parser(__doc__.splitlines()[0])
    add_class_grouping_arg(parser)
    add_normalization_arg(parser)
    add_frozen_sensors_arg(parser)
    add_run_arg(parser)
    args = parser.parse_args()
    started = datetime.now().astimezone()

    common = (
        args.model,
        args.task,
        args.normalization,
        args.frozen_mode,
        args.cv_group,
        args.eval,
        args.allow_overlap,
        args.keep_extreme_values,
    )
    tag = run_tag(*common, args.class_grouping)
    standard_tag = run_tag(*common, "standard")

    # Either label set is enough to report something, so the run only has to
    # hold one of the two evaluation tables; a missing strategy is reported
    # below rather than refusing the whole stage.
    run = resolve_run(
        args.run,
        must_contain=[
            f"metrics/{tag}_eval.parquet",
            f"metrics/{standard_tag}_eval.parquet",
        ],
    )

    if args.class_grouping == "standard":
        label_map = FAULT_CLASSES if args.task == "prediction" else WINDOW_CLASSES
        # (strategy name, tag to read, grouping to collapse the stored labels onto)
        wanted = [("full", tag, None)]
    else:
        label_map = grouping_label_map(args.class_grouping)
        wanted = [
            ("native", tag, None),
            ("collapse", standard_tag, args.class_grouping),
        ]

    print(f"Evaluation — {tag}")
    strategies = {}
    written = []
    for name, source_tag, collapse in wanted:
        preds = load_predictions(run.metrics, source_tag, collapse)
        if preds is None:
            print(
                f"  Strategy '{name}' skipped: {run.metrics / f'{source_tag}_eval.parquet'} "
                f"not found. Train it first:\n"
                f"    uv run scripts/02_train_val_test.py --model {args.model} "
                f"--task {args.task} --cv-group {args.cv_group} --eval {args.eval} "
                f"--class-grouping {'standard' if collapse else args.class_grouping}"
            )
            continue
        strategies[name] = {"source_run": source_tag, **score(preds, label_map)}

        artifact = tag if args.class_grouping == "standard" else f"{tag}_{name}"
        matrix_path = run.figures / f"{artifact}_confusion_matrix.png"
        plot_confusion_matrix(
            preds["y_true"].to_numpy(),
            preds["y_pred"].to_numpy(),
            label_map,
            title=f"Confusion matrix — {artifact} (row-normalized)",
            out_path=matrix_path,
        )
        written.append(matrix_path)

    if not strategies:
        sys.exit("Nothing to evaluate: no run of this configuration has been trained.")

    metrics = {
        "tag": tag,
        "class_grouping": args.class_grouping,
        "evaluation": args.eval,
        "cv_group": args.cv_group,
        "strategies": strategies,
    }
    for name, block in strategies.items():
        report(name, block, label_map, args.cv_group)

    if len(strategies) > 1:
        print("\n  Both strategies on the same held-out windows (F1-macro | accuracy):")
        for name, block in strategies.items():
            g = block["global"]
            print(
                f"    {name:<9} {g['f1_macro']:.4f} | {g['accuracy']:.4f}  (from {block['source_run']})"
            )
        gaps = {name: block["global"]["f1_macro"] for name, block in strategies.items()}
        best = max(gaps, key=gaps.get)
        other = next(n for n in gaps if n != best)
        print(
            f"    -> '{best}' leads by {gaps[best] - gaps[other]:+.4f} F1-macro; "
            "the two trained on different label sets, not on different data."
        )

    metrics_path = run.metrics / f"{tag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    written.append(metrics_path)
    print(f"\n  Saved: {metrics_path}")

    headline = {
        "class_grouping": args.class_grouping,
        "eval": args.eval,
        "cv_group": args.cv_group,
        **{
            f"{name}_f1_macro": round(block["global"]["f1_macro"], 4)
            for name, block in strategies.items()
        },
    }
    record_stage(run, "03_evaluate", tag, started, written, **headline)
    append_index(run, "03_evaluate", tag, headline)


if __name__ == "__main__":
    main()
