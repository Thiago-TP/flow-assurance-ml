"""Stage 5 — Compact decision tree on the top SHAP features.

Distills the ensemble's knowledge into a single interpretable tree: takes the
top-N features from the SHAP ranking written by the interpretation stage,
sweeps tree depths with grouped out-of-fold validation, refits the best depth
on all data, and exports the tree as a figure and as plain-text rules.

With ``--grouping hydrate`` the tree is judged on the coarser triage question —
Normal / Other Problem / Hydrate — using two strategies scored against the same
grouped truth, so their numbers are directly comparable:

- ``collapse`` — train on the full class set (as usual), then collapse the
  out-of-fold predictions into the three groups. Answers "how well does the
  existing tree already serve the triage question?"
- ``native`` — train directly on the three grouped labels, spending the whole
  depth budget on the distinction that matters.

Usage
-----
    uv run scripts/05_decision_tree.py [--model {rf,xgb}] [--task {prediction,detection}]
                                       [--filter {gaussian,statistical,none}]
                                       [--grouping {none,hydrate}]
                                       [--top-n N] [--depths 2,3,4,5,6]

``--model`` selects whose SHAP ranking to distill, not the tree itself.

Outputs (dtag = dt_<task>_<filter>_from_<model>, plus _<grouping> when grouping;
artifacts are strategy-suffixed when a grouping runs both strategies)
--------------------------------------------------------------------------------
    results/models/<dtag>.joblib             fitted imputer+tree pipeline
    results/metrics/<dtag>_metrics.json      per-depth OOF F1 + best-depth report
    results/metrics/<dtag>_rules.txt         the tree as if/else rules
    results/metrics/<dtag>_oof.parquet       out-of-fold predictions
    results/figures/<dtag>_tree.png
    results/figures/<dtag>_confusion_matrix.png
"""

import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from flowml.cli import add_grouping_arg, run_parser, run_tag
from flowml.config import (
    FAULT_CLASSES,
    FIGURES_DIR,
    HYDRATE_GROUP_NAMES,
    METRICS_DIR,
    MODELS_DIR,
    N_SPLITS_CV,
    RANDOM_STATE,
    WINDOW_CLASSES,
)
from flowml.evaluation import global_metrics, per_class_metrics, plot_confusion_matrix
from flowml.training import group_labels, load_task_data


def make_tree_pipeline(max_depth: int) -> Pipeline:
    """Build the imputer + decision-tree pipeline for one depth.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.

    Returns
    -------
    Pipeline
        Median imputer followed by a class-balanced decision tree.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def oof_predictions(
    max_depth: int, X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Compute grouped out-of-fold predictions for one tree depth.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    X : np.ndarray
        Feature matrix restricted to the selected top features.
    y : np.ndarray
        Training labels (decision trees accept non-contiguous integers).
    groups : np.ndarray
        ``instance_id`` per row for GroupKFold.

    Returns
    -------
    np.ndarray
        Out-of-fold predictions aligned with ``y``.
    """
    y_pred = np.empty_like(y)
    for train_idx, test_idx in GroupKFold(n_splits=N_SPLITS_CV).split(X, y, groups):
        pipe = make_tree_pipeline(max_depth)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = pipe.predict(X[test_idx])
    return y_pred


def run_strategy(
    name: str,
    depths: list[int],
    X: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    groups: np.ndarray,
    grouping: str,
    collapse_predictions: bool,
) -> dict:
    """Sweep tree depths for one training strategy and keep the best.

    Every strategy is scored against the same ``y_eval``, so strategies that
    train on different label sets stay directly comparable.

    Parameters
    ----------
    name : str
        Strategy name, used in log lines.
    depths : list[int]
        Tree depths to sweep.
    X : np.ndarray
        Feature matrix restricted to the selected top features.
    y_train : np.ndarray
        Labels the trees are fit on.
    y_eval : np.ndarray
        Labels the predictions are scored against.
    groups : np.ndarray
        ``instance_id`` per row for GroupKFold.
    grouping : str
        Grouping used to collapse predictions when ``collapse_predictions``.
    collapse_predictions : bool
        Whether predictions must be collapsed into ``y_eval``'s label space
        before scoring — true when the trees train on the finer label set.

    Returns
    -------
    dict
        ``sweep`` (depth -> OOF F1-macro), ``best_depth``, and ``y_pred``
        (the best depth's out-of-fold predictions in ``y_eval``'s space).
    """
    print(f"\n  Strategy '{name}' — depth sweep:")
    sweep: dict[int, float] = {}
    predictions: dict[int, np.ndarray] = {}

    for depth in depths:
        y_pred = oof_predictions(depth, X, y_train, groups)
        if collapse_predictions:
            y_pred = group_labels(y_pred, grouping)
        sweep[depth] = round(
            float(f1_score(y_eval, y_pred, average="macro", zero_division=0)), 4
        )
        predictions[depth] = y_pred
        print(f"    depth={depth}: OOF F1-macro = {sweep[depth]:.4f}")

    best_depth = max(sweep, key=sweep.get)
    print(f"    best depth: {best_depth} (F1-macro {sweep[best_depth]:.4f})")
    return {
        "sweep": sweep,
        "best_depth": best_depth,
        "y_pred": predictions[best_depth],
    }


def export_tree(
    X: np.ndarray,
    y_train: np.ndarray,
    best_depth: int,
    top_features: list[str],
    label_map: dict[int, str],
    name: str,
) -> None:
    """Refit the best tree on all data and export model, rules, and figure.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix restricted to the selected top features.
    y_train : np.ndarray
        Labels the tree is fit on.
    best_depth : int
        Depth selected by the sweep.
    top_features : list[str]
        Feature names aligned with ``X``.
    label_map : dict[int, str]
        Human-readable name per label the tree can output.
    name : str
        Base name for the exported artifacts.
    """
    pipe = make_tree_pipeline(best_depth)
    pipe.fit(X, y_train)
    tree = pipe.named_steps["clf"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")

    class_names = [label_map.get(c, str(c)) for c in tree.classes_]
    rules_path = METRICS_DIR / f"{name}_rules.txt"
    rules_path.write_text(
        export_text(tree, feature_names=top_features, class_names=class_names),
        encoding="utf-8",
    )
    print(f"  Saved: {rules_path}")

    fig, ax = plt.subplots(figsize=(max(14, 2.2**best_depth), 2.5 * best_depth + 3))
    plot_tree(
        tree,
        feature_names=top_features,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Decision tree (depth {best_depth}) — {name}", fontsize=13, pad=10)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tree_path = FIGURES_DIR / f"{name}_tree.png"
    plt.savefig(tree_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {tree_path}")


def main() -> None:
    """Select top SHAP features, sweep depths per strategy, and export trees."""
    parser = run_parser(__doc__.splitlines()[0])
    add_grouping_arg(parser)
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="number of top SHAP features to keep (default: 10)",
    )
    parser.add_argument(
        "--depths",
        default="2,3,4,5,6",
        help="comma-separated tree depths to sweep (default: 2,3,4,5,6)",
    )
    args = parser.parse_args()
    depths = [int(d) for d in args.depths.split(",")]

    tag = run_tag(args.model, args.task, args.filter_type)
    dtag = f"dt_{args.task}_{args.filter_type}_from_{args.model}"
    if args.grouping != "none":
        dtag = f"{dtag}_{args.grouping}"

    importance_path = METRICS_DIR / f"{tag}_importance.json"
    if not importance_path.exists():
        sys.exit(
            f"{importance_path} not found. Run the interpretation stage first:\n"
            f"  uv run scripts/04_interpret.py --model {args.model} "
            f"--task {args.task} --filter {args.filter_type}"
        )
    with open(importance_path, encoding="utf-8") as f:
        shap_scores = json.load(f)["rankings"]["shap"]
    top_features = (
        pd.Series(shap_scores)
        .sort_values(ascending=False)
        .head(args.top_n)
        .index.tolist()
    )

    print(f"Decision tree — {dtag}")
    print(f"  Top {args.top_n} SHAP features ({args.model}): {top_features}")

    data = load_task_data(args.filter_type, args.task)
    col_idx = [data.feature_cols.index(f) for f in top_features]
    X = data.X[:, col_idx]

    if args.grouping == "none":
        label_map = FAULT_CLASSES if args.task == "prediction" else WINDOW_CLASSES
        y_eval = data.y
        strategies = [("full", data.y, False)]
    else:
        label_map = HYDRATE_GROUP_NAMES
        y_eval = group_labels(data.y, args.grouping)
        strategies = [("collapse", data.y, True), ("native", y_eval, False)]
        print("\n  Grouped class distribution:")
        for cls, n in pd.Series(y_eval).value_counts().sort_index().items():
            share = 100 * n / len(y_eval)
            print(f"    {cls} {label_map[cls]:<16}: {n:>8,} ({share:.1f}%)")

    print(f"\n[1/2] Depth sweeps with GroupKFold({N_SPLITS_CV}) OOF...")
    results = {
        name: run_strategy(
            name, depths, X, y_train, y_eval, data.groups, args.grouping, collapse
        )
        for name, y_train, collapse in strategies
    }

    print("\n[2/2] Refitting best trees and exporting...")
    metrics = {
        "tag": dtag,
        "source_ranking": tag,
        "grouping": args.grouping,
        "top_features": top_features,
        "strategies": {},
    }

    for name, y_train, _collapse in strategies:
        result = results[name]
        artifact = dtag if args.grouping == "none" else f"{dtag}_{name}"

        export_tree(X, y_train, result["best_depth"], top_features, label_map, artifact)

        pd.DataFrame(
            {
                "instance_id": data.groups,
                "y_true": y_eval,
                "y_pred": result["y_pred"],
            }
        ).to_parquet(METRICS_DIR / f"{artifact}_oof.parquet", index=False)

        plot_confusion_matrix(
            y_eval,
            result["y_pred"],
            label_map,
            title=(
                f"Confusion matrix — {artifact}, "
                f"depth {result['best_depth']} (row-normalized)"
            ),
            out_path=FIGURES_DIR / f"{artifact}_confusion_matrix.png",
        )

        metrics["strategies"][name] = {
            "depth_sweep_oof_f1_macro": result["sweep"],
            "best_depth": result["best_depth"],
            "global": global_metrics(y_eval, result["y_pred"]),
            "per_class": per_class_metrics(y_eval, result["y_pred"], label_map),
        }

    metrics_path = METRICS_DIR / f"{dtag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {metrics_path}")

    print("\nSummary (out-of-fold, all strategies scored on the same labels):")
    for name, block in metrics["strategies"].items():
        g = block["global"]
        print(
            f"  {name:<9} depth {block['best_depth']}: "
            f"F1-macro {g['f1_macro']:.4f} | accuracy {g['accuracy']:.4f}"
        )
        for row in block["per_class"].values():
            print(
                f"      {row['name']:<16} precision {row['precision']:.3f} | "
                f"recall {row['recall']:.3f} | F1 {row['f1']:.3f} | n {row['support']:,}"
            )


if __name__ == "__main__":
    main()
