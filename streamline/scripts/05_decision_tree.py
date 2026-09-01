"""Stage 5 — Compact decision tree on the top SHAP features.

Distills the ensemble's knowledge into a single interpretable tree: takes the
top-N features from the SHAP ranking written by the interpretation stage,
sweeps tree depths with grouped out-of-fold validation, refits the best depth
on all data, and exports the tree as a figure and as plain-text rules.

Usage
-----
    uv run scripts/05_decision_tree.py [--model {rf,xgb}] [--task {prediction,detection}]
                                       [--filter {gaussian,statistical,none}]
                                       [--top-n N] [--depths 2,3,4,5,6]

``--model`` selects whose SHAP ranking to distill, not the tree itself.

Outputs (dtag = dt_<task>_<filter>_from_<model>)
------------------------------------------------
    results/models/<dtag>.joblib             fitted imputer+tree pipeline
    results/metrics/<dtag>_metrics.json      per-depth OOF F1 + best-depth report
    results/metrics/<dtag>_rules.txt         the tree as if/else rules
    results/figures/<dtag>_tree.png
    results/figures/<dtag>_confusion_matrix.png
"""

import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flowml.cli import run_parser, run_tag
from flowml.config import (
    FAULT_CLASSES,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    N_SPLITS_CV,
    RANDOM_STATE,
    WINDOW_CLASSES,
)
from flowml.evaluation import global_metrics, per_class_metrics, plot_confusion_matrix
from flowml.training import load_task_data
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


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


def oof_f1_for_depth(
    max_depth: int, X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[float, np.ndarray]:
    """Compute grouped out-of-fold predictions for one tree depth.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    X : np.ndarray
        Feature matrix restricted to the selected top features.
    y : np.ndarray
        Labels (decision trees accept non-contiguous integers directly).
    groups : np.ndarray
        ``instance_id`` per row for GroupKFold.

    Returns
    -------
    (float, np.ndarray)
        OOF F1-macro and the OOF prediction vector aligned with ``y``.
    """
    from sklearn.metrics import f1_score

    y_pred = np.empty_like(y)
    for train_idx, test_idx in GroupKFold(n_splits=N_SPLITS_CV).split(X, y, groups):
        pipe = make_tree_pipeline(max_depth)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = pipe.predict(X[test_idx])
    return f1_score(y, y_pred, average="macro", zero_division=0), y_pred


def main() -> None:
    """Select top SHAP features, sweep depths, and export the best tree."""
    parser = run_parser(__doc__.splitlines()[0])
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
    label_map = FAULT_CLASSES if args.task == "prediction" else WINDOW_CLASSES

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

    print(f"\n[1/2] Depth sweep with GroupKFold({N_SPLITS_CV}) OOF...")
    sweep: dict[int, float] = {}
    predictions: dict[int, np.ndarray] = {}
    for depth in depths:
        f1, y_pred = oof_f1_for_depth(depth, X, data.y, data.groups)
        sweep[depth] = round(float(f1), 4)
        predictions[depth] = y_pred
        print(f"  depth={depth}: OOF F1-macro = {f1:.4f}")

    best_depth = max(sweep, key=sweep.get)
    y_pred_best = predictions[best_depth]
    print(f"  Best depth: {best_depth} (F1-macro {sweep[best_depth]:.4f})")

    print("\n[2/2] Refitting best tree on all data and exporting...")
    pipe = make_tree_pipeline(best_depth)
    pipe.fit(X, data.y)
    tree = pipe.named_steps["clf"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / f"{dtag}.joblib")

    class_names = [label_map.get(c, str(c)) for c in tree.classes_]
    rules = export_text(tree, feature_names=top_features, class_names=class_names)
    rules_path = METRICS_DIR / f"{dtag}_rules.txt"
    rules_path.write_text(rules, encoding="utf-8")
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
    ax.set_title(f"Decision tree (depth {best_depth}) — {dtag}", fontsize=13, pad=10)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tree_path = FIGURES_DIR / f"{dtag}_tree.png"
    plt.savefig(tree_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {tree_path}")

    plot_confusion_matrix(
        data.y,
        y_pred_best,
        label_map,
        title=f"Confusion matrix — {dtag}, depth {best_depth} (row-normalized)",
        out_path=FIGURES_DIR / f"{dtag}_confusion_matrix.png",
    )

    metrics = {
        "tag": dtag,
        "source_ranking": tag,
        "top_features": top_features,
        "depth_sweep_oof_f1_macro": sweep,
        "best_depth": best_depth,
        "global": global_metrics(data.y, y_pred_best),
        "per_class": per_class_metrics(data.y, y_pred_best, label_map),
    }
    metrics_path = METRICS_DIR / f"{dtag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {metrics_path}")


if __name__ == "__main__":
    main()
