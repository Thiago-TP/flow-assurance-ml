"""Stage 5 — Compact decision tree on the top SHAP features.

Distills the ensemble's knowledge into a single interpretable tree: takes the
top-N features from the SHAP ranking written by the interpretation stage,
selects the tree depth with grouped out-of-fold validation on train+val only,
refits the chosen depth on train+val, and scores it once on the same seeded
grouped test set stage 2's holdout evaluation uses — so the tree and the
ensemble are judged on identical held-out data, and the depth sweep never
scores itself. The tree is exported as a figure and as plain-text rules.

With ``--class-grouping hydrate`` the tree is judged on the coarser triage question —
Normal / Other Problem / Hydrate — using two strategies scored against the same
grouped truth, so their numbers are directly comparable:

- ``collapse`` — train on the full class set (as usual), then collapse the
  test predictions into the three groups. Answers "how well does the
  existing tree already serve the triage question?"
- ``native`` — train directly on the three grouped labels, spending the whole
  depth budget on the distinction that matters.

``--class-grouping custom`` works the same way on the user-defined
``CUSTOM_CLASS_GROUPING`` from ``config.py``.

Usage
-----
    uv run scripts/05_decision_tree.py [--model {rf,xgb}] [--task {prediction,detection}]
                                       [--class-grouping {none,hydrate,custom}]
                                       [--eval {holdout,nested}]
                                       [--cv-group {instance_id,well_id}] [--no-normalization]
                                       [--top-n N] [--depths 2,3,4,5,6]

``--model`` selects whose SHAP ranking to distill, not the tree itself.
``--eval`` selects which stage-2 run's ranking to read; the tree itself is
always evaluated on the seeded grouped holdout split.

Outputs (dtag = dt_<task>_<norm>_from_<model>, plus _wellcv with well-level CV,
_nested when reading a nested run's ranking, and _<class-grouping> when
grouping; artifacts are strategy-suffixed when a grouping runs both strategies)
--------------------------------------------------------------------------------
    results/models/<dtag>.joblib             imputer+tree pipeline fit on train+val
    results/metrics/<dtag>_metrics.json      validation depth sweep + test report
    results/metrics/<dtag>_rules.txt         the tree as if/else rules
    results/metrics/<dtag>_eval.parquet      held-out test predictions
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

from flowml.cli import add_class_grouping_arg, run_parser, run_tag
from flowml.config import (
    FAULT_CLASSES,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    N_SPLITS_CV,
    RANDOM_STATE,
    WINDOW_CLASSES,
    norm_suffix,
)
from flowml.evaluation import global_metrics, per_class_metrics, plot_confusion_matrix
from flowml.train_val_test import (
    group_labels,
    grouping_label_map,
    holdout_split,
    load_task_data,
)


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


def val_predictions(max_depth: int, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Compute grouped out-of-fold validation predictions for one tree depth.

    Used only to *select* the depth on train+val; the selected depth is then
    scored on the held-out test set, never on these folds.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    X : np.ndarray
        Train+val feature matrix restricted to the selected top features.
    y : np.ndarray
        Train+val labels (decision trees accept non-contiguous integers).
    groups : np.ndarray
        Group key per row for GroupKFold.

    Returns
    -------
    np.ndarray
        Out-of-fold validation predictions aligned with ``y``.
    """
    y_pred = np.empty_like(y)
    for train_idx, val_idx in GroupKFold(n_splits=N_SPLITS_CV).split(X, y, groups):
        pipe = make_tree_pipeline(max_depth)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred[val_idx] = pipe.predict(X[val_idx])
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
    """Sweep tree depths on train+val for one training strategy.

    Every strategy is scored against the same ``y_eval``, so strategies that
    train on different label sets stay directly comparable. The sweep scores
    are validation scores — the winning depth is evaluated on the held-out
    test set afterwards.

    Parameters
    ----------
    name : str
        Strategy name, used in log lines.
    depths : list[int]
        Tree depths to sweep.
    X : np.ndarray
        Train+val feature matrix restricted to the selected top features.
    y_train : np.ndarray
        Train+val labels the trees are fit on.
    y_eval : np.ndarray
        Train+val labels the validation predictions are scored against.
    groups : np.ndarray
        Group key per row for GroupKFold.
    grouping : str
        Grouping used to collapse predictions when ``collapse_predictions``.
    collapse_predictions : bool
        Whether predictions must be collapsed into ``y_eval``'s label space
        before scoring — true when the trees train on the finer label set.

    Returns
    -------
    dict
        ``sweep`` (depth -> validation F1-macro) and ``best_depth``.
    """
    print(f"\n  Strategy '{name}' — depth sweep (validation):")
    sweep: dict[int, float] = {}

    for depth in depths:
        y_pred = val_predictions(depth, X, y_train, groups)
        if collapse_predictions:
            y_pred = group_labels(y_pred, grouping)
        sweep[depth] = round(float(f1_score(y_eval, y_pred, average="macro", zero_division=0)), 4)
        print(f"    depth={depth}: val F1-macro = {sweep[depth]:.4f}")

    best_depth = max(sweep, key=sweep.get)
    print(f"    best depth: {best_depth} (val F1-macro {sweep[best_depth]:.4f})")
    return {"sweep": sweep, "best_depth": best_depth}


def export_tree(
    pipe: Pipeline,
    best_depth: int,
    top_features: list[str],
    label_map: dict[int, str],
    name: str,
) -> None:
    """Export a fitted tree pipeline as model, rules, and figure.

    Parameters
    ----------
    pipe : Pipeline
        Fitted imputer + decision-tree pipeline.
    best_depth : int
        Depth selected by the sweep (figure title/size).
    top_features : list[str]
        Feature names the tree was fit on.
    label_map : dict[int, str]
        Human-readable name per label the tree can output.
    name : str
        Base name for the exported artifacts.
    """
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
    add_class_grouping_arg(parser)
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

    normalized = not args.no_normalization
    tag = run_tag(args.model, args.task, normalized, args.cv_group, args.eval)
    dtag = f"dt_{args.task}_{norm_suffix(normalized)}_from_{args.model}"
    if args.cv_group == "well_id":
        dtag = f"{dtag}_wellcv"
    if args.eval == "nested":
        dtag = f"{dtag}_nested"
    if args.class_grouping != "none":
        dtag = f"{dtag}_{args.class_grouping}"

    importance_path = METRICS_DIR / f"{tag}_importance.json"
    if not importance_path.exists():
        sys.exit(
            f"{importance_path} not found. Run the interpretation stage first:\n"
            f"  uv run scripts/04_interpret.py --model {args.model} "
            f"--task {args.task}"
        )
    with open(importance_path, encoding="utf-8") as f:
        shap_scores = json.load(f)["rankings"]["shap"]
    top_features = (
        pd.Series(shap_scores).sort_values(ascending=False).head(args.top_n).index.tolist()
    )

    print(f"Decision tree — {dtag}")
    print(f"  Top {args.top_n} SHAP features ({args.model}): {top_features}")

    data = load_task_data(args.task, normalized, args.cv_group)
    col_idx = [data.feature_cols.index(f) for f in top_features]
    X = data.X[:, col_idx]

    trainval_idx, test_idx = holdout_split(data)
    X_tv, X_test = X[trainval_idx], X[test_idx]
    groups_tv = data.groups[trainval_idx]
    print(
        f"  Holdout split: {pd.Series(groups_tv).nunique()} train+val groups | "
        f"{pd.Series(data.groups[test_idx]).nunique()} test groups"
    )

    if args.class_grouping == "none":
        label_map = FAULT_CLASSES if args.task == "prediction" else WINDOW_CLASSES
        y_eval = data.y
        strategies = [("full", data.y, False)]
    else:
        label_map = grouping_label_map(args.class_grouping)
        y_eval = group_labels(data.y, args.class_grouping)
        strategies = [("collapse", data.y, True), ("native", y_eval, False)]
        print("\n  Grouped class distribution:")
        for cls, n in pd.Series(y_eval).value_counts().sort_index().items():
            share = 100 * n / len(y_eval)
            print(f"    {cls} {label_map[cls]:<16}: {n:>8,} ({share:.1f}%)")

    print(f"\n[1/2] Depth sweeps on train+val (GroupKFold({N_SPLITS_CV}) validation)...")
    results = {
        name: run_strategy(
            name,
            depths,
            X_tv,
            y_train[trainval_idx],
            y_eval[trainval_idx],
            groups_tv,
            args.class_grouping,
            collapse,
        )
        for name, y_train, collapse in strategies
    }

    print("\n[2/2] Refitting best trees on train+val and scoring on the test set...")
    metrics = {
        "tag": dtag,
        "source_ranking": tag,
        "class_grouping": args.class_grouping,
        "top_features": top_features,
        "strategies": {},
    }

    for name, y_train, collapse in strategies:
        result = results[name]
        artifact = dtag if args.class_grouping == "none" else f"{dtag}_{name}"

        pipe = make_tree_pipeline(result["best_depth"])
        pipe.fit(X_tv, y_train[trainval_idx])
        y_pred_test = pipe.predict(X_test)
        if collapse:
            y_pred_test = group_labels(y_pred_test, args.class_grouping)
        y_true_test = y_eval[test_idx]

        export_tree(pipe, result["best_depth"], top_features, label_map, artifact)

        pd.DataFrame(
            {
                "group": data.groups[test_idx],
                "fold": 1,
                "y_true": y_true_test,
                "y_pred": y_pred_test,
            }
        ).to_parquet(METRICS_DIR / f"{artifact}_eval.parquet", index=False)

        plot_confusion_matrix(
            y_true_test,
            y_pred_test,
            label_map,
            title=(
                f"Confusion matrix — {artifact}, depth {result['best_depth']} "
                "(held-out test, row-normalized)"
            ),
            out_path=FIGURES_DIR / f"{artifact}_confusion_matrix.png",
        )

        metrics["strategies"][name] = {
            "depth_sweep_val_f1_macro": result["sweep"],
            "best_depth": result["best_depth"],
            "global": global_metrics(y_true_test, y_pred_test),
            "per_class": per_class_metrics(y_true_test, y_pred_test, label_map),
        }

    metrics_path = METRICS_DIR / f"{dtag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {metrics_path}")

    print("\nSummary (held-out test, all strategies scored on the same labels):")
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
