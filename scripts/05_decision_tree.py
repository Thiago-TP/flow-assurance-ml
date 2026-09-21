"""Stage 5 — Compact decision tree on the top SHAP features.

Distills the ensemble's knowledge into a single interpretable tree: takes the
top-N features from the SHAP ranking written by the interpretation stage,
selects the tree depth with grouped out-of-fold validation on training data
only, refits the chosen depth and scores it on data the sweep never saw. The
tree follows the run's ``--eval`` protocol, so it is judged on exactly the
held-out data the ensemble was:

- ``holdout`` — depth sweep on train+val, one refit on train+val, one score on
  the seeded grouped test set stage 2 held out;
- ``nested`` / ``leave-one-out`` — the seeded outer folds of stage 2: every
  outer fold runs its own depth sweep on its training part, refits the winner
  there and predicts its held-out part; the pooled out-of-fold predictions are
  scored, and with one group per fold every group gets a score of its own.
  The exported tree is then the winner of a final sweep on all data — the
  deployment tree, with a validation score only — just as stage 2 saves the
  model of its final search.

The tree is exported as a figure and as plain-text rules.

With ``--class-grouping hydrate`` the tree is judged on the coarser triage question —
Normal / Other Problem / Hydrate — using two strategies scored against the same
grouped truth, so their numbers are directly comparable:

- ``collapse`` — train on the full class set (as usual), then collapse the
  held-out predictions into the three groups. Answers "how well does the
  existing tree already serve the triage question?" Its exported rules and
  figure name the fine classes the tree actually predicts.
- ``native`` — train directly on the three grouped labels, spending the whole
  depth budget on the distinction that matters.

**Each strategy distills the ranking of the ensemble it corresponds to**: the
standard run's for ``collapse``, which trains on the same classes that
ensemble did, and the grouped run's for ``native``. So the two trees usually
start from different top-N features, and a native tree is no longer handicapped
by features selected for a question it is not asked. Running stages 2 and 4
with ``--class-grouping`` is what produces that second ranking; without it the
stage says so and stops.

``--class-grouping custom`` works the same way on the user-defined
``CUSTOM_CLASS_GROUPING`` from ``config.py``.

Usage
-----
    uv run scripts/05_decision_tree.py [--model {rf,xgb}] [--task {prediction,detection}]
                                       [--class-grouping {standard,hydrate,custom}]
                                       [--eval {holdout,nested,leave-one-out}]
                                       [--cv-group {instance_id,well_id}] [--normalization {none,instance,normal}]
                                       [--allow-overlap] [--top-n N]
                                       [--depths 2,3,4,5,6,7,8,9,10,11,12]

``--model`` selects whose SHAP ranking to distill, not the tree itself; it is
therefore one of ``rf`` / ``xgb``, and ``--model dt`` skips this stage, that
run being a decision tree already.

The rankings are read from — and the trees written into — the run directory
stage 4 wrote them to: the one ``main.py`` names in ``FLOWML_RUN_DIR``, the one
``--run`` points at, or otherwise the newest run holding this configuration's
ranking (see the ``runs`` module).

Outputs, inside the run directory (dtag = dt_<task>_<norm>[_overlap]_from_<model>,
plus _wellcv with well-level CV, _nested / _loo with those protocols, and
_<class-grouping> when grouping; artifacts are strategy-suffixed when a
grouping runs both strategies)
--------------------------------------------------------------------------------
    models/<dtag>.joblib             imputer+tree pipeline (the exported tree)
    metrics/<dtag>_metrics.json      depth sweeps + held-out report
    metrics/<dtag>_rules.txt         the tree as if/else rules
    metrics/<dtag>_eval.parquet      held-out predictions
    figures/<dtag>_tree.png          (+ .pdf when the canvas exceeds the raster cap)
    figures/<dtag>_confusion_matrix.png
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from flowml.cli import (
    add_class_grouping_arg,
    add_normalization_arg,
    add_run_arg,
    eval_suffix,
    grouping_suffix,
    run_parser,
    run_tag,
    skip_if_white_box,
)
from flowml.config import (
    N_SPLITS_CV,
    RANDOM_STATE,
    extreme_suffix,
    norm_suffix,
    overlap_suffix,
)
from flowml.evaluation import (
    global_metrics,
    per_class_metrics,
    per_fold_metrics,
    per_group_metrics,
    plot_confusion_matrix,
)
from flowml.interpretation import export_tree
from flowml.runs import append_index, record_stage, resolve_run
from flowml.train_val_test import (
    TaskData,
    coverage_folds,
    group_labels,
    group_sort_key,
    held_out_summary,
    holdout_split,
    load_task_data,
    outer_folds,
    split_composition,
)


@dataclass
class Strategy:
    """One way of reaching the labels the tree is scored on.

    Attributes
    ----------
    name : str
        ``"full"`` (standard grouping), ``"collapse"`` or ``"native"``.
    y_train : np.ndarray
        Labels the trees are fit on, for every row of the dataset.
    collapse : bool
        Whether predictions must be collapsed onto the scored label space
        before scoring — true when the trees train on the finer label set.
    source_ranking : str
        Tag of the stage-4 ranking the features came from: the run whose
        label set this strategy's trees are fit on.
    features : list[str]
        The top-N features of that ranking.
    X : np.ndarray
        Feature matrix restricted to ``features``, for every row.
    """

    name: str
    y_train: np.ndarray
    collapse: bool
    source_ranking: str
    features: list[str]
    X: np.ndarray


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

    Used only to *select* the depth on training data; the selected depth is
    then scored on held-out data, never on these folds. The folds are the
    class-coverage-repaired grouped folds of ``train_val_test.coverage_folds``
    — the same construction the ensemble search uses — so a class pinned to
    training there is pinned here too and keeps its out-of-fold prediction
    unset; such rows are excluded from the sweep score.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    X : np.ndarray
        Training feature matrix restricted to the selected top features.
    y : np.ndarray
        Training labels (decision trees accept non-contiguous integers).
    groups : np.ndarray
        Group key per row of the grouped folds.

    Returns
    -------
    np.ndarray
        Out-of-fold validation predictions aligned with ``y``; ``-1`` where a
        row was never held out.
    """
    y_pred = np.full_like(y, -1)
    folds = coverage_folds(y, groups, N_SPLITS_CV, np.random.default_rng(RANDOM_STATE), {})
    for train_idx, val_idx in folds:
        pipe = make_tree_pipeline(max_depth)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred[val_idx] = pipe.predict(X[val_idx])
    return y_pred


def sweep_depths(
    strategy: Strategy,
    depths: list[int],
    y_eval: np.ndarray,
    groups: np.ndarray,
    grouping: str,
    rows: np.ndarray,
    log: bool = True,
) -> tuple[dict[int, float], int]:
    """Sweep tree depths with grouped out-of-fold validation on ``rows``.

    Every strategy is scored against the same ``y_eval``, so strategies that
    train on different label sets stay directly comparable. The sweep scores
    are validation scores — the winning depth is refit and scored on held-out
    data afterwards. Rows a pinned group keeps in training throughout (see
    ``val_predictions``) have no validation prediction and are left out of
    the score.

    Parameters
    ----------
    strategy : Strategy
        Labels to fit on, features to fit them on, and whether to collapse
        the predictions.
    depths : list[int]
        Tree depths to sweep.
    y_eval : np.ndarray
        Labels the validation predictions are scored against, whole dataset.
    groups : np.ndarray
        Group key per row, whole dataset.
    grouping : str
        Grouping used to collapse predictions when the strategy asks for it.
    rows : np.ndarray
        Row indices the sweep is confined to (train+val, or the training
        part of an outer fold).
    log : bool
        Print one line per depth (default on; off inside outer folds, which
        print one line per strategy instead).

    Returns
    -------
    (dict[int, float], int)
        Validation F1-macro per depth, and the best depth.
    """
    sweep: dict[int, float] = {}
    for depth in depths:
        y_pred = val_predictions(depth, strategy.X[rows], strategy.y_train[rows], groups[rows])
        scored = y_pred != -1
        if strategy.collapse:
            y_pred[scored] = group_labels(y_pred[scored], grouping)
        sweep[depth] = round(
            float(f1_score(y_eval[rows][scored], y_pred[scored], average="macro", zero_division=0)),
            4,
        )
        if log:
            print(f"    depth={depth}: val F1-macro = {sweep[depth]:.4f}")

    best_depth = max(sweep, key=sweep.get)
    if log:
        print(f"    best depth: {best_depth} (val F1-macro {sweep[best_depth]:.4f})")
    return sweep, best_depth


def fit_predict(
    strategy: Strategy,
    depth: int,
    fit_rows: np.ndarray,
    predict_rows: np.ndarray,
    grouping: str,
) -> tuple[Pipeline, np.ndarray]:
    """Fit one tree on ``fit_rows`` and predict ``predict_rows`` in the scored label space.

    Parameters
    ----------
    strategy : Strategy
        Labels to fit on, features to fit them on, and whether to collapse
        the predictions.
    depth : int
        Tree depth.
    fit_rows, predict_rows : np.ndarray
        Row indices to fit on and to predict.
    grouping : str
        Grouping used to collapse predictions when the strategy asks for it.

    Returns
    -------
    (Pipeline, np.ndarray)
        The fitted pipeline and its predictions for ``predict_rows``.
    """
    pipe = make_tree_pipeline(depth)
    pipe.fit(strategy.X[fit_rows], strategy.y_train[fit_rows])
    y_pred = pipe.predict(strategy.X[predict_rows])
    if strategy.collapse:
        y_pred = group_labels(y_pred, grouping)
    return pipe, y_pred


def eval_table(
    groups: np.ndarray, fold: int, y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """One block of held-out predictions in the layout stage 2 writes."""
    return pd.DataFrame({"group": groups, "fold": fold, "y_true": y_true, "y_pred": y_pred})


def holdout_protocol(
    strategies: list[Strategy],
    depths: list[int],
    y_eval: np.ndarray,
    data: TaskData,
    grouping: str,
    verbose: bool,
) -> dict[str, dict]:
    """Sweep on train+val, refit there, score once on the seeded holdout test set.

    Parameters
    ----------
    strategies : list[Strategy]
        Strategies to run, all scored against ``y_eval``.
    depths : list[int]
        Tree depths to sweep.
    y_eval : np.ndarray
        Labels the trees are scored against, whole dataset.
    data : TaskData
        The dataset, for its groups, fine labels and holdout split.
    grouping : str
        The class grouping in force.
    verbose : bool
        Print the class-coverage repair of the split.

    Returns
    -------
    dict[str, dict]
        Per strategy name: ``sweep``, ``best_depth``, the fitted ``pipe``,
        its test ``eval_frame`` and ``outer_folds`` = ``None``.
    """
    trainval_idx, test_idx = holdout_split(data, verbose)
    print(
        f"\n[1/2] Depth sweeps on train+val "
        f"({N_SPLITS_CV} grouped folds, class coverage repaired)..."
    )
    results: dict[str, dict] = {}
    for strategy in strategies:
        print(f"\n  Strategy '{strategy.name}' (features from {strategy.source_ranking}):")
        sweep, best_depth = sweep_depths(
            strategy, depths, y_eval, data.groups, grouping, trainval_idx
        )
        results[strategy.name] = {"sweep": sweep, "best_depth": best_depth}

    print("\n[2/2] Refitting the best trees on train+val and scoring them on the test set...")
    for strategy in strategies:
        result = results[strategy.name]
        pipe, y_pred = fit_predict(strategy, result["best_depth"], trainval_idx, test_idx, grouping)
        result["pipe"] = pipe
        result["eval_frame"] = eval_table(data.groups[test_idx], 1, y_eval[test_idx], y_pred)
        result["outer_folds"] = None
    return results


def outer_fold_protocol(
    strategies: list[Strategy],
    depths: list[int],
    y_eval: np.ndarray,
    data: TaskData,
    grouping: str,
    eval_mode: str,
    verbose: bool,
) -> dict[str, dict]:
    """Sweep, refit and predict per outer fold; export the tree of a final sweep on all data.

    The folds are the seeded outer folds of stage 2 (``outer_folds``), so the
    pooled predictions cover the same rows the ensemble was scored on. Every
    fold logs what it holds out and one line per strategy; the composition
    table of each fold is printed under ``verbose``.

    Parameters
    ----------
    strategies : list[Strategy]
        Strategies to run, all scored against ``y_eval``.
    depths : list[int]
        Tree depths to sweep.
    y_eval : np.ndarray
        Labels the trees are scored against, whole dataset.
    data : TaskData
        The dataset, for its groups, fine labels and outer folds.
    grouping : str
        The class grouping in force.
    eval_mode : str
        ``"nested"`` or ``"leave-one-out"``.
    verbose : bool
        Print the class-coverage repair of the folds and each fold's
        composition table.

    Returns
    -------
    dict[str, dict]
        Per strategy name: the final ``sweep`` and ``best_depth`` on all
        data, the final fitted ``pipe``, the pooled out-of-fold
        ``eval_frame`` and one record per fold in ``outer_folds``.
    """
    folds = outer_folds(data, eval_mode, verbose)
    n_fits = len(folds) * len(depths) * N_SPLITS_CV * len(strategies)
    print(
        f"\n[1/2] Depth sweep, refit and prediction per outer fold ({len(folds)} folds x "
        f"{len(depths)} depths x {N_SPLITS_CV} inner folds x {len(strategies)} "
        f"strateg{'y' if len(strategies) == 1 else 'ies'} = {n_fits:,} tree fits)..."
    )
    parts: dict[str, list[pd.DataFrame]] = {s.name: [] for s in strategies}
    records: dict[str, list[dict]] = {s.name: [] for s in strategies}

    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        print(
            f"  Outer fold {fold}/{len(folds)} — held out "
            f"{held_out_summary(data.y, data.groups, test_idx, data.label_map)}"
        )
        if verbose:
            print(
                split_composition(
                    data.y,
                    data.groups,
                    {"train (depth sweep)": train_idx, "held out": test_idx},
                    data.label_map,
                )
            )
        held = sorted(set(np.unique(data.groups[test_idx])), key=group_sort_key)
        for strategy in strategies:
            sweep, best_depth = sweep_depths(
                strategy, depths, y_eval, data.groups, grouping, train_idx, log=False
            )
            _, y_pred = fit_predict(strategy, best_depth, train_idx, test_idx, grouping)
            y_true = y_eval[test_idx]
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            accuracy = float(np.mean(y_true == y_pred))
            print(
                f"    {strategy.name:<9} depth {best_depth:>2} "
                f"(val F1-macro {sweep[best_depth]:.4f}) "
                f"| test F1-macro {f1:.4f} | accuracy {accuracy:.4f}"
            )
            parts[strategy.name].append(eval_table(data.groups[test_idx], fold, y_true, y_pred))
            records[strategy.name].append(
                {
                    "fold": fold,
                    "n_held_out_groups": len(held),
                    **({"group": str(held[0])} if len(held) == 1 else {}),
                    "best_depth": best_depth,
                    "depth_sweep_val_f1_macro": sweep,
                    "val_f1_macro": sweep[best_depth],
                    "test_f1_macro": round(f1, 4),
                    "test_accuracy": round(accuracy, 4),
                    "n_test_windows": len(test_idx),
                }
            )

    print("\n[2/2] Final depth sweeps on all data (the exported trees; validation scores only)...")
    all_rows = np.arange(data.n_windows)
    results: dict[str, dict] = {}
    for strategy in strategies:
        print(f"\n  Strategy '{strategy.name}' (features from {strategy.source_ranking}):")
        sweep, best_depth = sweep_depths(strategy, depths, y_eval, data.groups, grouping, all_rows)
        pipe = make_tree_pipeline(best_depth).fit(strategy.X, strategy.y_train)
        results[strategy.name] = {
            "sweep": sweep,
            "best_depth": best_depth,
            "pipe": pipe,
            "eval_frame": pd.concat(parts[strategy.name], ignore_index=True),
            "outer_folds": records[strategy.name],
        }
    return results


def main() -> None:
    """Select top SHAP features, run the protocol per strategy, and export trees."""
    parser = run_parser(__doc__.splitlines()[0])
    add_class_grouping_arg(parser)
    add_normalization_arg(parser)
    add_run_arg(parser)
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="number of top SHAP features to keep (default: 10)",
    )
    parser.add_argument(
        "--depths",
        default="2,3,4,5,6,7,8,9,10,11,12",
        help="comma-separated tree depths to sweep (default: 2,3,4,5,6,7,8,9,10,11,12)",
    )
    args = parser.parse_args()
    skip_if_white_box(args.model, "distilling a compact tree")
    started = datetime.now().astimezone()
    depths = [int(d) for d in args.depths.split(",")]

    common = (
        args.model,
        args.task,
        args.normalization,
        args.cv_group,
        args.eval,
        args.allow_overlap,
        args.keep_extreme_values,
    )
    tag = run_tag(*common, args.class_grouping)
    standard_tag = run_tag(*common, "standard")
    dtag = (
        f"dt_{args.task}{norm_suffix(args.normalization)}{overlap_suffix(args.allow_overlap)}"
        f"{extreme_suffix(args.keep_extreme_values)}_from_{args.model}"
    )
    if args.cv_group == "well_id":
        dtag = f"{dtag}_wellcv"
    dtag += eval_suffix(args.eval) + grouping_suffix(args.class_grouping)

    # A grouped run distils two rankings and a standard one distils a single
    # ranking, so either is enough to pick the run; ``ranking_of`` says which
    # one is actually missing.
    run = resolve_run(
        args.run,
        must_contain=[
            f"metrics/{tag}_importance.json",
            f"metrics/{standard_tag}_importance.json",
        ],
    )

    data = load_task_data(
        args.task,
        args.normalization,
        args.cv_group,
        args.allow_overlap,
        args.keep_extreme_values,
        args.class_grouping,
    )

    def ranking_of(source_tag: str, grouping: str) -> tuple[list[str], np.ndarray]:
        """Top-N features of one stage-4 ranking, and the matrix restricted to them."""
        path = run.metrics / f"{source_tag}_importance.json"
        if not path.exists():
            sys.exit(
                f"{path} not found. Run stages 2 and 4 for that label set first:\n"
                f"  uv run scripts/02_train_val_test.py --model {args.model} --task {args.task} "
                f"--cv-group {args.cv_group} --eval {args.eval} --class-grouping {grouping}\n"
                f"  uv run scripts/04_interpret.py --model {args.model} --task {args.task} "
                f"--cv-group {args.cv_group} --eval {args.eval} --class-grouping {grouping}"
            )
        with open(path, encoding="utf-8") as f:
            shap_scores = json.load(f)["rankings"]["shap"]
        features = (
            pd.Series(shap_scores).sort_values(ascending=False).head(args.top_n).index.tolist()
        )
        return features, data.X[:, [data.feature_cols.index(f) for f in features]]

    fine_map = data.fine_label_map
    print(f"Decision tree — {dtag}")
    print(
        f"  Evaluation protocol: {args.eval} — the ensemble's, so tree and ensemble are "
        "judged on the same held-out data"
    )

    if args.class_grouping == "standard":
        label_map = fine_map
        y_eval = data.fine_y
        features, X = ranking_of(tag, "standard")
        strategies = [Strategy("full", data.fine_y, False, tag, features, X)]
    else:
        label_map = data.label_map
        y_eval = data.y
        # Each strategy distils the ranking of the ensemble trained on the same
        # labels it is: the standard run for 'collapse', the grouped one for
        # 'native'. Anything else hands a tree features chosen for a question
        # it is not being asked.
        collapse_features, collapse_X = ranking_of(standard_tag, "standard")
        native_features, native_X = ranking_of(tag, args.class_grouping)
        strategies = [
            Strategy("collapse", data.fine_y, True, standard_tag, collapse_features, collapse_X),
            Strategy("native", data.y, False, tag, native_features, native_X),
        ]
        print("\n  Grouped class distribution:")
        for cls, n in pd.Series(y_eval).value_counts().sort_index().items():
            share = 100 * n / len(y_eval)
            print(f"    {cls} {label_map[cls]:<16}: {n:>8,} ({share:.1f}%)")

    for strategy in strategies:
        print(f"\n  Top {args.top_n} SHAP features of {strategy.source_ranking}:")
        print(f"    {strategy.features}")

    if args.eval == "holdout":
        results = holdout_protocol(
            strategies, depths, y_eval, data, args.class_grouping, args.verbose
        )
        protocol = "held-out test"
    else:
        results = outer_fold_protocol(
            strategies, depths, y_eval, data, args.class_grouping, args.eval, args.verbose
        )
        protocol = "pooled outer folds" if args.eval == "nested" else "leave-one-out, pooled"

    print("\nWriting artifacts...")
    metrics = {
        "tag": dtag,
        "class_grouping": args.class_grouping,
        "evaluation": args.eval,
        "cv_group": args.cv_group,
        "strategies": {},
    }
    written = []

    for strategy in strategies:
        result = results[strategy.name]
        artifact = dtag if args.class_grouping == "standard" else f"{dtag}_{strategy.name}"

        # The rules and the drawing name the classes the tree itself predicts:
        # the fine classes for 'full' and 'collapse' — collapsing happens to
        # the predictions afterwards — and the groups for 'native'. Exporting
        # before the dump means the saved model is the pruned one they show.
        tree_label_map = label_map if strategy.name == "native" else fine_map
        written += export_tree(
            result["pipe"].named_steps["clf"],
            strategy.features,
            tree_label_map,
            artifact,
            metrics_dir=run.metrics,
            figures_dir=run.figures,
        )
        pipe_path = run.models / f"{artifact}.joblib"
        joblib.dump(result["pipe"], pipe_path)

        frame = result["eval_frame"]
        frame_path = run.metrics / f"{artifact}_eval.parquet"
        frame.to_parquet(frame_path, index=False)
        written += [pipe_path, frame_path]
        y_true, y_pred = frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()

        block = {
            "source_ranking": strategy.source_ranking,
            "top_features": strategy.features,
            "depth_sweep_val_f1_macro": result["sweep"],
            "best_depth": result["best_depth"],
            "global": global_metrics(y_true, y_pred),
            "per_class": per_class_metrics(y_true, y_pred, label_map),
        }
        per_group = per_group_metrics(frame)
        if per_group:
            block["per_group"] = per_group
        if result["outer_folds"] is not None:
            block["per_fold_f1_macro"] = per_fold_metrics(frame)
            block["outer_folds"] = result["outer_folds"]
        metrics["strategies"][strategy.name] = block

        if result["outer_folds"] is None:
            title = (
                f"Confusion matrix — {artifact}, depth {result['best_depth']} "
                f"({protocol}, row-normalized)"
            )
        else:
            fold_depths = [record["best_depth"] for record in result["outer_folds"]]
            title = (
                f"Confusion matrix — {artifact} ({protocol}; per-fold depths "
                f"{min(fold_depths)}-{max(fold_depths)}, row-normalized)"
            )
        matrix_path = run.figures / f"{artifact}_confusion_matrix.png"
        plot_confusion_matrix(
            y_true,
            y_pred,
            label_map,
            title=title,
            out_path=matrix_path,
        )
        written.append(matrix_path)

    metrics_path = run.metrics / f"{dtag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    written.append(metrics_path)
    print(f"  Saved: {metrics_path}")

    headline = {
        "class_grouping": args.class_grouping,
        "eval": args.eval,
        "cv_group": args.cv_group,
        "top_n": args.top_n,
    }
    for name, block in metrics["strategies"].items():
        headline[f"{name}_best_depth"] = block["best_depth"]
        headline[f"{name}_f1_macro"] = round(block["global"]["f1_macro"], 4)
    record_stage(run, "05_decision_tree", dtag, started, written, **headline)
    append_index(run, "05_decision_tree", dtag, headline)

    print(f"\nSummary ({protocol}; all strategies scored on the same labels):")
    for name, block in metrics["strategies"].items():
        g = block["global"]
        depth_note = (
            f"final depth {block['best_depth']}"
            if "outer_folds" in block
            else f"depth {block['best_depth']}"
        )
        line = (
            f"  {name:<9} {depth_note}: F1-macro {g['f1_macro']:.4f} | accuracy {g['accuracy']:.4f}"
        )
        if "per_fold_f1_macro" in block:
            folds = block["per_fold_f1_macro"]
            line += f" | per-fold F1-macro {folds['mean']:.4f} ± {folds['std']:.4f}"
        print(line)
        for row in block["per_class"].values():
            print(
                f"      {row['name']:<28} precision {row['precision']:.3f} | "
                f"recall {row['recall']:.3f} | F1 {row['f1']:.3f} | n {row['support']:,}"
            )
        if "per_group" in block:
            print(
                f"      per group ({args.cv_group}; F1-macro / accuracy / windows / true classes):"
            )
            for group, row in block["per_group"].items():
                classes = ", ".join(label_map.get(c, str(c)) for c in row["classes"])
                print(
                    f"        {group:>6} {row['f1_macro']:.3f} / {row['accuracy']:.3f} / "
                    f"{row['n_windows']:>7,} / {classes}"
                )


if __name__ == "__main__":
    main()
