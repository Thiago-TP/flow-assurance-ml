"""Stage 2 — Hyperparameter search (validation) and held-out evaluation.

Data used to select hyperparameters never evaluates the selection:

- ``--eval holdout`` (default with ``--cv-group instance_id``) — a grouped test
  set is split off first and never touches the search; a GroupKFold
  RandomizedSearchCV (F1-macro) runs on the remainder and the refit winner is
  scored once on the test set. The saved model is that winner — the exact
  model the test score describes.
- ``--eval nested`` — grouped nested CV: every outer fold runs its own inner
  search and predicts its held-out fold, evaluating the whole procedure on all
  data (~N_SPLITS_OUTER times slower). The saved model comes from one final
  search on all data; its tuning score is validation-only.
- ``--eval leave-one-out`` (default with ``--cv-group well_id``) — the nested
  protocol with one group per outer fold: as many inner searches as there are
  groups, and a score per group. With wells as groups that is a score per
  well, which a single seeded holdout of eight wells cannot give; with
  instances as groups it is a thousand searches, allowed but slow.

Every split is repaired so that each class is present on both of its sides
(``train_val_test.repair_holdout`` and ``coverage_folds``): a class that lives
in a single group stops the run under ``holdout`` and is pinned to training —
never evaluated — under ``nested`` and ``leave-one-out``. Every split also
prints its composition — windows, groups, and per class how much of the class
each side holds and what share of the side it makes up — since a legal split
can still be lopsided.

Evaluation and interpretation consume the artifacts written here, they never
retrain.

``--model dt`` fits a single decision tree instead of an ensemble. It goes
through the same search and the same evaluation, and additionally exports
itself as rules and a figure — being already interpretable, it skips stages 4
and 5, which exist to read a black box.

``--class-grouping`` selects the label set the model is *fit* on: the
dataset's own classes by default, their groups otherwise (e.g. ``hydrate`` =
Normal / Other Problem / Hydrate). A grouped run spends its whole capacity —
and its balanced class weights — on the coarser question instead of learning
distinctions it is not asked about, and it gets its own artifact tag, so it
sits beside the standard run rather than replacing it. The splits are built
from the dataset's own classes either way, so the two runs are evaluated on
exactly the same rows and stage 3 can compare them directly.

Usage
-----
    uv run scripts/02_train_val_test.py [--model {rf,xgb,dt}] [--task {prediction,detection}]
                                        [--class-grouping {standard,hydrate,custom}]
                                        [--eval {holdout,nested,leave-one-out}]
                                        [--cv-group {instance_id,well_id}]
                                        [--normalization {none,instance,normal}] [--allow-overlap]
                                        [--n-jobs N] [--verbose]

This is the stage that starts an experiment: with no ``--run`` and no
``FLOWML_RUN_DIR`` from ``main.py`` it creates a run directory of its own (see
the ``runs`` module), and the later stages find it again.

Outputs, inside the run directory (tag = <model>_<task>_<norm>; ``_overlap``
appended with --allow-overlap, ``_wellcv`` with --cv-group well_id,
``_nested`` with --eval nested, ``_loo`` with --eval leave-one-out,
``_<class-grouping>`` when grouping)
---------------------------------------
    models/<tag>.joblib             fitted imputer+classifier pipeline
    models/<tag>_label_encoder.joblib
    metrics/<tag>_eval.parquet      held-out test predictions
    metrics/<tag>_search.json       best params + validation/test scores
    metrics/<tag>_cv_results.csv    full search history
    metrics/<tag>_rules.txt         (dt) the tree as if/else rules
    figures/<tag>_tree.png          (dt) the tree drawn
"""

import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from flowml.cli import (
    add_class_grouping_arg,
    add_normalization_arg,
    add_run_arg,
    run_parser,
    run_tag,
)
from flowml.config import N_SPLITS_CV
from flowml.evaluation import global_metrics
from flowml.interpretation import export_tree
from flowml.runs import append_index, record_stage, resolve_run
from flowml.train_val_test import (
    WHITE_BOX_MODELS,
    holdout_evaluation,
    load_task_data,
    n_outer_splits,
    nested_evaluation,
    search_hyperparameters,
)


def main() -> None:
    """Parse arguments, run the evaluation protocol, and write the artifacts."""
    parser = run_parser(__doc__.splitlines()[0])
    add_class_grouping_arg(parser)
    add_normalization_arg(parser)
    add_run_arg(parser)
    args = parser.parse_args()
    tag = run_tag(
        args.model,
        args.task,
        args.normalization,
        args.cv_group,
        args.eval,
        args.allow_overlap,
        args.keep_extreme_values,
        args.class_grouping,
    )

    # Stage 2 is the head of the chain, so it opens a run when nothing names one.
    started = datetime.now().astimezone()
    run = resolve_run(args.run, create=True)
    print(f"Training {tag} | started {started:%Y-%m-%d %H:%M:%S}")

    print("\n[1/3] Loading dataset...")
    data = load_task_data(
        args.task,
        args.normalization,
        args.cv_group,
        args.allow_overlap,
        args.keep_extreme_values,
        args.class_grouping,
    )
    print(
        f"  {data.n_windows:,} windows | {len(data.feature_cols)} features "
        f"| {pd.Series(data.groups).nunique()} groups ({args.cv_group})"
    )
    if args.class_grouping != "standard":
        print(f"  Fit on the '{args.class_grouping}' grouping of the classes:")
        for cls, n in pd.Series(data.y).value_counts().sort_index().items():
            print(f"    {cls} {data.label_map[cls]:<16}: {n:>8,} ({100 * n / data.n_windows:.1f}%)")

    evaluation: dict = {"mode": args.eval}
    if args.eval == "holdout":
        print(
            f"\n[2/3] Holdout evaluation (grouped test split + GroupKFold({N_SPLITS_CV}) search)..."
        )
        search, encoder, eval_frame, info = holdout_evaluation(
            args.model, data, n_jobs=args.n_jobs, verbose=args.verbose
        )
        best_params = search.best_params_
        test_metrics = global_metrics(eval_frame["y_true"], eval_frame["y_pred"])
        evaluation.update(info)
        evaluation["test_f1_macro"] = test_metrics["f1_macro"]
        print(f"  Best F1-macro (validation CV): {search.best_score_:.4f}")
        print(f"  F1-macro (held-out test)     : {test_metrics['f1_macro']:.4f}")
    else:
        n_outer = n_outer_splits(args.eval, data)
        what = (
            f"Nested evaluation (GroupKFold({n_outer}) outer x inner searches)"
            if args.eval == "nested"
            else f"Leave-one-out evaluation ({n_outer} groups, one held out per outer fold)"
        )
        print(f"\n[2/3] {what}...")
        eval_frame, fold_records = nested_evaluation(
            args.model, data, n_jobs=args.n_jobs, verbose=args.verbose, eval_mode=args.eval
        )
        test_metrics = global_metrics(eval_frame["y_true"], eval_frame["y_pred"])
        evaluation["n_outer_folds"] = n_outer
        evaluation["outer_folds"] = fold_records
        evaluation["test_f1_macro"] = test_metrics["f1_macro"]
        print(f"  Pooled outer-fold F1-macro: {test_metrics['f1_macro']:.4f}")

        print("\n  Final search on all data (deployment model; tuning score only)...")
        encoder = LabelEncoder().fit(data.y)
        search = search_hyperparameters(
            args.model, data, encoder, n_jobs=args.n_jobs, verbose=args.verbose
        )
        best_params = search.best_params_
        print(f"  Best F1-macro (validation CV): {search.best_score_:.4f}")
    print(f"  Best params: {best_params}")

    print("\n[3/3] Writing artifacts...")
    written = []
    if args.model in WHITE_BOX_MODELS:
        # A single tree is its own explanation, so stages 4 and 5 are skipped
        # for it; export here what they would have produced. The tree was fit
        # on encoded labels, so the names are keyed by encoded value. Exporting
        # before the dump means the saved model is the pruned one the rules and
        # the drawing describe — same predictions, fewer splits.
        written += export_tree(
            search.best_estimator_.named_steps["clf"],
            data.feature_cols,
            {i: data.label_map.get(c, str(c)) for i, c in enumerate(encoder.classes_)},
            tag,
            metrics_dir=run.metrics,
            figures_dir=run.figures,
        )

    model_path = run.models / f"{tag}.joblib"
    encoder_path = run.models / f"{tag}_label_encoder.joblib"
    cv_path = run.metrics / f"{tag}_cv_results.csv"
    eval_path = run.metrics / f"{tag}_eval.parquet"
    joblib.dump(search.best_estimator_, model_path)
    joblib.dump(encoder, encoder_path)
    pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)
    eval_frame.to_parquet(eval_path, index=False)
    written += [model_path, encoder_path, cv_path, eval_path]

    summary = {
        "tag": tag,
        "model": args.model,
        "task": args.task,
        "normalization": args.normalization,
        "overlapping_instances": "kept" if args.allow_overlap else "dropped",
        "extreme_values": "kept" if args.keep_extreme_values else "masked",
        "cv_group": args.cv_group,
        "class_grouping": args.class_grouping,
        "trained_at": datetime.now().astimezone().isoformat(),
        "dataset": {
            "n_windows": data.n_windows,
            "n_features": len(data.feature_cols),
            "n_groups": int(pd.Series(data.groups).nunique()),
            "n_classes": len(encoder.classes_),
            "class_names": {str(c): data.label_map.get(c, str(c)) for c in encoder.classes_},
        },
        "search": {
            "strategy": (
                f"GroupKFold(n_splits={N_SPLITS_CV}, groups={args.cv_group}), "
                "repaired so every training fold holds every class"
            ),
            "scoring": "f1_macro",
            "best_score_validation": round(float(search.best_score_), 4),
        },
        "evaluation": evaluation,
        "best_params": {k.removeprefix("clf__"): v for k, v in best_params.items()},
    }
    search_path = run.metrics / f"{tag}_search.json"
    with open(search_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    written.append(search_path)

    headline = {
        "model": args.model,
        "task": args.task,
        "cv_group": args.cv_group,
        "eval": args.eval,
        "class_grouping": args.class_grouping,
        "n_windows": data.n_windows,
        "best_score_validation": round(float(search.best_score_), 4),
        "test_f1_macro": round(float(test_metrics["f1_macro"]), 4),
    }
    record_stage(run, "02_train_val_test", tag, started, written, **headline)
    append_index(run, "02_train_val_test", tag, headline)

    print(f"\nDone. Artifacts written under {run.path} with tag '{tag}'.")
    print(
        f"Next: uv run scripts/03_evaluate.py --model {args.model} --task {args.task} "
        f"--cv-group {args.cv_group} --eval {args.eval} "
        f"--class-grouping {args.class_grouping}"
    )


if __name__ == "__main__":
    main()
