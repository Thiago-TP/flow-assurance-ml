"""Stage 2 — Hyperparameter search (validation) and held-out evaluation.

Data used to select hyperparameters never evaluates the selection:

- ``--eval holdout`` (default) — a grouped test set is split off first and
  never touches the search; a GroupKFold RandomizedSearchCV (F1-macro) runs on
  the remainder and the refit winner is scored once on the test set. The
  saved model is that winner — the exact model the test score describes.
- ``--eval nested`` — grouped nested CV: every outer fold runs its own inner
  search and predicts its held-out fold, evaluating the whole procedure on all
  data (~N_SPLITS_OUTER times slower). The saved model comes from one final
  search on all data; its tuning score is validation-only.

Every split is repaired so that each class is present on both of its sides
(``train_val_test.repair_holdout`` and ``coverage_folds``): a class that lives
in a single group stops the run under ``holdout`` and is pinned to training —
never evaluated — under ``nested``.

Evaluation and interpretation consume the artifacts written here, they never
retrain.

``--model dt`` fits a single decision tree instead of an ensemble. It goes
through the same search and the same evaluation, and additionally exports
itself as rules and a figure — being already interpretable, it skips stages 4
and 5, which exist to read a black box.

Usage
-----
    uv run scripts/02_train_val_test.py [--model {rf,xgb,dt}] [--task {prediction,detection}]
                                        [--eval {holdout,nested}]
                                        [--cv-group {instance_id,well_id}]
                                        [--no-normalization] [--allow-overlap]
                                        [--n-jobs N] [--verbose]

Outputs (tag = <model>_<task>_<norm>; ``_overlap`` appended with
--allow-overlap, ``_wellcv`` with --cv-group well_id, ``_nested`` with --eval
nested)
---------------------------------------
    results/models/<tag>.joblib             fitted imputer+classifier pipeline
    results/models/<tag>_label_encoder.joblib
    results/metrics/<tag>_eval.parquet      held-out test predictions
    results/metrics/<tag>_search.json       best params + validation/test scores
    results/metrics/<tag>_cv_results.csv    full search history
    results/metrics/<tag>_rules.txt         (dt) the tree as if/else rules
    results/figures/<tag>_tree.png          (dt) the tree drawn
"""

import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from flowml.cli import run_parser, run_tag
from flowml.config import (
    METRICS_DIR,
    MODELS_DIR,
    N_SPLITS_CV,
    N_SPLITS_OUTER,
    norm_suffix,
)
from flowml.evaluation import global_metrics
from flowml.interpretation import export_tree
from flowml.train_val_test import (
    WHITE_BOX_MODELS,
    holdout_evaluation,
    load_task_data,
    nested_evaluation,
    search_hyperparameters,
)


def main() -> None:
    """Parse arguments, run the evaluation protocol, and write the artifacts."""
    args = run_parser(__doc__.splitlines()[0]).parse_args()
    normalized = not args.no_normalization
    tag = run_tag(
        args.model,
        args.task,
        normalized,
        args.cv_group,
        args.eval,
        args.allow_overlap,
        args.keep_extreme_values,
    )

    print(f"Training {tag} | started {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")

    print("\n[1/3] Loading dataset...")
    data = load_task_data(
        args.task, normalized, args.cv_group, args.allow_overlap, args.keep_extreme_values
    )
    print(
        f"  {data.n_windows:,} windows | {len(data.feature_cols)} features "
        f"| {pd.Series(data.groups).nunique()} groups ({args.cv_group})"
    )

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
        print(f"\n[2/3] Nested evaluation (GroupKFold({N_SPLITS_OUTER}) outer x inner searches)...")
        eval_frame, fold_records = nested_evaluation(
            args.model, data, n_jobs=args.n_jobs, verbose=args.verbose
        )
        test_metrics = global_metrics(eval_frame["y_true"], eval_frame["y_pred"])
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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, MODELS_DIR / f"{tag}.joblib")
    joblib.dump(encoder, MODELS_DIR / f"{tag}_label_encoder.joblib")
    pd.DataFrame(search.cv_results_).to_csv(METRICS_DIR / f"{tag}_cv_results.csv", index=False)
    eval_frame.to_parquet(METRICS_DIR / f"{tag}_eval.parquet", index=False)

    if args.model in WHITE_BOX_MODELS:
        # A single tree is its own explanation, so stages 4 and 5 are skipped
        # for it; export here what they would have produced. The tree was fit
        # on encoded labels, so the names are keyed by encoded value.
        export_tree(
            search.best_estimator_.named_steps["clf"],
            data.feature_cols,
            {i: data.label_map.get(c, str(c)) for i, c in enumerate(encoder.classes_)},
            tag,
        )

    summary = {
        "tag": tag,
        "model": args.model,
        "task": args.task,
        "normalization": norm_suffix(normalized),
        "overlapping_instances": "kept" if args.allow_overlap else "dropped",
        "extreme_values": "kept" if args.keep_extreme_values else "masked",
        "cv_group": args.cv_group,
        "trained_at": datetime.now().astimezone().isoformat(),
        "dataset": {
            "n_windows": data.n_windows,
            "n_features": len(data.feature_cols),
            "n_groups": int(pd.Series(data.groups).nunique()),
            "n_classes": len(encoder.classes_),
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
    with open(METRICS_DIR / f"{tag}_search.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Artifacts written under results/ with tag '{tag}'.")
    print(f"Next: uv run scripts/03_evaluate.py --model {args.model} --task {args.task}")


if __name__ == "__main__":
    main()
