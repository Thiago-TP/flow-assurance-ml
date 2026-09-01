"""Stage 2 — Hyperparameter search, final fit, and out-of-fold predictions.

Runs a GroupKFold RandomizedSearchCV (F1-macro) over the chosen model and
task, refits the best pipeline on all data, and regenerates leak-free
out-of-fold predictions with per-fold refits. Evaluation and interpretation
consume the artifacts written here — they never retrain.

Usage
-----
    uv run scripts/02_train.py [--model {rf,xgb}] [--task {prediction,detection}]
                               [--filter {gaussian,statistical,none}]

Outputs (tag = <model>_<task>_<filter>)
---------------------------------------
    results/models/<tag>.joblib             fitted imputer+classifier pipeline
    results/models/<tag>_label_encoder.joblib
    results/metrics/<tag>_oof.parquet       out-of-fold predictions
    results/metrics/<tag>_search.json       best params + CV score
    results/metrics/<tag>_cv_results.csv    full search history
"""

import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from flowml.cli import run_parser, run_tag
from flowml.config import METRICS_DIR, MODELS_DIR, N_SPLITS_CV
from flowml.training import load_task_data, oof_predict, search_hyperparameters


def main() -> None:
    """Parse arguments, run the search, and write the training artifacts."""
    args = run_parser(__doc__.splitlines()[0]).parse_args()
    tag = run_tag(args.model, args.task, args.filter_type)

    print(f"Training {tag} | started {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")

    print("\n[1/3] Loading dataset...")
    data = load_task_data(args.filter_type, args.task)
    encoder = LabelEncoder().fit(data.y)
    print(
        f"  {data.n_windows:,} windows | {len(data.feature_cols)} features "
        f"| {len(encoder.classes_)} classes"
    )

    print("\n[2/3] Hyperparameter search (GroupKFold RandomizedSearchCV)...")
    search = search_hyperparameters(args.model, data, encoder)
    best_params = search.best_params_
    print(f"  Best F1-macro (CV): {search.best_score_:.4f}")
    print(f"  Best params: {best_params}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, MODELS_DIR / f"{tag}.joblib")
    joblib.dump(encoder, MODELS_DIR / f"{tag}_label_encoder.joblib")
    pd.DataFrame(search.cv_results_).to_csv(
        METRICS_DIR / f"{tag}_cv_results.csv", index=False
    )

    print("\n[3/3] Out-of-fold predictions with best params...")
    oof = oof_predict(args.model, best_params, data, encoder)
    oof.to_parquet(METRICS_DIR / f"{tag}_oof.parquet", index=False)

    summary = {
        "tag": tag,
        "model": args.model,
        "task": args.task,
        "filter": args.filter_type,
        "trained_at": datetime.now().astimezone().isoformat(),
        "dataset": {
            "n_windows": data.n_windows,
            "n_features": len(data.feature_cols),
            "n_instances": int(pd.Series(data.groups).nunique()),
            "n_classes": len(encoder.classes_),
        },
        "cv": {
            "strategy": f"GroupKFold(n_splits={N_SPLITS_CV})",
            "scoring": "f1_macro",
            "best_score_cv": round(float(search.best_score_), 4),
        },
        "best_params": {k.removeprefix("clf__"): v for k, v in best_params.items()},
    }
    with open(METRICS_DIR / f"{tag}_search.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Artifacts written under results/ with tag '{tag}'.")
    print(
        f"Next: uv run scripts/03_evaluate.py --model {args.model} "
        f"--task {args.task} --filter {args.filter_type}"
    )


if __name__ == "__main__":
    main()
