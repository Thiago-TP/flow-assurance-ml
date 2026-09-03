"""Stage 4 — Feature-importance rankings (MDI / gain / permutation / SHAP).

Loads the trained pipeline and ranks features with every method available for
the model. The JSON output is the input for building compact decision trees
on the top SHAP features.

Usage
-----
    uv run scripts/04_interpret.py [--model {rf,xgb}] [--task {prediction,detection}]
                                   [--skip-permutation]

Outputs (tag = <model>_<task>)
---------------------------------------
    results/metrics/<tag>_importance.json   full rankings, every method
    results/figures/<tag>_mdi.png           (rf)  or  <tag>_gain.png (xgb)
    results/figures/<tag>_permutation.png
    results/figures/<tag>_shap.png
"""

import json
import sys

import joblib

from flowml.cli import run_parser, run_tag
from flowml.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, TOP_N_FEATURES
from flowml.interpretation import (
    mdi_importance,
    permutation_ranking,
    plot_permutation_boxplot,
    plot_ranking,
    shap_ranking,
    xgb_importance,
)
from flowml.training import load_task_data


def main() -> None:
    """Parse arguments, compute every applicable ranking, and save outputs."""
    parser = run_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="skip permutation importance (the slowest method)",
    )
    args = parser.parse_args()
    tag = run_tag(args.model, args.task)
    cmap = "Blues_r" if args.model == "rf" else "Oranges_r"

    model_path = MODELS_DIR / f"{tag}.joblib"
    if not model_path.exists():
        sys.exit(
            f"{model_path} not found. Train first:\n"
            f"  uv run scripts/02_train.py --model {args.model} "
            f"--task {args.task}"
        )
    pipe = joblib.load(model_path)
    encoder = joblib.load(MODELS_DIR / f"{tag}_label_encoder.joblib")
    clf = pipe.named_steps["clf"]

    print(f"Interpretation — {tag}")
    data = load_task_data(args.task)
    X_imputed = pipe.named_steps["imputer"].transform(data.X)
    rankings: dict[str, dict] = {}

    if args.model == "rf":
        print("\n[1/3] MDI (mean decrease in impurity)...")
        mdi = mdi_importance(clf, data.feature_cols)
        rankings["mdi"] = mdi.round(6).to_dict()
        plot_ranking(
            mdi,
            "RF — feature importance (MDI)",
            "Mean Gini-impurity decrease",
            FIGURES_DIR / f"{tag}_mdi.png",
            cmap,
        )
    else:
        print("\n[1/3] XGBoost gain / weight / cover...")
        native = xgb_importance(clf, data.feature_cols)
        for imp_type, series in native.items():
            rankings[imp_type] = series.round(6).to_dict()
        plot_ranking(
            native["gain"],
            "XGB — feature importance (gain)",
            "Mean accuracy gain per split",
            FIGURES_DIR / f"{tag}_gain.png",
            cmap,
        )

    if args.skip_permutation:
        print("\n[2/3] Permutation importance skipped (--skip-permutation).")
    else:
        print("\n[2/3] Permutation importance (slow)...")
        perm_mean, perm_raw = permutation_ranking(
            clf, X_imputed, encoder.transform(data.y), data.feature_cols
        )
        rankings["permutation"] = perm_mean.round(6).to_dict()
        plot_permutation_boxplot(
            perm_raw,
            data.feature_cols,
            f"{args.model.upper()} — permutation importance",
            FIGURES_DIR / f"{tag}_permutation.png",
        )

    print("\n[3/3] SHAP (TreeExplainer)...")
    shap_series = shap_ranking(clf, X_imputed, data.feature_cols)
    rankings["shap"] = shap_series.round(6).to_dict()
    plot_ranking(
        shap_series,
        f"{args.model.upper()} — SHAP global importance",
        "Mean |SHAP value|",
        FIGURES_DIR / f"{tag}_shap.png",
        cmap,
    )

    out = {
        "tag": tag,
        "top_shap_features": shap_series.head(TOP_N_FEATURES).index.tolist(),
        "rankings": rankings,
    }
    out_path = METRICS_DIR / f"{tag}_importance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Top {TOP_N_FEATURES} SHAP features: {out['top_shap_features']}")


if __name__ == "__main__":
    main()
