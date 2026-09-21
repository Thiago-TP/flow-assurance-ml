"""Stage 4 — Feature-importance rankings (MDI / gain / permutation / SHAP).

Loads the trained pipeline and ranks features with every method available for
the model. The JSON output is the input for building compact decision trees
on the top SHAP features.

``--model dt`` skips this stage: a single decision tree is already its own
explanation, exported as rules and a figure by stage 2.

``--class-grouping`` selects which stage-2 run to read, so a run trained on
grouped labels gets a ranking of its own: what drives *that* question, rather
than the dataset's own classes. Stage 5 pairs each ranking with the tree
strategy it belongs to.

Usage
-----
    uv run scripts/04_interpret.py [--model {rf,xgb}] [--task {prediction,detection}]
                                   [--class-grouping {standard,hydrate,custom}]
                                   [--eval {holdout,nested,leave-one-out}]
                                   [--cv-group {instance_id,well_id}]
                                   [--skip-permutation] [--normalization {none,instance,normal}]
                                   [--frozen-sensors {keep,flag,drop}] [--allow-overlap]
                                   [--n-jobs N]

``--eval`` only selects which stage-2 run's model to read (its tag); under
``nested`` and ``leave-one-out`` that is the deployment model of the final
search on all data.

The model is read from — and the rankings written into — the run directory
stage 2 wrote it to: the one ``main.py`` names in ``FLOWML_RUN_DIR``, the one
``--run`` points at, or otherwise the newest run holding this configuration's
model (see the ``runs`` module).

Outputs, inside the run directory (tag = <model>_<task>_<norm>, plus the
suffixes of stage 2)
---------------------------------------
    metrics/<tag>_importance.json   full rankings, every method
    figures/<tag>_mdi.png           (rf)  or  <tag>_gain.png (xgb)
    figures/<tag>_permutation.png
    figures/<tag>_shap.png
"""

import json
import sys
from datetime import datetime

import joblib

from flowml.cli import (
    add_class_grouping_arg,
    add_frozen_sensors_arg,
    add_normalization_arg,
    add_run_arg,
    run_parser,
    run_tag,
    skip_if_white_box,
)
from flowml.config import TOP_N_FEATURES
from flowml.interpretation import (
    mdi_importance,
    permutation_ranking,
    plot_permutation_boxplot,
    plot_ranking,
    shap_ranking,
    xgb_importance,
)
from flowml.runs import append_index, record_stage, resolve_run
from flowml.train_val_test import load_task_data


def main() -> None:
    """Parse arguments, compute every applicable ranking, and save outputs."""
    parser = run_parser(__doc__.splitlines()[0])
    add_class_grouping_arg(parser)
    add_normalization_arg(parser)
    add_frozen_sensors_arg(parser)
    add_run_arg(parser)
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="skip permutation importance (the slowest method)",
    )
    args = parser.parse_args()
    skip_if_white_box(args.model, "feature-importance ranking")
    started = datetime.now().astimezone()
    tag = run_tag(
        args.model,
        args.task,
        args.normalization,
        args.frozen_mode,
        args.cv_group,
        args.eval,
        args.allow_overlap,
        args.keep_extreme_values,
        args.class_grouping,
    )
    cmap = "Blues_r" if args.model == "rf" else "Oranges_r"

    run = resolve_run(args.run, must_contain=f"models/{tag}.joblib")
    model_path = run.models / f"{tag}.joblib"
    if not model_path.exists():
        sys.exit(
            f"{model_path} not found. Train first:\n"
            f"  uv run scripts/02_train_val_test.py --model {args.model} "
            f"--task {args.task} --cv-group {args.cv_group} --eval {args.eval} "
            f"--class-grouping {args.class_grouping}"
        )
    pipe = joblib.load(model_path)
    encoder = joblib.load(run.models / f"{tag}_label_encoder.joblib")
    clf = pipe.named_steps["clf"]

    print(f"Interpretation — {tag}")
    data = load_task_data(
        args.task,
        args.normalization,
        args.frozen_mode,
        args.cv_group,
        args.allow_overlap,
        args.keep_extreme_values,
        args.class_grouping,
    )
    X_imputed = pipe.named_steps["imputer"].transform(data.X)
    rankings: dict[str, dict] = {}
    written = []

    if args.model == "rf":
        print("\n[1/3] MDI (mean decrease in impurity)...")
        mdi = mdi_importance(clf, data.feature_cols)
        rankings["mdi"] = mdi.round(6).to_dict()
        written.append(run.figures / f"{tag}_mdi.png")
        plot_ranking(
            mdi,
            "RF — feature importance (MDI)",
            "Mean Gini-impurity decrease",
            written[-1],
            cmap,
        )
    else:
        print("\n[1/3] XGBoost gain / weight / cover...")
        native = xgb_importance(clf, data.feature_cols)
        for imp_type, series in native.items():
            rankings[imp_type] = series.round(6).to_dict()
        written.append(run.figures / f"{tag}_gain.png")
        plot_ranking(
            native["gain"],
            "XGB — feature importance (gain)",
            "Mean accuracy gain per split",
            written[-1],
            cmap,
        )

    if args.skip_permutation:
        print("\n[2/3] Permutation importance skipped (--skip-permutation).")
    else:
        print("\n[2/3] Permutation importance (slow)...")
        perm_mean, perm_raw = permutation_ranking(
            clf, X_imputed, encoder.transform(data.y), data.feature_cols, n_jobs=args.n_jobs
        )
        rankings["permutation"] = perm_mean.round(6).to_dict()
        written.append(run.figures / f"{tag}_permutation.png")
        plot_permutation_boxplot(
            perm_raw,
            data.feature_cols,
            f"{args.model.upper()} — permutation importance",
            written[-1],
        )

    print("\n[3/3] SHAP (TreeExplainer)...")
    shap_series = shap_ranking(clf, X_imputed, data.feature_cols)
    rankings["shap"] = shap_series.round(6).to_dict()
    written.append(run.figures / f"{tag}_shap.png")
    plot_ranking(
        shap_series,
        f"{args.model.upper()} — SHAP global importance",
        "Mean |SHAP value|",
        written[-1],
        cmap,
    )

    out = {
        "tag": tag,
        "class_grouping": args.class_grouping,
        "top_shap_features": shap_series.head(TOP_N_FEATURES).index.tolist(),
        "rankings": rankings,
    }
    out_path = run.metrics / f"{tag}_importance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    written.append(out_path)
    print(f"\n  Saved: {out_path}")
    print(f"  Top {TOP_N_FEATURES} SHAP features: {out['top_shap_features']}")

    headline = {
        "class_grouping": args.class_grouping,
        "methods": sorted(rankings),
        "top_shap_features": out["top_shap_features"],
    }
    record_stage(run, "04_interpret", tag, started, written, **headline)
    append_index(run, "04_interpret", tag, headline)


if __name__ == "__main__":
    main()
