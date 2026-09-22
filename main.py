"""Orchestrate the full pipeline: features -> train -> evaluate -> interpret -> tree.

Runs the numbered scripts in ``scripts/`` in order as subprocesses, passing the
shared switches through. Stage 1 (feature building) is skipped automatically
when its parquet already exists, unless ``--rebuild-features`` is given.

Stages 4 and 5 are skipped for ``--model dt``: a single decision tree is
already interpretable, and stage 2 exports its rules and figure itself.

Usage
-----
    uv run main.py [--model {rf,xgb,dt}] [--task {prediction,detection}]
                   [--class-grouping {standard,hydrate,custom}]
                   [--eval {holdout,nested,leave-one-out}]
                   [--cv-group {instance_id,well_id}] [--normalization {none,instance,normal-operation-values}]
                   [--frozen-sensors {keep,flag,drop}]
                   [--allow-overlap] [--keep-extreme-values] [--n-jobs N]
                   [--max-instances N] [--rebuild-features] [--verbose]
                   [--skip-permutation] [--top-n N] [--depths 2,3,4,5,6,7,8,9,10,11,12]
                   [--run ID]

Every artifact of the chain lands in one run directory under
``results/runs/``, named after the commit and the moment the chain started
(see ``flowml.runs``). It is created here and handed to each stage through
``FLOWML_RUN_DIR``, so the stages of one invocation always agree on where they
are writing — including the two label sets a class grouping trains.
``--run`` drives the chain into an existing run directory instead.

``--class-grouping`` reaches every stage but feature building, which the
grouping cannot change: stages 2 and 4 train and rank on the grouped labels,
stages 3 and 5 score in them. With a non-standard grouping the chain is run
twice — once on the dataset's own classes, once on the groups — because the
comparisons that make a grouping worth choosing (``collapse`` vs ``native``
in stages 3 and 5) need both runs, and stage 5 distils each strategy from the
ranking of the ensemble trained on the same labels. ``--eval`` left unset
defaults to ``holdout`` with instance grouping and to ``leave-one-out`` with
well grouping, and the resolved value is passed to every stage.

Examples
--------
    uv run main.py                          # full fault-prediction pipeline (XGB)
    uv run main.py --max-instances 3        # quick smoke test of every stage
    uv run main.py --model rf --task detection
    uv run main.py --model dt                # single tree; stages 4 and 5 skipped
"""

import os
import subprocess
import sys
from pathlib import Path

from flowml.cli import (
    add_class_grouping_arg,
    add_frozen_sensors_arg,
    add_normalization_arg,
    add_run_arg,
    run_parser,
)
from flowml.config import features_path
from flowml.runs import RUN_DIR_ENV, resolve_run
from flowml.train_val_test import WHITE_BOX_MODELS

SCRIPTS_DIR = Path(__file__).parent / "scripts"


def run_stage(script: str, extra_args: list[str]) -> None:
    """Run one pipeline script as a subprocess, aborting the chain on failure.

    Parameters
    ----------
    script : str
        Script filename inside ``scripts/``.
    extra_args : list[str]
        Command-line arguments forwarded to the script.
    """
    print(f"\n{'=' * 70}\n  {script}\n{'=' * 70}", flush=True)
    result = subprocess.run([sys.executable, str(SCRIPTS_DIR / script), *extra_args], check=False)
    if result.returncode != 0:
        sys.exit(f"{script} failed with exit code {result.returncode}; aborting.")


def main() -> None:
    """Parse the shared switches and run every stage in order."""
    parser = run_parser(__doc__.splitlines()[0])
    add_class_grouping_arg(parser)
    add_normalization_arg(parser)
    add_frozen_sensors_arg(parser)
    add_run_arg(parser)
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="cap instances per class in stage 1, for a quick smoke test",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="rerun stage 1 even when the features parquet already exists",
    )
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="skip permutation importance in stage 4 (the slowest method)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="top SHAP features for the stage-5 decision tree (default: 10)",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="2,3,4,5,6,7,8,9,10,11,12",
        help="tree depths swept in stage 5 (default: 2,3,4,5,6,7,8,9,10,11,12)",
    )
    args = parser.parse_args()

    # One run directory for the whole chain. Exporting it means every stage
    # subprocess — and both label sets of a class grouping — writes into the
    # same experiment instead of each opening one of its own.
    run = resolve_run(args.run, create=True)
    os.environ[RUN_DIR_ENV] = str(run.path)

    common = ["--verbose"] if args.verbose else []
    if args.allow_overlap:
        common.append("--allow-overlap")
    if args.keep_extreme_values:
        common.append("--keep-extreme-values")
    common += ["--n-jobs", str(args.n_jobs)]
    # Stage 1 has no normalization choice: the parquet it writes is raw, and
    # the reference is picked by the stages that load it.
    modeled = [
        *common,
        "--normalization",
        args.normalization,
        "--frozen-sensors",
        args.frozen_mode,
        "--model",
        args.model,
        "--task",
        args.task,
        "--cv-group",
        args.cv_group,
        "--eval",
        args.eval,
    ]

    parquet = features_path(args.allow_overlap, args.keep_extreme_values)
    if parquet.exists() and not args.rebuild_features:
        print(f"Stage 1 skipped: {parquet} already exists (use --rebuild-features).")
    else:
        stage1_args = list(common)
        if args.max_instances is not None:
            stage1_args += ["--max-instances", str(args.max_instances)]
        run_stage("01_build_features.py", stage1_args)

    standard = [*modeled, "--class-grouping", "standard"]
    grouped = [*modeled, "--class-grouping", args.class_grouping]
    # With a grouping, the training stages run once per label set: the
    # comparisons of stages 3 and 5 (collapse vs native) need the standard run
    # as well as the grouped one, and stage 5 distils each strategy from the
    # ranking of the ensemble trained on the same labels it is.
    label_sets = [standard] if args.class_grouping == "standard" else [standard, grouped]
    if len(label_sets) > 1:
        print(
            f"\n--class-grouping {args.class_grouping}: the training stages run twice, "
            "once on the dataset's own classes and once on the groups, so the scoring "
            "stages can compare collapsing against native training."
        )

    for label_set in label_sets:
        run_stage("02_train_val_test.py", label_set)
    run_stage("03_evaluate.py", grouped)

    if args.model in WHITE_BOX_MODELS:
        # Stages 4 and 5 read a black box: they rank what drives an ensemble
        # and distil that into a compact tree. A single tree is that already,
        # and stage 2 has just written its rules and figure.
        print(
            f"\nStages 4 and 5 skipped: --model {args.model} is already interpretable "
            "(stage 2 exported its rules and figure)."
        )
    else:
        for label_set in label_sets:
            stage4_args = list(label_set)
            if args.skip_permutation:
                stage4_args.append("--skip-permutation")
            run_stage("04_interpret.py", stage4_args)

        run_stage(
            "05_decision_tree.py",
            [*grouped, "--top-n", str(args.top_n), "--depths", args.depths],
        )

    print(f"\n{'=' * 70}\n  Pipeline complete: {run.path}\n{'=' * 70}")


if __name__ == "__main__":
    main()
