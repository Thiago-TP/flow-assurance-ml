"""Orchestrate the full pipeline: features -> train -> evaluate -> interpret -> tree.

Runs the numbered scripts in ``scripts/`` in order as subprocesses, passing the
shared switches through. Stage 1 (feature building) is skipped automatically
when its parquet already exists, unless ``--rebuild-features`` is given.

Usage
-----
    uv run main.py [--model {rf,xgb}] [--task {prediction,detection}]
                   [--class-grouping {none,hydrate,custom}] [--eval {holdout,nested}]
                   [--cv-group {instance_id,well_id}] [--no-normalization]
                   [--n-jobs N] [--max-instances N] [--rebuild-features] [--verbose]
                   [--skip-permutation] [--top-n N] [--depths 2,3,4,5,6]

``--class-grouping`` reaches the scoring stages (3 and 5) only: features and
the ensemble are always built on the full class set.

Examples
--------
    uv run main.py                          # full fault-prediction pipeline (XGB)
    uv run main.py --max-instances 3        # quick smoke test of every stage
    uv run main.py --model rf --task detection
"""

import subprocess
import sys
from pathlib import Path

from flowml.cli import add_class_grouping_arg, run_parser
from flowml.config import features_path

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
        default="2,3,4,5,6",
        help="tree depths swept in stage 5 (default: 2,3,4,5,6)",
    )
    args = parser.parse_args()

    common = ["--verbose"] if args.verbose else []
    if args.no_normalization:
        common.append("--no-normalization")
    common += ["--n-jobs", str(args.n_jobs)]
    modeled = [
        *common,
        "--model",
        args.model,
        "--task",
        args.task,
        "--cv-group",
        args.cv_group,
        "--eval",
        args.eval,
    ]

    parquet = features_path(not args.no_normalization)
    if parquet.exists() and not args.rebuild_features:
        print(f"Stage 1 skipped: {parquet} already exists (use --rebuild-features).")
    else:
        stage1_args = list(common)
        if args.max_instances is not None:
            stage1_args += ["--max-instances", str(args.max_instances)]
        run_stage("01_build_features.py", stage1_args)

    grouped = [*modeled, "--class-grouping", args.class_grouping]

    run_stage("02_train_val_test.py", modeled)
    run_stage("03_evaluate.py", grouped)

    stage4_args = list(modeled)
    if args.skip_permutation:
        stage4_args.append("--skip-permutation")
    run_stage("04_interpret.py", stage4_args)

    run_stage(
        "05_decision_tree.py",
        [*grouped, "--top-n", str(args.top_n), "--depths", args.depths],
    )

    print(f"\n{'=' * 70}\n  Pipeline complete.\n{'=' * 70}")


if __name__ == "__main__":
    main()
