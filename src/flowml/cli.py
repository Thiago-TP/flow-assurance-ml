"""Shared command-line plumbing for the pipeline scripts."""

import argparse

from flowml.config import (
    CLASS_GROUPINGS,
    CV_GROUPING,
    CV_GROUPINGS,
    EVAL_MODE,
    EVAL_MODES,
    N_JOBS,
    N_SPLITS_OUTER,
    TEST_SIZE,
    norm_suffix,
)
from flowml.train_val_test import MODEL_TYPES, TASKS


def run_parser(description: str, with_model: bool = True) -> argparse.ArgumentParser:
    """Build the argument parser shared by the pipeline scripts.

    Parameters
    ----------
    description : str
        Script description shown in ``--help``.
    with_model : bool
        Include ``--model`` and ``--task`` (all scripts except feature building).

    Returns
    -------
    argparse.ArgumentParser
        Parser with ``--verbose`` and, optionally, ``--model`` / ``--task``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print detailed progress (per class, per fold, per search candidate)",
    )
    parser.add_argument(
        "--no-normalization",
        action="store_true",
        help=(
            "skip per-instance z-score normalization: stage 1 builds raw features, "
            "later stages read and write the matching _raw artifacts"
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help=(
            "parallel workers for training and permutation importance; "
            "-1 uses all cores (default: min(6, cores - 2))"
        ),
    )
    if with_model:
        parser.add_argument(
            "--model",
            choices=MODEL_TYPES,
            default="xgb",
            help="classifier (default: xgb)",
        )
        parser.add_argument(
            "--task",
            choices=TASKS,
            default="prediction",
            help=(
                "prediction = which fault will the well develop, from normal-operation windows only; "
                "detection = current operational state, possibly including active faults "
                "(default: prediction)"
            ),
        )
        parser.add_argument(
            "--cv-group",
            choices=CV_GROUPINGS,
            default=CV_GROUPING,
            help=(
                "grouping column of every split: instance_id keeps windows of one recording "
                "together; well_id additionally keeps all recordings of one well together "
                "and drops simulated/drawn instances (default: instance_id)"
            ),
        )
        parser.add_argument(
            "--eval",
            choices=EVAL_MODES,
            default=EVAL_MODE,
            help=(
                f"evaluation protocol: holdout = grouped test split (fraction {TEST_SIZE} "
                f"of the groups) scored once; nested = grouped nested CV, unbiased but "
                f"~{N_SPLITS_OUTER}x slower (default: holdout)"
            ),
        )
    return parser


def add_class_grouping_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--class-grouping`` switch to a parser that scores predictions.

    Only the stages that read labels back (evaluation, decision tree) accept
    it; feature building and ensemble training always work on the full set of
    classes.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend in place.
    """
    parser.add_argument(
        "--class-grouping",
        choices=CLASS_GROUPINGS,
        default="none",
        help=(
            "collapse classes before scoring: hydrate = Normal / Other Problem / Hydrate; "
            "custom = user-defined CUSTOM_CLASS_GROUPING from config.py (default: none)"
        ),
    )


def run_tag(
    model: str,
    task: str,
    normalized: bool = True,
    cv_group: str = CV_GROUPING,
    eval_mode: str = EVAL_MODE,
) -> str:
    """Compose the artifact-name tag identifying one training run.

    The tag covers every switch that changes what stage 2 produces: model,
    task, normalization, CV grouping, and evaluation protocol. The defaults
    (instance grouping, holdout evaluation) add no suffix; well-level grouping
    appends ``_wellcv`` and nested evaluation ``_nested`` so all runs coexist.

    Parameters
    ----------
    model : str
        ``"rf"`` or ``"xgb"``.
    task : str
        ``"prediction"`` or ``"detection"``.
    normalized : bool
        Whether the run uses per-instance z-scored features.
    cv_group : str
        ``"instance_id"`` or ``"well_id"``.
    eval_mode : str
        ``"holdout"`` or ``"nested"``.

    Returns
    -------
    str
        E.g. ``"xgb_prediction_zscore"`` or ``"xgb_prediction_zscore_wellcv_nested"``.
    """
    tag = f"{model}_{task}_{norm_suffix(normalized)}"
    if cv_group == "well_id":
        tag += "_wellcv"
    if eval_mode == "nested":
        tag += "_nested"
    return tag
