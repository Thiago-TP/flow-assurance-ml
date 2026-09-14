"""Shared command-line plumbing for the pipeline scripts."""

import argparse

from flowml.config import (
    CLASS_GROUPINGS,
    CV_GROUPING,
    CV_GROUPINGS,
    EVAL_MODE,
    EVAL_MODES,
    EXTREME_VALUE_LIMIT,
    N_JOBS,
    N_SPLITS_OUTER,
    TEMPERATURE_LIMITS,
    TEST_SIZE,
    extreme_suffix,
    norm_suffix,
    overlap_suffix,
)
from flowml.train_val_test import MODEL_TYPES, TASKS, WHITE_BOX_MODELS


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
        "--allow-overlap",
        action="store_true",
        help=(
            "keep the real instances that overlap another recording of the same well in "
            "time; stage 1 drops them by default (they label the same samples twice), "
            "and with the flag every stage reads and writes the matching _overlap artifacts"
        ),
    )
    parser.add_argument(
        "--keep-extreme-values",
        action="store_true",
        help=(
            "keep sensor readings that cannot be measurements: beyond "
            f"|{EXTREME_VALUE_LIMIT:.0e}| in magnitude, negative pressures or choke openings, "
            f"or temperatures outside {list(TEMPERATURE_LIMITS)} C. Stage 1 masks them as "
            "missing data by "
            "default, and with the flag every stage reads and writes the matching "
            "_extremes artifacts"
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
            help=(
                "classifier: rf = random forest, xgb = gradient boosting, "
                "dt = a single decision tree, already interpretable, so stages 4 and 5 "
                "are skipped for it (default: xgb)"
            ),
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
        default="standard",
        help=(
            "collapse classes before scoring: hydrate = Normal / Other Problem / Hydrate; "
            "custom = user-defined CUSTOM_CLASS_GROUPING from config.py (default: standard)"
        ),
    )


def skip_if_white_box(model: str, stage: str) -> None:
    """Exit cleanly when an interpretation stage does not apply to the model.

    Stages 4 and 5 exist to read a black box: they rank what drives an
    ensemble's predictions and distil that ranking into a compact tree. A
    ``--model dt`` run is already that tree — stage 2 exports its rules and
    figure itself — so both stages have nothing left to do. Called from the
    scripts as well as skipped by ``main.py``, so that running one directly
    says why instead of failing on a missing artifact.

    Parameters
    ----------
    model : str
        The ``--model`` value of the run.
    stage : str
        What the calling stage would have done, for the message.
    """
    if model in WHITE_BOX_MODELS:
        print(
            f"Skipped: {stage} does not apply to --model {model}, which is already "
            "interpretable.\nIts rules and figure come from stage 2:\n"
            "  results/metrics/<tag>_rules.txt | results/figures/<tag>_tree.png"
        )
        raise SystemExit(0)


def run_tag(
    model: str,
    task: str,
    normalized: bool = True,
    cv_group: str = CV_GROUPING,
    eval_mode: str = EVAL_MODE,
    allow_overlap: bool = False,
    keep_extreme_values: bool = False,
) -> str:
    """Compose the artifact-name tag identifying one training run.

    The tag covers every switch that changes what stage 2 produces: model,
    task, normalization, overlap and extreme-value rules, CV grouping, and
    evaluation protocol. The defaults (overlapping instances dropped, extreme
    readings masked, instance grouping, holdout evaluation) add no suffix;
    keeping overlapping instances appends ``_overlap``, keeping extreme
    readings ``_extremes``, well-level grouping ``_wellcv`` and nested
    evaluation ``_nested`` so all runs coexist.

    Parameters
    ----------
    model : str
        ``"rf"``, ``"xgb"`` or ``"dt"``.
    task : str
        ``"prediction"`` or ``"detection"``.
    normalized : bool
        Whether the run uses per-instance z-scored features.
    cv_group : str
        ``"instance_id"`` or ``"well_id"``.
    eval_mode : str
        ``"holdout"`` or ``"nested"``.
    allow_overlap : bool
        Whether the features keep the real instances overlapping another of
        the same well.
    keep_extreme_values : bool
        Whether the features keep readings beyond ``EXTREME_VALUE_LIMIT``.

    Returns
    -------
    str
        E.g. ``"xgb_prediction_zscore"`` or
        ``"xgb_prediction_zscore_overlap_extremes_wellcv_nested"``.
    """
    tag = (
        f"{model}_{task}_{norm_suffix(normalized)}"
        f"{overlap_suffix(allow_overlap)}{extreme_suffix(keep_extreme_values)}"
    )
    if cv_group == "well_id":
        tag += "_wellcv"
    if eval_mode == "nested":
        tag += "_nested"
    return tag
