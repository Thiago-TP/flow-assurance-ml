"""Shared command-line plumbing for the pipeline scripts."""

import argparse

from flowml.config import (
    CLASS_GROUPINGS,
    CV_GROUPING,
    CV_GROUPINGS,
    EVAL_MODE_DEFAULTS,
    EVAL_MODES,
    EXTREME_VALUE_LIMIT,
    N_JOBS,
    N_SPLITS_OUTER,
    TEMPERATURE_LIMITS,
    TEST_SIZE,
    default_eval_mode,
    extreme_suffix,
    norm_suffix,
    overlap_suffix,
)
from flowml.runs import RUN_DIR_ENV
from flowml.train_val_test import MODEL_TYPES, TASKS, WHITE_BOX_MODELS


class RunParser(argparse.ArgumentParser):
    """Argument parser whose ``--eval`` default follows ``--cv-group``.

    The evaluation protocol that makes sense depends on what the groups are
    (see ``config.EVAL_MODE_DEFAULTS``), so ``--eval`` is parsed with no
    default and filled in here once ``--cv-group`` is known. Scripts keep
    calling ``parse_args()`` as usual.
    """

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:  # type: ignore[override]
        parsed = super().parse_args(args, namespace)
        if getattr(parsed, "eval", None) is None and getattr(parsed, "cv_group", None) in (
            EVAL_MODE_DEFAULTS
        ):
            parsed.eval = default_eval_mode(parsed.cv_group)
        return parsed


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
        With a model, ``--eval`` left unset resolves to the default of the
        chosen ``--cv-group`` at parse time (``RunParser``).
    """
    parser = RunParser(description=description)
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
            default=None,
            help=(
                f"evaluation protocol: holdout = grouped test split (fraction {TEST_SIZE} "
                f"of the groups) scored once; nested = grouped nested CV, unbiased but "
                f"~{N_SPLITS_OUTER}x slower; leave-one-out = nested CV with one group "
                "held out per outer fold, a score per group at about as many searches "
                "as there are groups (default: "
                + ", ".join(f"{mode} for {group}" for group, mode in EVAL_MODE_DEFAULTS.items())
                + ")"
            ),
        )
    return parser


def add_class_grouping_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--class-grouping`` switch to a parser that reads labels.

    Every stage from training onwards accepts it; only feature building does
    not, since the grouping changes labels and never features. In the
    training stages it selects the label set the model is *fit* on, and
    grouped runs get their own artifact tag so they coexist with the standard
    ones; in the scoring stages it additionally selects the label set
    predictions are judged in.

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
            "label set to model and score: hydrate = Normal / Other Problem / Hydrate; "
            "custom = user-defined CUSTOM_CLASS_GROUPING from config.py (default: standard)"
        ),
    )


def add_run_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--run`` switch selecting which run directory a stage uses.

    Every stage writes into one experiment directory under ``results/runs/``
    (see the ``runs`` module). Left unset, a stage takes the directory
    ``main.py`` put in ``FLOWML_RUN_DIR``, and failing that either starts a new
    run (stage 2) or picks the newest one holding the input it needs. The
    switch overrides both, which is how a stage is rerun against an older
    experiment.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend in place.
    """
    parser.add_argument(
        "--run",
        default=None,
        metavar="ID",
        help=(
            "run directory to read from and write into: a run id under results/runs/, "
            "a path, or 'latest' for the newest run holding this stage's input "
            f"(default: ${RUN_DIR_ENV} when set, else a new run for stage 2 and the "
            "newest matching run for the later stages)"
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
            "interpretable.\nIts rules and figure come from stage 2, inside that run's "
            "directory:\n"
            "  results/runs/<run>/metrics/<tag>_rules.txt | "
            "results/runs/<run>/figures/<tag>_tree.png"
        )
        raise SystemExit(0)


def run_tag(
    model: str,
    task: str,
    normalized: bool = True,
    cv_group: str = CV_GROUPING,
    eval_mode: str = "holdout",
    allow_overlap: bool = False,
    keep_extreme_values: bool = False,
    class_grouping: str = "standard",
) -> str:
    """Compose the artifact-name tag identifying one training run.

    The tag covers every switch that changes what stage 2 produces: model,
    task, normalization, overlap and extreme-value rules, CV grouping,
    evaluation protocol, and the label set the model is fit on. The defaults
    (overlapping instances dropped, extreme readings masked, instance
    grouping, holdout evaluation, the dataset's own classes) add no suffix;
    keeping overlapping instances appends ``_overlap``, keeping extreme
    readings ``_extremes``, well-level grouping ``_wellcv``, nested
    evaluation ``_nested``, leave-one-out evaluation ``_loo`` and a class
    grouping its own name, so all runs coexist. The protocol suffix is
    written even when the protocol is the default of the grouping, so a
    well-grouped run reads ``_wellcv_loo``.

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
        ``"holdout"``, ``"nested"`` or ``"leave-one-out"``.
    allow_overlap : bool
        Whether the features keep the real instances overlapping another of
        the same well.
    keep_extreme_values : bool
        Whether the features keep readings beyond ``EXTREME_VALUE_LIMIT``.
    class_grouping : str
        The label set the model is fit on: ``"standard"``, ``"hydrate"`` or
        ``"custom"``.

    Returns
    -------
    str
        E.g. ``"xgb_prediction_zscore"``, ``"xgb_prediction_raw_overlap_wellcv_loo"``,
        ``"xgb_prediction_raw_overlap_hydrate"`` or
        ``"xgb_prediction_zscore_overlap_extremes_wellcv_nested_custom"``.
    """
    tag = (
        f"{model}_{task}_{norm_suffix(normalized)}"
        f"{overlap_suffix(allow_overlap)}{extreme_suffix(keep_extreme_values)}"
    )
    if cv_group == "well_id":
        tag += "_wellcv"
    tag += eval_suffix(eval_mode) + grouping_suffix(class_grouping)
    return tag


def grouping_suffix(class_grouping: str) -> str:
    """Artifact-name suffix of a class grouping.

    Parameters
    ----------
    class_grouping : str
        ``"standard"``, ``"hydrate"`` or ``"custom"``.

    Returns
    -------
    str
        ``""`` for the dataset's own classes, ``"_<name>"`` otherwise.
    """
    if class_grouping not in CLASS_GROUPINGS:
        raise ValueError(f"Unknown class grouping: {class_grouping!r} (expected {CLASS_GROUPINGS})")
    return "" if class_grouping == "standard" else f"_{class_grouping}"


def eval_suffix(eval_mode: str) -> str:
    """Artifact-name suffix of an evaluation protocol.

    Parameters
    ----------
    eval_mode : str
        ``"holdout"``, ``"nested"`` or ``"leave-one-out"``.

    Returns
    -------
    str
        ``""`` for holdout, ``"_nested"`` for nested, ``"_loo"`` for
        leave-one-out. Shared by ``run_tag`` and the stage-5 tree tag.
    """
    if eval_mode not in EVAL_MODES:
        raise ValueError(f"Unknown eval mode: {eval_mode!r} (expected {EVAL_MODES})")
    return {"holdout": "", "nested": "_nested", "leave-one-out": "_loo"}[eval_mode]
