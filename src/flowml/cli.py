"""Shared command-line plumbing for the pipeline scripts."""

import argparse

from flowml.config import CLASS_GROUPINGS
from flowml.training import MODEL_TYPES, TASKS


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
        Parser with, optionally, ``--model`` / ``--task``.
    """
    parser = argparse.ArgumentParser(description=description)
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
    return parser


def add_class_grouping_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--grouping`` switch to a parser that scores predictions.

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
            "collapse classes before scoring: hydrate = Normal / Other Problem / Hydrate "
            "custom = user-defined grouping (default: none)"
        ),
    )


def run_tag(model: str, task: str) -> str:
    """Compose the artifact-name tag identifying one (model, task) run.

    Parameters
    ----------
    model : str
        ``"rf"`` or ``"xgb"``.
    task : str
        ``"prediction"`` or ``"detection"``.

    Returns
    -------
    str
        E.g. ``"xgb_prediction"``.
    """
    return f"{model}_{task}"
