"""Shared command-line plumbing for the pipeline scripts."""

import argparse

from flowml.config import GROUPINGS
from flowml.preprocessing import FILTER_TYPES
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
        Parser with ``--filter`` and, optionally, ``--model`` / ``--task``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--filter",
        choices=FILTER_TYPES,
        default="none",
        dest="filter_type",
        help="signal filter of the features parquet (default: none)",
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
                "prediction = which fault will the well develop, from normal-"
                "operation windows only; detection = current operational state "
                "(default: prediction)"
            ),
        )
    return parser


def add_grouping_arg(parser: argparse.ArgumentParser) -> None:
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
        "--grouping",
        choices=GROUPINGS,
        default="none",
        help=(
            "collapse classes before scoring: hydrate = Normal / Other Problem "
            "/ Hydrate (default: none)"
        ),
    )


def run_tag(model: str, task: str, filter_type: str) -> str:
    """Compose the artifact-name tag identifying one (model, task, filter) run.

    Parameters
    ----------
    model : str
        ``"rf"`` or ``"xgb"``.
    task : str
        ``"prediction"`` or ``"detection"``.
    filter_type : str
        Signal filter of the features used.

    Returns
    -------
    str
        E.g. ``"xgb_prediction_none"``.
    """
    return f"{model}_{task}_{filter_type}"
