"""Audit — what the pipeline's splits hold, class by class, under each CV grouping.

Reproduces the seeded splits of the modeling stages — ``holdout_split`` and,
for the out-of-fold protocols, ``outer_folds`` — for one features parquet and
prints their composition: windows, groups and, per class, how much of the class
each side holds and what share of the side it makes up. The same numbers on
the grouped labels of ``--class-grouping`` follow, then the per-well view that
explains a lopsided well split: windows per well on each side and which wells
carry normal operation at all.

The modeling stages print the holdout table themselves (``train_val_test.
split_composition``); this audit exists to look at a split without training
anything, and to compare the two groupings side by side — run it with
``--cv-group both`` (the default).

Usage
-----
    uv run scripts/audits/split_composition_auditing.py [--task {prediction,detection}]
        [--cv-group {instance_id,well_id,both}] [--eval {holdout,nested,leave-one-out}]
        [--class-grouping {standard,hydrate,custom}] [--normalization {none,instance,normal}] [--allow-overlap]
        [--keep-extreme-values] [--output-file PATH] [--verbose]

Output
------
    results/audits/split_composition_<task>_<norm>[_overlap][_extremes]_<timestamp>.txt
    (everything is printed to the terminal as well)
"""

import contextlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from flowml.cli import add_class_grouping_arg, add_normalization_arg, run_parser
from flowml.config import (
    CV_GROUPINGS,
    EVAL_MODES,
    RESULTS_DIR,
    default_eval_mode,
    extreme_suffix,
    norm_suffix,
    overlap_suffix,
)
from flowml.train_val_test import (
    TASKS,
    group_labels,
    group_sort_key,
    grouping_label_map,
    held_out_summary,
    holdout_split,
    load_task_data,
    outer_folds,
    split_composition,
)

AUDITS_DIR = RESULTS_DIR / "audits"


class Tee:
    """Write everything to several streams, so the audit is both shown and saved."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def audit_grouping(task: str, normalization: str, cv_group: str, eval_mode: str, args) -> None:
    """Print the split composition of one CV grouping under one protocol."""
    data = load_task_data(
        task, normalization, cv_group, args.allow_overlap, args.keep_extreme_values
    )
    print(f"\n{'=' * 100}")
    print(f"CV grouping: {cv_group} | evaluation: {eval_mode}")
    print(
        f"  {data.n_windows:,} windows | {pd.Series(data.groups).nunique()} groups "
        f"| classes: {', '.join(f'{c} {data.label_map.get(c, c)}' for c in np.unique(data.y))}"
    )

    if eval_mode == "holdout":
        trainval_idx, test_idx = holdout_split(data, args.verbose)  # prints its own table
        parts = {"train+val": trainval_idx, "test": test_idx}
    else:
        folds = outer_folds(data, eval_mode, args.verbose)
        print(f"  {len(folds)} outer folds; held-out part of each:")
        for fold, (train_idx, test_idx) in enumerate(folds, start=1):
            print(
                f"    fold {fold:>3}/{len(folds)}: "
                f"{held_out_summary(data.y, data.groups, test_idx, data.label_map)}"
            )
            if args.verbose:
                print(
                    split_composition(
                        data.y,
                        data.groups,
                        {"train": train_idx, "held out": test_idx},
                        data.label_map,
                    )
                )
        # The per-class view below needs two sides; show the first fold's.
        train_idx, test_idx = folds[0]
        parts = {"train (fold 1)": train_idx, "held out (fold 1)": test_idx}

    if args.class_grouping != "standard":
        grouped = group_labels(data.y, args.class_grouping)
        print(f"\n  Same split on the '{args.class_grouping}' grouping:")
        print(
            split_composition(grouped, data.groups, parts, grouping_label_map(args.class_grouping))
        )

    if cv_group == "well_id":
        side_of = np.full(data.n_windows, "", dtype=object)
        for name, idx in parts.items():
            side_of[idx] = name
        frame = pd.DataFrame({"well": data.groups, "y": data.y, "side": side_of})
        frame = frame[frame["side"] != ""]

        print("\n  Windows per well and side (largest first):")
        per_well = frame.groupby(["side", "well"]).size().sort_values(ascending=False)
        for (side, well), n in per_well.items():
            print(f"    {side:<20} well {well:>3}: {n:>8,}")

        normal = frame[frame["y"] == 0]
        if len(normal):
            print("\n  Normal-operation windows per well (all sides):")
            counts = normal.groupby("well").size()
            total = int(counts.sum())
            for well in sorted(counts.index, key=group_sort_key):
                n = int(counts[well])
                print(
                    f"    well {well:>3}: {n:>8,} ({100 * n / total:5.1f}% of normal) "
                    f"— {frame.loc[frame['well'] == well, 'side'].iloc[0]}"
                )


def main() -> None:
    """Parse the shared switches and audit the requested groupings."""
    parser = run_parser(__doc__.splitlines()[0], with_model=False)
    add_normalization_arg(parser)
    add_class_grouping_arg(parser)
    parser.add_argument("--task", choices=TASKS, default="prediction", help="(default: prediction)")
    parser.add_argument(
        "--cv-group",
        choices=(*CV_GROUPINGS, "both"),
        default="both",
        help="grouping(s) to audit (default: both, one after the other)",
    )
    parser.add_argument(
        "--eval",
        choices=EVAL_MODES,
        default=None,
        help="protocol whose split to show (default: the grouping's own default)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="where to save the audit (default: results/audits/split_composition_<...>.txt)",
    )
    args = parser.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out = args.output_file or AUDITS_DIR / (
        f"split_composition_{args.task}{norm_suffix(args.normalization)}"
        f"{overlap_suffix(args.allow_overlap)}{extreme_suffix(args.keep_extreme_values)}_{stamp}.txt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    groupings = list(CV_GROUPINGS) if args.cv_group == "both" else [args.cv_group]

    with open(out, "w", encoding="utf-8") as fh, contextlib.redirect_stdout(Tee(sys.stdout, fh)):
        print(f"Split composition audit — {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")
        print(
            f"task {args.task} | normalization {args.normalization}"
            f"{overlap_suffix(args.allow_overlap)}{extreme_suffix(args.keep_extreme_values)} "
            f"| class grouping {args.class_grouping}"
        )
        for cv_group in groupings:
            audit_grouping(
                args.task,
                args.normalization,
                cv_group,
                args.eval or default_eval_mode(cv_group),
                args,
            )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
