"""Audit — how much of an instance-grouped test set comes from wells seen in training.

Grouping the splits by ``instance_id`` keeps a recording on one side, but the
same well's other recordings can sit on the other side, so a model may score
well by recognizing the well rather than the coming fault. This audit measures
the exposure on the seeded holdout split of the instance grouping: how many
test windows, overall and per class, belong to a well that also appears in
train+val. It also shows why a well-grouped split cannot avoid being lopsided —
how the real instances of every class concentrate in a few wells — and how much
of each class is simulated or hand-drawn data, which the well grouping drops.

Usage
-----
    uv run scripts/audits/well_leakage_auditing.py [--task {prediction,detection}]
        [--no-normalization] [--allow-overlap] [--keep-extreme-values]
        [--output-file PATH] [--verbose]

Output
------
    results/audits/well_leakage_<task>_<norm>[_overlap][_extremes]_<timestamp>.txt
    (everything is printed to the terminal as well)
"""

import contextlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from flowml.cli import run_parser
from flowml.config import (
    FAULT_CLASSES,
    RESULTS_DIR,
    WINDOW_CLASSES,
    extreme_suffix,
    features_path,
    norm_suffix,
    overlap_suffix,
)
from flowml.train_val_test import TASKS, holdout_split, load_task_data

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


def audit(task: str, normalized: bool, args) -> None:
    """Print the source composition, the well exposure and the well concentration."""
    label_col = "fault_class" if task == "prediction" else "window_label"
    label_map = FAULT_CLASSES if task == "prediction" else WINDOW_CLASSES

    df = pd.read_parquet(features_path(normalized, args.allow_overlap, args.keep_extreme_values))
    if task == "prediction":
        df = df[df["window_label"] == 0]

    print("\nSource composition of the dataset (windows per class and instance source):")
    table = df.groupby([label_col, "source_type"]).size().unstack(fill_value=0)
    table["all sources"] = table.sum(axis=1)
    table.index = [f"{c} {label_map.get(c, '')}" for c in table.index]
    print(table.to_string())
    print("\n  windows by source:")
    for source, n in df["source_type"].value_counts().items():
        print(f"    {source:<10} {n:>8,} ({100 * n / len(df):.1f}%)")

    data = load_task_data(
        task, normalized, "instance_id", args.allow_overlap, args.keep_extreme_values
    )
    assert len(df) == data.n_windows, "features frame and task data disagree"
    print("\nInstance-grouped holdout split (the one stages 2 and 5 use):")
    trainval_idx, test_idx = holdout_split(data, args.verbose)
    wells = df["well_id"].to_numpy()

    trainval_wells, test_wells = set(wells[trainval_idx]), set(wells[test_idx])
    shared = test_wells & trainval_wells
    seen = np.isin(wells[test_idx], list(shared))
    print(
        f"\n  test wells: {len(test_wells)} | also in train+val: {len(shared)} "
        f"| test-only: {sorted(test_wells - shared)}"
    )
    print(
        f"  test windows whose well is also in train+val: {seen.sum():,}/{len(test_idx):,} "
        f"({100 * seen.mean():.1f}%)"
    )
    print("\n  per class, share of test windows whose well is also seen in training:")
    y_test = data.y[test_idx]
    for c in np.unique(y_test):
        m = y_test == c
        print(
            f"    {c:>3} {label_map.get(c, ''):<32} {100 * seen[m].mean():5.1f}%  (n={m.sum():,})"
        )

    real = df[df["source_type"] == "WELL"]
    print("\nWell concentration per class (real instances only):")
    for c, part in real.groupby(label_col):
        counts = part["well_id"].value_counts()
        top = counts.head(2)
        print(
            f"  {c:>3} {label_map.get(c, ''):<32} {counts.size:>2} wells | "
            f"top well {top.index[0]:>3} holds {100 * top.iloc[0] / counts.sum():3.0f}% | "
            f"top-2 {100 * top.sum() / counts.sum():3.0f}%"
        )


def main() -> None:
    """Parse the shared switches and run the audit."""
    parser = run_parser(__doc__.splitlines()[0], with_model=False)
    parser.add_argument("--task", choices=TASKS, default="prediction", help="(default: prediction)")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="where to save the audit (default: results/audits/well_leakage_<...>.txt)",
    )
    args = parser.parse_args()

    normalized = not args.no_normalization
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out = args.output_file or AUDITS_DIR / (
        f"well_leakage_{args.task}_{norm_suffix(normalized)}"
        f"{overlap_suffix(args.allow_overlap)}{extreme_suffix(args.keep_extreme_values)}_{stamp}.txt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fh, contextlib.redirect_stdout(Tee(sys.stdout, fh)):
        print(f"Well leakage audit — {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")
        print(
            f"task {args.task} | features {norm_suffix(normalized)}"
            f"{overlap_suffix(args.allow_overlap)}{extreme_suffix(args.keep_extreme_values)}"
        )
        audit(args.task, normalized, args)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
