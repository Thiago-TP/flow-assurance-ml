"""Audit — does per-instance z-scoring leak the coming fault into normal-operation windows?

The prediction task learns from windows of *normal operation only* and asks
which fault the instance later develops. Under ``--normalization instance``
its features are z-scores against the statistics of the **entire recording —
the fault period included** — so a normal-operation window is scaled by a
number the later fault helped produce. That was the pipeline's only behaviour
until TODO item 9; it is now one of three references a run can choose, kept so
the earlier results stay reproducible and so this audit still has something to
measure. The leak-free alternative this audit compares it against is the
``normal`` reference, over the normal-operation samples only.

A normal window's features therefore depend on what happens after it. If the
scale of the later fault differs by fault class — a hydrate plug moves the
wellhead pressure differently from a choke restriction — then the divisor
carries the label, and the model can read part of the answer off the
normalization rather than off the well's behaviour.

This audit quantifies the channel without asserting a verdict. Per instance it
compares the statistics stage 1 actually uses (whole recording) with the ones
a leak-free pipeline would use (the normal-operation prefix only), and reports
the ratio per fault class. A ratio far from 1, and *different between classes*,
is the leak: it means the same normal-operation samples are divided by
class-dependent numbers. It also reports how much of each instance is normal,
since an instance that is nearly all normal cannot leak much either way.

Only real and simulated instances with a normal-operation prefix are counted;
an instance whose ``class`` column never reads 0 has no prediction-task
windows to leak into.

Usage
-----
    uv run scripts/audits/normalization_leakage_auditing.py [--raw-dir PATH]
        [--max-instances N] [--sensors P-TPT,T-TPT,...] [--output-file PATH] [--verbose]

Output
------
    results/audits/normalization_leakage_<timestamp>.txt
    (everything is printed to the terminal as well)
"""

import argparse
import contextlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from flowml.config import (
    CONSTANT_THRESHOLD,
    FAULT_CLASSES,
    KEY_SENSORS,
    RAW_DATA_DIR,
    RESULTS_DIR,
)
from flowml.preprocessing import (
    clean_instance,
    list_raw_instances,
    load_raw_instances,
    mask_extreme_values,
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


def instance_rows(
    df: pd.DataFrame, fault_class: int, source: str, sensors: list[str]
) -> list[dict]:
    """Compare whole-recording statistics with normal-prefix ones, per sensor.

    Parameters
    ----------
    df : pd.DataFrame
        One cleaned instance, with the 3W ``class`` column.
    fault_class : int
        The fault the instance belongs to.
    source : str
        ``"WELL"``, ``"SIMULATED"`` or ``"DRAWN"``. Reported separately
        because a simulated instance starts from a steady state, so its
        normal period is nearly flat and its ratio is enormous for a reason
        that has nothing to do with real wells.
    sensors : list[str]
        Sensors to measure.

    Returns
    -------
    list[dict]
        One row per sensor that both parts have enough data for.
    """
    if "class" not in df.columns:
        return []
    labels = df["class"].to_numpy(dtype=float)
    normal = labels == 0
    if normal.sum() < 2 or normal.all():
        return []  # nothing to leak into, or nothing to leak from

    rows = []
    for sensor in sensors:
        if sensor not in df.columns:
            continue
        col = df[sensor].to_numpy(dtype=float)
        whole = col[~np.isnan(col)]
        prefix = col[normal & ~np.isnan(col)]
        if len(whole) < 2 or len(prefix) < 2:
            continue
        std_whole, std_prefix = float(whole.std()), float(prefix.std())
        if std_whole < CONSTANT_THRESHOLD or std_prefix < CONSTANT_THRESHOLD:
            continue
        rows.append(
            {
                "fault_class": fault_class,
                "source": source,
                "sensor": sensor,
                "std_ratio": std_whole / std_prefix,
                # How far the normal period's own mean sits from the divisor's
                # centre, in units of the divisor: the shift the leak adds.
                "mean_shift": abs(float(whole.mean()) - float(prefix.mean())) / std_whole,
                "normal_share": float(normal.mean()),
            }
        )
    return rows


def audit(raw_dir: Path, sensors: list[str], max_instances: int | None, verbose: bool) -> None:
    """Measure the normalization channel over the whole dataset and report it."""
    entries = list_raw_instances(raw_dir, list(FAULT_CLASSES), max_instances)
    rows: list[dict] = []
    n_used = 0
    current_class = None

    for fault_class, df_raw in load_raw_instances(entries):
        if fault_class != current_class:
            current_class = fault_class
            print(f"  Class {fault_class} {FAULT_CLASSES[fault_class]}...")
        df_raw, _ = mask_extreme_values(df_raw)
        df_clean = clean_instance(df_raw)
        if df_clean is None:
            continue
        instance_id = df_clean["instance_id"].iloc[0]
        found = instance_rows(df_clean, fault_class, df_clean["source_type"].iloc[0], sensors)
        if found:
            n_used += 1
            rows.extend(found)
            if verbose:
                worst = max(found, key=lambda r: r["std_ratio"])
                print(
                    f"    {instance_id}: normal {100 * worst['normal_share']:.0f}% | "
                    f"largest std ratio {worst['std_ratio']:.2f} ({worst['sensor']})"
                )

    if not rows:
        print("\nNo instance has both a normal-operation period and a labelled fault period.")
        return

    frame = pd.DataFrame(rows)
    print(f"\n{n_used} instances have a normal-operation period and a labelled fault period.")
    print(
        "\nstd ratio = std over the whole recording (what stage 1 divides by) / std over the\n"
        "normal-operation samples alone (what a leak-free pipeline would divide by).\n"
        "1.0 means the fault period does not change the divisor; the further from 1, and the\n"
        "more it differs between classes, the more the normalization encodes the label.\n"
    )

    def summarize(part: pd.DataFrame) -> pd.DataFrame:
        table = (
            part.groupby("fault_class")
            .agg(
                measurements=("std_ratio", "size"),
                median_std_ratio=("std_ratio", "median"),
                p90_std_ratio=("std_ratio", lambda s: s.quantile(0.9)),
                median_mean_shift=("mean_shift", "median"),
                median_normal_share=("normal_share", "median"),
            )
            .round(3)
        )
        table.index = [f"{c} {FAULT_CLASSES[c]}" for c in table.index]
        return table

    for source, part in frame.groupby("source"):
        print(f"\nPer fault class — {source} instances (over all sensors and instances):")
        print(summarize(part).to_string())

    real = frame[frame["source"] == "WELL"]
    if not real.empty:
        print("\nPer sensor, median std ratio by fault class — real instances only:")
        pivot = real.pivot_table(
            index="sensor", columns="fault_class", values="std_ratio", aggfunc="median"
        ).round(2)
        pivot.columns = [str(c) for c in pivot.columns]
        print(pivot.to_string())

        spread = summarize(real)["median_std_ratio"]
        print(
            f"\nReal instances: the per-class median std ratio spans {spread.min():.2f} to "
            f"{spread.max():.2f}.\n"
            "Classes whose medians differ are classes whose normal-operation windows are divided\n"
            "by systematically different numbers — a signal present in every z-scored feature of\n"
            "the prediction task and absent from the raw ones. Simulated instances start from a\n"
            "steady state, so their normal period is nearly flat and their ratios are far larger\n"
            "still; they are reported separately above for that reason."
        )


def main() -> None:
    """Parse arguments and run the audit."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="root of the 3W dataset (default: FLOWML_RAW_DATA_DIR or config)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="cap instances per class, for a quick look (default: all)",
    )
    parser.add_argument(
        "--sensors",
        default=",".join(KEY_SENSORS),
        help="comma-separated sensors to measure (default: the KEY_SENSORS of config.py)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="where to save the audit (default: results/audits/normalization_leakage_<...>.txt)",
    )
    parser.add_argument("--verbose", action="store_true", help="print one line per instance")
    args = parser.parse_args()

    sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out = args.output_file or AUDITS_DIR / f"normalization_leakage_{stamp}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fh, contextlib.redirect_stdout(Tee(sys.stdout, fh)):
        print(f"Normalization leakage audit — {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")
        print(f"sensors: {', '.join(sensors)}")
        audit(args.raw_dir, sensors, args.max_instances, args.verbose)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
