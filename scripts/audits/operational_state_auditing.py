"""Audit — which operational states the normal-operation samples of the real wells are in.

The prediction task learns from the samples labelled Normal (class 0) of every
recording. 3W's ``state`` column says what the well was doing at each sample
— Open, Shut-In, Flushing, Restart, ... (``WELL_STATES``). If shut-ins or
restarts hid inside the Normal label, the state would be a confounder: a
model could tell "normal before a fault" from "normal" by the well being
closed, and a well-invariant feature set would have to account for it. This
audit cross-tabulates state against class for the real instances, per fault
folder and per well, reading only the two label columns of every file.

Usage
-----
    uv run scripts/audits/operational_state_auditing.py [--raw-dir DIR] [--max-instances N]
        [--output-file PATH] [--verbose]

Output
------
    results/audits/operational_state_<timestamp>.txt
    (everything is printed to the terminal as well)
"""

import argparse
import contextlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from flowml.config import FAULT_CLASSES, RAW_DATA_DIR, RESULTS_DIR, SOURCE_TYPES, WELL_STATES
from flowml.preprocessing import list_raw_instances, parse_source_type, parse_well_id

AUDITS_DIR = RESULTS_DIR / "audits"
UNKNOWN_STATE = "Unknown"


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


def tally_states(raw_dir: Path, max_instances: int | None, verbose: bool) -> pd.DataFrame:
    """Count the samples of every (fault folder, well, class, state) over the real instances.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset.
    max_instances : int or None
        Cap on instances per fault folder, for a quick look.
    verbose : bool
        Print one line per instance.

    Returns
    -------
    pd.DataFrame
        Columns ``fault``, ``well``, ``class``, ``state`` (the state's name)
        and ``n``, one row per combination.
    """
    parts = []
    for fault, path in list_raw_instances(raw_dir, list(FAULT_CLASSES), max_instances):
        if parse_source_type(path.name) != SOURCE_TYPES["real"]:
            continue
        table = pq.read_table(path, columns=["class", "state"]).to_pandas()
        counts = table.groupby(["class", "state"], dropna=False).size().rename("n").reset_index()
        counts["fault"] = fault
        counts["well"] = int(parse_well_id(path.name))
        parts.append(counts)
        if verbose:
            print(
                f"  {path.name}: {len(table):,} samples, states {sorted(table['state'].dropna().unique())}"
            )
    if not parts:
        raise SystemExit("no real instance found under " + str(raw_dir))
    df = pd.concat(parts, ignore_index=True)
    df["state"] = df["state"].map(lambda s: UNKNOWN_STATE if pd.isna(s) else WELL_STATES[int(s)])
    return df[["fault", "well", "class", "state", "n"]]


def state_table(df: pd.DataFrame, index: str) -> pd.DataFrame:
    """Samples per ``index`` and state, with a total and the row-wise shares.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``tally_states``, already filtered to the rows of interest.
    index : str
        ``"fault"`` or ``"well"``.

    Returns
    -------
    pd.DataFrame
        Counts per state, a ``total`` column, then one ``% <state>`` column per state.
    """
    table = df.pivot_table(index=index, columns="state", values="n", aggfunc="sum", fill_value=0)
    order = [name for name in WELL_STATES.values() if name in table.columns]
    if UNKNOWN_STATE in table.columns:
        order.append(UNKNOWN_STATE)
    table = table[order]
    total = table.sum(axis=1)
    shares = (table.div(total, axis=0) * 100).round(1).add_prefix("% ")
    table["total"] = total
    return pd.concat([table, shares], axis=1)


def audit(df: pd.DataFrame) -> None:
    """Print the three cross-tabulations."""
    print(
        f"\n{df['n'].sum():,} samples over {df['well'].nunique()} real wells; "
        f"states seen: {sorted(df['state'].unique())}"
    )

    print("\nAll samples, by fault folder and state:")
    print(state_table(df, "fault").rename(index=FAULT_CLASSES).to_string())

    normal = df[df["class"] == 0]
    print(
        "\nNormal-operation samples only (class 0 — what the prediction task learns from), "
        "by fault folder and state:"
    )
    print(state_table(normal, "fault").rename(index=FAULT_CLASSES).to_string())

    print("\nNormal-operation samples only, by well and state:")
    per_well = state_table(normal, "well")
    print(per_well.to_string())
    open_name = WELL_STATES[0]
    not_open = per_well[per_well.get(f"% {open_name}", 0) < 100]
    if not_open.empty:
        print(f"\n  every well's normal-operation samples are entirely in state {open_name}")
    else:
        print(
            f"\n  wells whose normal-operation samples are not all {open_name}: "
            + ", ".join(f"{w} ({v:.1f} %)" for w, v in not_open[f"% {open_name}"].items())
        )


def main() -> None:
    """Parse arguments, tally the dataset, and write the summary."""
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
        help="cap instances per fault folder, for a quick look (default: all)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="where to save the audit (default: results/audits/operational_state_<timestamp>.txt)",
    )
    parser.add_argument("--verbose", action="store_true", help="print one line per instance")
    args = parser.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out = args.output_file or AUDITS_DIR / f"operational_state_{stamp}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fh, contextlib.redirect_stdout(Tee(sys.stdout, fh)):
        print(
            f"Operational state audit of {args.raw_dir} — {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}"
        )
        if args.max_instances is not None:
            print(f"capped at {args.max_instances} instances per fault folder")
        audit(tally_states(args.raw_dir, args.max_instances, args.verbose))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
