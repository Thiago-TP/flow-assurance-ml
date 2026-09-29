"""Audit — the real instances of every well: spans, overlaps, what the overlaps agree on, the recordings they merge into, and the label gaps inside those recordings.

3W cuts a well's history into instances that mostly overlap in time (986 of
the 1,119 real ones in 2.0.0 share samples with a neighbour). This audit lists
them per well, as it always did, and then asks the three questions the planned
reconstruction of the dataset (``ideas/2026-09-28_new_feature_dataset.md``)
depends on:

1. **Do overlapping instances disagree?** On every timestamp shared by two or
   more instances of a well, the ``class`` labels are compared — labelled in
   both and equal, labelled in both and different, labelled in one only,
   unlabelled in both — and so are the readings of the key sensors.
2. **What do they merge into?** Instances that overlap or touch (a gap of at
   most one second) are joined into recordings: how many, how long, and how
   many fault-class folders each one draws from.
3. **What label gaps do the merged timelines have?** Runs of unlabelled
   samples inside a contiguous 1 Hz stretch, classified by their flanks: the
   same class on both sides (the gaps the reconstruction would relabel),
   different classes, or the edge of the stretch. The same-class gaps longer
   than ``--gap-list-min`` are listed with the instances that cover them, so
   they can be inspected one by one; every gap is also measured against the
   length of the recording that holds it.

Only the timestamp, ``class`` and the key sensors of each file are read, so
the pass over the whole dataset takes seconds.

Usage
-----
    uv run scripts/audits/well_instances_auditing.py [--raw-dir PATH] [--well-ids all|1,2,3]
        [--gap-list-min SECONDS] [--output-dir DIR] [--quiet]

Outputs
-------
    results/audits/well_instances_<timestamp>.txt   the per-well listing and every table
    results/audits/well_instances_<timestamp>.pdf   instances against recordings per well, the
                                                    overlap agreement, recording lengths, label
                                                    gaps per class, gaps against their recordings
"""

import argparse
import contextlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator

from flowml.config import FAULT_CLASSES, KEY_SENSORS, RAW_DATA_DIR, RESULTS_DIR
from flowml.dataset.reconstruction import GAP_RELABEL_MAX_SECONDS
from flowml.preprocessing import (
    list_raw_instances,
    overlapping_mask,
    parse_source_type,
    parse_well_id,
)
from flowml.visualization.common import FAULT_COLORS

AUDITS_DIR = RESULTS_DIR / "audits"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
MERGE_TOLERANCE = pd.Timedelta(seconds=1)  # instances this close in time are one recording
SENSOR_TOLERANCE = 1e-9  # readings further apart than this count as different

# The same ink and grid tokens as ``sensor_distributions_auditing.py``, and the
# first two categorical slots of the project's chart palette for the two series
# that are not fault classes (instances, recordings); fault classes keep the
# colors every other figure of the repo gives them (``FAULT_COLORS``).
INK, INK_SECONDARY, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ("#2a78d6", "#eb6834")


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


@dataclass
class Recording:
    """A maximal run of instances of one well that overlap or touch in time."""

    well: int
    start: pd.Timestamp
    end: pd.Timestamp
    folders: set[int] = field(default_factory=set)
    files: list[str] = field(default_factory=list)

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600


@dataclass
class Gap:
    """A run of unlabelled samples inside a contiguous stretch of a merged timeline."""

    well: int
    start: pd.Timestamp
    end: pd.Timestamp  # last unlabelled second
    left: float  # class before the gap, NaN at the edge of a stretch
    right: float  # class after the gap, NaN at the edge of a stretch
    recording_seconds: int = 0  # length of the merged recording that holds the gap

    @property
    def seconds(self) -> int:
        return int((self.end - self.start).total_seconds()) + 1

    @property
    def ratio(self) -> float:
        """The gap's length as a fraction of its recording's, NaN when the recording is unknown."""
        return self.seconds / self.recording_seconds if self.recording_seconds else np.nan

    @property
    def kind(self) -> str:
        if np.isnan(self.left) or np.isnan(self.right):
            return "edge of a stretch"
        if self.left == self.right:
            return "same class both sides"
        return "different classes"


@dataclass
class WellAudit:
    """Everything the audit found about one well."""

    well: int
    spans: pd.DataFrame  # file, folder, start, end, overlaps
    rows: int
    unique: int
    agreement: Counter
    sensor_diff: Counter
    sensor_compared: int
    recordings: list[Recording]
    gaps: list[Gap]
    label_runs: list[tuple[float, pd.Timestamp, pd.Timestamp]]  # of the merged timeline


def class_name(label: float) -> str:
    """Human-readable name of a 3W ``class`` value (transients carry their fault's name)."""
    if np.isnan(label):
        return "unlabelled"
    code = int(label)
    if code >= 100:
        return f"{FAULT_CLASSES.get(code - 100, code)} (transient)"
    return FAULT_CLASSES.get(code, str(code))


def read_instance(path: Path) -> pd.DataFrame:
    """Read one instance's timestamp index, ``class`` and whichever key sensors it carries."""
    names = pq.read_schema(path).names
    columns = ["timestamp", "class"] + [s for s in KEY_SENSORS if s in names]
    return pq.read_table(path, columns=columns).to_pandas()


def merge_recordings(well: int, spans: pd.DataFrame) -> list[Recording]:
    """Join the instances that overlap or touch into recordings, in time order."""
    recordings: list[Recording] = []
    for row in spans.sort_values("start").itertuples(index=False):
        if recordings and row.start <= recordings[-1].end + MERGE_TOLERANCE:
            current = recordings[-1]
            current.end = max(current.end, row.end)
        else:
            current = Recording(well, row.start, row.end)
            recordings.append(current)
        current.folders.add(int(row.folder))
        current.files.append(row.file)
    return recordings


def label_gaps(well: int, timeline: pd.Series) -> list[Gap]:
    """Find the unlabelled runs inside the contiguous 1 Hz stretches of a merged timeline."""
    ts = timeline.index.to_numpy()
    vals = timeline.to_numpy(dtype=float)
    breaks = np.where(np.diff(ts).astype("timedelta64[s]").astype(int) != 1)[0] + 1
    gaps: list[Gap] = []
    for chunk_ts, chunk in zip(np.split(ts, breaks), np.split(vals, breaks)):
        isnan = np.isnan(chunk)
        if not isnan.any() or isnan.all():
            continue
        edges = np.diff(np.concatenate([[0], isnan.astype(int), [0]]))
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            left = chunk[a - 1] if a > 0 else np.nan
            right = chunk[b] if b < len(chunk) else np.nan
            gaps.append(
                Gap(well, pd.Timestamp(chunk_ts[a]), pd.Timestamp(chunk_ts[b - 1]), left, right)
            )
    return gaps


def label_runs(timeline: pd.Series) -> list[tuple[float, pd.Timestamp, pd.Timestamp]]:
    """Compress a merged timeline into runs of one label (NaN is a label of its own)."""
    vals = timeline.to_numpy(dtype=float)
    keys = np.where(np.isnan(vals), -1.0, vals)
    change = np.concatenate([[True], keys[1:] != keys[:-1]])
    starts = np.where(change)[0]
    ends = np.append(starts[1:], len(vals)) - 1
    return [(vals[a], timeline.index[a], timeline.index[b]) for a, b in zip(starts, ends)]


def audit_well(well: int, entries: list[tuple[int, Path]]) -> WellAudit:
    """Read one well's instances and answer the three questions for it."""
    frames, span_rows = [], []
    for folder, path in entries:
        frame = read_instance(path)
        frame["folder"] = folder
        frames.append(frame)
        span_rows.append(
            {
                "file": path.name,
                "folder": folder,
                "start": frame.index.min(),
                "end": frame.index.max(),
            }
        )
    spans = pd.DataFrame(span_rows).sort_values("file").reset_index(drop=True)
    spans["overlaps"] = overlapping_mask(
        spans["start"].to_numpy(dtype="datetime64[ns]"),
        spans["end"].to_numpy(dtype="datetime64[ns]"),
    )
    df = pd.concat(frames).sort_index(kind="stable")
    sensors = [s for s in KEY_SENSORS if s in df.columns]

    agreement, sensor_diff, sensor_compared = Counter(), Counter(), 0
    shared = df[df.index.duplicated(keep=False)]
    if len(shared):
        g = shared.groupby(level=0, sort=False)
        n_rows, n_labelled, n_distinct = g.size(), g["class"].count(), g["class"].nunique()
        agreement["timestamps shared"] = len(n_rows)
        agreement["labelled in one instance only"] = int(
            ((n_labelled > 0) & (n_labelled < n_rows)).sum()
        )
        agreement["unlabelled in all"] = int((n_labelled == 0).sum())
        agreement["labelled in all, equal"] = int(
            ((n_distinct == 1) & (n_labelled == n_rows)).sum()
        )
        agreement["labelled in all, different"] = int((n_distinct > 1).sum())
        for sensor in sensors:
            count = g[sensor].count()
            spread = (g[sensor].max() - g[sensor].min()).abs()
            sensor_compared += int((count >= 2).sum())
            sensor_diff[sensor] += int(((count >= 2) & (spread > SENSOR_TOLERANCE)).sum())

    timeline = df["class"].groupby(level=0).first().sort_index()
    recordings = merge_recordings(well, spans)
    gaps = label_gaps(well, timeline)
    for gap in gaps:
        holder = next((r for r in recordings if r.start <= gap.start <= r.end), None)
        if holder is not None:
            gap.recording_seconds = int((holder.end - holder.start).total_seconds()) + 1
    return WellAudit(
        well=well,
        spans=spans,
        rows=len(df),
        unique=len(timeline),
        agreement=agreement,
        sensor_diff=sensor_diff,
        sensor_compared=sensor_compared,
        recordings=recordings,
        gaps=gaps,
        label_runs=label_runs(timeline),
    )


def covering_files(audit: WellAudit, gap: Gap) -> list[str]:
    """The instances whose span touches a gap or its two flanking seconds."""
    lo, hi = gap.start - MERGE_TOLERANCE, gap.end + MERGE_TOLERANCE
    hit = (audit.spans["start"] <= hi) & (audit.spans["end"] >= lo)
    return audit.spans.loc[hit, "file"].tolist()


# ----------------------------------------------------------------------------- text report


def print_well(audit: WellAudit, quiet: bool) -> None:
    """The per-well block: the instance listing (unless quiet) and the well's summary lines."""
    n = len(audit.spans)
    print(f"\nWell {audit.well} has {n} instance{'s' * (n != 1)}")
    if not quiet:
        for row in audit.spans.itertuples(index=False):
            flag = "   (overlaps)" if row.overlaps else ""
            print(
                f"  - {row.file:<36} {row.start:{TIME_FORMAT}} - {row.end:{TIME_FORMAT}}"
                f"   folder {row.folder}{flag}"
            )
    n_over = int(audit.spans["overlaps"].sum())
    conflicts = audit.agreement["labelled in all, different"]
    disagreements = sum(audit.sensor_diff.values())
    print(
        f"  overlapping instances {n_over} ({100 * n_over / n:.1f} %) · timestamps shared "
        f"{audit.agreement['timestamps shared']:,} · label conflicts {conflicts:,} · "
        f"sensor disagreements {disagreements:,}"
    )
    print(
        f"  merged recordings {len(audit.recordings)} · unique samples {audit.unique:,} of "
        f"{audit.rows:,} rows · label gaps {len(audit.gaps)}"
    )


def print_dataset_tables(audits: list[WellAudit], gap_list_min: int) -> None:
    """The dataset-wide tables: agreement, recordings, gaps, and the gap list to inspect."""
    agreement, sensor_diff = Counter(), Counter()
    sensor_compared = 0
    for a in audits:
        agreement.update(a.agreement)
        sensor_diff.update(a.sensor_diff)
        sensor_compared += a.sensor_compared
    recordings = [r for a in audits for r in a.recordings]
    gaps = [g for a in audits for g in a.gaps]
    n_instances = sum(len(a.spans) for a in audits)
    rows, unique = sum(a.rows for a in audits), sum(a.unique for a in audits)

    print("\n" + "=" * 100)
    print("Overlap agreement — timestamps shared by two or more instances of the same well")
    for key in (
        "timestamps shared",
        "labelled in one instance only",
        "unlabelled in all",
        "labelled in all, equal",
        "labelled in all, different",
    ):
        print(f"  {key:32s} {agreement[key]:>12,}")
    print(
        f"  sensor disagreements (readings {SENSOR_TOLERANCE:g} apart or more), out of "
        f"{sensor_compared:,} sensor-timestamps compared:"
    )
    for sensor in KEY_SENSORS:
        print(f"    {sensor:11s} {sensor_diff[sensor]:>10,}")

    print("\n" + "=" * 100)
    print("Merged recordings — instances that overlap or touch, joined")
    print(
        f"  real instances {n_instances:,} -> recordings {len(recordings):,} · rows {rows:,} -> "
        f"unique samples {unique:,}"
    )
    hours = np.array([r.hours for r in recordings])
    print(
        f"  recording length (h): median {np.median(hours):.1f} · p90 "
        f"{np.percentile(hours, 90):.1f} · max {hours.max():.1f}"
    )
    per_count = Counter(len(r.folders - {0}) for r in recordings)
    print(
        "  fault-class folders per recording (0 = only Normal): "
        + ", ".join(f"{k}: {v}" for k, v in sorted(per_count.items()))
    )
    multi = [r for r in recordings if len(r.folders - {0}) > 1]
    if multi:
        print("  recordings drawing from more than one fault-class folder:")
        for r in multi:
            audit = next(a for a in audits if a.well == r.well)
            print(
                f"    well {r.well} · {r.start:{TIME_FORMAT}} - {r.end:{TIME_FORMAT}} "
                f"({r.hours:.1f} h) · folders {sorted(r.folders)} · instances: {', '.join(r.files)}"
            )
            for label, start, end in audit.label_runs:
                if r.start <= start <= r.end:
                    length = (end - start).total_seconds() + 1
                    print(
                        f"      {class_name(label):40s} {start:{TIME_FORMAT}} - {end:{TIME_FORMAT}}"
                        f"  ({length / 3600:.2f} h)"
                    )

    print("\n" + "=" * 100)
    print("Label gaps — unlabelled runs inside contiguous 1 Hz stretches of the merged timelines")
    kinds = Counter()
    for g in gaps:
        if g.kind == "same class both sides":
            kinds[f"same class both sides · {class_name(g.left)}"] += 1
        elif g.kind == "different classes":
            kinds[f"different classes · {class_name(g.left)} -> {class_name(g.right)}"] += 1
        else:
            kinds[g.kind] += 1
    for key, value in sorted(kinds.items()):
        print(f"  {key:70s} {value:>8,}")
    same = [g for g in gaps if g.kind == "same class both sides"]
    if same:
        print("  lengths (s) of the same-class gaps, per class: count / median / p90 / max")
        by_class = defaultdict(list)
        for g in same:
            by_class[g.left].append(g.seconds)
        for label, lengths in sorted(by_class.items()):
            arr = np.array(lengths)
            print(
                f"    {class_name(label):40s} {len(arr):>6,} / {np.median(arr):>7.0f} / "
                f"{np.percentile(arr, 90):>7.0f} / {arr.max():>8,}"
            )
        print(
            "  length of the same-class gaps relative to the recording that holds them (%): "
            "count / median / p90 / max / gaps above 1 %"
        )
        for label, gaps_of_class in sorted(by_class.items()):
            ratios = np.array(
                [100 * g.ratio for g in same if g.left == label and g.recording_seconds]
            )
            if len(ratios) == 0:
                continue
            print(
                f"    {class_name(label):40s} {len(ratios):>6,} / {np.median(ratios):>7.3f} / "
                f"{np.percentile(ratios, 90):>7.3f} / {ratios.max():>8.3f} / {(ratios > 1).sum():>6,}"
            )
        long_gaps = sorted((g for g in same if g.seconds >= gap_list_min), key=lambda g: -g.seconds)
        print(
            f"\n  same-class gaps of at least {gap_list_min} s, longest first "
            f"({len(long_gaps)} of {len(same)}), with the instances that cover them:"
        )
        for g in long_gaps:
            audit = next(a for a in audits if a.well == g.well)
            print(
                f"    well {g.well:>2} · {class_name(g.left):28s} · {g.start:{TIME_FORMAT}} - "
                f"{g.end:{TIME_FORMAT}} · {g.seconds:>6,} s · {100 * g.ratio:5.2f} % of its "
                f"recording · {', '.join(covering_files(audit, g))}"
            )


# ----------------------------------------------------------------------------- figures


def tint(color: str, strength: float = 0.35) -> tuple[float, float, float]:
    """Mix a color toward white; ``strength`` 1.0 keeps it untouched."""
    base = np.array(mcolors.to_rgb(color))
    return tuple(1.0 - (1.0 - base) * strength)


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """The recessive axes of every audit figure: no box, light grid, muted ticks."""
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(axis="both", labelsize=7, colors=INK_SECONDARY)
    ax.tick_params(axis="y", length=0)


def log_ticks(ax: plt.Axes, unit: str) -> None:
    """Human-readable ticks on a log axis of seconds or hours, within the current limits."""
    if unit == "seconds":
        marks = [
            (1, "1 s"),
            (10, "10 s"),
            (60, "1 min"),
            (300, "5 min"),
            (600, "10 min"),
            (1800, "30 min"),
            (3600, "1 h"),
            (10800, "3 h"),
            (43200, "12 h"),
        ]
    else:
        marks = [
            (1 / 60, "1 min"),
            (0.1, "6 min"),
            (0.5, "30 min"),
            (1, "1 h"),
            (6, "6 h"),
            (24, "1 d"),
            (72, "3 d"),
            (168, "1 w"),
            (336, "2 w"),
            (720, "30 d"),
        ]
    low, high = ax.get_xlim()
    ticks = [(value, text) for value, text in marks if low <= value <= high]
    ax.set_xticks([value for value, _ in ticks], [text for _, text in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def page_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    """A left-aligned bold title with a muted explanatory line under it."""
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.text(0.02, 0.905, subtitle, fontsize=7, color=INK_SECONDARY, va="top")


def draw_wells_page(audits: list[WellAudit]) -> plt.Figure:
    """Page 1: instances in 3W against recordings after merging, per well."""
    wells = [a.well for a in audits]
    instances = np.array([len(a.spans) for a in audits])
    recordings = np.array([len(a.recordings) for a in audits])
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.12)
    x, w = np.arange(len(wells)), 0.38
    ax.bar(x - w / 2, instances, width=w, color=SERIES[0], linewidth=0, label="instances in 3W")
    ax.bar(
        x + w / 2,
        recordings,
        width=w,
        color=SERIES[1],
        linewidth=0,
        label="recordings after merging overlaps",
    )
    # Selective direct labels: only the wells whose instance count stands out.
    for xi, (n, m) in enumerate(zip(instances, recordings)):
        if n >= 20:
            ax.text(
                xi - w / 2, n + 2, str(n), ha="center", va="bottom", fontsize=6, color=INK_SECONDARY
            )
            ax.text(
                xi + w / 2, m + 2, str(m), ha="center", va="bottom", fontsize=6, color=INK_SECONDARY
            )
    ax.set_xticks(x, [str(well) for well in wells])
    ax.set_xlim(-0.7, len(wells) - 0.3)
    style_axes(ax)
    ax.set_xlabel("well", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("count", fontsize=8, color=INK_SECONDARY)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    page_title(
        fig,
        f"Instances and merged recordings per well · {instances.sum():,} instances become "
        f"{recordings.sum():,} recordings",
        "instances of a well that overlap or touch in time (gap of at most one second) are joined "
        "into one recording; the number of samples changes far less than the number of files, "
        "since the overlaps are duplicated samples",
    )
    return fig


def draw_agreement_page(audits: list[WellAudit]) -> plt.Figure:
    """Page 2: what the overlaps agree on — a row of numbers, since the numbers are the finding."""
    agreement, sensor_diff = Counter(), Counter()
    compared = 0
    for a in audits:
        agreement.update(a.agreement)
        sensor_diff.update(a.sensor_diff)
        compared += a.sensor_compared
    shared = agreement["timestamps shared"] or 1
    tiles = [
        (
            f"{agreement['timestamps shared']:,}",
            "timestamps shared by two\nor more instances of a well",
        ),
        (
            f"{100 * agreement['labelled in one instance only'] / shared:.2f} %",
            "labelled in one of them only\n(the other is unlabelled)",
        ),
        (f"{agreement['unlabelled in all']:,}", "unlabelled in all of them"),
        (f"{agreement['labelled in all, equal']:,}", "labelled in all, equal"),
        (
            f"{agreement['labelled in all, different']:,}",
            "labelled in all, different\n(a label conflict)",
        ),
        (
            f"{sum(sensor_diff.values()):,}",
            f"differing key-sensor readings\nout of {compared:,} compared",
        ),
    ]
    fig = plt.figure(figsize=(11.5, 3.6))
    page_title(
        fig,
        "Overlap agreement — what two instances say about the same second",
        "labels: 3W's class column, compared on every shared timestamp · sensors: the eight key "
        f"sensors, readings compared at {SENSOR_TOLERANCE:g} tolerance",
    )
    left, width = 0.03, 0.94 / len(tiles)
    for i, (value, caption) in enumerate(tiles):
        x = left + i * width
        fig.text(x, 0.52, value, fontsize=20, fontweight="bold", color=INK, va="center")
        fig.text(x, 0.36, caption, fontsize=7, color=INK_SECONDARY, va="top")
        fig.add_artist(plt.Line2D([x, x + width - 0.02], [0.66, 0.66], color=GRID, linewidth=0.8))
    return fig


def draw_recordings_page(audits: list[WellAudit]) -> plt.Figure:
    """Page 3: how long the merged recordings are and how many fault folders each draws from."""
    recordings = [r for a in audits for r in a.recordings]
    hours = np.array([r.hours for r in recordings])
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11.5, 7.0), gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.55}
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.09)

    low = max(hours.min(), 1 / 60)
    bins = np.logspace(np.log10(low), np.log10(hours.max() * 1.05), 36)
    top.hist(hours, bins=bins, color=tint(SERIES[0]), edgecolor=SERIES[0], linewidth=0.6)
    top.set_xscale("log")
    for value, name in ((np.median(hours), "median"), (np.percentile(hours, 90), "p90")):
        top.axvline(value, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
        top.text(
            value,
            top.get_ylim()[1] * 0.97,
            f" {name} {value:.1f} h",
            fontsize=7,
            color=INK,
            va="top",
        )
    style_axes(top)
    log_ticks(top, "hours")
    top.set_xlabel("recording length (log scale)", fontsize=8, color=INK_SECONDARY)
    top.set_ylabel("recordings", fontsize=8, color=INK_SECONDARY)

    counts = Counter(len(r.folders - {0}) for r in recordings)
    ks = list(range(max(max(counts), 2) + 1))
    values = [counts.get(k, 0) for k in ks]
    bottom.bar(ks, values, width=0.55, color=SERIES[0], linewidth=0)
    for k, v in zip(ks, values):
        bottom.text(
            k,
            v + max(values) * 0.02,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK_SECONDARY,
        )
    bottom.set_xticks(
        ks, ["only Normal" if k == 0 else f"{k} fault folder{'s' * (k != 1)}" for k in ks]
    )
    style_axes(bottom)
    bottom.set_ylabel("recordings", fontsize=8, color=INK_SECONDARY)
    page_title(
        fig,
        f"Merged recordings · {len(recordings):,} recordings from "
        f"{sum(len(a.spans) for a in audits):,} instances",
        "top: length of every merged recording · bottom: how many fault-class folders (1 to 9) "
        "the instances of each recording came from — a recording that spans two events is the "
        "one case the reconstruction has to place explicitly",
    )
    return fig


def draw_gaps_page(audits: list[WellAudit], gap_list_min: int) -> plt.Figure:
    """Page 4: the unlabelled gaps of the merged timelines — lengths per class, and their kinds."""
    gaps = [g for a in audits for g in a.gaps]
    same = [g for g in gaps if g.kind == "same class both sides"]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11.5, 7.6), gridspec_kw={"height_ratios": [3, 1.8], "hspace": 0.6}
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.08)
    # The kind labels of the lower chart are long; give that axes a wider left margin.
    position = bottom.get_position()
    bottom.set_position([0.20, position.y0, 0.78, position.height])

    if same:
        lengths = np.array([g.seconds for g in same])
        bins = np.logspace(0, np.log10(lengths.max() * 1.1), 40)
        handles = []
        for label in sorted({g.left for g in same}):
            subset = np.array([g.seconds for g in same if g.left == label])
            color = FAULT_COLORS.get(int(label) % 100, INK_MUTED)
            top.hist(subset, bins=bins, color=tint(color, 0.45), edgecolor=color, linewidth=0.7)
            handles.append(
                Patch(
                    facecolor=tint(color, 0.45),
                    edgecolor=color,
                    label=f"{class_name(label)} ({len(subset):,})",
                )
            )
        top.set_xscale("log")
        p90 = np.percentile(lengths, 90)
        for value, name, height in (
            (p90, f"p90 {p90:.0f} s", 0.97),
            (gap_list_min, f"--gap-list-min {gap_list_min} s", 0.88),
        ):
            top.axvline(value, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
            top.text(value, top.get_ylim()[1] * height, f" {name}", fontsize=7, color=INK, va="top")
        # Above the axes, so no bar can run under it.
        top.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=len(handles),
            fontsize=7,
            frameon=False,
        )
    else:
        top.text(
            0.5, 0.5, "no same-class gap", transform=top.transAxes, ha="center", color=INK_MUTED
        )
    style_axes(top)
    if same:
        log_ticks(top, "seconds")
    top.set_xlabel("gap length (log scale)", fontsize=8, color=INK_SECONDARY)
    top.set_ylabel("gaps", fontsize=8, color=INK_SECONDARY)

    kinds: Counter[str] = Counter()
    kind_colors: dict[str, str | tuple] = {}
    for g in gaps:
        if g.kind == "same class both sides":
            key = f"same class · {class_name(g.left)}"
            kind_colors[key] = FAULT_COLORS.get(int(g.left) % 100, INK_MUTED)
        else:
            key = g.kind
            kind_colors[key] = INK_MUTED
        kinds[key] += 1
    names = sorted(kinds, key=lambda k: -kinds[k])
    values = [kinds[k] for k in names]
    y = np.arange(len(names))
    bottom.barh(y, values, height=0.6, color=[kind_colors[k] for k in names], linewidth=0)
    for yi, v in zip(y, values):
        bottom.text(
            v + max(values) * 0.01, yi, f"{v:,}", va="center", fontsize=7, color=INK_SECONDARY
        )
    bottom.set_yticks(y, names)
    bottom.invert_yaxis()
    style_axes(bottom, grid_axis="x")
    bottom.set_xlabel("gaps", fontsize=8, color=INK_SECONDARY)
    page_title(
        fig,
        f"Label gaps inside the merged timelines · {len(gaps):,} unlabelled runs, "
        f"{len(same):,} with the same class on both sides",
        "top: length of every gap whose two flanks carry the same class (the class the "
        "reconstruction would give it) · bottom: every gap by kind — flanks agree, flanks "
        "disagree, or one flank only (the edge of a contiguous stretch)",
    )
    return fig


def draw_gap_ratio_page(audits: list[WellAudit], gap_list_min: int) -> plt.Figure:
    """Page 5: the same-class gaps against the recordings that hold them."""
    same = [
        g
        for a in audits
        for g in a.gaps
        if g.kind == "same class both sides" and g.recording_seconds
    ]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11.5, 7.8), gridspec_kw={"height_ratios": [2.2, 2.6], "hspace": 0.5}
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.08)
    if same:
        classes = sorted({g.left for g in same})
        colors = {label: FAULT_COLORS.get(int(label) % 100, INK_MUTED) for label in classes}
        ratios = np.array([100 * g.ratio for g in same])
        bins = np.logspace(np.log10(max(ratios.min(), 1e-4)), np.log10(ratios.max() * 1.1), 40)
        handles = []
        for label in classes:
            subset = np.array([100 * g.ratio for g in same if g.left == label])
            top.hist(
                subset,
                bins=bins,
                color=tint(colors[label], 0.45),
                edgecolor=colors[label],
                linewidth=0.7,
            )
            handles.append(
                Patch(
                    facecolor=tint(colors[label], 0.45),
                    edgecolor=colors[label],
                    label=f"{class_name(label)} ({len(subset):,})",
                )
            )
        top.set_xscale("log")
        p90 = np.percentile(ratios, 90)
        top.axvline(p90, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
        top.text(
            p90, top.get_ylim()[1] * 0.97, f" p90 {p90:.2f} %", fontsize=7, color=INK, va="top"
        )
        top.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=len(handles),
            fontsize=7,
            frameon=False,
        )
        top.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g} %"))
        top.xaxis.set_minor_locator(NullLocator())

        for label in classes:
            points = [g for g in same if g.left == label]
            bottom.scatter(
                [g.recording_seconds / 3600 for g in points],
                [g.seconds for g in points],
                s=14,
                facecolor=tint(colors[label], 0.55),
                edgecolor=colors[label],
                linewidth=0.6,
            )
        bottom.set_xscale("log")
        bottom.set_yscale("log")
        for value, name in (
            (gap_list_min, f"--gap-list-min {gap_list_min} s"),
            (GAP_RELABEL_MAX_SECONDS, f"relabel bound {GAP_RELABEL_MAX_SECONDS} s"),
        ):
            bottom.axhline(value, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
            bottom.text(bottom.get_xlim()[0], value, f" {name}", fontsize=7, color=INK, va="bottom")
        log_ticks(bottom, "hours")
        low, high = bottom.get_ylim()
        marks = [
            (1, "1 s"),
            (10, "10 s"),
            (60, "1 min"),
            (300, "5 min"),
            (1800, "30 min"),
            (3600, "1 h"),
        ]
        ticks = [(value, text) for value, text in marks if low <= value <= high]
        bottom.set_yticks([value for value, _ in ticks], [text for _, text in ticks])
        bottom.yaxis.set_minor_locator(NullLocator())
    else:
        top.text(
            0.5, 0.5, "no same-class gap", transform=top.transAxes, ha="center", color=INK_MUTED
        )
    style_axes(top)
    top.set_xlabel(
        "gap length as a share of its recording (log scale)", fontsize=8, color=INK_SECONDARY
    )
    top.set_ylabel("gaps", fontsize=8, color=INK_SECONDARY)
    style_axes(bottom, grid_axis="both")
    bottom.set_xlabel(
        "length of the recording that holds the gap (log scale)", fontsize=8, color=INK_SECONDARY
    )
    bottom.set_ylabel("gap length (log scale)", fontsize=8, color=INK_SECONDARY)
    page_title(
        fig,
        f"Same-class gaps against their recordings · {len(same):,} gaps",
        "top: each gap's length divided by the length of the merged recording it sits in · bottom: "
        "the same gaps as points, recording length against gap length, with the list cutoff and the "
        "relabel bound of the reconstruction",
    )
    return fig


# ----------------------------------------------------------------------------- entry point


def parse_well_ids(text: str, available: list[int]) -> list[int]:
    """``all`` or a comma-separated list, checked against the wells the dataset has."""
    if text.strip().lower() == "all":
        return available
    wanted = sorted({int(part) for part in text.split(",") if part.strip()})
    missing = [w for w in wanted if w not in available]
    if missing:
        raise SystemExit(f"no real instance of well(s) {missing}; available: {available}")
    return wanted


def main() -> None:
    """Parse arguments, audit every requested well, and write the report and the figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="root of the 3W dataset (default: FLOWML_RAW_DATA_DIR or config)",
    )
    parser.add_argument(
        "--well-ids",
        default="all",
        help="comma-separated well ids to audit, or 'all' (default: all)",
    )
    parser.add_argument(
        "--gap-list-min",
        type=int,
        default=300,
        help="list the same-class gaps at least this long, in seconds (default: 300)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AUDITS_DIR,
        help="where the report and the PDF go (default: results/audits)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="skip the per-instance listing of every well"
    )
    args = parser.parse_args()

    by_well: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for folder, path in list_raw_instances(args.raw_dir, list(FAULT_CLASSES)):
        if parse_source_type(path.name) == "WELL":
            by_well[int(parse_well_id(path.name))].append((folder, path))
    wells = parse_well_ids(args.well_ids, sorted(by_well))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    txt_path = args.output_dir / f"well_instances_{stamp}.txt"
    pdf_path = args.output_dir / f"well_instances_{stamp}.pdf"

    with (
        open(txt_path, "w", encoding="utf-8") as fh,
        contextlib.redirect_stdout(Tee(sys.stdout, fh)),
    ):
        print(
            f"Well instances audit of {args.raw_dir} — {datetime.now().astimezone():{TIME_FORMAT}}"
        )
        print(
            f"wells {wells[0]}..{wells[-1]} ({len(wells)}) · key sensors {', '.join(KEY_SENSORS)}"
        )
        audits = []
        for well in wells:
            audit = audit_well(well, by_well[well])
            audits.append(audit)
            print_well(audit, args.quiet)
        print_dataset_tables(audits, args.gap_list_min)

    with PdfPages(pdf_path) as pdf:
        for figure in (
            draw_wells_page(audits),
            draw_agreement_page(audits),
            draw_recordings_page(audits),
            draw_gaps_page(audits, args.gap_list_min),
            draw_gap_ratio_page(audits, args.gap_list_min),
        ):
            pdf.savefig(figure)
            plt.close(figure)
    print(f"\nSaved: {txt_path}\nSaved: {pdf_path}")


if __name__ == "__main__":
    main()
