"""Fault timeline: how the real instances of a well tile its history, one page per well."""

import re
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.textpath import TextToPath

from flowml.config import FAULT_CLASSES, RAW_DATA_DIR, WELL_HISTORY_FIGURES_DIR
from flowml.preprocessing import overlapping_mask, pack_lanes, parse_well_id
from flowml.visualization.common import FAULT_COLORS, _tint, _write_pdf, list_instances

# Tint strength of an instance bar in the faults-per-well timeline, keyed by
# how far the fault developed inside that instance (see ``_fault_reach``): a
# window that closes before the fault installs itself is worth telling apart
# from one that records the fault in place.
FAULT_REACH_TINTS = {"steady": 1.00, "transient": 0.55, "normal": 0.25}
FAULT_REACH_LABELS = {
    "steady": "steady state reached",
    "transient": "transient state reached",
    "normal": "no fault reached",
}

# Layout of a well timeline page (see ``_plot_well_timeline``). The side
# margins are fractions of the page width, fixed rather than fitted so the
# width of the axis is known before drawing; the strip right of ``right``
# holds the legend, wide enough for a fault name and its reach on one line.
# The room for the title above the axis and for the dates below it is instead
# absolute, in inches, so it stays sufficient however few stack levels the
# page has, as does the height of a level.
TIMELINE_WIDTH = 13.0
TIMELINE_MARGINS = {"left": 0.055, "right": 0.755}
TIMELINE_INSETS = {"top": 1.00, "bottom": 1.15}
TIMELINE_LANE_HEIGHT = 0.62
MAX_TIMELINE_LANES = 8  # levels a page shares with the others before growing

# Measures text with the actual font outlines and no renderer; see
# ``_text_width_pt``.
_TEXT_MEASURE = TextToPath()


def _filename_stamp(filename: str, fallback: pd.Timestamp) -> pd.Timestamp:
    """Read the timestamp a real instance filename is keyed by.

    Real instances are named ``WELL-000{id}_{YYYYMMDDhhmmss}.parquet``, the
    stamp being the first timestamp of the recording. Labeling a bar with the
    stamp rather than with the data's own first index keeps the label a
    literal piece of the filename, so a suspicious bar can be looked up.

    Parameters
    ----------
    filename : str
        Name of the raw parquet file.
    fallback : pd.Timestamp
        Value returned when the name carries no readable stamp.

    Returns
    -------
    pd.Timestamp
        The parsed stamp, or ``fallback``.
    """
    match = re.search(r"_(\d{14})", Path(filename).stem)
    if not match:
        return fallback
    stamp = pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S", errors="coerce")
    return fallback if pd.isna(stamp) else stamp


def _fault_reach(class_values: np.ndarray) -> str:
    """Tell how far a fault developed inside one instance, from its labels.

    A 3W fault is first labeled transient (``class`` 101-109) while it
    installs itself and then steady (``class`` 1-9) once it has, so an
    instance whose window closes early carries only the weaker label — or no
    fault label at all, which happens when the recording sits entirely in the
    normal period preceding the event.

    Parameters
    ----------
    class_values : np.ndarray
        The ``class`` column of the instance as floats, NaN included.

    Returns
    -------
    str
        ``"steady"``, ``"transient"``, or ``"normal"``, the strongest state
        the labels reach. Faults 3 and 4 have no transient period in the 3W
        dataset, so their instances are never ``"transient"``.
    """
    present = class_values[~np.isnan(class_values)]
    if np.any((present >= 1) & (present <= 9)):
        return "steady"
    if np.any((present >= 101) & (present <= 109)):
        return "transient"
    return "normal"


def _bar_color(fault_class: int, reach: str) -> tuple[float, float, float]:
    """Fill color of one instance bar: its fault hue, tinted by fault reach.

    Parameters
    ----------
    fault_class : int
        Fault-class folder the instance comes from, which fixes the hue.
    reach : str
        Strongest labeled state of the instance, from ``_fault_reach``.

    Returns
    -------
    (float, float, float)
        RGB fill. Normal instances (folder 0) keep full strength whatever
        their labels say: they have no fault to develop, and tinting them
        would render the largest group of the dataset as near-white bars.
    """
    strength = 1.0 if fault_class == 0 else FAULT_REACH_TINTS[reach]
    return _tint(FAULT_COLORS[fault_class], strength)


def _text_color(rgb: tuple[float, float, float]) -> str:
    """Pick black or white text for a background, whichever reads better."""
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "#ffffff" if luminance < 0.55 else "#1a1a1a"


def _text_width_pt(text: str, fontsize: float) -> float:
    """Width of a label in points, as matplotlib's default font sets it.

    Deciding whether a label fits inside its bar needs the label width before
    anything is drawn. ``TextToPath`` measures it from the font outlines
    themselves, with no renderer involved, so the answer is exact and does not
    depend on the backend.

    Parameters
    ----------
    text : str
        Label to measure.
    fontsize : float
        Font size in points.

    Returns
    -------
    float
        Advance width of the label, in points.
    """
    width, _, _ = _TEXT_MEASURE.get_text_width_height_descent(
        text, FontProperties(size=fontsize), ismath=False
    )
    return width


def _recording_blocks(
    starts: np.ndarray, ends: np.ndarray, gap_hours: float, gap_share: float = 0.12
) -> pd.DataFrame:
    """Group the instances into recording blocks laid on a gap-compressed axis.

    A well is recorded in bursts: well 2 spans nearly four years but holds
    only 4 % of them as samples, so on a true calendar axis every instance
    shrinks to an invisible sliver. Instances closer than ``gap_hours`` are
    merged into one block, blocks are laid side by side in chronological
    order, and each silence between them collapses to a fixed narrow blank.
    The time scale stays uniform inside and across blocks — a bar length is
    still a duration, and two bars overlap on the page exactly when the
    instances overlap in time — only the silences lose their real width.

    Parameters
    ----------
    starts, ends : np.ndarray
        First and last timestamp of every instance, ``datetime64``.
    gap_hours : float
        A silence this long or longer breaks the recording into two blocks.
    gap_share : float
        Fraction of the drawn width given to the collapsed silences in total,
        split evenly between them, so a well recorded in many bursts does not
        spend the page on blanks.

    Returns
    -------
    pd.DataFrame
        One row per block, chronological, with its ``start`` and ``end``
        timestamps, its ``hours`` of real duration, and ``x0``, the position
        of its start on the compressed axis (also in hours).
    """
    limit = np.timedelta64(int(gap_hours * 3600), "s")
    blocks: list[list[np.datetime64]] = []
    for i in np.argsort(starts, kind="stable"):
        if blocks and starts[i] - blocks[-1][1] < limit:
            blocks[-1][1] = max(blocks[-1][1], ends[i])
        else:
            blocks.append([starts[i], ends[i]])

    frame = pd.DataFrame(blocks, columns=["start", "end"])
    hours = (frame["end"] - frame["start"]).dt.total_seconds().to_numpy() / 3600.0
    gap = gap_share * max(hours.sum(), 1.0) / max(len(frame) - 1, 1)
    frame["hours"] = hours
    frame["x0"] = np.concatenate(([0.0], np.cumsum(hours + gap)[:-1]))
    return frame


def _block_positions(blocks: pd.DataFrame, timestamps: np.ndarray) -> np.ndarray:
    """Map timestamps onto the compressed axis of ``_recording_blocks``.

    Parameters
    ----------
    blocks : pd.DataFrame
        Blocks from ``_recording_blocks``.
    timestamps : np.ndarray
        Timestamps to place, each inside one block by construction.

    Returns
    -------
    np.ndarray
        Position of every timestamp, in hours from the start of the axis.
    """
    block_starts = blocks["start"].to_numpy()
    index = np.clip(np.searchsorted(block_starts, timestamps, side="right") - 1, 0, len(blocks) - 1)
    return blocks["x0"].to_numpy()[index] + (timestamps - block_starts[index]) / np.timedelta64(
        1, "h"
    )


def _timeline_legend(fig, present: Iterable[tuple[int, str]], top: float) -> None:
    """Key the exact colors of a well timeline page, one entry per color drawn.

    A bar carries a fault hue tinted by how far the fault got, and naming the
    hue and the tint in two separate legends leaves the reader to imagine
    their product — awkward for the paler tints and hopeless for the greyish
    hues. Every combination present on the page therefore gets its own swatch,
    in the very color its bars carry. Normal instances are never tinted (see
    ``_bar_color``), so their entry names no reach.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure the legend is added to.
    present : Iterable[(int, str)]
        Fault class and reach of every color drawn on the page, in the order
        the entries should be listed.
    top : float
        Top edge of the axis in figure coordinates, which the legend aligns
        with; it sits in the strip right of the axis.
    """
    fig.legend(
        handles=[
            Patch(
                facecolor=_bar_color(fault_class, reach),
                edgecolor=FAULT_COLORS[fault_class],
                lw=0.4,
                label=FAULT_CLASSES[fault_class]
                + (f" ({FAULT_REACH_LABELS[reach]})" if fault_class != 0 else ""),
            )
            for fault_class, reach in present
        ],
        title="Fault type (severity reach)",
        loc="upper left",
        bbox_to_anchor=(TIMELINE_MARGINS["right"] + 0.015, top),
        fontsize=7,
        title_fontsize=7,
        alignment="left",
        frameon=False,
    )


def _plot_well_timeline(
    well_id: int,
    rows: pd.DataFrame,
    lane_slots: int | None = None,
    gap_hours: float = 12.0,
    label_fontsize: float = 5.6,
):
    """Build the one-page figure of how one well's instances tile its history.

    Parameters
    ----------
    well_id : int
        Well number shown in the title.
    rows : pd.DataFrame
        The instances of this well, one row each, with the columns
        ``fault_class``, ``stamp``, ``start``, ``end`` and ``reach``.
    lane_slots : int | None
        Stack levels the y axis holds even when this well fills fewer, so
        every page of the PDF comes out the same size; ``None`` fits the axis
        to the well.
    gap_hours : float
        Silence that breaks the axis into two blocks (see
        ``_recording_blocks``).
    label_fontsize : float
        Size of the timestamp written inside a bar; smaller fits more bars.

    Returns
    -------
    matplotlib.figure.Figure
        The finished figure, ready to save.
    """
    rows = rows.sort_values("start").reset_index(drop=True)
    starts = rows["start"].to_numpy(dtype="datetime64[ns]")
    ends = rows["end"].to_numpy(dtype="datetime64[ns]")

    blocks = _recording_blocks(starts, ends, gap_hours)
    x0 = _block_positions(blocks, starts)
    x1 = _block_positions(blocks, ends)
    lane_of = pack_lanes(starts, ends)
    n_lanes = int(lane_of.max()) + 1

    span = float(blocks["x0"].iloc[-1] + blocks["hours"].iloc[-1]) or 1.0
    pad = 0.01 * span

    # Every page of the PDF gets the same number of stack levels — the deepest
    # pile-up of the dataset — so the pages are the same size and a well that
    # never stacks is visibly shallower than one that does.
    slots = max(n_lanes, lane_slots or n_lanes)
    height = TIMELINE_LANE_HEIGHT * (slots + 0.4) + sum(TIMELINE_INSETS.values())
    fig, ax = plt.subplots(figsize=(TIMELINE_WIDTH, height))
    fig.subplots_adjust(
        **TIMELINE_MARGINS,
        top=1 - TIMELINE_INSETS["top"] / height,
        bottom=TIMELINE_INSETS["bottom"] / height,
    )
    # Deciding whether a timestamp fits inside its bar needs the axis width.
    axis_width_pt = 72 * TIMELINE_WIDTH * (TIMELINE_MARGINS["right"] - TIMELINE_MARGINS["left"])

    for block in blocks.itertuples():  # a band per burst of recording
        ax.axvspan(block.x0, block.x0 + block.hours, color="#f4f4f4", lw=0, zorder=0)
    for previous, block in zip(blocks.itertuples(), blocks.iloc[1:].itertuples()):
        ax.axvline(
            0.5 * (previous.x0 + previous.hours + block.x0),
            color="#9a9a9a",
            lw=0.6,
            ls=(0, (2, 2)),
            zorder=1,
        )
    ax.hlines(range(n_lanes), -pad, span + pad, color="#dcdcdc", lw=0.6, zorder=1)

    for row, left, right, lane in zip(rows.itertuples(), x0, x1, lane_of):
        width = max(right - left, 0.001 * span)  # keep a very short instance visible
        color = _bar_color(row.fault_class, row.reach)
        ax.broken_barh(
            [(left, width)],
            (lane - 0.31, 0.62),
            facecolors=[color],
            edgecolor=FAULT_COLORS[row.fault_class],
            lw=0.35,
            zorder=2,
        )
        # The bar carries the instance's own name — the timestamp its filename
        # is keyed by — in the longest format the bar has room for.
        room_pt = width / (span + 2 * pad) * axis_width_pt - 2.0
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m-%d %H:%M", "%H:%M"):
            text = row.stamp.strftime(fmt)
            if _text_width_pt(text, label_fontsize) <= room_pt:
                ax.text(
                    left + 0.5 * width,
                    lane,
                    text,
                    ha="center",
                    va="center",
                    fontsize=label_fontsize,
                    color=_text_color(color),
                    zorder=3,
                )
                break

    # Ticks sit at the block starts: they are the only positions whose date the
    # compressed axis can state without misleading about the collapsed blanks.
    first, last = pd.Timestamp(starts.min()), pd.Timestamp(ends.max())
    calendar_span = (last - first) / pd.Timedelta(days=1)
    fmt = "%Y-%m-%d %H:%M" if calendar_span < 2 else "%Y-%m-%d"
    # Short bursts can start a few hours apart on the compressed axis, close
    # enough for their labels to pile up, so a tick is kept only once there is
    # room for its own label — the dashed separators still mark every block.
    tick_fontsize = 7.0
    label_room = _text_width_pt(first.strftime(fmt), tick_fontsize) * np.cos(np.radians(35)) + 2
    min_step = label_room / axis_width_pt * (span + 2 * pad)
    ticks: list[float] = []
    tick_labels: list[str] = []
    for x, stamp in zip(blocks["x0"], blocks["start"]):
        if not ticks or x - ticks[-1] >= min_step:
            ticks.append(x)
            tick_labels.append(stamp.strftime(fmt))
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right")
    ax.set_xlim(-pad, span + pad)
    ax.set_xlabel("Time (dashed lines mark collapsed silences between recordings)", fontsize=9)
    ax.set_yticks(range(n_lanes))
    ax.set_yticklabels([str(lane + 1) for lane in range(n_lanes)])
    ax.set_ylim(-0.7, slots - 0.3)
    ax.set_ylabel("stack level", fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(labelsize=tick_fontsize)

    overlapping = overlapping_mask(starts, ends)

    recorded = float(blocks["hours"].sum())
    faults = sorted(rows["fault_class"].unique())
    subtitle = (
        f"{overlapping.sum()} of {len(rows)} instances overlap another "
        f"(deepest pile-up {n_lanes}) | "
        f"{recorded:,.1f} h recorded in {len(blocks)} bursts over "
        f"{calendar_span:,.0f} days ({recorded / max(24 * calendar_span, 1e-9):.1%}) | "
        f"{first:%Y-%m-%d} to {last:%Y-%m-%d}\n"
        f"fault folders: {', '.join(FAULT_CLASSES[f] for f in faults)}"
    )
    # Only a fault has a severity to reach, so the tally skips the normal
    # instances, exactly as their legend entry skips the reach.
    reaches = rows.loc[rows["fault_class"] != 0, "reach"].value_counts()
    if len(reaches):
        subtitle += " | severity reach: " + ", ".join(
            f"{reaches[r]} {FAULT_REACH_LABELS[r]}" for r in FAULT_REACH_TINTS if r in reaches
        )
    fig.suptitle(
        f"WELL-{well_id:05d} | Real instances across time", fontsize=11, y=1 - 0.28 / height
    )
    fig.text(
        0.5,
        1 - 0.52 / height,
        subtitle,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#444444",
    )
    # Every color drawn gets its own entry, strongest tint of a fault first.
    present = sorted(
        {
            (fault_class, reach if fault_class != 0 else "steady")
            for fault_class, reach in zip(rows["fault_class"], rows["reach"])
        },
        key=lambda entry: (entry[0], -FAULT_REACH_TINTS[entry[1]]),
    )
    _timeline_legend(fig, present, top=1 - TIMELINE_INSETS["top"] / height)
    return fig


def plot_faults_per_well(
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = WELL_HISTORY_FIGURES_DIR,
    verbose: bool = False,
) -> None:
    """Plot how the instances of every well tile its history, one page per well.

    Each page is one well: every real instance recorded on it is a horizontal
    bar spanning from its first to its last timestamp, and instances that
    overlap in time are stacked on top of each other, so the page shows at a
    glance how much of the well is recorded twice — the overlap that leaks
    between train and test when the split is made per instance rather than
    per well. Bars wide enough carry the timestamp keying their filename
    (the ``*`` of ``WELL-000{well_id}_*.parquet``), which names the offending
    instance outright. Stage 1 stacks the instances with the very same rule
    (``preprocessing.pack_lanes``) and, unless ``--allow-overlap`` is given,
    drops every instance above the bottom level, so the page also shows what
    the default dataset leaves out.

    Recordings cover only a few percent of a well's calendar span, in bursts
    separated by months of silence, so the axis compresses those silences to
    narrow marked blanks (see ``_recording_blocks``) instead of shrinking
    every instance to an invisible sliver. The time scale is uniform
    everywhere else: bar length is instance duration, and bars overlap on the
    page exactly when the instances overlap in time. The PDF is vector
    graphics, so the densest wells resolve on zoom.

    A bar's hue is its fault-class folder, and its tint says how far the fault
    got inside that window: full strength once the steady fault state is
    labeled, lighter when only the transient is, lightest when the window
    carries no fault label at all — a distinction worth seeing, since 43 of
    the instances filed under hydrate in the service line never leave normal
    operation. Normal instances stay full strength (see ``_bar_color``). The
    legend keys the hue and the tint together, one entry per color the page
    actually draws (see ``_timeline_legend``). Simulated and hand-drawn
    instances are skipped.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset.
    out_dir : Path
        Destination directory of the PDF (named ``faults_per_well.pdf``).
    verbose : bool
        Print per-class and per-page progress (default off).
    """
    records = []
    for fault_class in FAULT_CLASSES:
        files = list_instances(fault_class, "real", raw_dir)
        if verbose:
            print(f"  Class {fault_class} ({FAULT_CLASSES[fault_class]}): {len(files)} instances")
        for filepath in files:
            labels = pd.read_parquet(filepath, columns=["class"])
            start, end = labels.index.min(), labels.index.max()
            records.append(
                {
                    "well": int(parse_well_id(filepath.name)),
                    "fault_class": fault_class,
                    "file": filepath.name,
                    "stamp": _filename_stamp(filepath.name, start),
                    "start": start,
                    "end": end,
                    "reach": _fault_reach(labels["class"].to_numpy(dtype=float)),
                }
            )
    if not records:
        raise FileNotFoundError(f"No real (WELL-*) instances under {raw_dir}")
    spans = pd.DataFrame(records)

    wells = sorted(spans["well"].unique())
    # The y axis of every page holds as many stack levels as the deepest
    # pile-up in the dataset, so the pages are the same size and comparable;
    # a pathologically deep well would only stretch its own page.
    per_well = {well_id: rows for well_id, rows in spans.groupby("well")}
    lane_slots = min(
        max(
            int(pack_lanes(rows["start"].to_numpy(), rows["end"].to_numpy()).max()) + 1
            for rows in per_well.values()
        ),
        MAX_TIMELINE_LANES,
    )

    out_path = out_dir / "faults_per_well.pdf"

    def pages():
        for i, well_id in enumerate(wells, start=1):
            rows = per_well[well_id]
            if verbose:
                print(f"  [{i}/{len(wells)}] WELL-{well_id:05d}: {len(rows)} instances")
            yield _plot_well_timeline(well_id, rows, lane_slots)

    _write_pdf(out_path, pages())
    print(f"  Saved: {out_path} ({len(spans)} instances on {len(wells)} wells, one page each)")
