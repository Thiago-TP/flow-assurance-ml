"""Raw-data visualization: per-fault instance histories.

Plots work directly on the raw 3W parquet files (not on the extracted
features), so they show exactly what the pipeline consumes. Only real
instances (``WELL-*`` files) are plotted; simulated and hand-drawn ones are
skipped.

Sensor units are read from the 3W ``dataset.ini`` shipped with the dataset.
"""

import configparser
import re
from collections.abc import Iterable
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from flowml.config import (
    FAULT_CLASSES,
    FIGURES_DIR,
    KEY_SENSORS,
    RAW_DATA_DIR,
    WELL_STATES,
)
from flowml.preprocessing import parse_source_type, parse_well_id

# Background colors for the class label of each stretch of a time series.
LABEL_COLORS = {
    "normal": "#dcefdc",  # light green
    "transient": "#fdf0c6",  # light yellow
    "active": "#f5c1bd",  # light red
    "unknown": "#e9e9e9",  # grey
}

# One color per fault class for the faults-per-well timeline. Normal is green
# (matching the healthy Open state); the faults get distinct categorical colors.
FAULT_COLORS = {
    0: "#4c9e4c",
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#7f7f7f",
    8: "#bcbd22",
    9: "#17becf",
}

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

# Advance width, in em, of the characters a timestamp label is made of, taken
# from DejaVu Sans (matplotlib's default face); see ``_text_width_pt``.
_DIGIT_EM = 0.636
_CHAR_EM = {"-": 0.361, ":": 0.337, " ": 0.318}

# Band colors for the well operational status ("state" column). Open is the
# healthy status; the remaining codes cycle through a categorical palette.
STATE_COLORS = {
    None: "#d9d9d9",  # unknown
    0: "#4c9e4c",
    1: "#d95f5f",
    2: "#c9a227",
    3: "#8f7ee6",
    4: "#5fa8d3",
    5: "#e6a23c",
    6: "#7f8c8d",
    7: "#3fbf9f",
    8: "#d47fb8",
}


def resolve_fault(fault: str) -> tuple[int, str]:
    """Turn a fault given by number or name into ``(class number, name)``.

    Parameters
    ----------
    fault : str
        Fault-class folder number (``"9"``) or fault name, case-insensitive
        (``"Hydrate in Service Line"``).

    Returns
    -------
    (int, str)
        The fault-class number and its canonical name.
    """
    s = str(fault).strip()
    if s.isdigit() and int(s) in FAULT_CLASSES:
        return int(s), FAULT_CLASSES[int(s)]
    for number, name in FAULT_CLASSES.items():
        if name.lower() == s.lower():
            return number, name
    raise ValueError(
        f"Unknown fault: {fault!r} (expected 0-9 or one of {list(FAULT_CLASSES.values())})"
    )


def load_sensor_units(raw_dir: Path = RAW_DATA_DIR) -> dict[str, str]:
    """Read the physical unit of every variable from the 3W ``dataset.ini``.

    Units are the trailing ``[...]`` of each variable description in the
    ``PARQUET_FILE_PROPERTIES`` section. Enumerated "units" of valve-state
    columns (``[0, 0.5, or 1]``) collapse to ``-``, and ASCII spellings are
    prettified (``oC`` -> ``°C``, ``m3/s`` -> ``m³/s``).

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset (contains ``dataset.ini``).

    Returns
    -------
    dict[str, str]
        Unit per variable name; empty dict when the ini file is missing.
    """
    ini_path = raw_dir / "dataset.ini"
    if not ini_path.exists():
        return {}
    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    if "PARQUET_FILE_PROPERTIES" not in parser:
        return {}

    units = {}
    for key, desc in parser["PARQUET_FILE_PROPERTIES"].items():
        match = re.search(r"\[([^\[\]]+)\]\s*$", desc)
        if not match:
            continue
        unit = match.group(1)
        if "," in unit or " or " in unit:
            unit = "-"
        unit = unit.replace("oC", "°C").replace("m3/s", "m³/s")
        units[key.upper()] = unit
    return units


def load_sensor_names(raw_dir: Path = RAW_DATA_DIR) -> list[str]:
    """Read the canonical list of sensor variables from the 3W ``dataset.ini``.

    The ``PARQUET_FILE_PROPERTIES`` section declares every variable an
    instance file may carry, in dataset order. Returning that full list —
    rather than whichever columns a given well happens to populate — is what
    lets every well history have the same pages: a well missing a sensor
    still gets its page, drawn empty.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset (contains ``dataset.ini``).

    Returns
    -------
    list[str]
        Variable names excluding ``timestamp``, ``class`` and ``state``;
        falls back to ``KEY_SENSORS`` when the ini file is missing.
    """
    ini_path = raw_dir / "dataset.ini"
    if not ini_path.exists():
        return list(KEY_SENSORS)
    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    if "PARQUET_FILE_PROPERTIES" not in parser:
        return list(KEY_SENSORS)
    return [
        key.upper()
        for key in parser["PARQUET_FILE_PROPERTIES"]
        if key.lower() not in ("timestamp", "class", "state")
    ]


def list_well_ids(raw_dir: Path = RAW_DATA_DIR) -> list[int]:
    """List the numbers of every real well present in the dataset.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset.

    Returns
    -------
    list[int]
        Sorted well numbers taken from the ``WELL-*`` filenames; simulated
        and hand-drawn instances have no physical well and are ignored.
    """
    wells = set()
    for fault_class in FAULT_CLASSES:
        class_dir = raw_dir / str(fault_class)
        if not class_dir.exists():
            continue
        for filepath in class_dir.glob("*.parquet"):
            if parse_source_type(filepath.name) != "WELL":
                continue
            well = parse_well_id(filepath.name)
            if well.isdigit():
                wells.add(int(well))
    return sorted(wells)


def _label_kind(value: float) -> str:
    """Classify one 3W ``class`` value as normal, transient, active, or unknown."""
    if np.isnan(value):
        return "unknown"
    if value == 0:
        return "normal"
    if 1 <= value <= 9:
        return "active"
    if 101 <= value <= 109:
        return "transient"
    return "unknown"


def _label_name(value: float) -> str:
    """Human-readable name of one 3W ``class`` value."""
    kind = _label_kind(value)
    if kind == "unknown":
        return "Unknown"
    if kind == "normal":
        return "Normal Operation"
    if kind == "active":
        return FAULT_CLASSES[int(value)]
    return f"{FAULT_CLASSES[int(value) - 100]} - Transient"


def _segments(values: np.ndarray) -> list[tuple[int, int, float]]:
    """Split a 1-D array into runs of constant value (NaN counts as a value).

    Parameters
    ----------
    values : np.ndarray
        Float array, possibly containing NaN.

    Returns
    -------
    list[(int, int, float)]
        ``(start, end, value)`` per run, with ``end`` exclusive.
    """
    codes = np.where(np.isnan(values), -1.0, values)
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(codes)]))
    return [(int(s), int(e), values[s]) for s, e in zip(starts, ends)]


def _draw_band(ax, x, segments, colors: dict, names: dict, label: str) -> None:
    """Draw one thin colored band (state or class) with centered segment names.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Band axis (no y data).
    x : np.ndarray
        Time axis values.
    segments : list[(int, int, float)]
        Runs from ``_segments``.
    colors : dict
        Segment value -> color.
    names : dict
        Segment value -> display name.
    label : str
        Y-axis label of the band (``"state"`` or ``"class"``).
    """
    span = len(x)
    for start, end, value in segments:
        key = None if np.isnan(value) else value
        x0, x1 = x[start], x[min(end, span - 1)]
        ax.axvspan(x0, x1, color=colors.get(key, "#d9d9d9"), lw=0)
        if end - start > 0.05 * span:  # only name segments wide enough to fit text
            center = x[(start + end) // 2]
            ax.text(center, 0.5, names.get(key, "Unknown"), ha="center", va="center", fontsize=6)
    ax.set_yticks([])
    ax.set_ylabel(label, fontsize=7, rotation=90)
    ax.set_ylim(0, 1)


def _plot_instance(df: pd.DataFrame, fault_name: str, filename: str, units: dict[str, str]):
    """Build the one-page figure of a single raw instance.

    Parameters
    ----------
    df : pd.DataFrame
        One raw instance, timestamp-indexed, with ``class`` and ``state``.
    fault_name : str
        Fault name shown in the title.
    filename : str
        Instance filename shown in the title.
    units : dict[str, str]
        Unit per variable name (see ``load_sensor_units``).

    Returns
    -------
    matplotlib.figure.Figure
        The finished figure, ready to save.
    """
    sensors = [c for c in df.columns if c not in ("class", "state") and df[c].notna().any()]
    x = df.index.to_numpy()

    class_values = (
        df["class"].to_numpy(dtype=float) if "class" in df.columns else np.full(len(df), np.nan)
    )
    state_values = (
        df["state"].to_numpy(dtype=float) if "state" in df.columns else np.full(len(df), np.nan)
    )
    class_segments = _segments(class_values)
    state_segments = _segments(state_values)

    n = len(sensors)
    fig, axes = plt.subplots(
        n + 2,
        1,
        sharex=True,
        figsize=(11, 1.0 + 1.35 * n),
        gridspec_kw={"height_ratios": [0.18, 0.18] + [1.0] * n},
    )
    fig.suptitle(f"{fault_name} | Instance history | {filename}", fontsize=10)

    state_names = {k: v for k, v in WELL_STATES.items()} | {None: "Unknown"}
    _draw_band(axes[0], x, state_segments, STATE_COLORS, state_names, "state")

    class_colors = {
        (None if np.isnan(v) else v): LABEL_COLORS[_label_kind(v)] for _, _, v in class_segments
    }
    class_names = {(None if np.isnan(v) else v): _label_name(v) for _, _, v in class_segments}
    _draw_band(axes[1], x, class_segments, class_colors, class_names, "class")

    for ax, sensor in zip(axes[2:], sensors):
        for start, end, value in class_segments:
            ax.axvspan(
                x[start],
                x[min(end, len(x) - 1)],
                color=LABEL_COLORS[_label_kind(value)],
                lw=0,
                zorder=0,
            )
        col = df[sensor].to_numpy(dtype=float)
        ax.plot(x, col, lw=0.8, zorder=2)

        unit = units.get(sensor, "")
        ax.set_ylabel(f"{sensor} [{unit}]" if unit else sensor, fontsize=7)
        ax.tick_params(labelsize=7)

        valid = col[~np.isnan(col)]
        delta = valid.max() - valid.min() if len(valid) else float("nan")
        ax.text(
            0.995,
            0.95,
            f"Δ = {delta:.3g} {unit}".rstrip(),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "black", "lw": 0.4, "pad": 2},
            zorder=3,
        )

    axes[-1].set_xlabel("Time", fontsize=9)
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return fig


def plot_fault(
    fault: str,
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = FIGURES_DIR,
    verbose: bool = False,
) -> None:
    """Plot every real instance of one fault into a multi-page PDF.

    One page per instance: two bands on top showing the well operational
    status (``state``) and the label (``class``) over time, then one subplot
    per sensor that has any data. Sensor backgrounds are shaded by label —
    light green for normal operation, light yellow for the transient period,
    light red for the active fault, grey for unlabeled stretches — and each
    panel reports the total variation of the signal (Δ = max - min).
    Simulated and hand-drawn instances are skipped.

    Parameters
    ----------
    fault : str
        Fault-class folder number (``"9"``) or fault name, case-insensitive
        (``"Hydrate in Service Line"``).
    raw_dir : Path
        Root of the 3W dataset.
    out_dir : Path
        Destination directory of the PDF (named
        ``fault_<number>_real_instances.pdf``).
    verbose : bool
        Print per-instance progress (default off).
    """
    fault_class, fault_name = resolve_fault(fault)
    class_dir = raw_dir / str(fault_class)
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")

    files = [f for f in sorted(class_dir.glob("*.parquet")) if parse_source_type(f.name) == "WELL"]
    if not files:
        raise FileNotFoundError(f"No real (WELL-*) instances under {class_dir}")

    units = load_sensor_units(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fault_{fault_class}_real_instances.pdf"

    with PdfPages(out_path) as pdf:
        for i, filepath in enumerate(files, start=1):
            if verbose:
                print(f"  [{i}/{len(files)}] {filepath.name}")
            df = pd.read_parquet(filepath)
            fig = _plot_instance(df, fault_name, filepath.name, units)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Saved: {out_path} ({len(files)} instances)")


def _class_palette(values: np.ndarray) -> tuple[dict, dict]:
    """Build the shading color and display name of every ``class`` value present.

    Normal operation and unlabeled stretches keep the ``plot_fault`` colors
    (light green, grey). Faults instead get a pale tint of their own
    ``FAULT_COLORS`` entry, so a history carrying several faults shows each
    one in its own hue: the active period at full tint strength and the
    preceding transient at half strength.

    Parameters
    ----------
    values : np.ndarray
        The ``class`` values appearing in the history, NaN included.

    Returns
    -------
    (dict, dict)
        Color per value and display name per value, both keyed by the value
        itself (``None`` for NaN) so they can feed ``_draw_band``.
    """
    colors, names = {}, {}
    for value in values:
        key = None if np.isnan(value) else value
        kind = _label_kind(value)
        if kind in ("normal", "unknown"):
            colors[key] = LABEL_COLORS[kind]
        else:
            fault = int(value) - 100 if kind == "transient" else int(value)
            strength = 0.30 if kind == "active" else 0.15
            base = np.array(mcolors.to_rgb(FAULT_COLORS[fault]))
            colors[key] = tuple(1.0 - (1.0 - base) * strength)
        names[key] = _label_name(value)
    return colors, names


def load_well_history(
    well_id: int, raw_dir: Path = RAW_DATA_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join every real instance of one well into a single chronological timeline.

    All ``WELL-000{well_id}_*.parquet`` files across the fault-class folders
    belong to the same physical well and are stitched into one series ordered
    by timestamp. The instances overlap heavily — they are sliding windows
    over a continuous recording, so roughly a fifth of the timestamps appear
    in two files — and duplicates are resolved by keeping, per timestamp, the
    record with the fewest missing values. That is safe because overlapping
    instances never disagree on a label where both are labeled: the only
    conflict is a missing value against a real one.

    Parameters
    ----------
    well_id : int
        Well number, as it appears in the filename (``5`` for ``WELL-00005``).
    raw_dir : Path
        Root of the 3W dataset.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        The joined timeline, timestamp-indexed with the ``class`` and
        ``state`` columns preserved, and one row per source instance with its
        ``file``, ``fault_class``, ``start``, ``end`` and ``n_samples``.
    """
    frames, spans = [], []
    for fault_class in FAULT_CLASSES:
        class_dir = raw_dir / str(fault_class)
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        for filepath in sorted(class_dir.glob(f"WELL-{well_id:05d}_*.parquet")):
            df = pd.read_parquet(filepath)
            sensors = [c for c in df.columns if c not in ("class", "state")]
            df[sensors] = df[sensors].astype("float32")  # halve the memory of long wells
            frames.append(df)
            spans.append(
                {
                    "file": filepath.name,
                    "fault_class": fault_class,
                    "start": df.index.min(),
                    "end": df.index.max(),
                    "n_samples": len(df),
                }
            )
    if not frames:
        raise FileNotFoundError(f"No real instances of WELL-{well_id:05d} under {raw_dir}")

    joined = pd.concat(frames)
    del frames

    # Per timestamp keep the most complete record: ordering by timestamp first
    # and by descending completeness second makes the survivor the first row.
    completeness = joined.notna().sum(axis=1).to_numpy()
    joined = joined.iloc[np.lexsort((-completeness, joined.index.to_numpy()))]
    joined = joined[~joined.index.duplicated(keep="first")]

    spans = pd.DataFrame(spans).sort_values("start").reset_index(drop=True)
    return joined, spans


def _envelope(timeline: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Reduce a long timeline to a plottable min/max envelope.

    A well history holds millions of samples, far more than a page can
    resolve. Samples are bucketed by position — so resolution follows data
    density rather than calendar time — and each bucket contributes the
    minimum and maximum of every sensor, which keeps spikes visible instead
    of letting plain subsampling drop them. Labels are taken as the first
    non-missing value of the bucket, unambiguous because overlapping
    instances agree.

    Parameters
    ----------
    timeline : pd.DataFrame
        Joined history from ``load_well_history``.
    max_points : int
        Target number of buckets; the timeline is returned untouched when it
        is already shorter.

    Returns
    -------
    pd.DataFrame
        Timestamp-indexed frame with ``<sensor>_min`` / ``<sensor>_max`` per
        sensor plus ``class`` and ``state``.
    """
    sensors = [c for c in timeline.columns if c not in ("class", "state")]
    step = max(1, len(timeline) // max_points)

    if step == 1:
        env = pd.DataFrame(index=timeline.index)
        for sensor in sensors:
            env[f"{sensor}_min"] = timeline[sensor]
            env[f"{sensor}_max"] = timeline[sensor]
    else:
        bucket = np.arange(len(timeline)) // step
        grouped = timeline.groupby(bucket, sort=True)
        agg = grouped[sensors].agg(["min", "max"])
        agg.columns = [f"{sensor}_{stat}" for sensor, stat in agg.columns]
        env = agg.set_index(timeline.index.to_series().groupby(bucket, sort=True).first())

    for column in ("class", "state"):
        if column in timeline.columns:
            source = timeline[column]
            env[column] = (
                source.to_numpy()
                if step == 1
                else source.groupby(np.arange(len(timeline)) // step, sort=True).first().to_numpy()
            )
        else:
            env[column] = np.nan

    return env


def _compress_gaps(env: pd.DataFrame, factor: float = 20.0, gap_slots: float = 5.0) -> pd.DataFrame:
    """Lay the buckets on a gap-compressed axis, one slot per bucket.

    A well is recorded in bursts: well 1 spans three and a half years but
    holds only weeks of samples, so on a true calendar axis the recordings
    shrink to invisible slivers. Plotting against bucket position instead
    gives every bucket the same width, and each long silence collapses to a
    fixed narrow blank marked by a separator row. Chronological order and the
    real timestamps are preserved — the axis ticks still carry real dates —
    only the empty stretches lose their proportional width.

    Parameters
    ----------
    env : pd.DataFrame
        Envelope from ``_envelope``, timestamp-indexed.
    factor : float
        A silence counts as a gap when it exceeds ``factor`` times the median
        spacing between buckets.
    gap_slots : float
        Width of a collapsed gap, in bucket slots.

    Returns
    -------
    pd.DataFrame
        The envelope with a ``position`` column for the x axis, a
        ``timestamp`` column keeping the real time of each bucket, and a
        boolean ``separator`` column flagging the inserted gap rows (whose
        values are all missing, so lines break across them).
    """
    frame = env.copy()
    frame["timestamp"] = frame.index
    frame["separator"] = False

    if len(frame) < 3:
        frame["position"] = np.arange(len(frame), dtype=float)
        return frame

    seconds = np.diff(env.index.to_numpy()).astype("timedelta64[s]").astype(float)
    positive = seconds[seconds > 0]
    gaps = (
        np.flatnonzero(seconds > factor * np.median(positive))
        if len(positive)
        else np.array([], int)
    )

    if not len(gaps):
        frame["position"] = np.arange(len(frame), dtype=float)
        return frame

    # One separator row per gap, placed between the buckets it separates.
    separators = pd.DataFrame(np.nan, index=range(len(gaps)), columns=frame.columns)
    separators["separator"] = True
    separators["timestamp"] = pd.NaT
    separators["_order"] = gaps + 0.5

    frame["_order"] = np.arange(len(frame), dtype=float)
    combined = pd.concat([frame, separators]).sort_values("_order").reset_index(drop=True)

    # Each real bucket advances one slot; each separator advances gap_slots.
    steps = np.where(combined["separator"].to_numpy(), gap_slots, 1.0)
    combined["position"] = np.concatenate(([0.0], np.cumsum(steps)[:-1]))
    return combined.drop(columns="_order")


def _plot_well_sensor(
    env: pd.DataFrame,
    sensor: str,
    well_id: int,
    unit: str,
    subtitle: str,
    class_style: tuple[dict, dict],
    coverage: tuple[int, int],
):
    """Build the one-page figure of a single sensor across a well's history.

    Parameters
    ----------
    env : pd.DataFrame
        Gap-compressed envelope from ``_compress_gaps``.
    sensor : str
        Sensor whose page this is.
    well_id : int
        Well number shown in the title.
    unit : str
        Physical unit of the sensor (see ``load_sensor_units``).
    subtitle : str
        History summary line shown under the title.
    class_style : (dict, dict)
        Color and name per ``class`` value, from ``_class_palette``.
    coverage : (int, int)
        Samples carrying a reading and total samples of the joined timeline,
        reported on the page. A sensor the well never recorded gets an empty
        panel instead of a signal.

    Returns
    -------
    matplotlib.figure.Figure
        The finished figure, ready to save.
    """
    x = env["position"].to_numpy(dtype=float)
    separator = env["separator"].to_numpy(dtype=bool)
    class_values = env["class"].to_numpy(dtype=float)
    state_values = env["state"].to_numpy(dtype=float)
    class_colors, class_names = class_style
    state_colors = STATE_COLORS | {None: "#ffffff"}  # gaps read as blank, not grey

    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(13, 5.4),
        gridspec_kw={"height_ratios": [0.13, 0.13, 1.0]},
    )
    fig.suptitle(
        f"WELL-{well_id:05d} | Joined history | {sensor}{f' [{unit}]' if unit else ''}",
        fontsize=11,
    )

    # Shade each run of constant label over exactly the slots it occupies, so a
    # collapsed gap stays blank instead of being bridged by its neighbours.
    def shade(ax, values, colors):
        for start, end, value in _segments(values):
            if separator[start]:
                continue
            key = None if np.isnan(value) else value
            ax.axvspan(
                x[start],
                x[end - 1] + 1.0,
                color=colors.get(key, LABEL_COLORS["unknown"]),
                lw=0,
                zorder=0,
            )

    for ax, values, colors, label in (
        (axes[0], state_values, state_colors, "state"),
        (axes[1], class_values, class_colors, "class"),
    ):
        shade(ax, values, colors)
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=7, rotation=90)
        ax.set_ylim(0, 1)

    ax = axes[2]
    shade(ax, class_values, class_colors)

    if f"{sensor}_min" in env.columns:
        low = env[f"{sensor}_min"].to_numpy(dtype=float)
        high = env[f"{sensor}_max"].to_numpy(dtype=float)
    else:  # the well's files do not even carry the column
        low = high = np.full(len(x), np.nan)
    ax.fill_between(x, low, high, color="#1f4e79", lw=0.4, edgecolor="#1f4e79", zorder=2)

    for position in x[separator]:  # mark where a silence was collapsed
        for band in axes:
            band.axvline(position, color="#9a9a9a", lw=0.6, ls=(0, (2, 2)), zorder=1)

    ax.set_ylabel(f"{sensor} [{unit}]" if unit else sensor, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(x[0], x[-1] + 1.0)
    ax.set_xlabel("Time (dashed lines mark collapsed silences)", fontsize=9)

    real = np.flatnonzero(~separator)
    ticks = real[np.linspace(0, len(real) - 1, min(9, len(real))).astype(int)]
    stamps = [pd.Timestamp(env["timestamp"].iloc[i]) for i in ticks]
    # Instances can be hours long, so a date-only tick would repeat itself;
    # add the clock time whenever consecutive ticks fall on the same day.
    fmt = (
        "%Y-%m-%d %H:%M"
        if len(stamps) > 1 and min(np.diff(stamps)) < pd.Timedelta(days=1)
        else "%Y-%m-%d"
    )
    ax.set_xticks(x[ticks])
    ax.set_xticklabels([s.strftime(fmt) for s in stamps], rotation=30, ha="right")

    valid_high = high[~np.isnan(high)]
    valid_low = low[~np.isnan(low)]
    n_valid, n_total = coverage
    share = 100 * n_valid / n_total if n_total else 0.0
    report = f"coverage {share:.1f}% ({n_valid:,} of {n_total:,} samples)"
    if len(valid_high) and len(valid_low):
        report += f"\nΔ = {valid_high.max() - valid_low.min():.3g} {unit}".rstrip()
    else:
        ax.text(
            0.5,
            0.5,
            "No data recorded for this sensor",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#8a8a8a",
            zorder=3,
        )
    ax.text(
        0.995,
        0.96,
        report,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        linespacing=1.5,
        bbox={"facecolor": "white", "edgecolor": "black", "lw": 0.4, "pad": 2},
        zorder=3,
    )

    # One legend per band, each titled with the column it explains, so the
    # operational status colors are named as well as the label colors.
    def legend_handles(values, colors, names, missing_label):
        real_values = values[~separator]
        handles = [
            Patch(color=colors[v], label=names.get(v, str(v)))
            for v in sorted({v for v in real_values if not np.isnan(v)})
        ]
        if np.isnan(real_values).any():
            handles.append(
                Patch(color=LABEL_COLORS["unknown"], ec="#bbbbbb", lw=0.4, label=missing_label)
            )
        return handles

    class_handles = legend_handles(class_values, class_colors, class_names, "Unlabeled")
    state_handles = legend_handles(state_values, state_colors, dict(WELL_STATES), "Unknown status")

    fig.legend(
        handles=class_handles,
        title="class — label",
        loc="lower left",
        bbox_to_anchor=(0.055, 0.0),
        ncol=min(4, max(1, len(class_handles))),
        fontsize=7,
        title_fontsize=7,
        alignment="left",
        frameon=False,
    )
    fig.legend(
        handles=state_handles,
        title="state — well operational status",
        loc="lower right",
        bbox_to_anchor=(0.99, 0.0),
        ncol=min(3, max(1, len(state_handles))),
        fontsize=7,
        title_fontsize=7,
        alignment="left",
        frameon=False,
    )

    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=7.5, color="#444444")
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0.115, 1, 0.93))
    return fig


def plot_well_history(
    well_id: int,
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = FIGURES_DIR,
    max_points: int = 20_000,
    verbose: bool = False,
) -> None:
    """Plot the whole recorded history of one well into a multi-page PDF.

    Every real instance of the well, across all fault-class folders, is joined
    into a single chronological timeline (see ``load_well_history``) and drawn
    one time series per page, so the pages show how the recordings tile the
    well's lifetime and where the faults sit. Long silences between recordings
    are collapsed to a narrow marked blank (see ``_compress_gaps``), because a
    well's samples cover only a small fraction of its calendar span; ticks
    still carry real dates.

    Coloring follows the ``plot_fault`` scheme, extended because a whole
    history usually carries several different faults: each page shows a band
    of the well operational status (``state``), a band of the label
    (``class``), and the sensor itself over a background shaded by label.
    Normal operation stays light green and unlabeled stretches grey, while
    every fault gets a pale tint of its own color — stronger for the active
    period, fainter for the preceding transient. Both bands get their own
    named legend. Sensors are drawn as a min/max envelope over
    position-bucketed samples, which keeps spikes visible instead of letting
    subsampling drop them.

    Every well gets a page for every sensor the dataset declares, in dataset
    order, including the ones this well never recorded: those pages are drawn
    empty. Each page reports how much of the timeline the sensor actually
    covers, so the missing data is visible rather than silently absent, and
    the PDFs are directly comparable page by page across wells.

    Parameters
    ----------
    well_id : int
        Well number, as it appears in the filename (``5`` for ``WELL-00005``).
    raw_dir : Path
        Root of the 3W dataset.
    out_dir : Path
        Destination directory of the PDF (named ``well_<id>_history.pdf``).
    max_points : int
        Envelope resolution in buckets per page; lower it for smaller PDFs.
    verbose : bool
        Print per-sensor progress and the instance overlap report (default off).
    """
    timeline, spans = load_well_history(well_id, raw_dir)

    overlaps = int((spans["start"] < spans["end"].shift()).sum())
    duplicated = int(spans["n_samples"].sum() - len(timeline))
    faults = sorted(spans["fault_class"].unique())
    subtitle = (
        f"{len(spans)} instances ({overlaps + 1} overlapping) | "
        f"{len(timeline):,} unique timestamps ({duplicated:,} deduplicated) | "
        f"{spans['start'].min():%Y-%m-%d} to {spans['end'].max():%Y-%m-%d} | "
        f"fault folders: {', '.join(FAULT_CLASSES[f] for f in faults)}"
    )
    if verbose:
        print(f"  WELL-{well_id:05d}: {subtitle}")

    units = load_sensor_units(raw_dir)
    sensors = load_sensor_names(raw_dir)

    # Coverage must be counted before decimation: a bucket's min/max is present
    # as soon as one sample in it is, which would overstate a sparse sensor.
    n_total = len(timeline)
    coverage = {
        sensor: (int(timeline[sensor].notna().sum()) if sensor in timeline.columns else 0, n_total)
        for sensor in sensors
    }

    env = _compress_gaps(_envelope(timeline, max_points))
    del timeline

    class_style = _class_palette(np.unique(env["class"].to_numpy(dtype=float)))
    n_empty = sum(1 for sensor in sensors if coverage[sensor][0] == 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"well_{well_id}_history.pdf"
    with PdfPages(out_path) as pdf:
        for i, sensor in enumerate(sensors, start=1):
            if verbose:
                share = 100 * coverage[sensor][0] / n_total if n_total else 0.0
                print(f"  [{i}/{len(sensors)}] {sensor} ({share:.1f}% coverage)")
            fig = _plot_well_sensor(
                env,
                sensor,
                well_id,
                units.get(sensor, ""),
                subtitle,
                class_style,
                coverage[sensor],
            )
            pdf.savefig(fig)
            plt.close(fig)

    print(
        f"  Saved: {out_path} ({len(sensors)} time series, {n_empty} of them empty, "
        f"{len(spans)} instances joined)"
    )


def plot_wells_histories(
    well_ids: Iterable[int] | None = None,
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = FIGURES_DIR,
    max_points: int = 20_000,
    verbose: bool = False,
) -> None:
    """Plot the joined history of several wells, one multi-page PDF each.

    Wells are processed in ascending order. A well that fails — unreadable
    files, say — is reported and skipped rather than aborting the batch,
    since a full run covers every well in the dataset and takes a while; the
    failures are listed again at the end.

    Parameters
    ----------
    well_ids : Iterable[int] | None
        Well numbers to plot; ``None`` plots every real well in the dataset
        (see ``list_well_ids``).
    raw_dir : Path
        Root of the 3W dataset.
    out_dir : Path
        Destination directory of the PDFs.
    max_points : int
        Envelope resolution in buckets per page; lower it for smaller PDFs.
    verbose : bool
        Print per-sensor progress of each well (default off).
    """
    wells = sorted(set(list_well_ids(raw_dir) if well_ids is None else well_ids))
    if not wells:
        raise ValueError(f"No wells to plot under {raw_dir}")

    failures: list[tuple[int, str]] = []
    for i, well_id in enumerate(wells, start=1):
        print(f"\n[{i}/{len(wells)}] WELL-{well_id:05d} history...")
        try:
            plot_well_history(well_id, raw_dir, out_dir, max_points, verbose)
        except Exception as error:  # noqa: BLE001 - one bad well must not stop the batch
            failures.append((well_id, f"{type(error).__name__}: {error}"))
            print(f"  FAILED: {failures[-1][1]}")

    print(f"\n{len(wells) - len(failures)} of {len(wells)} wells plotted into {out_dir}")
    for well_id, reason in failures:
        print(f"  WELL-{well_id:05d} failed — {reason}")


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


def _tint(color: str, strength: float) -> tuple[float, float, float]:
    """Mix one color toward white, ``strength`` 1.0 keeping it untouched."""
    base = np.array(mcolors.to_rgb(color))
    return tuple(1.0 - (1.0 - base) * strength)


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
    """Width of a timestamp label in points, from DejaVu Sans advances.

    Deciding whether a label fits inside its bar needs the label width before
    anything is drawn, and asking the renderer for it would tie the decision
    to the backend. The labels only ever hold digits and the three separators
    of a timestamp, whose advances in matplotlib's default face are known, so
    the width is summed directly.

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
    em = sum(_DIGIT_EM if char.isdigit() else _CHAR_EM.get(char, 0.6) for char in text)
    return em * fontsize


def _pack_lanes(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Stack overlapping instances, one lane per level of simultaneity.

    Instances are placed in chronological order, each one taking the lowest
    lane whose last instance has already ended; a new lane opens only when
    every existing one is still busy. Two instances therefore share a lane
    exactly when they do not overlap, so the number of lanes is the deepest
    pile-up of the well and a chain of sliding windows alternates between
    two lanes, its overlaps visible as the horizontal offset between them.

    Parameters
    ----------
    starts, ends : np.ndarray
        First and last timestamp of every instance, ``datetime64``.

    Returns
    -------
    np.ndarray
        Lane index (0-based) per instance, in the input order.
    """
    lane_of = np.zeros(len(starts), dtype=int)
    lane_ends: list[np.datetime64] = []
    for i in np.argsort(starts, kind="stable"):
        for lane, lane_end in enumerate(lane_ends):
            if starts[i] > lane_end:
                lane_of[i] = lane
                lane_ends[lane] = ends[i]  # starts are sorted, so this only grows
                break
        else:
            lane_of[i] = len(lane_ends)
            lane_ends.append(ends[i])
    return lane_of


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
    lane_of = _pack_lanes(starts, ends)
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

    # An instance overlaps another when it starts before the latest end so far,
    # which marks both members of the pair.
    reach_before = np.maximum.accumulate(ends)
    overlaps_earlier = starts[1:] < reach_before[:-1]
    overlapping = np.zeros(len(rows), dtype=bool)
    overlapping[1:] |= overlaps_earlier
    overlapping[:-1] |= overlaps_earlier

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
    out_dir: Path = FIGURES_DIR,
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
    instance outright.

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
        class_dir = raw_dir / str(fault_class)
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        files = [
            f for f in sorted(class_dir.glob("*.parquet")) if parse_source_type(f.name) == "WELL"
        ]
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
            int(_pack_lanes(rows["start"].to_numpy(), rows["end"].to_numpy()).max()) + 1
            for rows in per_well.values()
        ),
        MAX_TIMELINE_LANES,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "faults_per_well.pdf"
    with PdfPages(out_path) as pdf:
        for i, well_id in enumerate(wells, start=1):
            rows = per_well[well_id]
            if verbose:
                print(f"  [{i}/{len(wells)}] WELL-{well_id:05d}: {len(rows)} instances")
            fig = _plot_well_timeline(well_id, rows, lane_slots)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Saved: {out_path} ({len(spans)} instances on {len(wells)} wells, one page each)")
