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
import matplotlib.dates as mdates
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
        f"{len(spans)} instances ({overlaps} overlapping) | "
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


def plot_faults_per_well(
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = FIGURES_DIR,
    verbose: bool = False,
) -> None:
    """Plot which faults occur in each well across time.

    Each well is one horizontal line spanning the shared time axis, and every
    real instance recorded on that well appears as a colored segment whose
    length is the instance duration and whose color is its fault class
    (normal recordings included). Instance durations are short compared to
    the years-long axis, so the segments look like "nicks" on the well lines;
    the PDF is vector graphics, so zooming resolves them. Simulated and
    hand-drawn instances are skipped.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset.
    out_dir : Path
        Destination directory of the PDF (named ``faults_per_well.pdf``).
    verbose : bool
        Print per-class progress (default off).
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
            index = pd.read_parquet(filepath, columns=[]).index
            records.append(
                {
                    "well": int(parse_well_id(filepath.name)),
                    "fault_class": fault_class,
                    "start": index.min(),
                    "end": index.max(),
                }
            )
    if not records:
        raise FileNotFoundError(f"No real (WELL-*) instances under {raw_dir}")
    spans = pd.DataFrame(records)

    wells = sorted(spans["well"].unique())
    y_of = {well: i for i, well in enumerate(wells)}
    t_min, t_max = spans["start"].min(), spans["end"].max()

    fig, ax = plt.subplots(figsize=(12, 0.32 * len(wells) + 1.8))
    ax.hlines(list(y_of.values()), t_min, t_max, color="#c9c9c9", lw=0.7, zorder=1)
    for row in spans.itertuples():
        x0, x1 = mdates.date2num(row.start), mdates.date2num(row.end)
        ax.broken_barh(
            [(x0, x1 - x0)],
            (y_of[row.well] - 0.35, 0.7),
            color=FAULT_COLORS[row.fault_class],
            lw=0,
            zorder=2,
        )

    ax.set_yticks(list(y_of.values()))
    ax.set_yticklabels([f"WELL-{well:05d}" for well in wells], fontsize=7)
    ax.set_ylim(-1, len(wells))
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Time")
    ax.set_title("3W real instances — faults per well across time", fontsize=11)

    present = sorted(spans["fault_class"].unique())
    ax.legend(
        handles=[Patch(color=FAULT_COLORS[k], label=FAULT_CLASSES[k]) for k in present],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        frameon=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "faults_per_well.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path} ({len(spans)} instances on {len(wells)} wells)")
