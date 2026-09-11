"""Shared ground of the dataset plots: palettes, metadata, labels, envelopes, pages.

Everything more than one plot family needs lives here: the colors of the
``class`` and ``state`` bands, the readers of the 3W ``dataset.ini``, the
discovery of instance files, the helpers that turn a label column into runs,
colored bands and background shading, the min/max envelope that keeps long
series plottable, and the PDF writer every family saves its pages through.
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

from flowml.config import FAULT_CLASSES, KEY_SENSORS, RAW_DATA_DIR, SOURCE_TYPES, WELL_STATES
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

# A signal whose range is this small relative to its own level counts as flat.
# Real 3W instances are full of frozen variables (9.77 % of them, per Table 7
# of the paper) and of readings that drift by a few float digits; left to
# autoscale, both fill their axis with amplified noise that reads like a
# signal. See ``_hold_if_flat``.
FLAT_SPAN = 1e-4


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


def _dataset_properties(raw_dir: Path) -> dict[str, str] | None:
    """Read the ``PARQUET_FILE_PROPERTIES`` section of the 3W ``dataset.ini``.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset (contains ``dataset.ini``).

    Returns
    -------
    dict[str, str] | None
        Description per upper-cased variable name, in dataset order; ``None``
        when the ini file or the section is missing.
    """
    ini_path = raw_dir / "dataset.ini"
    if not ini_path.exists():
        return None
    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    if "PARQUET_FILE_PROPERTIES" not in parser:
        return None
    return {key.upper(): desc for key, desc in parser["PARQUET_FILE_PROPERTIES"].items()}


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
    units = {}
    for name, desc in (_dataset_properties(raw_dir) or {}).items():
        match = re.search(r"\[([^\[\]]+)\]\s*$", desc)
        if not match:
            continue
        unit = match.group(1)
        if "," in unit or " or " in unit:
            unit = "-"
        units[name] = unit.replace("oC", "°C").replace("m3/s", "m³/s")
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
    properties = _dataset_properties(raw_dir)
    if properties is None:
        return list(KEY_SENSORS)
    return [name for name in properties if name not in ("TIMESTAMP", "CLASS", "STATE")]


def list_instances(
    fault_class: int, source: str = "real", raw_dir: Path = RAW_DATA_DIR
) -> list[Path]:
    """List the instance files of one fault folder that come from one source.

    Parameters
    ----------
    fault_class : int
        Fault-class folder number.
    source : str
        Instance source, a key of ``SOURCE_TYPES``: ``"real"``,
        ``"simulated"`` or ``"drawn"``.
    raw_dir : Path
        Root of the 3W dataset.

    Returns
    -------
    list[Path]
        The matching parquet files, sorted by name. Empty when the fault has
        no instance of that source, which is common: the hand-drawn ones exist
        only for faults 1 and 7, and normal operation was never simulated.
    """
    if source not in SOURCE_TYPES:
        raise ValueError(f"Unknown instance source: {source!r} (expected {list(SOURCE_TYPES)})")
    class_dir = raw_dir / str(fault_class)
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")
    return [
        f
        for f in sorted(class_dir.glob("*.parquet"))
        if parse_source_type(f.name) == SOURCE_TYPES[source]
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
        for filepath in list_instances(fault_class, "real", raw_dir):
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


def _label_style(class_segments: list[tuple[int, int, float]]) -> tuple[dict, dict]:
    """Shading color and display name of every ``class`` run, for ``_draw_band``.

    Parameters
    ----------
    class_segments : list[(int, int, float)]
        Runs of the ``class`` column from ``_segments``.

    Returns
    -------
    (dict, dict)
        Color per value and name per value, ``None`` standing for NaN.
    """
    colors, names = {}, {}
    for _, _, value in class_segments:
        key = None if np.isnan(value) else value
        colors[key] = LABEL_COLORS[_label_kind(value)]
        names[key] = _label_name(value)
    return colors, names


def _draw_label_bands(band_axes, x, class_values, state_values) -> list[tuple[int, int, float]]:
    """Draw the two bands every instance page opens with: state, then class.

    Parameters
    ----------
    band_axes : sequence of matplotlib.axes.Axes
        The two thin axes above the plotting area, top first.
    x : np.ndarray
        Time axis values.
    class_values, state_values : np.ndarray
        The ``class`` and ``state`` columns as floats, NaN included.

    Returns
    -------
    list[(int, int, float)]
        The runs of ``class_values``, for the caller to shade its plotting
        area with (see ``_shade_by_label``).
    """
    state_names = dict(WELL_STATES) | {None: "Unknown"}
    _draw_band(band_axes[0], x, _segments(state_values), STATE_COLORS, state_names, "state")
    class_segments = _segments(class_values)
    colors, names = _label_style(class_segments)
    _draw_band(band_axes[1], x, class_segments, colors, names, "class")
    return class_segments


def _shade_by_label(ax, x, class_segments: list[tuple[int, int, float]]) -> None:
    """Shade the background of ``ax`` by label, one ``LABEL_COLORS`` tint per run."""
    for start, end, value in class_segments:
        ax.axvspan(
            x[start],
            x[min(end, len(x) - 1)],
            color=LABEL_COLORS[_label_kind(value)],
            lw=0,
            zorder=0,
        )


def _tint(color: str, strength: float) -> tuple[float, float, float]:
    """Mix one color toward white, ``strength`` 1.0 keeping it untouched."""
    base = np.array(mcolors.to_rgb(color))
    return tuple(1.0 - (1.0 - base) * strength)


def _hold_if_flat(ax, bottom: float, top: float) -> bool:
    """Pin a flat signal to the middle of a padded y axis; tell whether it was.

    A frozen or barely moving variable left to autoscale fills its axis with
    amplified noise that reads like a signal. When the range ``bottom``..``top``
    is below ``FLAT_SPAN`` of the signal's own level, the axis is instead
    centered on the signal with a margin of 1 % of that level (at least 0.5),
    so the line reads as the flat line it is and the ticks state its level.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis carrying the signal.
    bottom, top : float
        Minimum and maximum of the signal.

    Returns
    -------
    bool
        ``True`` when the signal was flat and the limits were set.
    """
    center = 0.5 * (bottom + top)
    if top - bottom > FLAT_SPAN * max(abs(center), 1.0):
        return False
    pad = max(abs(center) * 0.01, 0.5)
    ax.set_ylim(center - pad, center + pad)
    return True


def _write_pdf(out_path: Path, figures: Iterable) -> None:
    """Write one figure per page to ``out_path``, closing each figure once saved.

    Parameters
    ----------
    out_path : Path
        Destination file; its directory is created when missing.
    figures : Iterable[matplotlib.figure.Figure]
        The pages in order — typically a generator, so that only one figure
        is alive at a time however many pages the PDF has.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)


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
        Target number of buckets; a timeline already shorter than that gets
        one bucket per sample, so its envelope is the signal itself.

    Returns
    -------
    pd.DataFrame
        Frame indexed by the first timestamp of each bucket, with
        ``<sensor>_min`` / ``<sensor>_max`` per sensor plus ``class`` and
        ``state``.
    """
    sensors = [c for c in timeline.columns if c not in ("class", "state")]
    step = max(1, len(timeline) // max_points)
    grouped = timeline.groupby(np.arange(len(timeline)) // step, sort=True)

    env = grouped[sensors].agg(["min", "max"])
    env.columns = [f"{sensor}_{stat}" for sensor, stat in env.columns]
    for column in ("class", "state"):
        env[column] = grouped[column].first().to_numpy() if column in timeline.columns else np.nan
    env.index = timeline.index[::step]  # bucket k starts at sample k * step
    return env
