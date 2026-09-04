"""Raw-data visualization: per-fault instance histories.

Plots work directly on the raw 3W parquet files (not on the extracted
features), so they show exactly what the pipeline consumes. Only real
instances (``WELL-*`` files) are plotted; simulated and hand-drawn ones are
skipped.

Sensor units are read from the 3W ``dataset.ini`` shipped with the dataset.
"""

import configparser
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from flowml.config import FAULT_CLASSES, FIGURES_DIR, RAW_DATA_DIR, WELL_STATES
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
    raise ValueError(f"Unknown fault: {fault!r} (expected 0-9 or one of {list(FAULT_CLASSES.values())})")


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

    class_values = df["class"].to_numpy(dtype=float) if "class" in df.columns else np.full(len(df), np.nan)
    state_values = df["state"].to_numpy(dtype=float) if "state" in df.columns else np.full(len(df), np.nan)
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
        (None if np.isnan(v) else v): LABEL_COLORS[_label_kind(v)]
        for _, _, v in class_segments
    }
    class_names = {
        (None if np.isnan(v) else v): _label_name(v) for _, _, v in class_segments
    }
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
