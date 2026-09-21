"""Raw-data loading, instance selection, and cleaning.

The 3W dataset stores one parquet file per instance (a continuous recording of
one well), organized in folders ``0/`` .. ``9/`` named after the fault class.
This module turns those raw files into clean sensor series ready for feature
extraction. It no longer normalizes them: scaling is applied to the window
features at load time, against a reference the run chooses (see the
``normalization`` module).

Two defaults guard against known defects of the raw data, each with a switch
that turns it off:

- real instances of one well are windows cut from the same continuous
  recording and often overlap in time, so the shared samples would enter the
  dataset twice, under different labels — the overlapping instances are
  dropped (see ``select_instances``; ``allow_overlap`` keeps them all);
- some sensors report values that cannot be measurements — magnitudes up to
  1e42, negative absolute pressures, temperatures of 30,000 °C — and those
  readings become NaN and are imputed later (see ``mask_extreme_values``;
  ``keep_extreme_values`` keeps them all).
"""

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from flowml.config import (
    CRITICAL_SENSOR,
    EXTREME_VALUE_LIMIT,
    FFILL_LIMIT,
    KEY_SENSORS,
    MAX_MISSING_RATIO,
    OPENING_MIN,
    OPENING_SENSORS,
    PRESSURE_MIN,
    PRESSURE_SENSORS,
    TEMPERATURE_LIMITS,
    TEMPERATURE_SENSORS,
)

# Columns of a loaded instance that are not sensor readings: the two 3W label
# columns and the metadata ``load_raw_instances`` attaches.
NON_SENSOR_COLUMNS = ("class", "state", "instance_id", "well_id", "fault_class", "source_type")


def parse_source_type(filename: str) -> str:
    """Classify an instance file as real, simulated, or hand-drawn data.

    Parameters
    ----------
    filename : str
        Name of the raw parquet file.

    Returns
    -------
    str
        ``"WELL"`` (real field data), ``"SIMULATED"``, or ``"DRAWN"``.
    """
    name = Path(filename).stem.upper()
    if "SIMULATED" in name:
        return "SIMULATED"
    if "DRAWN" in name:
        return "DRAWN"
    return "WELL"


def parse_well_id(filename: str) -> str:
    """Extract the well identifier from an instance filename.

    Real instances are named ``WELL-000{id}_...`` (e.g.
    ``WELL-00026_20170608230000.parquet`` comes from well 26). Simulated and
    hand-drawn instances have no physical well, so they get the fictitious
    IDs ``"simulated"`` and ``"drawn"``.

    Parameters
    ----------
    filename : str
        Name of the raw parquet file.

    Returns
    -------
    str
        The well number without leading zeros (e.g. ``"26"``), or
        ``"simulated"`` / ``"drawn"``.
    """
    name = Path(filename).stem.upper()
    if "SIMULATED" in name:
        return "simulated"
    if "DRAWN" in name:
        return "drawn"
    match = re.match(r"WELL-(\d+)", name)
    if match:
        return str(int(match.group(1)))
    return name  # unrecognized pattern: keep the stem as its own group


def list_raw_instances(
    raw_dir: Path,
    fault_classes: list[int],
    max_instances_per_class: int | None = None,
) -> list[tuple[int, Path]]:
    """List the instance files to load, class folder by class folder.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset (contains folders ``0/`` .. ``9/``).
    fault_classes : list[int]
        Fault-class folders to read.
    max_instances_per_class : int | None
        Cap on instances per class; ``None`` lists everything. Useful for a
        quick smoke test of the pipeline.

    Returns
    -------
    list[(int, Path)]
        The fault class and path of every file, in folder order and sorted by
        name within a folder.
    """
    entries: list[tuple[int, Path]] = []
    for fault_class in fault_classes:
        class_dir = raw_dir / str(fault_class)
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        files = sorted(class_dir.glob("*.parquet"))
        if max_instances_per_class is not None:
            files = files[:max_instances_per_class]
        entries.extend((fault_class, filepath) for filepath in files)
    return entries


def instance_spans(files: Iterable[Path]) -> pd.DataFrame:
    """Read the time span of every real instance among ``files``.

    Only the index of each file is needed, so a single column is read; the
    pass over the whole dataset takes seconds. Simulated and hand-drawn
    instances have no well and are left out.

    Parameters
    ----------
    files : Iterable[Path]
        Instance parquet files.

    Returns
    -------
    pd.DataFrame
        One row per real instance with its ``file``, ``well_id`` (as parsed
        by ``parse_well_id``), ``start`` and ``end`` timestamps.
    """
    rows = []
    for filepath in files:
        if parse_source_type(filepath.name) != "WELL":
            continue
        index = pd.read_parquet(filepath, columns=["class"]).index
        rows.append(
            {
                "file": filepath,
                "well_id": parse_well_id(filepath.name),
                "start": index.min(),
                "end": index.max(),
            }
        )
    return pd.DataFrame(rows, columns=["file", "well_id", "start", "end"])


def pack_lanes(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Stack overlapping instances, one lane per level of simultaneity.

    Instances are placed in chronological order, each one taking the lowest
    lane whose last instance has already ended; a new lane opens only when
    every existing one is still busy. Two instances therefore share a lane
    exactly when they do not overlap, so the number of lanes is the deepest
    pile-up of the well and a chain of sliding windows alternates between
    two lanes, its overlaps visible as the horizontal offset between them.

    This one function serves both the fault timeline of
    ``visualization.plot_faults_per_well`` — the lane is the page's stack
    level — and the default instance selection of ``select_instances``, which
    keeps the bottom lane only, so what the plot shows removed is exactly what
    the pipeline removes.

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


def overlapping_mask(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Flag every instance that shares at least one timestamp with another.

    Parameters
    ----------
    starts, ends : np.ndarray
        First and last timestamp of every instance, ``datetime64``.

    Returns
    -------
    np.ndarray
        Boolean per instance, in the input order. Both members of an
        overlapping pair are flagged; touching at a single shared second
        counts, since that second is then labeled twice.
    """
    starts, ends = np.asarray(starts), np.asarray(ends)
    hits = (starts[:, None] <= ends[None, :]) & (starts[None, :] <= ends[:, None])
    np.fill_diagonal(hits, False)
    return hits.any(axis=1)


def instance_overlaps(files: Iterable[Path]) -> pd.DataFrame:
    """Stack the real instances of every well and flag the overlapping ones.

    Parameters
    ----------
    files : Iterable[Path]
        Instance parquet files; simulated and hand-drawn ones are ignored.

    Returns
    -------
    pd.DataFrame
        The spans of ``instance_spans`` plus, per instance, its ``lane`` in
        the stacking of its well (see ``pack_lanes``) and whether it
        ``overlaps`` another instance of that well.
    """
    spans = instance_spans(files)
    spans["lane"] = 0
    spans["overlaps"] = False
    for idx in spans.groupby("well_id").groups.values():
        starts = spans.loc[idx, "start"].to_numpy(dtype="datetime64[ns]")
        ends = spans.loc[idx, "end"].to_numpy(dtype="datetime64[ns]")
        spans.loc[idx, "lane"] = pack_lanes(starts, ends)
        spans.loc[idx, "overlaps"] = overlapping_mask(starts, ends)
    return spans


def _well_label(well_id: str) -> str:
    """Filename-style name of a well (``WELL-00026``) from its parsed id."""
    return f"WELL-{int(well_id):05d}" if well_id.isdigit() else well_id


def _overlap_report(table: pd.DataFrame, allow_overlap: bool, n_other: int) -> list[str]:
    """Lines of the verbose summary of ``select_instances``."""
    n_real, n_wells = len(table), table["well_id"].nunique()
    n_over = int(table["overlaps"].sum())
    wells_over = table.loc[table["overlaps"], "well_id"].nunique()
    lines = [
        (
            f"Overlap check: {n_real} real instances on {n_wells} wells; {n_over} overlap "
            f"another instance of the same well ({wells_over} wells); {n_other} "
            f"simulated/drawn instances not subject to the rule"
        )
    ]
    if allow_overlap:
        lines.append("  all kept (--allow-overlap)")
    else:
        n_removed = int((~table["kept"]).sum())
        lines.append(
            f"  {n_removed} removed (stack level 2 or higher in faults_per_well.pdf), "
            f"{n_real - n_removed} kept"
        )
    wells = sorted(
        table["well_id"].unique(), key=lambda w: (not w.isdigit(), int(w) if w.isdigit() else w)
    )
    for well in wells:
        rows = table[table["well_id"] == well]
        if not rows["overlaps"].any():
            continue
        lines.append(
            f"  {_well_label(well)}: {len(rows)} instances, {int(rows['overlaps'].sum())} "
            f"overlapping, {int((~rows['kept']).sum())} removed"
        )
    return lines


def select_instances(
    entries: list[tuple[int, Path]],
    allow_overlap: bool = False,
    verbose: bool = False,
) -> tuple[list[tuple[int, Path]], pd.DataFrame]:
    """Drop the real instances that overlap another of the same well.

    Real 3W instances are windows cut from one continuous recording of a
    well, and most of them — 986 of the 1119 in 3W 2.0.0 — overlap a
    neighbour in time, typically as a chain of sliding windows. The shared
    samples would enter the dataset twice — and under different labels, since
    the end of one instance is the start of the next: a pressure labeled as
    the steady fault state in the earlier instance is normal operation in the
    later one. Stacking the instances of a well as ``pack_lanes`` does, only
    the bottom lane survives, so the kept instances of a well never overlap
    each other and every removed one is on stack level 2 or higher of
    ``faults_per_well.pdf``. A chain of sliding windows alternates between
    two lanes, so the rule keeps every other instance of it: on the full
    dataset 478 real instances go and 641 stay. Simulated and hand-drawn
    instances have no well and always pass.

    Parameters
    ----------
    entries : list[(int, Path)]
        Fault class and path of every candidate file, from
        ``list_raw_instances``.
    allow_overlap : bool
        Keep every instance regardless of overlap (default off).
    verbose : bool
        Print how many instances overlap and how many were removed, in total
        and per well (default off).

    Returns
    -------
    (list[(int, Path)], pd.DataFrame)
        The entries kept, in their original order, and the table of
        ``instance_overlaps`` with a ``kept`` column added.
    """
    table = instance_overlaps(path for _, path in entries)
    table["kept"] = True if allow_overlap else table["lane"].eq(0)
    dropped = set(table.loc[~table["kept"], "file"])
    kept = [(fault_class, path) for fault_class, path in entries if path not in dropped]
    if verbose:
        for line in _overlap_report(table, allow_overlap, n_other=len(entries) - len(table)):
            print(f"  {line}")
    return kept, table


def load_raw_instances(entries: Iterable[tuple[int, Path]]) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield raw instances one at a time, keeping memory usage flat.

    Parameters
    ----------
    entries : Iterable[(int, Path)]
        Fault class and path of every file to load.

    Yields
    ------
    (int, pd.DataFrame)
        The fault class and one instance DataFrame with the metadata columns
        ``instance_id``, ``well_id``, ``fault_class``, and ``source_type``
        attached.
    """
    for fault_class, filepath in entries:
        df = pd.read_parquet(filepath)
        df["instance_id"] = filepath.stem
        df["well_id"] = parse_well_id(filepath.name)
        df["fault_class"] = fault_class
        df["source_type"] = parse_source_type(filepath.name)
        yield fault_class, df


def iter_raw_instances(
    raw_dir: Path,
    fault_classes: list[int],
    max_instances_per_class: int | None = None,
    allow_overlap: bool = False,
    verbose: bool = False,
) -> Iterator[tuple[int, pd.DataFrame]]:
    """List, select and load the raw instances in one go.

    Convenience composition of ``list_raw_instances``, ``select_instances``
    and ``load_raw_instances``; callers that want the selection counts back
    (as ``features.build_features`` does for its summary) use the three steps
    directly.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset (contains folders ``0/`` .. ``9/``).
    fault_classes : list[int]
        Fault-class folders to read.
    max_instances_per_class : int | None
        Cap on instances per class; ``None`` loads everything.
    allow_overlap : bool
        Keep the real instances that overlap another of the same well
        (default off, see ``select_instances``).
    verbose : bool
        Print the overlap report (default off).

    Yields
    ------
    (int, pd.DataFrame)
        The fault class and one instance DataFrame with the metadata columns
        attached, as ``load_raw_instances`` yields them.
    """
    entries = list_raw_instances(raw_dir, fault_classes, max_instances_per_class)
    entries, _ = select_instances(entries, allow_overlap, verbose)
    yield from load_raw_instances(entries)


def mask_extreme_values(
    df: pd.DataFrame,
    limit: float = EXTREME_VALUE_LIMIT,
    pressure_min: float = PRESSURE_MIN,
    opening_min: float = OPENING_MIN,
    temperature_limits: tuple[float, float] = TEMPERATURE_LIMITS,
) -> tuple[pd.DataFrame, dict[str, Counter]]:
    """Replace readings that cannot be measurements with NaN.

    Four rules, each deliberately loose enough that no genuine signal is at
    risk — real spikes are fault signatures, not noise:

    ``magnitude``
        any sensor beyond ``limit`` in absolute value, infinities included.
        Real 3W instances carry sensors frozen at absurd levels (P-PDG at
        -1.2e42 Pa for three whole instances of well 6) and signals off by
        orders of magnitude (P-JUS-CKP around 1.4e9 Pa on well 26).
    ``negative pressure``
        a sensor of ``PRESSURE_SENSORS`` below ``pressure_min``. The 3W
        pressures are absolute, so a negative reading is a broken or
        mis-mapped tag; zero is left alone, being the frozen-at-zero case
        instead.
    ``negative opening``
        a sensor of ``OPENING_SENSORS`` below ``opening_min``: a choke
        opening is a percentage, and the one negative value the dataset
        carries (-99.99 %) is a sentinel of the source system.
    ``temperature range``
        a sensor of ``TEMPERATURE_SENSORS`` outside ``temperature_limits``,
        which catches the -999 and -99.99 sentinels of the source system as
        well as readings of 30,000 °C.

    No model can use such values: a z-score computed on them flattens every
    other value of the instance, and magnitudes beyond the float32 range break
    XGBoost outright. Masking rather than dropping keeps the timeline intact —
    the reading becomes missing data like any other, forward-filled if the gap
    is short (see ``clean_instance``) and imputed by the model pipeline
    otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        One raw instance.
    limit : float
        Magnitude from which any reading counts as extreme (see
        ``EXTREME_VALUE_LIMIT``).
    pressure_min : float
        Smallest acceptable reading of a pressure sensor.
    opening_min : float
        Smallest acceptable reading of a choke-opening sensor.
    temperature_limits : (float, float)
        Acceptable band of a temperature sensor, in °C.

    Returns
    -------
    (pd.DataFrame, dict[str, Counter])
        The instance with the failing readings masked — the input itself when
        there are none — and, per sensor that lost readings, how many each
        rule masked. A sensor left entirely NaN is the normal outcome for one
        frozen at an impossible value, and makes its instance a candidate for
        the quality gate of ``clean_instance``.
    """
    sensors = [
        c
        for c in df.columns
        if c not in NON_SENSOR_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    values = df[sensors]
    readings = values.to_numpy(dtype=float)
    low, high = temperature_limits

    # NaN fails every comparison, so missing data is never counted as masked.
    failures = {"magnitude": np.abs(readings) > limit}
    for rule, members, test in (
        ("negative pressure", PRESSURE_SENSORS, lambda col: col < pressure_min),
        ("negative opening", OPENING_SENSORS, lambda col: col < opening_min),
        ("temperature range", TEMPERATURE_SENSORS, lambda col: (col < low) | (col > high)),
    ):
        failed = np.zeros_like(failures["magnitude"])
        for i, sensor in enumerate(sensors):
            if sensor in members:
                failed[:, i] = test(readings[:, i])
        failures[rule] = failed & ~failures["magnitude"]  # every reading counted once

    counts: dict[str, Counter] = {}
    for rule, failed in failures.items():
        for sensor, n in zip(sensors, failed.sum(axis=0)):
            if n:
                counts.setdefault(sensor, Counter())[rule] = int(n)
    if not counts:
        return df, {}

    df = df.copy()
    df[sensors] = values.mask(np.logical_or.reduce(list(failures.values())))
    return df, counts


def clean_instance(df: pd.DataFrame, sensors: list[str] | None = None) -> pd.DataFrame | None:
    """Forward-fill short gaps and drop instances with a too-sparse critical sensor.

    The fill is causal (past values only) and capped at ``FFILL_LIMIT`` samples,
    so no future information leaks into a window. Instances whose critical
    sensor (``P-TPT`` by default) is missing in more than ``MAX_MISSING_RATIO`` of the
    samples are considered unusable and discarded. Readings masked by
    ``mask_extreme_values`` count as missing here, so an instance whose
    critical sensor is mostly garbage is discarded by this gate.

    Parameters
    ----------
    df : pd.DataFrame
        One raw instance.
    sensors : list[str] | None
        Sensor columns to clean; defaults to the available ``KEY_SENSORS``.

    Returns
    -------
    pd.DataFrame | None
        The cleaned instance, or ``None`` when the instance is discarded.
    """
    if sensors is None:
        sensors = [s for s in KEY_SENSORS if s in df.columns]

    df = df.copy()
    df[sensors] = df[sensors].ffill(limit=FFILL_LIMIT)

    if CRITICAL_SENSOR in df.columns and df[CRITICAL_SENSOR].isna().mean() > MAX_MISSING_RATIO:
        return None
    return df


# Per-instance z-scoring used to live here, applied to the signal before
# windowing. It now happens to the window *features* instead, at load time and
# against a reference the run chooses — see the ``normalization`` module, which
# also documents why the old whole-recording reference leaked the label.
