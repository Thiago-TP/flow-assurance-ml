"""Cut the recordings into windows and write one features parquet per window length.

Every row is one window of one recording (a merged real recording, or a
simulated or hand-drawn instance read from the raw dataset as it is). The
columns come in four groups, and the distinction is the deployability rule of
``ideas/2026-09-28_new_feature_dataset.md``: a feature is what a deployed model
can compute from the signals and the well's own past; anything that needs the
labels is metadata and never an input.

keys
    ``recording_id``, ``well_id``, ``source_type``, ``window_start`` and
    ``window_end`` (timestamps), ``n_valid`` (valid samples of the critical
    sensor in the window).
features
    the eleven statistics of ``features.window_features`` per sensor;
    ``<sensor>_frozen`` and ``<sensor>_missing`` flags; ``state_mode``, the
    well's operational status over the window; ``time_since_start``, seconds
    from the recording's first sample to the window's last.
references
    ``run__<sensor>_n``, ``_mean`` and ``_var``: the causal running statistics
    of the recording up to the window's first sample. The burn-in reference
    of length *K* windows is the running statistics at window *K*; the
    whole-history reference is the current window's own — both become
    load-time choices of the train pipeline.
metadata
    ``fault_class`` (the folder), ``window_label`` (the majority label of the
    window), ``time_to_event`` (seconds from the window's last sample to the
    next fault-labelled sample, transient or steady; NaN when none follows)
    and ``time_since_event`` (seconds from the last fault-labelled sample
    before the window to its first sample; NaN when none precedes).

Windows are cut on the recording's 1 Hz grid; they never span a discontinuity
because the reconstruction makes every recording contiguous. A window is one
operating condition, so two kinds are dropped rather than labelled by
majority: windows with any unlabelled sample, and windows that mix normal
operation (class 0) with abnormal operation (a fault's transient or steady
label). A window mixing a fault's transient and steady samples is kept and
carries the majority label. The counts of dropped windows go to the manifest.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from flowml.config import (
    CRITICAL_SENSOR,
    DATA_DIR,
    FAULT_CLASSES,
    FEATURE_STATS,
    FROZEN_STD_THRESHOLD,
    KEY_SENSORS,
    SOURCE_TYPES,
)
from flowml.dataset.manifest import code_state, file_sha256, read_manifest, write_manifest
from flowml.dataset.reconstruction import (
    MANIFEST_NAME,
    list_instances_by_source,
    merged_dir_name,
)
from flowml.features import window_features
from flowml.preprocessing import clean_instance, mask_extreme_values, parse_well_id

WINDOW_LENGTH = 512  # seconds; tentative, see the ideas document
WINDOW_OVERLAP = 0  # percent of the window shared with the next one
SOURCES = ("merged", "original")
RUN_PREFIX = "run__"
RUN_STATS = ("n", "mean", "var")
KEY_COLUMNS = ["recording_id", "well_id", "source_type", "window_start", "window_end", "n_valid"]
CONTEXT_FEATURES = ["state_mode", "time_since_start"]
METADATA_COLUMNS = ["fault_class", "window_label", "time_to_event", "time_since_event"]
DROP_REASONS = ("unlabelled", "mixed_operation")


def features_name(length: int, overlap: int, max_instances_per_class: int | None) -> str:
    """``features_w<length>_o<overlap>[_n<N>]``, the stem of a features parquet and its manifest."""
    cap = "" if max_instances_per_class is None else f"_n{max_instances_per_class}"
    return f"features_w{length}_o{overlap}{cap}"


def window_starts(n_samples: int, length: int, overlap: int) -> np.ndarray:
    """First sample of every window that fits in a recording.

    Parameters
    ----------
    n_samples : int
        Length of the recording.
    length : int
        Window length in samples.
    overlap : int
        Percent of the window shared with the next one, in ``[0, 100)``.

    Returns
    -------
    np.ndarray
        Start indices; empty when the recording is shorter than one window.
    """
    if not 0 <= overlap < 100:
        raise ValueError(f"overlap must be in [0, 100), got {overlap}")
    step = max(1, round(length * (1 - overlap / 100)))
    if n_samples < length:
        return np.empty(0, dtype=int)
    return np.arange(0, n_samples - length + 1, step)


def running_statistics(
    values: np.ndarray, starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count, mean and population variance of the valid samples before each start.

    Sums are taken on values shifted by the first valid sample, so a pressure
    of 1e7 Pa varying by a few Pa keeps its variance exact.

    Parameters
    ----------
    values : np.ndarray
        One sensor over the whole recording, NaN where missing.
    starts : np.ndarray
        Window start indices; the statistics cover ``values[:start]``.

    Returns
    -------
    (n, mean, var)
        Arrays aligned with ``starts``; mean is NaN for an empty prefix and
        var is NaN for a prefix of fewer than two valid samples.
    """
    valid = ~np.isnan(values)
    offset = values[np.argmax(valid)] if valid.any() else 0.0
    shifted = np.where(valid, values - offset, 0.0)
    count = np.concatenate([[0], np.cumsum(valid)])[starts]
    total = np.concatenate([[0.0], np.cumsum(shifted)])[starts]
    squares = np.concatenate([[0.0], np.cumsum(shifted * shifted)])[starts]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = total / count
        var = np.maximum(squares / count - mean * mean, 0.0)
    mean = np.where(count > 0, mean + offset, np.nan)
    var = np.where(count > 1, var, np.nan)
    return count.astype(float), mean, var


def mode_and_share(values: np.ndarray) -> tuple[float, float]:
    """The most frequent non-NaN value (the smallest on a tie) and its share of the non-NaN values."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan, np.nan
    uniques, counts = np.unique(valid, return_counts=True)
    best = int(np.argmax(counts))  # first maximum, and uniques are sorted: the smallest wins a tie
    return float(uniques[best]), float(counts[best] / len(valid))


def fault_neighbours(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every sample, the index of the next and of the previous fault-labelled sample.

    A sample counts as fault-labelled when its label is set and is not Normal
    (transients included). ``-1`` where there is none in that direction.
    """
    fault = ~np.isnan(labels) & (labels != 0)
    n = len(labels)
    positions = np.where(fault, np.arange(n), -1)
    previous = np.maximum.accumulate(positions)
    ahead = np.where(fault, np.arange(n), n)
    following = np.minimum.accumulate(ahead[::-1])[::-1]
    return np.where(following < n, following, -1), previous


def extract_windows(
    df: pd.DataFrame,
    recording_id: str,
    well_id: str,
    source_type: str,
    fault_class: int,
    length: int = WINDOW_LENGTH,
    overlap: int = WINDOW_OVERLAP,
    sensors: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Turn one cleaned recording into its window rows.

    Parameters
    ----------
    df : pd.DataFrame
        The recording on its 1 Hz grid: sensor columns, ``class`` and
        ``state`` (either may be absent), a ``DatetimeIndex``.
    recording_id, well_id, source_type : str
        The keys every row carries.
    fault_class : int
        The recording's folder.
    length, overlap : int
        Window length in seconds and overlap in percent.
    sensors : list[str] | None
        Sensors to featurize; defaults to ``KEY_SENSORS``. A sensor the
        recording lacks is treated as entirely missing.

    Returns
    -------
    (pd.DataFrame, dict[str, int])
        One row per kept window, and how many windows were dropped for each
        reason in ``DROP_REASONS``.
    """
    sensors = list(sensors or KEY_SENSORS)
    n = len(df)
    starts = window_starts(n, length, overlap)
    dropped = dict.fromkeys(DROP_REASONS, 0)
    if len(starts) == 0:
        return pd.DataFrame(), dropped
    nan = np.full(n, np.nan)
    arrays = {s: df[s].to_numpy(dtype=float) if s in df.columns else nan for s in sensors}
    labels = df["class"].to_numpy(dtype=float) if "class" in df.columns else nan
    states = df["state"].to_numpy(dtype=float) if "state" in df.columns else nan
    following, previous = fault_neighbours(labels)
    running = {s: running_statistics(arrays[s], starts) for s in sensors}
    index = df.index

    rows = []
    for k, start in enumerate(starts):
        end = start + length
        window_labels = labels[start:end]
        if np.isnan(window_labels).any():
            dropped["unlabelled"] += 1
            continue
        abnormal = window_labels != 0
        if abnormal.any() and not abnormal.all():
            dropped["mixed_operation"] += 1
            continue
        label = mode_and_share(window_labels)[0]
        row = {
            "recording_id": recording_id,
            "well_id": well_id,
            "source_type": source_type,
            "window_start": index[start],
            "window_end": index[end - 1],
            "n_valid": int(np.isfinite(arrays[CRITICAL_SENSOR][start:end]).sum())
            if CRITICAL_SENSOR in arrays
            else 0,
        }
        for sensor in sensors:
            stats = window_features(arrays[sensor][start:end], sensor)
            row.update(stats)
            std = stats[f"{sensor}_std"]
            row[f"{sensor}_frozen"] = float(np.isfinite(std) and std < FROZEN_STD_THRESHOLD)
            row[f"{sensor}_missing"] = float(np.isnan(std))
        row["state_mode"] = mode_and_share(states[start:end])[0]
        row["time_since_start"] = float(end - 1)
        for sensor in sensors:
            for name, values in zip(RUN_STATS, running[sensor]):
                row[f"{RUN_PREFIX}{sensor}_{name}"] = float(values[k])
        row["fault_class"] = fault_class
        row["window_label"] = int(label)
        nxt, prv = following[end - 1], previous[start]
        row["time_to_event"] = float(nxt - (end - 1)) if nxt >= 0 else np.nan
        row["time_since_event"] = float(start - prv) if prv >= 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows), dropped


def column_groups(sensors: list[str]) -> dict[str, list[str]]:
    """The four column groups a features parquet holds, by name."""
    return {
        "keys": KEY_COLUMNS,
        "features": [f"{s}_{stat}" for s in sensors for stat in FEATURE_STATS]
        + [f"{s}_{flag}" for s in sensors for flag in ("frozen", "missing")]
        + CONTEXT_FEATURES,
        "references": [f"{RUN_PREFIX}{s}_{name}" for s in sensors for name in RUN_STATS],
        "metadata": METADATA_COLUMNS,
    }


def list_recordings(
    source: str, raw_dir: Path, merged_dir: Path | None, max_instances_per_class: int | None
) -> tuple[list[tuple[Path, str, int, str]], dict | None]:
    """The recordings to featurize: real ones from the chosen source, synthetic ones from 3W.

    Returns
    -------
    (list[(path, well_id, folder, source_type)], dict | None)
        The entries, and the merged manifest when ``source`` is ``merged``.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r} (expected {SOURCES})")
    entries: list[tuple[Path, str, int, str]] = []
    manifest = None
    if source == "merged":
        merged_dir = merged_dir or DATA_DIR / merged_dir_name(max_instances_per_class)
        manifest = read_manifest(merged_dir / MANIFEST_NAME)
        for record in manifest["recordings"]:
            entries.append(
                (merged_dir / record["file"], str(record["well"]), int(record["folder"]), "WELL")
            )
    for folder, path, kind in list_instances_by_source(raw_dir, max_instances_per_class):
        if kind == SOURCE_TYPES["real"] and source == "merged":
            continue
        entries.append((path, parse_well_id(path.name), folder, kind))
    return entries, manifest


def build_features(
    source: str = "merged",
    raw_dir: Path | None = None,
    merged_dir: Path | None = None,
    lengths: list[int] | None = None,
    overlap: int = WINDOW_OVERLAP,
    sensors: list[str] | None = None,
    keep_extreme_values: bool = False,
    max_instances_per_class: int | None = None,
    output_dir: Path = DATA_DIR,
    verbose: bool = False,
) -> list[Path]:
    """Featurize every recording into one parquet per window length, each with its manifest.

    Parameters
    ----------
    source : str
        ``merged`` reads the reconstruction (``merged_dir``), ``original``
        reads 3W's real instances as they are; synthetic instances always
        come from ``raw_dir``.
    raw_dir : Path | None
        Root of the 3W dataset (``config.RAW_DATA_DIR`` by default).
    merged_dir : Path | None
        The reconstruction to read; defaults to ``data/merged`` or
        ``data/merged_n<N>`` when capped.
    lengths : list[int] | None
        Window lengths in seconds, one parquet each (``[WINDOW_LENGTH]``).
    overlap : int
        Percent of the window shared with the next one.
    sensors : list[str] | None
        Sensors to featurize (``KEY_SENSORS``).
    keep_extreme_values : bool
        Keep the readings that cannot be measurements instead of masking them.
    max_instances_per_class : int | None
        Cap per fault folder and source: the reconstruction's recordings come
        capped already (``merged_n<N>``), the synthetic instances are capped here.
    output_dir : Path
        Where the parquets and manifests go.
    verbose : bool
        Print one line per recording.

    Returns
    -------
    list[Path]
        The manifests written, one per length.
    """
    from flowml.config import RAW_DATA_DIR  # resolved at call time, so the env override applies

    raw_dir = raw_dir or RAW_DATA_DIR
    lengths = lengths or [WINDOW_LENGTH]
    sensors = list(sensors or KEY_SENSORS)
    entries, merged_manifest = list_recordings(source, raw_dir, merged_dir, max_instances_per_class)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = {L: features_name(L, overlap, max_instances_per_class) for L in lengths}
    writers: dict[int, pq.ParquetWriter | None] = dict.fromkeys(lengths)
    windows: dict[int, Counter] = {L: Counter() for L in lengths}
    dropped_windows: dict[int, Counter] = {L: Counter() for L in lengths}
    processed, gated = 0, 0
    try:
        for path, well_id, folder, kind in entries:
            df = pd.read_parquet(path).astype("float64")
            if not keep_extreme_values:
                df, _ = mask_extreme_values(df)
            df = clean_instance(df)
            if df is None:
                gated += 1
                continue
            processed += 1
            for L in lengths:
                frame, reasons = extract_windows(
                    df, path.stem, well_id, kind, folder, L, overlap, sensors
                )
                dropped_windows[L].update(reasons)
                if frame.empty:
                    continue
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writers[L] is None:
                    writers[L] = pq.ParquetWriter(
                        output_dir / f"{stems[L]}.parquet", table.schema, compression="snappy"
                    )
                writers[L].write_table(table)
                windows[L][f"{kind}/{folder}"] += len(frame)
            if verbose:
                print(
                    f"  {path.stem:<32} {kind:<10} folder {folder} · {len(df):>9,} samples",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()

    manifests = []
    for L in lengths:
        parquet = output_dir / f"{stems[L]}.parquet"
        by_source: Counter = Counter()
        by_class: Counter = Counter()
        for key, count in windows[L].items():
            kind, folder = key.split("/")
            by_source[kind] += count
            by_class[f"{folder} {FAULT_CLASSES[int(folder)]}"] += count
        manifests.append(
            write_manifest(
                output_dir / f"{stems[L]}.json",
                {
                    "kind": "3w-features",
                    "code": code_state(),
                    "source": {
                        "real_instances": source,
                        "raw_dir": str(raw_dir),
                        "merged_manifest": None
                        if merged_manifest is None
                        else {
                            "path": str(
                                (merged_dir or DATA_DIR / merged_dir_name(max_instances_per_class))
                                / MANIFEST_NAME
                            ),
                            "sha256": file_sha256(
                                (merged_dir or DATA_DIR / merged_dir_name(max_instances_per_class))
                                / MANIFEST_NAME
                            ),
                            "created": merged_manifest.get("created"),
                        },
                    },
                    "parameters": {
                        "window_length_seconds": L,
                        "window_overlap_percent": overlap,
                        "sensors": sensors,
                        "keep_extreme_values": keep_extreme_values,
                        "max_instances_per_class": max_instances_per_class,
                    },
                    "output": {
                        "parquet": parquet.name,
                        "size_bytes": parquet.stat().st_size if parquet.exists() else 0,
                        "windows": int(sum(windows[L].values())),
                        "windows_by_source": dict(by_source),
                        "windows_by_fault_class": dict(sorted(by_class.items())),
                        "windows_dropped": {
                            reason: int(dropped_windows[L][reason]) for reason in DROP_REASONS
                        },
                        "recordings_featurized": processed,
                        "recordings_dropped_by_quality_gate": gated,
                    },
                    "columns": column_groups(sensors),
                },
            )
        )
    return manifests
