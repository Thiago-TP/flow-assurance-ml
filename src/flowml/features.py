"""Sliding-window feature extraction and labeling.

Each instance is split into fixed-size windows (300 s, 50 % overlap) and every
window becomes one row: 11 statistics per sensor (8 sensors -> 88 features)
plus metadata. Every row carries BOTH labels used by the two tasks:

- ``window_label``: mode of the 3W ``class`` column inside the window
  (0 = normal, 1-9 = active event, 101-109 = transient). Used by the
  *detection* task.
- ``fault_class``: the fault the instance eventually develops (its 3W
  folder number). Used by the *prediction* task, which keeps only rows
  with ``window_label == 0``.

Because both labels are always present, a single features parquet per signal
filter serves both tasks.
"""

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import kurtosis, skew

from flowml.config import (
    CONSTANT_THRESHOLD,
    FAULT_CLASSES,
    FEATURE_STATS,
    KEY_SENSORS,
    MIN_VALID_SAMPLES,
    RAW_DATA_DIR,
    STEP_SIZE,
    WINDOW_SIZE,
)
from flowml.preprocessing import (
    clean_instance,
    iter_raw_instances,
    normalize_instance,
)


def window_features(window: np.ndarray, sensor: str) -> dict:
    """Compute the 11 statistical features of one window of one sensor.

    Outliers are deliberately preserved: pressure spikes are a fault signature,
    not noise, and are captured by ``max_zscore``. A constant window (stuck or
    switched-off sensor) gets skewness, kurtosis, and max_zscore of exactly 0
    instead of the NaN scipy would produce through catastrophic cancellation.

    Parameters
    ----------
    window : np.ndarray
        1-D slice of a (filtered, z-scored) sensor series.
    sensor : str
        Sensor name, used to prefix the feature keys.

    Returns
    -------
    dict
        ``{f"{sensor}_{stat}": value}`` for the 11 stats in ``FEATURE_STATS``;
        all NaN when the window has fewer than ``MIN_VALID_SAMPLES`` valid points.
    """
    valid = window[~np.isnan(window)]
    if len(valid) < MIN_VALID_SAMPLES:
        return {f"{sensor}_{stat}": np.nan for stat in FEATURE_STATS}

    mean = valid.mean()
    std = valid.std()
    diff1 = np.diff(valid)
    diff2 = np.diff(diff1)
    q75, q25 = np.percentile(valid, [75, 25])

    if std < CONSTANT_THRESHOLD:
        skew_v, kurt_v, max_z = 0.0, 0.0, 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skew_v = float(skew(valid))
            kurt_v = float(kurtosis(valid))
        max_z = float((np.abs(valid - mean) / std).max())

    return {
        f"{sensor}_mean": mean,
        f"{sensor}_std": std,
        f"{sensor}_min": valid.min(),
        f"{sensor}_max": valid.max(),
        f"{sensor}_median": float(np.median(valid)),
        f"{sensor}_iqr": float(q75 - q25),
        f"{sensor}_skewness": skew_v,
        f"{sensor}_kurtosis": kurt_v,
        f"{sensor}_diff1_std": float(diff1.std()) if len(diff1) > 1 else np.nan,
        f"{sensor}_diff2_std": float(diff2.std()) if len(diff2) > 1 else np.nan,
        f"{sensor}_max_zscore": max_z,
    }


def extract_instance_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> pd.DataFrame:
    """Turn one cleaned instance into a DataFrame of windowed feature rows.

    The instance is z-scored, then each window of each sensor is reduced to 11 statistics.
    Windows whose 3W ``class`` column is entirely NaN
    (unlabeled pre-event stretches) are discarded.

    Parameters
    ----------
    df : pd.DataFrame
        One cleaned instance with ``instance_id``, ``fault_class``,
        ``source_type``, and the 3W ``class`` column.
    sensors : list[str] | None
        Sensor columns to use; defaults to the available ``KEY_SENSORS``.
    window_size : int
        Window length in samples.
    step_size : int
        Stride between consecutive windows.

    Returns
    -------
    pd.DataFrame
        One row per window with metadata + 11 features per sensor.
    """
    if sensors is None:
        sensors = [s for s in KEY_SENSORS if s in df.columns]

    instance_id = df["instance_id"].iloc[0]
    fault_class = int(df["fault_class"].iloc[0])
    source_type = df["source_type"].iloc[0]

    df = normalize_instance(df, sensors)
    sensor_arrays = {s: df[s].to_numpy(dtype=float) for s in sensors}
    state = df["class"].to_numpy() if "class" in df.columns else None

    num_windows = (len(df) - window_size) // step_size + 1
    rows = [{} for _ in range(num_windows)]  # preallocate for speed
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size

        if state is not None:
            window_states = pd.Series(state[start:end]).dropna()
            if window_states.empty:
                continue
            window_label = int(window_states.mode().iloc[0])
        else:
            window_label = fault_class

        row = {
            "instance_id": instance_id,
            "fault_class": fault_class,
            "window_label": window_label,
            "source_type": source_type,
            "window_start": start,
        }
        for sensor in sensors:
            sensor_features = window_features(sensor_arrays[sensor][start:end], sensor)
            row.update(sensor_features)
        rows[i] = row

    return pd.DataFrame(rows)


def build_features(
    output_path: Path,
    raw_dir: Path = RAW_DATA_DIR,
    max_instances_per_class: int | None = None,
    verbose: bool = True,
) -> None:
    """Run the full raw -> features pass and write one parquet incrementally.

    Instances are processed one at a time and flushed to disk through a
    PyArrow writer, so peak memory stays at one instance regardless of
    dataset size.

    Parameters
    ----------
    output_path : Path
        Destination parquet file (overwritten if present).
    raw_dir : Path
        Root of the 3W dataset.
    max_instances_per_class : int | None
        Cap per class for quick smoke tests; ``None`` processes everything.
    verbose : bool
        Print per-class progress.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_windows = 0
    n_kept = 0
    n_dropped = 0

    try:
        current_class = None
        for fault_class, df_raw in iter_raw_instances(
            raw_dir, list(FAULT_CLASSES), max_instances_per_class
        ):
            if verbose and fault_class != current_class:
                current_class = fault_class
                print(f"  Class {fault_class}: {FAULT_CLASSES[fault_class]}")

            df_clean = clean_instance(df_raw)
            del df_raw
            if df_clean is None:
                n_dropped += 1
                continue
            n_kept += 1

            df_feat = extract_instance_features(df_clean)
            del df_clean
            if df_feat.empty:
                continue

            table = pa.Table.from_pandas(df_feat, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_windows += len(df_feat)
            del df_feat, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    if verbose:
        print(
            f"\nDone: {total_windows:,} windows from {n_kept} instances "
            f"({n_dropped} dropped by the quality filter)"
        )
        print(f"  -> {output_path}")
