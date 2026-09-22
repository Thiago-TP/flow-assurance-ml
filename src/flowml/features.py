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

Because both labels are always present, a single features parquet serves both
tasks.
"""

import gc
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import kurtosis, skew

from flowml.config import (
    CONSTANT_THRESHOLD,
    EXTREME_VALUE_LIMIT,
    FAULT_CLASSES,
    FEATURE_STATS,
    KEY_SENSORS,
    MIN_VALID_SAMPLES,
    NORMAL_OPERATION_REFERENCE,
    NORMALIZATION_REFERENCES,
    OPENING_MIN,
    PRESSURE_MIN,
    RAW_DATA_DIR,
    STEP_SIZE,
    TEMPERATURE_LIMITS,
    WINDOW_SIZE,
    ref_col,
)
from flowml.preprocessing import (
    clean_instance,
    list_raw_instances,
    load_raw_instances,
    mask_extreme_values,
    select_instances,
)


def window_features(window: np.ndarray, sensor: str) -> dict:
    """Compute the 11 statistical features of one raw window of one sensor.

    Outliers are deliberately preserved: pressure spikes are a fault signature,
    not noise, and are captured by ``max_zscore``, computed within the window
    itself. A constant window (stuck or switched-off sensor) gets skewness and
    kurtosis of exactly 0 instead of the NaN scipy would produce through
    catastrophic cancellation.

    The window is always raw here. Z-scoring a sensor against an instance-level
    reference is applied to these statistics afterwards, at load time, where
    the reference is a choice of the run rather than of the dataset (see the
    ``normalization`` module).

    Parameters
    ----------
    window : np.ndarray
        1-D slice of a raw sensor series.
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
        skew_v, kurt_v = 0.0, 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skew_v = float(skew(valid))
            kurt_v = float(kurtosis(valid))

    if std < CONSTANT_THRESHOLD:
        max_z = 0.0
    else:
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


def reference_statistics(df: pd.DataFrame, sensors: list[str]) -> dict[str, float]:
    """Compute the (mu, sigma) of every normalization reference of one instance.

    These travel in the features parquet beside the raw window features, so a
    run can z-score against any of them at load time without a rebuild (see
    the ``normalization`` module). Two references are stored:

    - ``instance`` — over the whole recording. This is what the pipeline used
      to apply at build time, kept so the earlier results stay reproducible,
      and the leak of TODO item 9: the fault period inflates the divisor of
      the normal-operation windows that the prediction task learns from.
    - ``normal-operation-values`` — over the samples the 3W ``class`` column
      marks as normal operation (0), so the fault period cannot reach the
      statistic.

    A sensor with fewer than two valid samples under a reference gets NaN for
    both statistics, which the normalizer reads as "leave this sensor raw" —
    the behaviour the build-time normalizer had.

    Parameters
    ----------
    df : pd.DataFrame
        One cleaned instance, including the 3W ``class`` column when present.
    sensors : list[str]
        Sensor columns to summarize.

    Returns
    -------
    dict[str, float]
        ``{ref_col(sensor, stat, reference): value}`` for every sensor,
        ``"mean"``/``"std"`` and reference.
    """
    normal_mask = df["class"].to_numpy() == 0 if "class" in df.columns else None
    stats: dict[str, float] = {}
    for sensor in sensors:
        column = df[sensor].to_numpy(dtype=float)
        for reference in NORMALIZATION_REFERENCES:
            if reference == NORMAL_OPERATION_REFERENCE and normal_mask is not None:
                sample = column[normal_mask]
            else:
                sample = column
            valid = sample[~np.isnan(sample)]
            mean, std = (np.nan, np.nan) if len(valid) < 2 else (valid.mean(), valid.std())
            stats[ref_col(sensor, "mean", reference)] = float(mean)
            stats[ref_col(sensor, "std", reference)] = float(std)
    return stats


def extract_instance_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> pd.DataFrame:
    """Turn one cleaned instance into a DataFrame of windowed feature rows.

    Each window of each sensor is reduced to 11 statistics of the **raw**
    signal, and every row additionally carries the instance's reference
    statistics (``reference_statistics``) so that a run can z-score the
    features against a reference of its choosing at load time. Windows whose
    3W ``class`` column is entirely NaN (unlabeled pre-event stretches) are
    discarded.

    Parameters
    ----------
    df : pd.DataFrame
        One cleaned instance with ``instance_id``, ``well_id``,
        ``fault_class``, ``source_type``, and the 3W ``class`` column.
    sensors : list[str] | None
        Sensor columns to use; defaults to the available ``KEY_SENSORS``.
    window_size : int
        Window length in samples.
    step_size : int
        Stride between consecutive windows.

    Returns
    -------
    pd.DataFrame
        One row per window with metadata, 11 raw features per sensor, and the
        instance's reference statistics repeated on every row.
    """
    if sensors is None:
        sensors = [s for s in KEY_SENSORS if s in df.columns]

    instance_id = df["instance_id"].iloc[0]
    well_id = df["well_id"].iloc[0]
    fault_class = int(df["fault_class"].iloc[0])
    source_type = df["source_type"].iloc[0]

    references = reference_statistics(df, sensors)
    sensor_arrays = {s: df[s].to_numpy(dtype=float) for s in sensors}
    state = df["class"].to_numpy() if "class" in df.columns else None

    num_windows = (len(df) - window_size) // step_size + 1
    rows = []
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
            "well_id": well_id,
            "fault_class": fault_class,
            "window_label": window_label,
            "source_type": source_type,
            "window_start": start,
        }
        for sensor in sensors:
            row.update(window_features(sensor_arrays[sensor][start:end], sensor))
        row.update(references)
        rows.append(row)

    return pd.DataFrame(rows)


def build_features(
    output_path: Path,
    raw_dir: Path = RAW_DATA_DIR,
    max_instances_per_class: int | None = None,
    allow_overlap: bool = False,
    keep_extreme_values: bool = False,
    verbose: bool = False,
) -> None:
    """Run the full raw -> features pass and write one parquet incrementally.

    The real instances that overlap another of the same well are dropped
    first (see ``preprocessing.select_instances``), unless ``allow_overlap``
    is set. The remaining instances are processed one at a time and flushed
    to disk through a PyArrow writer, so peak memory stays at one instance
    regardless of dataset size. Each one has the readings that cannot be
    measurements — beyond ``EXTREME_VALUE_LIMIT``, negative pressures or choke
    openings, temperatures outside ``TEMPERATURE_LIMITS`` — masked as missing
    data (see
    ``preprocessing.mask_extreme_values``) unless ``keep_extreme_values`` is
    set, and is then cleaned, which can drop it when the masking left its
    critical sensor mostly empty.

    Parameters
    ----------
    output_path : Path
        Destination parquet file (overwritten if present).
    raw_dir : Path
        Root of the 3W dataset.
    max_instances_per_class : int | None
        Cap per class for quick smoke tests; ``None`` processes everything.
        The cap applies before the overlap rule, so a capped run may see
        fewer overlaps than the full dataset has.
    allow_overlap : bool
        Keep the overlapping real instances instead of dropping them
        (default off).
    keep_extreme_values : bool
        Keep the readings that cannot be measurements instead of masking them
        (default off).
    verbose : bool
        Print the overlap and masking reports, per-class progress and the
        final summary (default off).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    entries = list_raw_instances(raw_dir, list(FAULT_CLASSES), max_instances_per_class)
    entries, overlap_table = select_instances(entries, allow_overlap, verbose)
    n_overlapping = int((~overlap_table["kept"]).sum())

    writer: pq.ParquetWriter | None = None
    total_windows = 0
    n_kept = 0
    n_dropped = 0
    masked_per_sensor: Counter[str] = Counter()
    masked_per_rule: Counter[str] = Counter()
    emptied_per_sensor: Counter[str] = Counter()
    n_masked_instances = 0

    try:
        current_class = None
        for fault_class, df_raw in load_raw_instances(entries):
            if verbose and fault_class != current_class:
                current_class = fault_class
                print(f"  Class {fault_class}: {FAULT_CLASSES[fault_class]}")

            if not keep_extreme_values:
                df_raw, masked = mask_extreme_values(df_raw)
                if masked:
                    n_masked_instances += 1
                    per_sensor = {s: sum(rules.values()) for s, rules in masked.items()}
                    masked_per_sensor.update(per_sensor)
                    for rules in masked.values():
                        masked_per_rule.update(rules)
                    emptied = [s for s in masked if df_raw[s].isna().all()]
                    emptied_per_sensor.update(emptied)
                    if verbose:
                        detail = ", ".join(
                            f"{s} {n:,}{' (all)' if s in emptied else ''}"
                            for s, n in sorted(per_sensor.items())
                        )
                        print(
                            f"    {df_raw['instance_id'].iloc[0]}: "
                            f"{sum(per_sensor.values()):,} implausible readings masked: {detail}"
                        )

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
        if keep_extreme_values:
            print("\nImplausible values: kept (--keep-extreme-values)")
        else:
            low, high = TEMPERATURE_LIMITS
            print(
                f"\nImplausible values: {sum(masked_per_sensor.values()):,} readings masked "
                f"in {n_masked_instances} of {len(entries)} instances"
            )
            print(
                f"  by rule: magnitude beyond |{EXTREME_VALUE_LIMIT:.0e}| "
                f"{masked_per_rule['magnitude']:,} | pressure below {PRESSURE_MIN:g} "
                f"{masked_per_rule['negative pressure']:,} | opening below {OPENING_MIN:g} "
                f"{masked_per_rule['negative opening']:,} | temperature outside "
                f"[{low:g}, {high:g}] C {masked_per_rule['temperature range']:,}"
            )
            for sensor, count in masked_per_sensor.most_common():
                emptied = emptied_per_sensor[sensor]
                plural = "instance" if emptied == 1 else "instances"
                print(
                    f"  {sensor}: {count:,} readings"
                    + (f" ({emptied} {plural} left with no {sensor} at all)" if emptied else "")
                )
        print(
            f"\nDone: {total_windows:,} windows from {n_kept} instances "
            f"({n_overlapping} removed as overlapping, {n_dropped} dropped by the quality filter)"
        )
        print(f"  -> {output_path}")
