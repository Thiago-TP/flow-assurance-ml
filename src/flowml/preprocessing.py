"""Raw-data loading, cleaning, normalization, and signal filtering.

The 3W dataset stores one parquet file per instance (a continuous recording of
one well), organized in folders ``0/`` .. ``9/`` named after the fault class.
This module turns those raw files into clean, per-instance-normalized,
optionally filtered sensor series ready for feature extraction.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from flowml.config import (
    CONSTANT_THRESHOLD,
    CRITICAL_SENSOR,
    FFILL_LIMIT,
    KEY_SENSORS,
    MAX_MISSING_RATIO,
)


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


def iter_raw_instances(
    raw_dir: Path,
    fault_classes: list[int],
    max_instances_per_class: int | None = None,
):
    """Yield raw instances one at a time, keeping memory usage flat.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset (contains folders ``0/`` .. ``9/``).
    fault_classes : list[int]
        Fault-class folders to read.
    max_instances_per_class : int | None
        Cap on instances per class; ``None`` loads everything. Useful for a
        quick smoke test of the pipeline.

    Yields
    ------
    (int, pd.DataFrame)
        The fault class and one instance DataFrame with the metadata columns
        ``instance_id``, ``fault_class``, and ``source_type`` attached.
    """
    for fault_class in fault_classes:
        class_dir = raw_dir / str(fault_class)
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        files = sorted(class_dir.glob("*.parquet"))
        if max_instances_per_class is not None:
            files = files[:max_instances_per_class]
        for filepath in files:
            df = pd.read_parquet(filepath)
            df["instance_id"] = filepath.stem
            df["fault_class"] = fault_class
            df["source_type"] = parse_source_type(filepath.name)
            yield fault_class, df


def clean_instance(df: pd.DataFrame, sensors: list[str] | None = None) -> pd.DataFrame | None:
    """Forward-fill short gaps and drop instances with a too-sparse critical sensor.

    The fill is causal (past values only) and capped at ``FFILL_LIMIT`` samples,
    so no future information leaks into a window. Instances whose critical
    sensor (``P-TPT`` by default) is missing in more than ``MAX_MISSING_RATIO`` of the
    samples are considered unusable and discarded.

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


def normalize_instance(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Z-score each sensor within one instance.

    Wells operate at very different absolute levels (e.g. 50 bar vs 200 bar),
    so per-instance normalization makes the model learn *patterns of change*
    relative to each well's own baseline instead of absolute values. Constant
    sensors (stuck or switched off) are set to 0 so their absolute level
    cannot leak into the features.

    Parameters
    ----------
    df : pd.DataFrame
        One cleaned instance.
    sensors : list[str]
        Sensor columns to normalize.

    Returns
    -------
    pd.DataFrame
        Copy of the instance with normalized sensors.
    """
    df = df.copy()
    for sensor in sensors:
        col = df[sensor].to_numpy(dtype=float)
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        std = valid.std()
        if std < CONSTANT_THRESHOLD:
            df[sensor] = 0.0
            continue
        df[sensor] = (col - valid.mean()) / std
    return df
