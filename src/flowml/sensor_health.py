"""Frozen sensors: making instrument status an explicit feature, at load time.

A sensor that never moves — stuck, switched off, or a dead transmitter — makes
every dispersion statistic of its windows exactly zero. ``P-TPT`` reads exactly
0 for 100 % of the samples of wells 35, 36 and 40 and of every hand-drawn
instance, and sits at a constant 817 bar for well 29
(``scripts/audits/sensor_distributions_auditing.py``). Those exact zeros are
crisp and highly predictive, so the trees split on them at the root: report 2
§6 found *every* top split of the z-scored instance tree to be a test of
whether some standard deviation is zero. The model is reading instrument
status, and calling it flow physics (TODO item 10).

Three policies, selected per run because the window's own standard deviation is
already in the features parquet, so none of them needs a rebuild:

``keep``
    Leave the zeros. What the pipeline did before this item; kept so the
    earlier results stay reproducible and so a report can measure the others
    against it.
``flag``
    Replace every statistic of a frozen sensor's window with NaN — the
    per-fold ``SimpleImputer`` fills them, as it does any other missing
    reading — and add an explicit ``<sensor>_frozen`` indicator. Instrument
    status still reaches the model, which is right, because for these wells it
    is most of what there is to know; it just arrives through one feature that
    says so rather than through eleven that pretend to be measurements.
``drop``
    Discard whole instances whose ``CRITICAL_SENSOR`` never moves across the
    recording, and leave everything else alone.

The last two are the two remedies the TODO item proposes, kept as alternatives
rather than combined so that each differs from ``keep`` in exactly one way.

Why all eleven statistics go, not only the dispersion ones
----------------------------------------------------------
Blanking ``std``, ``iqr`` and the derivative dispersions alone would leave
``mean``, ``min``, ``max`` and ``median`` — all equal to the value the sensor
is stuck at. For wells 35, 36 and 40 that value is 0, so the shortcut would
simply move from the dispersion statistics to the level ones and the model
would go on identifying those wells by instrument status. Removing the whole
sensor for that window is what leaves the indicator as the only channel, which
is the point of the exercise. This follows the 3W Toolkit's ``CleanSignals``,
which likewise blanks a frozen signal's entire column for the event it is
frozen in.

Note that ``frozen`` here is a property of the *window*, evaluated on the raw
statistics, not of the whole recording. A recording-wide freeze flags every one
of its windows; a sensor that dies halfway through flags only the windows after
it did. That granularity is also what keeps the indicator leak-free for the
prediction task: deciding it from the whole recording would let the fault
period decide the flag of a normal-operation window, which is exactly the
mistake item 9 removed.
"""

import numpy as np
import pandas as pd

from flowml.config import (
    CRITICAL_SENSOR,
    FEATURE_STATS,
    FROZEN_MODES,
    FROZEN_STD_THRESHOLD,
    ref_col,
    sensors_with_features,
)

FROZEN_SUFFIX = "_frozen"


def frozen_indicator(sensor: str) -> str:
    """Name of the indicator feature marking a sensor as frozen in a window."""
    return f"{sensor}{FROZEN_SUFFIX}"


def featured_sensors(frame: pd.DataFrame) -> list[str]:
    """Sensors that have a full set of window features in this frame.

    Parameters
    ----------
    frame : pd.DataFrame
        A features parquet loaded into memory.

    Returns
    -------
    list[str]
        Sensor names, in column order (see ``config.sensors_with_features``
        for why a suffix match alone will not do).
    """
    return sensors_with_features(frame.columns)


def frozen_windows(frame: pd.DataFrame, sensor: str) -> np.ndarray:
    """Flag the windows in which one sensor never moves.

    Parameters
    ----------
    frame : pd.DataFrame
        Features frame carrying ``<sensor>_std``.
    sensor : str
        Sensor to test.

    Returns
    -------
    np.ndarray
        Boolean per row. A window whose statistics are missing altogether (too
        few valid samples) is *not* called frozen: nothing was measured, which
        is a different condition and already reaches the model as NaN.
    """
    dispersion = frame[f"{sensor}_std"].to_numpy(dtype=float)
    return np.isfinite(dispersion) & (dispersion < FROZEN_STD_THRESHOLD)


def frozen_instances(frame: pd.DataFrame, sensor: str = CRITICAL_SENSOR) -> np.ndarray:
    """Flag the rows whose instance never moved one sensor across the recording.

    Uses the stored whole-recording standard deviation, the same statistic the
    ``instance`` normalization reference is built from, so no rebuild is needed
    to answer it.

    Parameters
    ----------
    frame : pd.DataFrame
        Features frame carrying the reference statistics.
    sensor : str
        Sensor to test; the critical sensor by default.

    Returns
    -------
    np.ndarray
        Boolean per row, true where the row's instance has that sensor frozen
        throughout. All false when the frame does not carry the statistic.
    """
    column = ref_col(sensor, "std", "instance")
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    dispersion = frame[column].to_numpy(dtype=float)
    return np.isfinite(dispersion) & (dispersion < FROZEN_STD_THRESHOLD)


def apply_frozen_policy(frame: pd.DataFrame, frozen_mode: str) -> pd.DataFrame:
    """Apply one frozen-sensor policy to a features frame.

    Parameters
    ----------
    frame : pd.DataFrame
        Features frame, already normalized if the run normalizes.
    frozen_mode : str
        One of ``FROZEN_MODES``.

    Returns
    -------
    pd.DataFrame
        ``keep`` returns the frame unchanged; ``flag`` returns a copy with the
        frozen windows blanked and the indicators appended; ``drop`` returns
        the rows whose instances keep the critical sensor alive.

    Raises
    ------
    ValueError
        When the mode is unknown.
    """
    if frozen_mode not in FROZEN_MODES:
        raise ValueError(f"Unknown frozen-sensor mode: {frozen_mode!r} (expected {FROZEN_MODES})")
    if frozen_mode == "keep":
        return frame
    if frozen_mode == "drop":
        return frame[~frozen_instances(frame)]

    out = frame.copy()
    for sensor in featured_sensors(frame):
        frozen = frozen_windows(frame, sensor)
        out[frozen_indicator(sensor)] = frozen.astype(float)
        if frozen.any():
            out.loc[frozen, [f"{sensor}_{stat}" for stat in FEATURE_STATS]] = np.nan
    return out


def frozen_report(frame: pd.DataFrame, frozen_mode: str) -> str:
    """One line per policy describing what it did, for the stage logs.

    Parameters
    ----------
    frame : pd.DataFrame
        The features frame *before* the policy is applied.
    frozen_mode : str
        The policy about to be applied.

    Returns
    -------
    str
        Human-readable summary, empty under ``keep``.
    """
    if frozen_mode == "keep":
        return ""
    if frozen_mode == "drop":
        dropped = frozen_instances(frame)
        n_instances = frame.loc[dropped, "instance_id"].nunique() if dropped.any() else 0
        return (
            f"  Frozen sensors: dropped {n_instances} instance(s) "
            f"({dropped.sum():,} windows) whose {CRITICAL_SENSOR} never moves."
        )
    affected = {
        sensor: int(frozen_windows(frame, sensor).sum()) for sensor in featured_sensors(frame)
    }
    listed = ", ".join(f"{s} {n:,}" for s, n in sorted(affected.items()) if n) or "none"
    return (
        f"  Frozen sensors: blanked and flagged per sensor ({listed}); "
        f"{len(affected)} indicator features added."
    )
