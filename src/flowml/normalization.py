"""Z-scoring the window features against a per-instance reference, at load time.

The pipeline used to z-score each sensor *before* windowing, and bake the
result into the features parquet. That had two costs. It made normalization a
property of the dataset, so comparing with and without it meant rebuilding
231 MB of features; and, with the whole recording as the reference, it leaked
the label — a normal-operation window of a well that later developed a
Spurious DHSV Closure was divided by a standard deviation 372x larger than the
leak-free one, so the feature encoded the coming fault (TODO item 9).

Both go away once the transform is applied to the *features* rather than the
signals. Z-scoring a sensor by a constant (mu, sigma) is an affine map, and
every one of the eleven statistics in ``FEATURE_STATS`` has a closed form
under it:

===================================  ====================================
statistic                            value after z-scoring
===================================  ====================================
``mean`` ``min`` ``max`` ``median``  ``(v - mu) / sigma``
``std`` ``iqr`` ``diff1_std``        ``v / sigma``
``diff2_std``
``skewness`` ``kurtosis``            unchanged
``max_zscore``                       ``max(|max - mu|, |min - mu|) / sigma``
===================================  ====================================

``max_zscore`` is the only one that is not a plain rescale, because its
definition differs between modes: on z-scored data it is the largest absolute
value of the window (already a z-score against the instance baseline, never
re-normalized within the window), and the largest ``|x - mu|`` over a window
is attained at one of its two extremes — so the raw ``min`` and ``max`` give
it exactly.

Stage 1 therefore stores raw features plus, per instance and sensor, the
(mu, sigma) of every reference in ``NORMALIZATION_REFERENCES``. Choosing a
reference costs one vectorized pass over the feature matrix — milliseconds —
instead of a rebuild, and the ``instance`` reference reproduces the old
z-scored artifacts, which is what makes the leaky baseline comparable in the
re-made reports.

This is a row-wise transform: every row uses only its own instance's
statistics, so it neither needs nor benefits from being fitted inside a
cross-validation fold. That is worth being explicit about, because it means
moving normalization to load time does *not* by itself fix the leak — the
contaminated divisor and the window it divides belong to the same instance and
always land on the same side of any split. What fixes the leak is choosing a
reference that never saw the fault period: ``normal``, or ``none``.
"""

import numpy as np
import pandas as pd

from flowml.config import (
    CONSTANT_THRESHOLD,
    INVARIANT_STATS,
    LOCATION_SCALE_STATS,
    NORMALIZATIONS,
    REF_COL_PREFIX,
    SCALE_STATS,
    ref_col,
    sensors_with_features,
)


def reference_columns(frame: pd.DataFrame) -> list[str]:
    """The reference-statistic columns of a features frame.

    Parameters
    ----------
    frame : pd.DataFrame
        A features parquet loaded into memory.

    Returns
    -------
    list[str]
        Every column holding a stored (mu, sigma); these are inputs to the
        normalization, never features in their own right.
    """
    return [c for c in frame.columns if c.startswith(REF_COL_PREFIX)]


def normalized_sensors(frame: pd.DataFrame, reference: str) -> list[str]:
    """Sensors this frame can actually be z-scored against one reference.

    A sensor qualifies only when the frame carries everything
    ``_normalize_sensor`` will read: both stored statistics of the reference,
    *and* a complete set of window features. Stripping the
    ``ref__<sensor>_mean_<reference>`` name is not enough on its own — sensor
    names are free text, so a suffix match is an assumption about which names
    happen not to collide rather than a fact about the frame. It is the
    assumption that broke ``sensor_health.featured_sensors``, where ``_std``
    also matched ``diff1_std`` and invented a sensor called ``P-TPT_diff1``.
    Verifying the columns instead makes ``normalize_features`` total: every
    sensor it is handed can be transformed completely.

    Parameters
    ----------
    frame : pd.DataFrame
        A features parquet loaded into memory.
    reference : str
        One of ``NORMALIZATION_REFERENCES``.

    Returns
    -------
    list[str]
        Sensor names, in column order.
    """
    prefix, suffix = REF_COL_PREFIX, f"_mean_{reference}"
    named = [
        c[len(prefix) : -len(suffix)]
        for c in frame.columns
        if c.startswith(prefix) and c.endswith(suffix)
    ]
    complete = set(sensors_with_features(frame.columns))
    present = set(frame.columns)
    return [s for s in named if s in complete and ref_col(s, "std", reference) in present]


def normalize_features(frame: pd.DataFrame, normalization: str) -> pd.DataFrame:
    """Z-score a features frame in place of its raw values.

    Parameters
    ----------
    frame : pd.DataFrame
        Features parquet contents: metadata, raw window features and the
        stored reference statistics.
    normalization : str
        One of ``NORMALIZATIONS``. ``"none"`` returns the frame unchanged.

    Returns
    -------
    pd.DataFrame
        A copy with the eleven statistics of every sensor expressed against
        the chosen reference. The reference columns are left in place; the
        caller drops them when selecting features.

    Raises
    ------
    ValueError
        When the normalization is unknown, or the frame does not carry the
        statistics the chosen reference needs.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization: {normalization!r} (expected {NORMALIZATIONS})")
    if normalization == "none":
        return frame

    sensors = normalized_sensors(frame, normalization)
    if not sensors:
        raise ValueError(
            f"{normalization!r} normalization needs the '{ref_col('<sensor>', 'mean', normalization)}' "
            "columns, which this features parquet does not have. Rebuild it:\n"
            "  uv run scripts/01_build_features.py"
        )

    out = frame.copy()
    for sensor in sensors:
        _normalize_sensor(out, frame, sensor, normalization)
    return out


def _normalize_sensor(out: pd.DataFrame, raw: pd.DataFrame, sensor: str, reference: str) -> None:
    """Rewrite one sensor's eleven features against its reference, in place.

    Three cases, mirroring what the build-time normalizer used to do to the
    signal itself:

    - **no usable reference** (fewer than two valid samples, so the stored
      statistics are NaN): the sensor is left raw, as ``normalize_instance``
      left it.
    - **flat reference** (sigma below ``CONSTANT_THRESHOLD``: a stuck or
      switched-off sensor): every statistic becomes exactly 0, because the old
      normalizer replaced the whole column with zeros. That includes windows
      whose features were NaN for want of valid samples — zeroing the column
      gave them 300 valid zeros — so the zeros are written unconditionally to
      keep the reproduction faithful. This manufactured, perfectly flat feature
      is what puts "is this standard deviation exactly zero?" at the root of
      the z-scored trees, and is TODO item 10's subject.
    - **otherwise**: the closed forms of the module docstring.

    Parameters
    ----------
    out : pd.DataFrame
        Frame being rewritten.
    raw : pd.DataFrame
        The untouched frame, read for the pre-transform ``min`` and ``max``.
    sensor : str
        Sensor whose features to rewrite.
    reference : str
        Reference whose stored statistics to use.
    """
    mu = raw[ref_col(sensor, "mean", reference)].to_numpy(dtype=float)
    sigma = raw[ref_col(sensor, "std", reference)].to_numpy(dtype=float)

    usable = np.isfinite(mu) & np.isfinite(sigma)
    flat = usable & (sigma < CONSTANT_THRESHOLD)
    scaled = usable & ~flat

    # From the raw extremes, before they are overwritten: the largest |x - mu|
    # over a window is reached at one of its two ends.
    low = raw[f"{sensor}_min"].to_numpy(dtype=float)
    high = raw[f"{sensor}_max"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        max_zscore = np.maximum(np.abs(high - mu), np.abs(low - mu)) / sigma

    for stat in LOCATION_SCALE_STATS:
        column = f"{sensor}_{stat}"
        values = raw[column].to_numpy(dtype=float)
        out.loc[scaled, column] = (values[scaled] - mu[scaled]) / sigma[scaled]
    for stat in SCALE_STATS:
        column = f"{sensor}_{stat}"
        values = raw[column].to_numpy(dtype=float)
        out.loc[scaled, column] = values[scaled] / sigma[scaled]
    out.loc[scaled, f"{sensor}_max_zscore"] = max_zscore[scaled]

    # The constant-window guard is re-evaluated on the rescaled dispersion,
    # since that is the number the old code saw: a window that is flat in
    # z-scored units gets skewness and kurtosis of exactly 0 rather than the
    # NaN scipy would produce through catastrophic cancellation.
    rescaled_std = out[f"{sensor}_std"].to_numpy(dtype=float)
    degenerate = scaled & np.isfinite(rescaled_std) & (rescaled_std < CONSTANT_THRESHOLD)
    for stat in INVARIANT_STATS:
        out.loc[degenerate, f"{sensor}_{stat}"] = 0.0

    if flat.any():
        for stat in (*LOCATION_SCALE_STATS, *SCALE_STATS, *INVARIANT_STATS, "max_zscore"):
            out.loc[flat, f"{sensor}_{stat}"] = 0.0
