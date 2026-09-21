"""Tests of load-time normalization: equivalence with the old build-time path.

The pipeline used to z-score the *signal* before windowing; it now z-scores the
*window features* afterwards, from statistics stored alongside them. The two
must agree, or the earlier results stop being comparable — which is the whole
reason the ``instance`` reference exists. The first test here is that
equivalence, checked against a transcription of the code that was removed.
"""

import numpy as np
import pandas as pd
import pytest

from flowml.config import CONSTANT_THRESHOLD, FEATURE_STATS, ref_col
from flowml.features import extract_instance_features, reference_statistics, window_features
from flowml.normalization import normalize_features, normalized_sensors, reference_columns

SENSORS = ["P-TPT", "T-TPT"]


def legacy_normalize_instance(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """The build-time normalizer this change replaced, kept as the oracle.

    Transcribed verbatim from ``preprocessing.normalize_instance`` as it stood
    at commit b1a2016, so the equivalence test compares against what actually
    ran rather than against a restatement of it.
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


def legacy_features(df: pd.DataFrame, sensors: list[str], window_size=64, step_size=32):
    """Window features the old way: normalize the signal, then cut the windows.

    Mirrors ``extract_instance_features`` before the change, including its
    ``max_zscore`` definition on normalized data (the largest absolute value of
    the window, never re-normalized within it).
    """
    normalized = legacy_normalize_instance(df, sensors)
    rows = []
    for i in range((len(df) - window_size) // step_size + 1):
        start, end = i * step_size, i * step_size + window_size
        row = {}
        for sensor in sensors:
            window = normalized[sensor].to_numpy(dtype=float)[start:end]
            row.update(window_features(window, sensor))
            valid = window[~np.isnan(window)]
            # The old code's normalized branch, which this rewrite folds into
            # `max(|max - mu|, |min - mu|) / sigma`.
            row[f"{sensor}_max_zscore"] = float(np.abs(valid).max()) if len(valid) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def synthetic_instance(seed: int = 0, n: int = 200, flat_sensor: bool = False) -> pd.DataFrame:
    """One instance with a normal-operation prefix and a fault period after it."""
    rng = np.random.default_rng(seed)
    normal = rng.normal(100.0, 2.0, size=n // 2)
    # The fault period moves and widens the signal: the mechanism that inflates
    # a whole-recording divisor above the normal-operation one.
    fault = rng.normal(130.0, 20.0, size=n - n // 2)
    series = np.concatenate([normal, fault])
    frame = pd.DataFrame(
        {
            "P-TPT": series,
            "T-TPT": np.zeros(n) if flat_sensor else series * 0.5 + rng.normal(0, 1, size=n),
            "class": np.concatenate([np.zeros(n // 2), np.full(n - n // 2, 8.0)]),
            "instance_id": "WELL-00001_x",
            "well_id": "1",
            "fault_class": 8,
            "source_type": "WELL",
        }
    )
    return frame


def built(frame: pd.DataFrame, window_size=64, step_size=32) -> pd.DataFrame:
    """Stage-1 output for one instance: raw features plus reference statistics."""
    return extract_instance_features(
        frame, sensors=SENSORS, window_size=window_size, step_size=step_size
    )


# -- Equivalence with the removed build-time path -------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_instance_reference_reproduces_the_old_build_time_normalization(seed):
    frame = synthetic_instance(seed)

    new = normalize_features(built(frame), "instance")
    old = legacy_features(frame, SENSORS)

    for sensor in SENSORS:
        for stat in FEATURE_STATS:
            column = f"{sensor}_{stat}"
            np.testing.assert_allclose(
                new[column].to_numpy(),
                old[column].to_numpy(),
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"{column} diverged from the build-time result",
            )


def test_instance_reference_reproduces_the_old_handling_of_a_flat_sensor():
    # The old normalizer replaced a constant sensor's whole column with zeros,
    # so every one of its features came out exactly 0.
    frame = synthetic_instance(flat_sensor=True)

    new = normalize_features(built(frame), "instance")
    old = legacy_features(frame, SENSORS)

    flat = [f"T-TPT_{stat}" for stat in FEATURE_STATS]
    assert (new[flat].to_numpy() == 0.0).all()
    np.testing.assert_allclose(new[flat].to_numpy(), old[flat].to_numpy())


# -- The references themselves --------------------------------------------------


def test_stage_one_stores_both_references_for_every_sensor():
    frame = built(synthetic_instance())

    assert normalized_sensors(frame, "instance") == SENSORS
    assert normalized_sensors(frame, "normal") == SENSORS
    assert len(reference_columns(frame)) == len(SENSORS) * 2 * 2


def test_a_reference_column_without_its_pair_does_not_name_a_sensor():
    # ref__<sensor>_mean_<reference> alone is not enough: _normalize_sensor
    # reads the standard deviation too, so a half-written frame must not
    # advertise the sensor as normalizable.
    frame = built(synthetic_instance()).drop(columns=[ref_col("T-TPT", "std", "instance")])

    assert normalized_sensors(frame, "instance") == ["P-TPT"]


def test_a_reference_without_window_features_does_not_name_a_sensor():
    # The mirror of the bug that bit `featured_sensors`: the reference columns
    # are not evidence that the eleven statistics the transform rewrites exist.
    frame = built(synthetic_instance()).drop(columns=["T-TPT_diff2_std"])

    assert normalized_sensors(frame, "instance") == ["P-TPT"]


def test_a_sensor_named_after_a_statistic_is_still_resolved_correctly():
    # Sensor names are free text, so nothing stops one from ending in a
    # statistic name; the suffix arithmetic must not mis-split it.
    frame = built(synthetic_instance()).rename(
        columns=lambda c: c.replace("T-TPT", "T-TPT_mean") if "T-TPT" in c else c
    )

    assert normalized_sensors(frame, "instance") == ["P-TPT", "T-TPT_mean"]


def test_the_normal_reference_ignores_the_fault_period():
    frame = synthetic_instance()
    stats = reference_statistics(frame, ["P-TPT"])

    normal_only = frame.loc[frame["class"] == 0, "P-TPT"].to_numpy()
    assert stats[ref_col("P-TPT", "mean", "normal")] == pytest.approx(normal_only.mean())
    assert stats[ref_col("P-TPT", "std", "normal")] == pytest.approx(normal_only.std())


def test_the_fault_period_inflates_the_instance_divisor_above_the_normal_one():
    # The leak of TODO item 9, in miniature: this ratio is what the audit
    # measures at 10x to 372x on the real classes.
    stats = reference_statistics(synthetic_instance(), ["P-TPT"])

    leaky = stats[ref_col("P-TPT", "std", "instance")]
    leak_free = stats[ref_col("P-TPT", "std", "normal")]
    assert leaky > 3 * leak_free


def test_a_sensor_with_too_few_valid_samples_gets_no_reference():
    frame = synthetic_instance()
    frame["P-TPT"] = np.nan

    stats = reference_statistics(frame, ["P-TPT"])

    assert np.isnan(stats[ref_col("P-TPT", "mean", "instance")])
    assert np.isnan(stats[ref_col("P-TPT", "std", "instance")])


def test_a_sensor_without_a_usable_reference_is_left_raw():
    frame = synthetic_instance()
    frame["T-TPT"] = np.nan
    raw = built(frame)

    out = normalize_features(raw, "instance")

    # P-TPT is rescaled; T-TPT has no reference, so its (NaN) features stand.
    assert not np.allclose(out["P-TPT_mean"], raw["P-TPT_mean"])
    assert out["T-TPT_mean"].isna().all()


# -- Selection and guards -------------------------------------------------------


def test_none_returns_the_frame_untouched():
    raw = built(synthetic_instance())

    assert normalize_features(raw, "none") is raw


def test_an_unknown_normalization_is_refused():
    with pytest.raises(ValueError, match="Unknown normalization"):
        normalize_features(built(synthetic_instance()), "per-well")


def test_a_frame_without_the_reference_columns_says_to_rebuild():
    raw = built(synthetic_instance()).drop(columns=reference_columns(built(synthetic_instance())))

    with pytest.raises(ValueError, match="01_build_features"):
        normalize_features(raw, "instance")


def test_normalizing_leaves_the_reference_columns_in_place_for_the_caller_to_drop():
    raw = built(synthetic_instance())

    out = normalize_features(raw, "normal")

    assert reference_columns(out) == reference_columns(raw)
    pd.testing.assert_frame_equal(out[reference_columns(out)], raw[reference_columns(raw)])
