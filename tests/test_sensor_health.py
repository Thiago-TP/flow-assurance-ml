"""Tests of the frozen-sensor policies.

The problem being fixed is that a sensor which never moves gives every
dispersion statistic of its window an exact zero, which the trees then split on
at the root — instrument status arriving disguised as flow physics. These tests
pin down what each policy does about it, and in particular that ``flag`` leaves
the indicator as the *only* channel: blanking the dispersion statistics alone
would just move the shortcut into the level statistics.
"""

import numpy as np
import pandas as pd
import pytest

from flowml.config import CRITICAL_SENSOR, FEATURE_STATS, FROZEN_STD_THRESHOLD, ref_col
from flowml.sensor_health import (
    apply_frozen_policy,
    featured_sensors,
    frozen_indicator,
    frozen_instances,
    frozen_report,
    frozen_windows,
)

SENSORS = ["P-TPT", "QGL"]


def features_frame(frozen_rows: list[int], stuck_at: float = 0.0) -> pd.DataFrame:
    """Four windows of two sensors, with P-TPT frozen in the given rows."""
    n = 4
    frame = pd.DataFrame({"instance_id": ["A", "A", "B", "B"], "well_id": ["1", "1", "2", "2"]})
    for sensor in SENSORS:
        for stat in FEATURE_STATS:
            frame[f"{sensor}_{stat}"] = np.linspace(1.0, 4.0, n)
        # A frozen window: no dispersion, and every level statistic sits at the
        # value the sensor is stuck at.
        for row in frozen_rows if sensor == CRITICAL_SENSOR else []:
            for stat in ("std", "iqr", "diff1_std", "diff2_std", "max_zscore"):
                frame.loc[row, f"{sensor}_{stat}"] = 0.0
            for stat in ("mean", "min", "max", "median"):
                frame.loc[row, f"{sensor}_{stat}"] = stuck_at
        frame[ref_col(sensor, "mean", "instance")] = 0.0
        frame[ref_col(sensor, "std", "instance")] = 1.0
    return frame


# -- Detection -----------------------------------------------------------------


def test_featured_sensors_are_read_from_the_dispersion_columns():
    assert featured_sensors(features_frame([])) == SENSORS


def test_a_window_with_no_dispersion_is_frozen():
    frame = features_frame(frozen_rows=[1, 2])

    assert frozen_windows(frame, CRITICAL_SENSOR).tolist() == [False, True, True, False]
    assert not frozen_windows(frame, "QGL").any()


def test_a_window_with_dispersion_just_above_the_threshold_is_not_frozen():
    frame = features_frame([])
    frame.loc[0, f"{CRITICAL_SENSOR}_std"] = FROZEN_STD_THRESHOLD * 2

    assert not frozen_windows(frame, CRITICAL_SENSOR)[0]


def test_a_window_with_no_measurements_at_all_is_not_called_frozen():
    # All-NaN features mean nothing was measured, which is a different
    # condition from a sensor that was measured and never moved.
    frame = features_frame([])
    frame.loc[0, [f"{CRITICAL_SENSOR}_{stat}" for stat in FEATURE_STATS]] = np.nan

    assert not frozen_windows(frame, CRITICAL_SENSOR)[0]


def test_instance_level_freeze_is_read_from_the_stored_reference():
    frame = features_frame([])
    frame.loc[[2, 3], ref_col(CRITICAL_SENSOR, "std", "instance")] = 0.0

    assert frozen_instances(frame).tolist() == [False, False, True, True]


def test_instance_level_freeze_is_false_without_the_reference_columns():
    frame = features_frame([]).drop(columns=[ref_col(CRITICAL_SENSOR, "std", "instance")])

    assert not frozen_instances(frame).any()


# -- Policies ------------------------------------------------------------------


def test_keep_returns_the_frame_untouched():
    frame = features_frame([1])

    assert apply_frozen_policy(frame, "keep") is frame


def test_flag_blanks_every_statistic_of_a_frozen_window_not_only_dispersion():
    # The point of the policy: leaving mean/min/max/median in place would keep
    # the stuck value — 0 Pa for wells 35, 36 and 40 — as a second shortcut.
    frame = features_frame(frozen_rows=[1])

    out = apply_frozen_policy(frame, "flag")

    blanked = out.loc[1, [f"{CRITICAL_SENSOR}_{stat}" for stat in FEATURE_STATS]]
    assert blanked.isna().all()
    assert out.loc[0, f"{CRITICAL_SENSOR}_mean"] == frame.loc[0, f"{CRITICAL_SENSOR}_mean"]


def test_flag_adds_one_indicator_per_sensor_and_marks_only_the_frozen_windows():
    frame = features_frame(frozen_rows=[1, 2])

    out = apply_frozen_policy(frame, "flag")

    assert out[frozen_indicator(CRITICAL_SENSOR)].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert out[frozen_indicator("QGL")].tolist() == [0.0] * 4


def test_flag_leaves_a_healthy_sensor_alone():
    frame = features_frame(frozen_rows=[1])

    out = apply_frozen_policy(frame, "flag")

    healthy = [f"QGL_{stat}" for stat in FEATURE_STATS]
    pd.testing.assert_frame_equal(out[healthy], frame[healthy])


def test_flag_is_what_distinguishes_two_wells_stuck_at_different_values():
    # Well 29 sits at a constant 817 bar where wells 35/36/40 sit at 0. After
    # blanking, neither is distinguishable by its features — only by the flag,
    # which is the honest encoding of "this instrument was dead".
    at_zero = apply_frozen_policy(features_frame([1], stuck_at=0.0), "flag")
    at_817 = apply_frozen_policy(features_frame([1], stuck_at=817e5), "flag")

    columns = [f"{CRITICAL_SENSOR}_{stat}" for stat in FEATURE_STATS]
    assert at_zero.loc[1, columns].isna().all()
    pd.testing.assert_series_equal(at_zero.loc[1, columns], at_817.loc[1, columns])


def test_drop_removes_the_instances_whose_critical_sensor_never_moves():
    frame = features_frame([])
    frame.loc[[2, 3], ref_col(CRITICAL_SENSOR, "std", "instance")] = 0.0

    out = apply_frozen_policy(frame, "drop")

    assert out["instance_id"].tolist() == ["A", "A"]
    # `drop` is the TODO's other remedy, so it changes rows and nothing else.
    assert frozen_indicator(CRITICAL_SENSOR) not in out.columns


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="Unknown frozen-sensor mode"):
        apply_frozen_policy(features_frame([]), "impute")


# -- Reporting -----------------------------------------------------------------


def test_keep_reports_nothing():
    assert frozen_report(features_frame([1]), "keep") == ""


def test_flag_reports_the_affected_window_count_per_sensor():
    report = frozen_report(features_frame(frozen_rows=[1, 2]), "flag")

    assert f"{CRITICAL_SENSOR} 2" in report
    assert "QGL" not in report  # no frozen windows, so not listed


def test_drop_reports_the_instances_it_removes():
    frame = features_frame([])
    frame.loc[[2, 3], ref_col(CRITICAL_SENSOR, "std", "instance")] = 0.0

    report = frozen_report(frame, "drop")

    assert "1 instance(s)" in report
    assert "2 windows" in report
