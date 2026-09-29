"""Tests of the dataset package: merging, gap relabelling, splitting, windows, running statistics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flowml.dataset.extraction import (
    extract_windows,
    fault_neighbours,
    mode_and_share,
    running_statistics,
    window_starts,
)
from flowml.dataset.reconstruction import (
    label_events,
    merge_spans,
    relabel_gaps,
    split_indices,
)

T0 = pd.Timestamp("2020-01-01 00:00:00")


def ts(seconds: int) -> pd.Timestamp:
    return T0 + pd.Timedelta(seconds=seconds)


# ----------------------------------------------------------------------------- reconstruction


def test_merge_spans_joins_overlapping_and_touching_instances_only():
    instances = [
        (ts(0), ts(100), 0, Path("a")),
        (ts(90), ts(200), 0, Path("b")),  # overlaps a
        (ts(201), ts(300), 1, Path("c")),  # touches b (one second later)
        (ts(400), ts(500), 0, Path("d")),  # separate
    ]
    plans = merge_spans(instances, well=7)
    assert [[p.name for p in plan.files] for plan in plans] == [["a", "b", "c"], ["d"]]
    assert plans[0].start == ts(0) and plans[0].end == ts(300)
    assert plans[0].folders == [0, 0, 1]


def test_relabel_gaps_fills_short_same_class_gaps_only():
    labels = np.array([0, 0, np.nan, np.nan, 0, np.nan, 1, 1, np.nan, np.nan, np.nan, 1, np.nan])
    filled, stats = relabel_gaps(labels, max_seconds=2)
    # the two-sample gap between zeros is filled; the 0|NaN|1 gap is not (flanks differ);
    # the three-sample gap between ones exceeds the bound; the trailing NaN has one flank.
    expected = np.array([0, 0, 0, 0, 0, np.nan, 1, 1, np.nan, np.nan, np.nan, 1, np.nan])
    np.testing.assert_array_equal(filled, expected)
    assert stats == {0: {"gaps": 1, "seconds": 2}}
    unbounded, stats = relabel_gaps(labels, max_seconds=None)
    assert np.isnan(unbounded[5]) and unbounded[8:11].tolist() == [1, 1, 1]
    assert stats[1] == {"gaps": 1, "seconds": 3}
    disabled, stats = relabel_gaps(labels, max_seconds=0)
    np.testing.assert_array_equal(disabled, labels)
    assert stats == {}


def test_split_indices_cuts_after_the_first_event_at_the_next_labelled_sample():
    # normal, event 1 (transient then steady), unlabelled, normal, event 7
    labels = np.array([0, 0, 101, 101, 1, np.nan, np.nan, 0, 0, 107, 7], dtype=float)
    assert split_indices(labels, folders=[1, 7]) == [(0, 1), (7, 7)]
    assert split_indices(labels, folders=[1, 1]) == [(0, 1)]
    assert split_indices(np.zeros(5), folders=[0]) == [(0, 0)]


def test_label_events_lists_the_runs_of_labels():
    labels = np.array([np.nan, 0, 0, 105, 105, 5], dtype=float)
    index = pd.date_range(T0, periods=6, freq="s")
    events = label_events(labels, index)
    assert [(e["class"], e["seconds"]) for e in events] == [(0, 2), (105, 2), (5, 1)]
    assert events[0]["start"] == ts(1).isoformat()


# ----------------------------------------------------------------------------- extraction


def test_window_starts_respect_length_and_overlap():
    assert window_starts(1000, 300, 0).tolist() == [0, 300, 600]
    assert window_starts(1000, 300, 50).tolist() == [0, 150, 300, 450, 600]
    assert window_starts(200, 300, 0).size == 0
    with pytest.raises(ValueError):
        window_starts(1000, 300, 100)


def test_running_statistics_match_numpy_on_the_prefix():
    rng = np.random.default_rng(0)
    values = 1e7 + rng.normal(0, 3, 1000)
    values[[5, 6, 500]] = np.nan
    starts = np.array([0, 1, 10, 600])
    n, mean, var = running_statistics(values, starts)
    assert n.tolist() == [0, 1, 8, 597]
    assert np.isnan(mean[0]) and np.isnan(var[0]) and np.isnan(var[1])
    for k, s in enumerate(starts[1:], start=1):
        prefix = values[:s][~np.isnan(values[:s])]
        assert mean[k] == pytest.approx(prefix.mean(), rel=1e-12)
        if len(prefix) > 1:
            assert var[k] == pytest.approx(prefix.var(), rel=1e-9)


def test_mode_and_share_and_fault_neighbours():
    assert mode_and_share(np.array([np.nan, 0, 0, 105, 105])) == (0.0, 0.5)  # tie: smallest wins
    assert np.isnan(mode_and_share(np.array([np.nan, np.nan]))[0])
    following, previous = fault_neighbours(np.array([np.nan, 0, 0, 105, 0, 5], dtype=float))
    assert following.tolist() == [3, 3, 3, 3, 5, 5]
    assert previous.tolist() == [-1, -1, -1, 3, 3, 5]


def recording(n: int, labels: np.ndarray, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "P-TPT": 2e7 + rng.normal(0, 100, n),
            "T-TPT": 60.0,  # frozen throughout
            "class": labels,
            "state": 0.0,
        },
        index=pd.date_range(T0, periods=n, freq="s"),
    )


def test_extract_windows_rows_keys_features_references_and_metadata():
    n = 3000
    labels = np.concatenate(
        [np.full(100, np.nan), np.zeros(1900), np.full(500, 105.0), np.full(500, 5.0)]
    )
    frame = recording(n, labels)
    rows, dropped = extract_windows(
        frame, "REC", "9", "WELL", 5, length=500, overlap=0, sensors=["P-TPT", "T-TPT", "QGL"]
    )
    # the first window touches the unlabelled prefix and is dropped; the other five are kept
    assert dropped == {"unlabelled": 1, "mixed_operation": 0}
    assert len(rows) == 5 and "label_purity" not in rows.columns
    first, last = rows.iloc[0], rows.iloc[-1]
    assert first["window_start"] == ts(500) and first["window_end"] == ts(999)
    assert first["window_label"] == 0 and first["n_valid"] == 500
    assert first["T-TPT_frozen"] == 1.0 and first["P-TPT_frozen"] == 0.0
    assert first["QGL_missing"] == 1.0 and np.isnan(first["QGL_mean"])
    assert first["state_mode"] == 0.0 and first["time_since_start"] == 999.0
    # references are causal: the first kept window sees the 500 samples before it
    assert first["run__P-TPT_n"] == 500.0
    assert first["run__P-TPT_mean"] == pytest.approx(frame["P-TPT"].iloc[:500].mean(), rel=1e-12)
    # metadata: the next fault-labelled sample is index 2000 (the transient)
    assert first["time_to_event"] == 2000 - 999 and np.isnan(first["time_since_event"])
    assert rows.iloc[3]["window_label"] == 105 and rows.iloc[3]["time_to_event"] == 0.0
    assert last["window_label"] == 5 and last["time_since_event"] == 0.0
    assert rows["fault_class"].eq(5).all()


def test_extract_windows_drops_mixed_windows_but_keeps_transient_with_steady():
    # 0..199 unlabelled, 200..1099 normal, 1100..1399 transient, 1400..1799 steady
    labels = np.concatenate(
        [np.full(200, np.nan), np.zeros(900), np.full(300, 105.0), np.full(400, 5.0)]
    )
    rows, dropped = extract_windows(
        recording(1800, labels), "REC", "9", "WELL", 5, length=300, overlap=0, sensors=["P-TPT"]
    )
    # windows: [0,300) unlabelled · [300,600) [600,900) normal · [900,1200) normal+transient: mixed
    # · [1200,1500) transient+steady: kept, majority transient · [1500,1800) steady
    assert dropped == {"unlabelled": 1, "mixed_operation": 1}
    assert rows["window_start"].tolist() == [ts(300), ts(600), ts(1200), ts(1500)]
    assert rows["window_label"].tolist() == [0, 0, 105, 5]
