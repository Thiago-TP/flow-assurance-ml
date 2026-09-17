"""Tests for ``flowml.visualization.timeline``."""

import numpy as np
import pandas as pd
import pytest

from flowml.visualization import timeline
from flowml.visualization.common import FAULT_COLORS, _tint
from tests.visualization.helpers import make_all_fault_dirs, real_name, write_instance

# -- _filename_stamp -----------------------------------------------------------------


def test_filename_stamp_parses_real_instance_name():
    fallback = pd.Timestamp("2000-01-01")
    stamp = timeline._filename_stamp("WELL-00005_20170608230000.parquet", fallback)
    assert stamp == pd.Timestamp("2017-06-08 23:00:00")


def test_filename_stamp_falls_back_when_unreadable():
    fallback = pd.Timestamp("2000-01-01")
    assert timeline._filename_stamp("SIMULATED_A.parquet", fallback) == fallback


# -- _fault_reach ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([0.0, 1.0, 9.0]), "steady"),
        (np.array([0.0, 103.0, np.nan]), "transient"),
        (np.array([0.0, 0.0, np.nan]), "normal"),
        (np.array([np.nan, np.nan]), "normal"),
        (np.array([101.0, 1.0]), "steady"),  # steady beats transient when both present
    ],
)
def test_fault_reach(values, expected):
    assert timeline._fault_reach(values) == expected


# -- _bar_color / _text_color -----------------------------------------------------------


def test_bar_color_normal_always_full_strength_regardless_of_reach():
    assert timeline._bar_color(0, "normal") == pytest.approx(_tint(FAULT_COLORS[0], 1.0))


def test_bar_color_fault_tints_by_reach():
    assert timeline._bar_color(3, "steady") == pytest.approx(_tint(FAULT_COLORS[3], 1.00))
    assert timeline._bar_color(3, "transient") == pytest.approx(_tint(FAULT_COLORS[3], 0.55))
    assert timeline._bar_color(3, "normal") == pytest.approx(_tint(FAULT_COLORS[3], 0.25))


def test_text_color_picks_white_on_dark_background():
    assert timeline._text_color((0.0, 0.0, 0.0)) == "#ffffff"


def test_text_color_picks_dark_on_light_background():
    assert timeline._text_color((1.0, 1.0, 1.0)) == "#1a1a1a"


# -- _text_width_pt ----------------------------------------------------------------------


def test_text_width_pt_grows_with_text_length():
    short = timeline._text_width_pt("A", 8.0)
    long = timeline._text_width_pt("A much longer label", 8.0)
    assert short > 0
    assert long > short


def test_text_width_pt_grows_with_fontsize():
    small = timeline._text_width_pt("label", 6.0)
    large = timeline._text_width_pt("label", 12.0)
    assert large > small


# -- _recording_blocks / _block_positions -------------------------------------------------


def test_recording_blocks_merges_close_instances_into_one_block():
    starts = np.array(["2020-01-01T00:00:00", "2020-01-01T00:10:00"], dtype="datetime64[s]")
    ends = np.array(["2020-01-01T00:05:00", "2020-01-01T00:15:00"], dtype="datetime64[s]")
    blocks = timeline._recording_blocks(starts, ends, gap_hours=1.0)
    assert len(blocks) == 1
    assert blocks["start"].iloc[0] == pd.Timestamp("2020-01-01T00:00:00")
    assert blocks["end"].iloc[0] == pd.Timestamp("2020-01-01T00:15:00")


def test_recording_blocks_splits_on_long_silence():
    starts = np.array(["2020-01-01T00:00:00", "2020-06-01T00:00:00"], dtype="datetime64[s]")
    ends = np.array(["2020-01-01T01:00:00", "2020-06-01T01:00:00"], dtype="datetime64[s]")
    blocks = timeline._recording_blocks(starts, ends, gap_hours=12.0)
    assert len(blocks) == 2
    assert blocks["x0"].iloc[1] > blocks["hours"].iloc[0]  # a gap was inserted after block 0


def test_recording_blocks_x0_is_cumulative_hours_plus_gap():
    starts = np.array(
        ["2020-01-01T00:00:00", "2020-01-05T00:00:00", "2020-01-10T00:00:00"],
        dtype="datetime64[s]",
    )
    ends = starts + np.timedelta64(1, "h")
    blocks = timeline._recording_blocks(starts, ends, gap_hours=1.0)
    assert len(blocks) == 3
    assert blocks["x0"].iloc[0] == 0.0
    assert (blocks["x0"].diff().dropna() > blocks["hours"].iloc[:-1].to_numpy()).all()


def test_block_positions_places_timestamp_relative_to_its_block():
    starts = np.array(["2020-01-01T00:00:00", "2020-06-01T00:00:00"], dtype="datetime64[s]")
    ends = np.array(["2020-01-01T02:00:00", "2020-06-01T02:00:00"], dtype="datetime64[s]")
    blocks = timeline._recording_blocks(starts, ends, gap_hours=1.0)

    timestamps = np.array(["2020-01-01T01:00:00"], dtype="datetime64[ns]")
    positions = timeline._block_positions(blocks, timestamps)
    assert positions[0] == pytest.approx(blocks["x0"].iloc[0] + 1.0)


# -- _plot_well_timeline -----------------------------------------------------------------


def _timeline_rows(n=3, well_id=1, overlap=False):
    if overlap:
        starts = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:00"])
        ends = pd.to_datetime(["2020-01-01 01:00:00", "2020-01-01 01:00:00"])
        fault_classes = [0, 3]
    else:
        starts = pd.to_datetime([f"2020-01-0{i + 1} 00:00:00" for i in range(n)])
        ends = starts + pd.Timedelta(hours=1)
        fault_classes = [0] * n
    return pd.DataFrame(
        {
            "fault_class": fault_classes,
            "stamp": starts,
            "start": starts,
            "end": ends,
            "reach": ["steady"] * len(starts),
        }
    )


def test_plot_well_timeline_builds_one_lane_per_non_overlapping_group():
    rows = _timeline_rows(overlap=False)
    fig = timeline._plot_well_timeline(1, rows)
    ax = fig.axes[0]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]  # y axis is inverted
    assert len(ax.get_yticks()) == 1  # no overlap: everything fits in a single lane


def test_plot_well_timeline_stacks_overlapping_instances():
    rows = _timeline_rows(overlap=True)
    fig = timeline._plot_well_timeline(2, rows)
    ax = fig.axes[0]
    assert len(ax.get_yticks()) == 2  # two overlapping instances need two lanes


def test_plot_well_timeline_legend_has_one_entry_per_color_drawn():
    # Faults 0 (normal) and 3 with two different reaches: three distinct
    # colors must appear, and the legend must key every one of them (see the
    # "a legend must show every color exactly as drawn" invariant).
    rows = pd.DataFrame(
        {
            "fault_class": [0, 3, 3],
            "stamp": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "start": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "end": pd.to_datetime(["2020-01-01 01:00", "2020-02-01 01:00", "2020-03-01 01:00"]),
            "reach": ["steady", "steady", "transient"],
        }
    )
    fig = timeline._plot_well_timeline(3, rows)
    legend = fig.legends[0]
    assert len(legend.legend_handles) == 3
    labels = {h.get_label() for h in legend.legend_handles}
    assert "Normal" in labels
    assert "Severe Slugging (steady state reached)" in labels
    assert "Severe Slugging (transient state reached)" in labels


def test_plot_well_timeline_respects_minimum_lane_slots():
    rows = _timeline_rows(overlap=False, n=1)
    fig = timeline._plot_well_timeline(1, rows, lane_slots=5)
    ax = fig.axes[0]
    # ylim spans slots (5) even though this well only fills one lane.
    assert ax.get_ylim()[0] == pytest.approx(5 - 0.3)


# -- plot_faults_per_well -----------------------------------------------------------------


def test_plot_faults_per_well_raises_when_no_real_instances(tmp_path):
    make_all_fault_dirs(tmp_path)
    with pytest.raises(FileNotFoundError, match="No real"):
        timeline.plot_faults_per_well(raw_dir=tmp_path)


def test_plot_faults_per_well_writes_one_pdf_with_pages(tmp_path):
    raw_dir = tmp_path / "raw"
    make_all_fault_dirs(raw_dir)
    from tests.visualization.helpers import make_instance

    write_instance(raw_dir / "0" / real_name(1, "20170101000000"), make_instance("2017-01-01", n=5))
    write_instance(raw_dir / "3" / real_name(2, "20170102000000"), make_instance("2017-01-02", n=5))

    out_dir = tmp_path / "out"
    timeline.plot_faults_per_well(raw_dir=raw_dir, out_dir=out_dir)

    out_path = out_dir / "faults_per_well.pdf"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_faults_per_well_verbose_reports_per_class_counts(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    make_all_fault_dirs(raw_dir)
    from tests.visualization.helpers import make_instance

    write_instance(raw_dir / "0" / real_name(1, "20170101000000"), make_instance("2017-01-01", n=5))

    timeline.plot_faults_per_well(raw_dir=raw_dir, out_dir=tmp_path / "out", verbose=True)

    captured = capsys.readouterr()
    assert "Class 0 (Normal): 1 instances" in captured.out
    assert "WELL-00001" in captured.out
