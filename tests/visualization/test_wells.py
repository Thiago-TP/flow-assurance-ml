"""Tests for ``flowml.visualization.wells``."""

import numpy as np
import pandas as pd
import pytest

from flowml.visualization import wells
from flowml.visualization.common import LABEL_COLORS, _envelope
from tests.visualization.helpers import (
    make_all_fault_dirs,
    make_instance,
    real_name,
    write_dataset_ini,
    write_instance,
)

# -- _class_palette ------------------------------------------------------------------


def test_class_palette_normal_and_unknown_keep_label_colors():
    colors, names = wells._class_palette(np.array([0.0, np.nan]))
    assert colors[0.0] == LABEL_COLORS["normal"]
    assert colors[None] == LABEL_COLORS["unknown"]
    assert names[0.0] == "Normal Operation"
    assert names[None] == "Unknown"


def test_class_palette_active_fault_gets_stronger_tint_than_transient():
    from flowml.visualization.common import FAULT_COLORS, _tint

    colors, _ = wells._class_palette(np.array([3.0, 103.0]))
    assert colors[3.0] == pytest.approx(_tint(FAULT_COLORS[3], 0.30))
    assert colors[103.0] == pytest.approx(_tint(FAULT_COLORS[3], 0.15))


def test_class_palette_names_transient_and_active_distinctly():
    _, names = wells._class_palette(np.array([9.0, 109.0]))
    assert names[9.0] == "Hydrate in Service Line"
    assert names[109.0] == "Hydrate in Service Line - Transient"


# -- load_well_history -----------------------------------------------------------------


def test_load_well_history_raises_when_well_absent(tmp_path):
    make_all_fault_dirs(tmp_path)
    with pytest.raises(FileNotFoundError, match="WELL-00007"):
        wells.load_well_history(7, raw_dir=tmp_path)


def test_load_well_history_joins_instances_across_fault_folders(tmp_path):
    make_all_fault_dirs(tmp_path)
    write_instance(
        tmp_path / "0" / real_name(5, "20170101000000"),
        make_instance("2017-01-01", n=5, freq="1s"),
    )
    write_instance(
        tmp_path / "3" / real_name(5, "20170101000010"),
        make_instance("2017-01-01 00:00:10", n=5, freq="1s"),
    )

    timeline, spans = wells.load_well_history(5, raw_dir=tmp_path)

    assert len(spans) == 2
    assert len(timeline) == 10  # no overlap between the two windows
    assert list(spans["fault_class"]) == [0, 3]  # sorted by start


def test_load_well_history_keeps_most_complete_row_on_overlap(tmp_path):
    make_all_fault_dirs(tmp_path)
    # Two instances share the same 5-second window; the second is missing a
    # sensor reading, so the first (more complete) record must survive.
    complete = make_instance(
        "2017-01-01", n=5, sensors={"P-PDG": np.arange(5.0), "T-TPT": np.arange(5.0)}
    )
    sparse = make_instance(
        "2017-01-01",
        n=5,
        sensors={"P-PDG": np.full(5, np.nan), "T-TPT": np.arange(100.0, 105.0)},
    )
    write_instance(tmp_path / "0" / real_name(1, "20170101000000"), complete)
    write_instance(tmp_path / "1" / real_name(1, "20170101000000"), sparse)

    timeline, spans = wells.load_well_history(1, raw_dir=tmp_path)

    assert len(timeline) == 5
    assert not timeline["P-PDG"].isna().any()
    assert int(spans["n_samples"].sum() - len(timeline)) == 5  # 5 duplicated timestamps


# -- _compress_gaps ---------------------------------------------------------------------


def _env(index, sensor_values):
    frame = pd.DataFrame(
        {"sensor": sensor_values, "class": np.zeros(len(index)), "state": np.zeros(len(index))},
        index=index,
    )
    return _envelope(frame, max_points=len(index))


def test_compress_gaps_no_gap_gives_sequential_positions():
    index = pd.date_range("2020-01-01", periods=5, freq="1s")
    env = _env(index, np.arange(5.0))
    compressed = wells._compress_gaps(env, factor=20.0)
    assert list(compressed["position"]) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert not compressed["separator"].any()


def test_compress_gaps_inserts_separator_row_for_long_silence():
    index = pd.DatetimeIndex(
        ["2020-01-01 00:00:00", "2020-01-01 00:00:01", "2020-01-01 00:00:02"]
    ).append(pd.DatetimeIndex(["2020-06-01 00:00:00", "2020-06-01 00:00:01"]))
    env = _env(index, np.arange(5.0))
    compressed = wells._compress_gaps(env, factor=20.0, gap_slots=5.0)

    assert compressed["separator"].sum() == 1
    assert len(compressed) == len(env) + 1
    separator_row = compressed[compressed["separator"]]
    assert np.isnan(separator_row["sensor_min"].iloc[0])
    assert compressed["position"].is_monotonic_increasing


def test_compress_gaps_short_frame_falls_back_to_index_positions():
    index = pd.date_range("2020-01-01", periods=2, freq="1s")
    env = _env(index, np.array([1.0, 2.0]))
    compressed = wells._compress_gaps(env)
    assert list(compressed["position"]) == [0.0, 1.0]


# -- _plot_well_sensor ------------------------------------------------------------------


def _well_env(n=10, with_gap=False):
    if with_gap:
        index = pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-01 00:00:01"]).append(
            pd.DatetimeIndex(["2020-06-01 00:00:00", "2020-06-01 00:00:01"])
        )
        n = len(index)
    else:
        index = pd.date_range("2020-01-01", periods=n, freq="1s")
    frame = pd.DataFrame(
        {
            "sensor_min": np.arange(n, dtype=float),
            "sensor_max": np.arange(n, dtype=float) + 1.0,
            "class": np.zeros(n),
            "state": np.zeros(n),
        },
        index=index,
    )
    compressed = wells._compress_gaps(frame.assign())
    return compressed


def test_plot_well_sensor_reports_coverage_and_delta():
    env = _well_env(n=10)
    class_style = wells._class_palette(np.unique(env["class"].to_numpy(dtype=float)))
    fig = wells._plot_well_sensor(env, "sensor", 5, "Pa", "subtitle", class_style, (8, 10))

    report_text = fig.axes[2].texts[0].get_text()
    assert "coverage 80.0%" in report_text
    assert "8 of 10" in report_text


def test_plot_well_sensor_marks_never_recorded_sensor():
    n = 5
    index = pd.date_range("2020-01-01", periods=n, freq="1s")
    frame = pd.DataFrame(
        {"class": np.zeros(n), "state": np.zeros(n)}, index=index
    )  # no <sensor>_min/_max columns at all
    env = wells._compress_gaps(frame)
    class_style = wells._class_palette(np.unique(env["class"].to_numpy(dtype=float)))

    fig = wells._plot_well_sensor(env, "missing_sensor", 1, "", "subtitle", class_style, (0, 5))

    texts = [t.get_text() for t in fig.axes[2].texts]
    assert any("No data recorded" in t for t in texts)


def test_plot_well_sensor_draws_dashed_separator_for_gap():
    env = _well_env(with_gap=True)
    class_style = wells._class_palette(np.unique(env["class"].to_numpy(dtype=float)))
    fig = wells._plot_well_sensor(env, "sensor", 2, "Pa", "subtitle", class_style, (4, 4))

    # One dashed axvline per separator, drawn on every one of the 3 bands.
    dashed_lines = [
        line for ax in fig.axes for line in ax.lines if line.get_linestyle() not in ("-", "None")
    ]
    assert len(dashed_lines) == 3


# -- plot_well_history --------------------------------------------------------------------


def test_plot_well_history_writes_pdf_with_one_page_per_declared_sensor(tmp_path):
    raw_dir = tmp_path / "raw"
    write_dataset_ini(raw_dir, {"P-PDG": "Downhole pressure gauge [Pa]"})
    make_all_fault_dirs(raw_dir)
    write_instance(raw_dir / "0" / real_name(8, "20170101000000"), make_instance("2017-01-01", n=5))

    out_dir = tmp_path / "out"
    wells.plot_well_history(8, raw_dir=raw_dir, out_dir=out_dir)

    out_path = out_dir / "well_8_history.pdf"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_well_history_verbose_reports_coverage(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    write_dataset_ini(raw_dir, {"P-PDG": "Downhole pressure gauge [Pa]"})
    make_all_fault_dirs(raw_dir)
    write_instance(raw_dir / "0" / real_name(3, "20170101000000"), make_instance("2017-01-01", n=5))

    wells.plot_well_history(3, raw_dir=raw_dir, out_dir=tmp_path / "out", verbose=True)

    captured = capsys.readouterr()
    assert "P-PDG" in captured.out
    assert "%" in captured.out


# -- plot_wells_histories --------------------------------------------------------------------


def test_plot_wells_histories_raises_when_no_wells(tmp_path):
    make_all_fault_dirs(tmp_path)
    with pytest.raises(ValueError, match="No wells to plot"):
        wells.plot_wells_histories(raw_dir=tmp_path)


def test_plot_wells_histories_reports_failure_and_continues(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    write_dataset_ini(raw_dir, {"P-PDG": "Downhole pressure gauge [Pa]"})
    make_all_fault_dirs(raw_dir)
    write_instance(raw_dir / "0" / real_name(1, "20170101000000"), make_instance("2017-01-01", n=5))

    out_dir = tmp_path / "out"
    # Well 2 is requested but never written: plot_well_history raises for it,
    # and the batch must still complete well 1.
    wells.plot_wells_histories(well_ids=[1, 2], raw_dir=raw_dir, out_dir=out_dir)

    assert (out_dir / "well_1_history.pdf").exists()
    captured = capsys.readouterr()
    assert "WELL-00002 failed" in captured.out
    assert "1 of 2 wells plotted" in captured.out
