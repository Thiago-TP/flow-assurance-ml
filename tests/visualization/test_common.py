"""Tests for ``flowml.visualization.common``."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb

from flowml.visualization import common
from tests.visualization.helpers import (
    DATASET_INI_VARIABLES,
    drawn_name,
    make_all_fault_dirs,
    make_instance,
    real_name,
    simulated_name,
    write_dataset_ini,
    write_instance,
)

# -- resolve_fault -------------------------------------------------------------


@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        ("9", (9, "Hydrate in Service Line")),
        (9, (9, "Hydrate in Service Line")),
        ("0", (0, "Normal")),
        ("Hydrate in Service Line", (9, "Hydrate in Service Line")),
        ("hydrate in service line", (9, "Hydrate in Service Line")),
        ("  9  ", (9, "Hydrate in Service Line")),
    ],
)
def test_resolve_fault_accepts_number_or_name(fault, expected):
    assert common.resolve_fault(fault) == expected


@pytest.mark.parametrize("fault", ["99", "not a fault", ""])
def test_resolve_fault_rejects_unknown(fault):
    with pytest.raises(ValueError, match="Unknown fault"):
        common.resolve_fault(fault)


# -- _dataset_properties / load_sensor_units / load_sensor_names ---------------


def test_dataset_properties_missing_ini_returns_none(tmp_path):
    assert common._dataset_properties(tmp_path) is None


def test_dataset_properties_missing_section_returns_none(tmp_path):
    (tmp_path / "dataset.ini").write_text("[OTHER_SECTION]\nfoo = bar\n", encoding="utf-8")
    assert common._dataset_properties(tmp_path) is None


def test_dataset_properties_reads_uppercased_keys(tmp_path):
    write_dataset_ini(tmp_path)
    properties = common._dataset_properties(tmp_path)
    assert properties["P-PDG"] == DATASET_INI_VARIABLES["P-PDG"]
    assert "p-pdg" not in properties


def test_load_sensor_units_missing_ini_returns_empty_dict(tmp_path):
    assert common.load_sensor_units(tmp_path) == {}


def test_load_sensor_units_parses_and_prettifies(tmp_path):
    write_dataset_ini(tmp_path)
    units = common.load_sensor_units(tmp_path)
    assert units["P-PDG"] == "Pa"
    assert units["T-PDG"] == "°C"  # oC -> °C
    assert units["ABER-CKP"] == "-"  # enumerated valve state collapses to "-"


def test_load_sensor_units_collapses_multivalued_units(tmp_path):
    write_dataset_ini(tmp_path, {"QGL": "Gas-lift flow rate [m3/s]"})
    units = common.load_sensor_units(tmp_path)
    assert units["QGL"] == "m³/s"


def test_load_sensor_units_skips_descriptions_without_brackets(tmp_path):
    write_dataset_ini(tmp_path, {"WEIRD": "No unit here"})
    units = common.load_sensor_units(tmp_path)
    assert "WEIRD" not in units


def test_load_sensor_names_missing_ini_falls_back_to_key_sensors(tmp_path):
    from flowml.config import KEY_SENSORS

    assert common.load_sensor_names(tmp_path) == list(KEY_SENSORS)


def test_load_sensor_names_excludes_timestamp_class_state(tmp_path):
    write_dataset_ini(tmp_path)
    names = common.load_sensor_names(tmp_path)
    assert "TIMESTAMP" not in names
    assert "CLASS" not in names
    assert "STATE" not in names
    assert set(names) == set(DATASET_INI_VARIABLES)


# -- list_instances / list_well_ids ---------------------------------------------


def test_list_instances_rejects_unknown_source(tmp_path):
    (tmp_path / "0").mkdir(parents=True)
    with pytest.raises(ValueError, match="Unknown instance source"):
        common.list_instances(0, "bogus", tmp_path)


def test_list_instances_missing_class_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        common.list_instances(3, "real", tmp_path)


def test_list_instances_filters_by_source_and_sorts(tmp_path):
    class_dir = tmp_path / "2"
    write_instance(class_dir / real_name(7, "20170101000000"), make_instance("2017-01-01"))
    write_instance(class_dir / real_name(3, "20170101000000"), make_instance("2017-01-01"))
    write_instance(class_dir / simulated_name("a"), make_instance("2017-01-01"))
    write_instance(class_dir / drawn_name("a"), make_instance("2017-01-01"))

    real_files = common.list_instances(2, "real", tmp_path)
    assert [f.name for f in real_files] == sorted(
        [real_name(7, "20170101000000"), real_name(3, "20170101000000")]
    )

    assert [f.name for f in common.list_instances(2, "simulated", tmp_path)] == [
        simulated_name("a")
    ]
    assert [f.name for f in common.list_instances(2, "drawn", tmp_path)] == [drawn_name("a")]


def test_list_instances_empty_when_no_matching_source(tmp_path):
    class_dir = tmp_path / "0"
    write_instance(class_dir / real_name(1, "20170101000000"), make_instance("2017-01-01"))
    assert common.list_instances(0, "simulated", tmp_path) == []


def test_list_well_ids_gathers_unique_sorted_real_wells(tmp_path):
    # list_well_ids scans every fault-class folder, so all of them must exist,
    # as they do in a real (even if partially downloaded) 3W dataset.
    make_all_fault_dirs(tmp_path)

    write_instance(tmp_path / "0" / real_name(5, "20170101000000"), make_instance("2017-01-01"))
    write_instance(tmp_path / "1" / real_name(2, "20170102000000"), make_instance("2017-01-02"))
    write_instance(
        tmp_path / "1" / real_name(5, "20170103000000"), make_instance("2017-01-03")
    )  # well 5 again, from a different fault folder
    write_instance(tmp_path / "1" / simulated_name("a"), make_instance("2017-01-01"))

    assert common.list_well_ids(tmp_path) == [2, 5]


# -- _label_kind / _label_name --------------------------------------------------


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (float("nan"), "unknown"),
        (0.0, "normal"),
        (1.0, "active"),
        (9.0, "active"),
        (101.0, "transient"),
        (109.0, "transient"),
        (50.0, "unknown"),
        (110.0, "unknown"),
        (-1.0, "unknown"),
    ],
)
def test_label_kind(value, kind):
    assert common._label_kind(value) == kind


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "Unknown"),
        (0.0, "Normal Operation"),
        (3.0, "Severe Slugging"),
        (103.0, "Severe Slugging - Transient"),
    ],
)
def test_label_name(value, expected):
    assert common._label_name(value) == expected


# -- _segments -------------------------------------------------------------------


def test_segments_splits_constant_runs_including_nan():
    values = np.array([0.0, 0.0, 1.0, 1.0, 1.0, np.nan, np.nan, 2.0])
    segments = common._segments(values)
    assert [(s, e) for s, e, _ in segments] == [(0, 2), (2, 5), (5, 7), (7, 8)]
    assert [v for _, _, v in segments[:2]] + [segments[3][2]] == [0.0, 1.0, 2.0]
    assert np.isnan(segments[2][2])


def test_segments_single_run():
    values = np.zeros(5)
    assert common._segments(values) == [(0, 5, 0.0)]


def test_segments_every_sample_its_own_run():
    values = np.array([0.0, 1.0, 2.0])
    segments = common._segments(values)
    assert [(s, e) for s, e, _ in segments] == [(0, 1), (1, 2), (2, 3)]


# -- _draw_band ------------------------------------------------------------------


def test_draw_band_sets_axis_cosmetics(subplots):
    _, ax = subplots()
    x = np.arange(10, dtype=float)
    segments = [(0, 5, 0.0), (5, 10, 1.0)]
    colors = {0.0: "#111111", 1.0: "#222222"}
    names = {0.0: "A", 1.0: "B"}

    common._draw_band(ax, x, segments, colors, names, "class")

    assert ax.get_ylabel() == "class"
    assert ax.get_ylim() == (0.0, 1.0)
    assert list(ax.get_yticks()) == []
    assert len(ax.patches) == 2  # one axvspan per segment


def test_draw_band_only_labels_wide_enough_segments(subplots):
    _, ax = subplots()
    n = 100
    x = np.arange(n, dtype=float)
    # First segment spans more than 5% of the axis and gets a name; the second
    # is a single sample and stays unlabeled.
    segments = [(0, 90, 0.0), (90, 91, 1.0), (91, 100, 2.0)]
    colors = {0.0: "#111111", 1.0: "#222222", 2.0: "#333333"}
    names = {0.0: "Wide", 1.0: "Narrow", 2.0: "AlsoWide"}

    common._draw_band(ax, x, segments, colors, names, "state")

    texts = {t.get_text() for t in ax.texts}
    assert texts == {"Wide", "AlsoWide"}


def test_draw_band_falls_back_to_default_color_and_name(subplots):
    _, ax = subplots()
    x = np.arange(4, dtype=float)
    segments = [(0, 4, 5.0)]  # 5.0 is not in colors/names
    common._draw_band(ax, x, segments, {}, {}, "state")
    assert ax.patches[0].get_facecolor()[:3] == pytest.approx((217 / 255, 217 / 255, 217 / 255))


# -- _label_style / _draw_label_bands / _shade_by_label --------------------------


def test_label_style_maps_colors_and_names_per_kind():
    segments = common._segments(np.array([0.0, 0.0, 3.0, 103.0, np.nan]))
    colors, names = common._label_style(segments)

    assert colors[0.0] == common.LABEL_COLORS["normal"]
    assert colors[3.0] == common.LABEL_COLORS["active"]
    assert colors[103.0] == common.LABEL_COLORS["transient"]
    assert colors[None] == common.LABEL_COLORS["unknown"]
    assert names[0.0] == "Normal Operation"
    assert names[3.0] == "Severe Slugging"
    assert names[103.0] == "Severe Slugging - Transient"
    assert names[None] == "Unknown"


def test_draw_label_bands_returns_class_segments_and_labels_axes(subplots):
    fig, _ = subplots(nrows=2)
    band_axes = fig.axes
    x = np.arange(6, dtype=float)
    class_values = np.array([0.0, 0.0, 3.0, 3.0, 3.0, 3.0])
    state_values = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    class_segments = common._draw_label_bands(band_axes, x, class_values, state_values)

    assert class_segments == common._segments(class_values)
    assert band_axes[0].get_ylabel() == "state"
    assert band_axes[1].get_ylabel() == "class"


def test_shade_by_label_draws_one_span_per_segment(subplots):
    _, ax = subplots()
    x = np.arange(6, dtype=float)
    values = np.array([0.0, 0.0, 0.0, 9.0, 9.0, 9.0])
    segments = common._segments(values)

    common._shade_by_label(ax, x, segments)

    assert len(ax.patches) == len(segments) == 2
    colors = [patch.get_facecolor()[:3] for patch in ax.patches]
    assert colors[0] == pytest.approx(to_rgb(common.LABEL_COLORS["normal"]))
    assert colors[1] == pytest.approx(to_rgb(common.LABEL_COLORS["active"]))


# -- _tint -------------------------------------------------------------------------


def test_tint_full_strength_keeps_color_untouched():
    assert common._tint("#ff0000", 1.0) == pytest.approx((1.0, 0.0, 0.0))


def test_tint_zero_strength_is_white():
    assert common._tint("#ff0000", 0.0) == pytest.approx((1.0, 1.0, 1.0))


def test_tint_half_strength_mixes_toward_white():
    assert common._tint("#ff0000", 0.5) == pytest.approx((1.0, 0.5, 0.5))


# -- _hold_if_flat ------------------------------------------------------------------


def test_hold_if_flat_pins_axis_when_range_is_negligible(subplots):
    _, ax = subplots()
    was_flat = common._hold_if_flat(ax, 100.0, 100.0)
    assert was_flat is True
    assert ax.get_ylim() == pytest.approx((99.0, 101.0))


def test_hold_if_flat_leaves_axis_untouched_when_signal_varies(subplots):
    _, ax = subplots()
    default_ylim = ax.get_ylim()
    was_flat = common._hold_if_flat(ax, 0.0, 1000.0)
    assert was_flat is False
    assert ax.get_ylim() == default_ylim


def test_hold_if_flat_pads_by_at_least_half(subplots):
    _, ax = subplots()
    common._hold_if_flat(ax, 0.0, 0.0)
    assert ax.get_ylim() == pytest.approx((-0.5, 0.5))


# -- _write_pdf ----------------------------------------------------------------------


def test_write_pdf_creates_parent_dirs_and_file(tmp_path):
    figs = [plt.figure(), plt.figure()]
    numbers = [fig.number for fig in figs]
    out_path = tmp_path / "nested" / "out.pdf"

    common._write_pdf(out_path, iter(figs))

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert all(not plt.fignum_exists(n) for n in numbers)


def test_write_pdf_accepts_empty_iterable_without_raising(tmp_path):
    out_path = tmp_path / "empty.pdf"
    common._write_pdf(out_path, iter([]))  # matplotlib writes no file for zero pages
    assert out_path.parent.exists()


# -- _envelope ------------------------------------------------------------------------


def test_envelope_buckets_min_max_and_first_label(tmp_path):
    index = pd.date_range("2020-01-01", periods=10, freq="1s")
    timeline = pd.DataFrame(
        {
            "sensor": np.arange(10, dtype=float),
            "class": [np.nan, np.nan, 1.0] + [np.nan] * 7,
            "state": [0.0] * 10,
        },
        index=index,
    )

    env = common._envelope(timeline, max_points=3)

    # step = max(1, 10 // 3) = 3 -> buckets of size 3,3,3,1
    assert len(env) == 4
    assert list(env["sensor_min"]) == [0.0, 3.0, 6.0, 9.0]
    assert list(env["sensor_max"]) == [2.0, 5.0, 8.0, 9.0]
    assert env["class"].iloc[0] == 1.0  # first non-missing value of bucket 0
    assert env.index[0] == index[0]
    assert env.index[1] == index[3]


def test_envelope_short_timeline_is_bucket_per_sample():
    index = pd.date_range("2020-01-01", periods=5, freq="1s")
    timeline = pd.DataFrame(
        {"sensor": [1.0, 2.0, 3.0, 4.0, 5.0], "class": [0.0] * 5, "state": [0.0] * 5}, index=index
    )

    env = common._envelope(timeline, max_points=100)

    assert len(env) == len(timeline)
    assert list(env["sensor_min"]) == list(env["sensor_max"]) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_envelope_all_missing_bucket_label_is_nan():
    index = pd.date_range("2020-01-01", periods=4, freq="1s")
    timeline = pd.DataFrame(
        {"sensor": [1.0, 2.0, 3.0, 4.0], "class": [np.nan] * 4, "state": [0.0] * 4}, index=index
    )

    env = common._envelope(timeline, max_points=1)

    assert np.isnan(env["class"].iloc[0])
