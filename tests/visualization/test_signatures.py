"""Tests for ``flowml.visualization.signatures``."""

import numpy as np
import pandas as pd
import pytest

from flowml.visualization import signatures
from tests.visualization.helpers import (
    drawn_name,
    real_name,
    simulated_name,
    write_dataset_ini,
    write_instance,
)


def _make_signature_df(n=20, variables=("P-MON-CKP", "P-PDG", "P-TPT", "T-TPT"), missing=()):
    index = pd.date_range("2017-01-01", periods=n, freq="1s", name="timestamp")
    data = {}
    for i, var in enumerate(variables):
        if var in missing:
            continue
        data[var] = np.linspace(1.0e7 + i * 1e5, 1.1e7 + i * 1e5, n)
    data["class"] = np.zeros(n)
    data["state"] = np.zeros(n)
    return pd.DataFrame(data, index=index)


# -- _CompactFormatter -------------------------------------------------------------


def test_compact_formatter_scales_by_magnitude():
    formatter = signatures._CompactFormatter()
    formatter.set_locs([1.0e6, 2.0e6, 3.0e6])
    assert formatter(2.0e6) == "2M"


def test_compact_formatter_uses_finer_precision_for_close_ticks():
    formatter = signatures._CompactFormatter()
    formatter.set_locs([1.30e7, 1.31e7, 1.32e7])  # 0.01M apart
    label = formatter(1.305e7)
    assert label.endswith("M")
    assert "." in label  # needs decimals to stay distinct at this spacing


def test_compact_formatter_no_suffix_below_thousand():
    formatter = signatures._CompactFormatter()
    formatter.set_locs([1.0, 2.0, 3.0])
    assert formatter(2.0) == "2"  # unit spacing needs no decimals to stay distinct


def test_compact_formatter_handles_single_or_empty_locs():
    formatter = signatures._CompactFormatter()
    formatter.set_locs([5.0])  # no spacing to infer precision from: falls back to 2 decimals
    assert formatter(5.0) == "5.00"
    formatter.set_locs([])
    assert formatter(0.0) == "0.00"


# -- _signature_axis ----------------------------------------------------------------


def test_signature_axis_first_variable_reuses_base(subplots):
    _, base = subplots()
    ax, side = signatures._signature_axis(base, index=0, n_variables=3, offset_pt=10.0)
    assert ax is base
    assert side == "left"
    assert base.spines["right"].get_visible() is False


def test_signature_axis_first_variable_keeps_right_spine_when_alone(subplots):
    _, base = subplots()
    ax, _ = signatures._signature_axis(base, index=0, n_variables=1, offset_pt=10.0)
    assert ax is base
    assert base.spines["right"].get_visible() is True


def test_signature_axis_alternates_sides(subplots):
    _, base = subplots()
    _, side0 = signatures._signature_axis(base, 0, 4, 10.0)
    _, side1 = signatures._signature_axis(base, 1, 4, 10.0)
    _, side2 = signatures._signature_axis(base, 2, 4, 10.0)
    _, side3 = signatures._signature_axis(base, 3, 4, 10.0)
    assert [side0, side1, side2, side3] == ["left", "right", "left", "right"]


def test_signature_axis_steps_outward_on_reuse(subplots):
    _, base = subplots()
    ax_first, _ = signatures._signature_axis(base, 0, 4, 10.0)  # left, outward 0
    ax_second, _ = signatures._signature_axis(base, 2, 4, 10.0)  # left again, outward 10
    assert ax_first is base
    assert ax_second is not base
    assert ax_second.spines["left"].get_position() == ("outward", 10.0)


# -- _plot_signature ------------------------------------------------------------------


def test_plot_signature_draws_one_handle_per_variable():
    from flowml.visualization.common import _envelope

    df = _make_signature_df()
    env = _envelope(df, max_points=100)
    fig = signatures._plot_signature(
        env, "Spurious DHSV Closure", "f.parquet", ("P-MON-CKP", "P-PDG", "P-TPT", "T-TPT"), {}, ""
    )
    legend = fig.legends[0]
    assert len(legend.legend_handles) == 4


def test_plot_signature_marks_missing_variable_as_not_recorded():
    from flowml.visualization.common import _envelope

    variables = ("P-MON-CKP", "P-PDG", "P-TPT", "T-TPT")
    df = _make_signature_df(variables=variables, missing=("T-TPT",))
    env = _envelope(df, max_points=100)
    fig = signatures._plot_signature(env, "Fault", "f.parquet", variables, {}, "")

    labels = [h.get_label() for h in fig.legends[0].legend_handles]
    assert any("no data" in label for label in labels)


def test_plot_signature_flags_flat_variable():
    variables = ("P-PDG",)
    n = 10
    index = pd.date_range("2017-01-01", periods=n, freq="1s")
    df = pd.DataFrame(
        {"P-PDG": np.full(n, 5.0e6), "class": np.zeros(n), "state": np.zeros(n)}, index=index
    )
    from flowml.visualization.common import _envelope

    env = _envelope(df, max_points=100)
    fig = signatures._plot_signature(env, "Fault", "f.parquet", variables, {"P-PDG": "Pa"}, "")

    label = fig.legends[0].legend_handles[0].get_label()
    assert "(flat)" in label


# -- plot_fault_signatures ------------------------------------------------------------


def test_plot_fault_signatures_rejects_undocumented_fault(tmp_path):
    (tmp_path / "1").mkdir(parents=True)
    with pytest.raises(ValueError, match="No documented signature"):
        signatures.plot_fault_signatures(1, raw_dir=tmp_path)


def test_plot_fault_signatures_raises_when_no_instances_of_source(tmp_path):
    (tmp_path / "2").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        signatures.plot_fault_signatures(2, source="simulated", raw_dir=tmp_path)


def test_plot_fault_signatures_writes_pdf_under_source_subdirectory(tmp_path):
    raw_dir = tmp_path / "raw"
    write_dataset_ini(raw_dir)
    class_dir = raw_dir / "2"
    variables = signatures.FAULT_SIGNATURES[2]
    write_instance(
        class_dir / real_name(1, "20170101000000"), _make_signature_df(variables=variables)
    )

    out_dir = tmp_path / "out"
    signatures.plot_fault_signatures(2, source="real", raw_dir=raw_dir, out_dir=out_dir)

    out_path = out_dir / "real" / "fault_2_real_signatures.pdf"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_fault_signatures_accepts_simulated_and_drawn_sources(tmp_path):
    raw_dir = tmp_path / "raw"
    write_dataset_ini(raw_dir)
    class_dir = raw_dir / "0"
    variables = signatures.FAULT_SIGNATURES[0]
    write_instance(class_dir / simulated_name("a"), _make_signature_df(variables=variables))
    write_instance(class_dir / drawn_name("b"), _make_signature_df(variables=variables))

    out_dir = tmp_path / "out"
    signatures.plot_fault_signatures(0, source="simulated", raw_dir=raw_dir, out_dir=out_dir)
    signatures.plot_fault_signatures(0, source="drawn", raw_dir=raw_dir, out_dir=out_dir)

    assert (out_dir / "simulated" / "fault_0_simulated_signatures.pdf").exists()
    assert (out_dir / "drawn" / "fault_0_drawn_signatures.pdf").exists()


def test_plot_fault_signatures_verbose_reports_missing_variables(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    write_dataset_ini(raw_dir)
    class_dir = raw_dir / "2"
    variables = signatures.FAULT_SIGNATURES[2]
    write_instance(
        class_dir / real_name(1, "20170101000000"),
        _make_signature_df(variables=variables, missing=(variables[0],)),
    )

    signatures.plot_fault_signatures(2, raw_dir=raw_dir, out_dir=tmp_path / "out", verbose=True)

    captured = capsys.readouterr()
    assert "3/4 variables recorded" in captured.out


# -- plot_all_fault_signatures ---------------------------------------------------------


def test_plot_all_fault_signatures_rejects_empty_sources(tmp_path):
    with pytest.raises(ValueError, match="No sources to plot"):
        signatures.plot_all_fault_signatures(sources=(), raw_dir=tmp_path)


def test_plot_all_fault_signatures_skips_missing_and_plots_available(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    write_dataset_ini(raw_dir)
    # Only fault 2 gets a real instance; the other documented faults (0, 3, 6, 8)
    # have no folder at all, which must be reported as a failure/skip rather
    # than aborting the batch.
    class_dir = raw_dir / "2"
    variables = signatures.FAULT_SIGNATURES[2]
    write_instance(
        class_dir / real_name(1, "20170101000000"), _make_signature_df(variables=variables)
    )

    out_dir = tmp_path / "out"
    signatures.plot_all_fault_signatures(sources=("real",), raw_dir=raw_dir, out_dir=out_dir)

    assert (out_dir / "real" / "fault_2_real_signatures.pdf").exists()
    captured = capsys.readouterr()
    assert "1 of 5 fault signature PDFs plotted" in captured.out
