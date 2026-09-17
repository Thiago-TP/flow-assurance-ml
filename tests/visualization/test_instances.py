"""Tests for ``flowml.visualization.instances``."""

import numpy as np
import pandas as pd
import pytest

from flowml.visualization import instances
from tests.visualization.helpers import (
    real_name,
    simulated_name,
    write_dataset_ini,
    write_instance,
)


def _make_df(n=20, flat_sensor=True):
    index = pd.date_range("2017-01-01", periods=n, freq="1s", name="timestamp")
    data = {
        "P-PDG": np.linspace(1.0e7, 1.1e7, n),
        "T-TPT": np.full(n, 80.0) if flat_sensor else np.linspace(80.0, 90.0, n),
        "ALL-NAN": np.full(n, np.nan),  # a sensor never recorded
        "class": np.array([0.0] * (n // 2) + [3.0] * (n - n // 2)),
        "state": np.zeros(n),
    }
    return pd.DataFrame(data, index=index)


# -- _plot_instance --------------------------------------------------------------


def test_plot_instance_skips_all_nan_sensors_and_labels_axes():
    df = _make_df()
    fig = instances._plot_instance(df, "Severe Slugging", "WELL-00001_x.parquet", {"P-PDG": "Pa"})

    # 2 label bands + 2 real sensors (ALL-NAN is dropped).
    assert len(fig.axes) == 4
    assert "Severe Slugging" in fig._suptitle.get_text()
    assert "WELL-00001_x.parquet" in fig._suptitle.get_text()

    sensor_axes = fig.axes[2:]
    ylabels = [ax.get_ylabel() for ax in sensor_axes]
    assert ylabels == ["P-PDG [Pa]", "T-TPT"]  # unit shown only when known


def test_plot_instance_flags_flat_sensor_in_annotation():
    df = _make_df(flat_sensor=True)
    fig = instances._plot_instance(df, "Normal", "f.parquet", {})

    t_tpt_ax = fig.axes[3]  # P-PDG then T-TPT
    annotation = t_tpt_ax.texts[-1].get_text()
    assert "(flat)" in annotation


def test_plot_instance_reports_delta_for_varying_sensor():
    df = _make_df(flat_sensor=False)
    fig = instances._plot_instance(df, "Normal", "f.parquet", {"T-TPT": "°C"})

    t_tpt_ax = fig.axes[3]
    annotation = t_tpt_ax.texts[-1].get_text()
    assert "(flat)" not in annotation
    assert "10" in annotation  # delta of 90-80 = 10


# -- plot_fault --------------------------------------------------------------------


def test_plot_fault_raises_when_no_real_instances(tmp_path):
    (tmp_path / "3").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No real"):
        instances.plot_fault("3", raw_dir=tmp_path, out_dir=tmp_path / "out")


def test_plot_fault_only_plots_real_instances_and_names_output(tmp_path):
    raw_dir = tmp_path / "raw"
    write_dataset_ini(raw_dir)
    class_dir = raw_dir / "3"
    write_instance(class_dir / real_name(1, "20170101000000"), _make_df(n=10))
    write_instance(class_dir / real_name(2, "20170102000000"), _make_df(n=10))
    write_instance(class_dir / simulated_name("a"), _make_df(n=10))  # must be skipped

    out_dir = tmp_path / "out"
    instances.plot_fault("3", raw_dir=raw_dir, out_dir=out_dir)

    out_path = out_dir / "fault_3_real_instances.pdf"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_fault_accepts_fault_name(tmp_path):
    raw_dir = tmp_path / "raw"
    write_dataset_ini(raw_dir)
    class_dir = raw_dir / "0"
    write_instance(class_dir / real_name(9, "20170101000000"), _make_df(n=5))

    out_dir = tmp_path / "out"
    instances.plot_fault("Normal", raw_dir=raw_dir, out_dir=out_dir)

    assert (out_dir / "fault_0_real_instances.pdf").exists()


def test_plot_fault_verbose_prints_progress(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    write_dataset_ini(raw_dir)
    class_dir = raw_dir / "1"
    write_instance(class_dir / real_name(4, "20170101000000"), _make_df(n=5))

    instances.plot_fault("1", raw_dir=raw_dir, out_dir=tmp_path / "out", verbose=True)

    captured = capsys.readouterr()
    assert "1/1" in captured.out
    assert real_name(4, "20170101000000") in captured.out
