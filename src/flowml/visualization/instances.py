"""Instance histories: every sensor of every real instance of a fault, one page each."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flowml.config import INSTANCE_FIGURES_DIR, RAW_DATA_DIR
from flowml.visualization.common import (
    _draw_label_bands,
    _hold_if_flat,
    _shade_by_label,
    _write_pdf,
    list_instances,
    load_sensor_units,
    resolve_fault,
)


def _plot_instance(df: pd.DataFrame, fault_name: str, filename: str, units: dict[str, str]):
    """Build the one-page figure of a single raw instance.

    Parameters
    ----------
    df : pd.DataFrame
        One raw instance, timestamp-indexed, with ``class`` and ``state``.
    fault_name : str
        Fault name shown in the title.
    filename : str
        Instance filename shown in the title.
    units : dict[str, str]
        Unit per variable name (see ``load_sensor_units``).

    Returns
    -------
    matplotlib.figure.Figure
        The finished figure, ready to save.
    """
    sensors = [c for c in df.columns if c not in ("class", "state") and df[c].notna().any()]
    x = df.index.to_numpy()
    labels = {
        column: (
            df[column].to_numpy(dtype=float) if column in df.columns else np.full(len(df), np.nan)
        )
        for column in ("class", "state")
    }

    n = len(sensors)
    fig, axes = plt.subplots(
        n + 2,
        1,
        sharex=True,
        figsize=(11, 1.0 + 1.35 * n),
        gridspec_kw={"height_ratios": [0.18, 0.18] + [1.0] * n},
    )
    fig.suptitle(f"{fault_name} | Instance history | {filename}", fontsize=10)
    class_segments = _draw_label_bands(axes[:2], x, labels["class"], labels["state"])

    for ax, sensor in zip(axes[2:], sensors):
        _shade_by_label(ax, x, class_segments)
        col = df[sensor].to_numpy(dtype=float)
        ax.plot(x, col, lw=0.8, zorder=2)

        unit = units.get(sensor, "")
        ax.set_ylabel(f"{sensor} [{unit}]" if unit else sensor, fontsize=7)
        ax.tick_params(labelsize=7)

        valid = col[~np.isnan(col)]
        delta, note = float("nan"), ""
        if len(valid):
            delta = valid.max() - valid.min()
            note = " (flat)" if _hold_if_flat(ax, valid.min(), valid.max()) else ""
        ax.text(
            0.995,
            0.95,
            f"Δ = {delta:.3g} {unit}".rstrip() + note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "black", "lw": 0.4, "pad": 2},
            zorder=3,
        )

    axes[-1].set_xlabel("Time", fontsize=9)
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return fig


def plot_fault(
    fault: str,
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = INSTANCE_FIGURES_DIR,
    verbose: bool = False,
) -> None:
    """Plot every real instance of one fault into a multi-page PDF.

    One page per instance: two bands on top showing the well operational
    status (``state``) and the label (``class``) over time, then one subplot
    per sensor that has any data. Sensor backgrounds are shaded by label —
    light green for normal operation, light yellow for the transient period,
    light red for the active fault, grey for unlabeled stretches — and each
    panel reports the total variation of the signal (Δ = max - min). A sensor
    that never moves — frozen, or drifting by a few float digits — is held on
    a padded axis and marked flat instead of being autoscaled into noise.
    Simulated and hand-drawn instances are skipped.

    Parameters
    ----------
    fault : str
        Fault-class folder number (``"9"``) or fault name, case-insensitive
        (``"Hydrate in Service Line"``).
    raw_dir : Path
        Root of the 3W dataset.
    out_dir : Path
        Destination directory of the PDF (named
        ``fault_<number>_real_instances.pdf``).
    verbose : bool
        Print per-instance progress (default off).
    """
    fault_class, fault_name = resolve_fault(fault)
    files = list_instances(fault_class, "real", raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No real (WELL-*) instances of fault {fault_class} under {raw_dir}"
        )

    units = load_sensor_units(raw_dir)
    out_path = out_dir / f"fault_{fault_class}_real_instances.pdf"

    def pages():
        for i, filepath in enumerate(files, start=1):
            if verbose:
                print(f"  [{i}/{len(files)}] {filepath.name}")
            yield _plot_instance(pd.read_parquet(filepath), fault_name, filepath.name, units)

    _write_pdf(out_path, pages())
    print(f"  Saved: {out_path} ({len(files)} instances)")
