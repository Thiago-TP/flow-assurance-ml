"""Fault signatures: the variables that identify a fault, drawn together per instance."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import Formatter

from flowml.config import FAULT_CLASSES, RAW_DATA_DIR, SIGNATURE_FIGURES_DIR, SOURCE_TYPES
from flowml.visualization.common import (
    _draw_label_bands,
    _envelope,
    _hold_if_flat,
    _shade_by_label,
    _write_pdf,
    list_instances,
    load_sensor_units,
    resolve_fault,
)

# Variables that carry the visual signature of a fault, i.e. the ones whose
# joint behavior an expert reads to recognize it. Each entry reproduces the
# variable set of the corresponding example figure of the 3W Dataset 2.0.0
# paper (https://doi.org/10.1038/s41597-026-07225-z), which is why only the
# faults illustrated there — figures 3 to 7 — have a signature: the remaining
# ones have no published reference set to copy. Adding a fault here is enough
# for ``plot_all_fault_signatures`` to pick it up.
FAULT_SIGNATURES: dict[int, tuple[str, ...]] = {
    0: ("ABER-CKP", "ESTADO-SDV-P", "ESTADO-W1", "T-TPT"),  # figure 7
    2: ("P-MON-CKP", "P-PDG", "P-TPT", "T-TPT"),  # figure 3
    3: ("P-MON-CKP", "P-PDG", "P-TPT", "T-JUS-CKP"),  # figure 6
    6: ("ABER-CKP", "P-MON-CKP", "P-PDG", "P-TPT"),  # figure 4
    8: ("P-MON-CKP", "P-PDG", "P-TPT", "T-TPT"),  # figure 5
}

# Line color of each signature variable, in the order the variables are listed.
# They are saturated on purpose: a signature page draws them over the pale
# class shading, which the lighter members of a categorical palette sink into.
SIGNATURE_LINE_COLORS = ["#1f4e79", "#c0392b", "#1b7837", "#7b3294", "#b8860b", "#00707f"]

# Layout of a signature page (see ``_plot_signature``). Every variable keeps
# its own scale, so each one gets its own y axis: the axes alternate between
# the left and the right edge and step outward by ``SIGNATURE_AXIS_STEP`` of
# the page width each time a side is reused. The margins are therefore a
# function of how many axes a side holds, which is why they are fractions
# fitted here rather than left to ``tight_layout``.
SIGNATURE_WIDTH = 13.0
SIGNATURE_HEIGHT = 6.2
SIGNATURE_AXIS_MARGIN = 0.075  # room for the first axis of a side
SIGNATURE_AXIS_STEP = 0.07  # extra room per axis stacked outside it


class _CompactFormatter(Formatter):
    """Tick labels scaled by magnitude and precise enough to stay distinct.

    Pressures are stored in Pa and run to tens of millions, which matplotlib
    would otherwise label with a shared exponent printed once at the corner of
    the axis — unreadable on a page carrying four axes at four different
    scales. Each label therefore carries its own magnitude suffix (``13.3M``),
    as the figures of the 3W paper do.

    A fixed number of significant digits will not do, though: a signature is
    often a small excursion on a large base — 78 kPa of TPT pressure on a
    13.3 MPa line — and three digits would print every tick of that axis as
    ``13.3M``. The number of decimals is picked instead from the spacing of
    the ticks actually placed, which matplotlib hands to ``set_locs`` before
    asking for any label, so the labels are as short as they can be while
    still differing from one another.
    """

    _MAGNITUDES = ((1e9, "G"), (1e6, "M"), (1e3, "k"))

    def __init__(self) -> None:
        self._scale, self._suffix, self._decimals = 1.0, "", 2

    def set_locs(self, locs) -> None:
        """Fit the scale and the precision to the tick positions in ``locs``."""
        finite = np.sort(np.asarray([loc for loc in locs if np.isfinite(loc)], dtype=float))
        peak = np.abs(finite).max() if len(finite) else 0.0
        self._scale, self._suffix = next(
            ((scale, suffix) for scale, suffix in self._MAGNITUDES if peak >= scale), (1.0, "")
        )
        steps = np.diff(finite)
        step = steps[steps > 0].min() / self._scale if np.any(steps > 0) else 0.0
        self._decimals = int(np.clip(np.ceil(-np.log10(step)), 0, 4)) if step > 0 else 2

    def __call__(self, value: float, pos=None) -> str:
        return f"{value / self._scale:.{self._decimals}f}{self._suffix}"


def _signature_axis(base, index: int, n_variables: int, offset_pt: float):
    """Give one signature variable its own y axis on the shared time axis.

    Axes alternate sides — even variables left, odd variables right — and each
    reuse of a side pushes the axis one step further outward, so no two scales
    share a spine.

    Parameters
    ----------
    base : matplotlib.axes.Axes
        Axis holding the first variable, the shading and the time axis.
    index : int
        Position of the variable in the signature.
    n_variables : int
        Size of the signature, needed to know whether the right edge of
        ``base`` belongs to a variable or is a plain frame.
    offset_pt : float
        Outward step between two axes on the same side, in points.

    Returns
    -------
    (matplotlib.axes.Axes, str)
        The axis to draw on and the side its spine sits on.
    """
    side = "left" if index % 2 == 0 else "right"
    outward = (index // 2) * offset_pt

    if index == 0:
        if n_variables > 1:  # the right edge belongs to the second variable
            base.spines["right"].set_visible(False)
        return base, side

    ax = base.twinx()
    ax.spines["right" if side == "left" else "left"].set_visible(False)
    ax.yaxis.set_label_position(side)
    ax.yaxis.set_ticks_position(side)
    ax.spines[side].set_position(("outward", outward))
    return ax, side


def _plot_signature(
    env: pd.DataFrame,
    fault_name: str,
    filename: str,
    variables: tuple[str, ...],
    units: dict[str, str],
    subtitle: str,
):
    """Build the one-page figure of a single instance's fault signature.

    Parameters
    ----------
    env : pd.DataFrame
        Envelope of the instance from ``_envelope``, timestamp-indexed, with
        ``<variable>_min`` / ``<variable>_max`` per signature variable plus
        ``class`` and ``state``.
    fault_name : str
        Fault name shown in the title.
    filename : str
        Instance filename shown in the title.
    variables : tuple[str, ...]
        Signature variables, drawn in this order.
    units : dict[str, str]
        Unit per variable name (see ``load_sensor_units``).
    subtitle : str
        Instance summary line shown under the title.

    Returns
    -------
    matplotlib.figure.Figure
        The finished figure, ready to save.
    """
    x = env.index.to_numpy()

    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(SIGNATURE_WIDTH, SIGNATURE_HEIGHT),
        gridspec_kw={"height_ratios": [0.12, 0.12, 1.0]},
    )
    fig.suptitle(f"{fault_name} | Fault signature | {filename}", fontsize=11)
    class_segments = _draw_label_bands(
        axes[:2], x, env["class"].to_numpy(dtype=float), env["state"].to_numpy(dtype=float)
    )

    # The label shades the plotting area, as in ``plot_fault``; the well
    # operational status stays confined to its band, where it informs without
    # competing with four signals for the reader's attention.
    base = axes[2]
    _shade_by_label(base, x, class_segments)

    offset_pt = SIGNATURE_AXIS_STEP * SIGNATURE_WIDTH * 72.0
    handles = []
    for i, variable in enumerate(variables):
        color = SIGNATURE_LINE_COLORS[i % len(SIGNATURE_LINE_COLORS)]
        ax, side = _signature_axis(base, i, len(variables), offset_pt)
        unit = units.get(variable, "")
        label = f"{variable} [{unit}]" if unit else variable

        low = (
            env[f"{variable}_min"].to_numpy(dtype=float)
            if f"{variable}_min" in env.columns
            else np.full(len(x), np.nan)
        )
        high = (
            env[f"{variable}_max"].to_numpy(dtype=float)
            if f"{variable}_max" in env.columns
            else np.full(len(x), np.nan)
        )

        if np.isnan(low).all():  # the well never recorded this variable
            ax.set_yticks([])
            ax.set_ylabel(f"{label} (not recorded)", color="#8a8a8a", fontsize=8)
            ax.spines[side].set_color("#cccccc")
            handles.append(
                Line2D([], [], color="#bbbbbb", lw=1.4, ls=(0, (2, 2)), label=f"{label} — no data")
            )
            continue

        # Long instances are drawn as a min/max envelope (see ``_envelope``),
        # which keeps the fast oscillations of e.g. severe slugging visible
        # instead of letting decimation alias them away; on a short instance
        # the two bounds coincide and this is an ordinary line. The envelope
        # is a single line alternating the two bounds of each bucket — the
        # shape a dense raw signal draws anyway — rather than a filled band,
        # because matplotlib simplifies a line path and not a filled one,
        # which divides the size of the PDF of a populous fault by about five.
        bounds = np.empty(2 * len(x))
        bounds[0::2], bounds[1::2] = low, high
        ax.plot(np.repeat(x, 2), bounds, color=color, lw=0.9, zorder=2 + i)
        ax.set_ylabel(label, color=color, fontsize=8)
        ax.spines[side].set_color(color)
        ax.tick_params(axis="y", colors=color, labelsize=7)
        ax.yaxis.set_major_formatter(_CompactFormatter())

        bottom, top = np.nanmin(low), np.nanmax(high)
        delta = top - bottom
        note = " (flat)" if _hold_if_flat(ax, bottom, top) else ""
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                lw=2.0,
                label=f"{label} — Δ = {delta:.3g} {unit}".rstrip() + note,
            )
        )

    base.set_xlim(x[0], x[-1])
    base.set_xlabel("Time", fontsize=9)
    base.tick_params(axis="x", labelsize=7)
    locator = mdates.AutoDateLocator()
    base.xaxis.set_major_locator(locator)
    base.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    n_left = len(variables) - len(variables) // 2
    n_right = len(variables) // 2
    fig.subplots_adjust(
        left=SIGNATURE_AXIS_MARGIN + SIGNATURE_AXIS_STEP * (n_left - 1),
        right=1.0 - (SIGNATURE_AXIS_MARGIN + SIGNATURE_AXIS_STEP * max(n_right - 1, 0)),
        top=0.885,
        bottom=0.16,
        hspace=0.14,
    )
    fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=7.5, color="#444444")
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=min(len(handles), 4),
        fontsize=7.5,
        frameon=False,
    )
    return fig


def plot_fault_signatures(
    fault: int,
    source: str = "real",
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = SIGNATURE_FIGURES_DIR,
    max_points: int = 5_000,
    verbose: bool = False,
) -> None:
    """Plot the signature of one fault across the instances of one source.

    A signature is the handful of variables whose joint behavior identifies
    the fault — the ones the 3W Dataset 2.0.0 paper puts on its example
    figures, listed in ``FAULT_SIGNATURES``. Where ``plot_fault`` spends one
    subplot per sensor and therefore one page per instance on stacked panels,
    this view draws the whole signature on a single time axis, each variable
    on its own y axis in its own color, so the trends can be read against each
    other: the PDG pressure rising while the choke pressure falls, say.

    Unlike the well-keyed plots of this module, signatures are drawn for any
    instance source, one PDF each: the same event seen in the field, in an
    OLGA simulation and in an expert's drawing is exactly the comparison the
    three sources invite. Each source writes into its own subdirectory of
    ``out_dir``.

    One page per instance, ordered by filename. Each page carries the same two
    bands as ``plot_fault`` on top — the well operational status (``state``)
    and the label (``class``) — and shades the plotting area by label alone,
    in the same colors: light green for normal operation, light yellow for the
    transient period, light red for the active fault, grey for unlabeled
    stretches. The legend reports the unit and the total variation
    (Δ = max - min) of every variable, and a variable the instance does not
    carry keeps its axis, marked as such, so the pages stay comparable. That
    last case is the rule rather than the exception outside real data: the
    simulations left several variables out entirely, so their pages are
    expected to be missing part of the signature (ABER-CKP for fault 6, say).

    Parameters
    ----------
    fault : int
        Fault-class number; must be one of ``FAULT_SIGNATURES`` (a fault name
        is accepted too, see ``resolve_fault``).
    source : str
        Instance source to plot, a key of ``SOURCE_TYPES``: ``"real"``,
        ``"simulated"`` or ``"drawn"``.
    raw_dir : Path
        Root of the 3W dataset.
    out_dir : Path
        Parent of the destination directory; the PDF is written to
        ``<out_dir>/<source>/fault_<number>_<source>_signatures.pdf``.
    max_points : int
        Envelope resolution in buckets per page; lower it for smaller PDFs.
    verbose : bool
        Print per-instance progress (default off).
    """
    fault_class, fault_name = resolve_fault(fault)
    if fault_class not in FAULT_SIGNATURES:
        known = ", ".join(f"{f} ({FAULT_CLASSES[f]})" for f in sorted(FAULT_SIGNATURES))
        raise ValueError(
            f"No documented signature for fault {fault_class} ({fault_name}); expected one of: {known}"
        )
    variables = FAULT_SIGNATURES[fault_class]

    files = list_instances(fault_class, source, raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No {source} ({SOURCE_TYPES[source]}*) instances of fault {fault_class} "
            f"({fault_name}) under {raw_dir / str(fault_class)}"
        )

    units = load_sensor_units(raw_dir)
    out_path = out_dir / source / f"fault_{fault_class}_{source}_signatures.pdf"
    complete = 0

    def pages():
        nonlocal complete
        for i, filepath in enumerate(files, start=1):
            df = pd.read_parquet(filepath)
            columns = [c for c in (*variables, "class", "state") if c in df.columns]
            missing = [v for v in variables if v not in df.columns or df[v].isna().all()]
            complete += not missing
            if verbose:
                print(
                    f"  [{i}/{len(files)}] {filepath.name} "
                    f"({len(variables) - len(missing)}/{len(variables)} variables recorded)"
                )

            env = _envelope(df[columns], max_points)
            first, last = df.index.min(), df.index.max()
            subtitle = (
                f"{len(df):,} samples | {(last - first).total_seconds() / 3600:,.1f} h | "
                f"{first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M}"
            )
            if len(env) < len(df):
                subtitle += f" | min/max envelope over {len(env):,} buckets"
            if missing:
                subtitle += f" | not recorded: {', '.join(missing)}"
            yield _plot_signature(env, fault_name, filepath.name, variables, units, subtitle)

    _write_pdf(out_path, pages())
    print(
        f"  Saved: {out_path} ({len(files)} {source} instances, {complete} with all "
        f"{len(variables)} signature variables recorded)"
    )


def plot_all_fault_signatures(
    sources: Iterable[str] = tuple(SOURCE_TYPES),
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = SIGNATURE_FIGURES_DIR,
    max_points: int = 5_000,
    verbose: bool = False,
) -> None:
    """Plot the signature PDF of every fault that has one, per instance source.

    Covers the faults listed in ``FAULT_SIGNATURES`` in ascending order, once
    per requested source, into one subdirectory per source. A fault with no
    instance of a source is reported as skipped rather than failed — the
    normal case rather than an error, since the 3W dataset simulates no normal
    operation and hand-draws only faults 1 and 7, neither of which has a
    published signature. A fault that genuinely fails is reported and skipped
    too, so one bad folder cannot abort the batch, and both lists are repeated
    at the end.

    Parameters
    ----------
    sources : Iterable[str]
        Instance sources to plot, keys of ``SOURCE_TYPES``; by default all
        three (real, simulated and hand-drawn).
    raw_dir : Path
        Root of the 3W dataset.
    out_dir : Path
        Parent of the per-source destination directories.
    max_points : int
        Envelope resolution in buckets per page; lower it for smaller PDFs.
    verbose : bool
        Print per-instance progress of each fault (default off).
    """
    jobs = [(source, fault_class) for source in sources for fault_class in sorted(FAULT_SIGNATURES)]
    if not jobs:
        raise ValueError(f"No sources to plot (expected any of {list(SOURCE_TYPES)})")

    plotted = 0
    skipped: list[tuple[str, int]] = []
    failures: list[tuple[str, int, str]] = []
    for i, (source, fault_class) in enumerate(jobs, start=1):
        name = FAULT_CLASSES[fault_class]
        print(f"\n[{i}/{len(jobs)}] {name} — {source} signature...")
        try:
            if not list_instances(fault_class, source, raw_dir):
                skipped.append((source, fault_class))
                print(f"  Skipped: the dataset has no {source} instance of {name}")
                continue
            plot_fault_signatures(fault_class, source, raw_dir, out_dir, max_points, verbose)
            plotted += 1
        except Exception as error:  # noqa: BLE001 - one bad fault must not stop the batch
            failures.append((source, fault_class, f"{type(error).__name__}: {error}"))
            print(f"  FAILED: {failures[-1][2]}")

    print(
        f"\n{plotted} of {len(jobs)} fault signature PDFs plotted into {out_dir} "
        f"({len(skipped)} skipped for lack of instances, {len(failures)} failed)"
    )
    for source, fault_class in skipped:
        print(f"  {FAULT_CLASSES[fault_class]} skipped — no {source} instance")
    for source, fault_class, reason in failures:
        print(f"  {FAULT_CLASSES[fault_class]} ({source}) failed — {reason}")
