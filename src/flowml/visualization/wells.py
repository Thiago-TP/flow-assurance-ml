"""Well histories: the real instances of a well joined in time, one page per sensor."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from flowml.config import FAULT_CLASSES, RAW_DATA_DIR, WELL_HISTORY_FIGURES_DIR, WELL_STATES
from flowml.preprocessing import overlapping_mask, parse_well_id
from flowml.visualization.common import (
    FAULT_COLORS,
    LABEL_COLORS,
    STATE_COLORS,
    _envelope,
    _label_kind,
    _label_name,
    _segments,
    _tint,
    _write_pdf,
    list_instances,
    list_well_ids,
    load_sensor_names,
    load_sensor_units,
)


def _class_palette(values: np.ndarray) -> tuple[dict, dict]:
    """Build the shading color and display name of every ``class`` value present.

    Normal operation and unlabeled stretches keep the ``plot_fault`` colors
    (light green, grey). Faults instead get a pale tint of their own
    ``FAULT_COLORS`` entry, so a history carrying several faults shows each
    one in its own hue: the active period at full tint strength and the
    preceding transient at half strength.

    Parameters
    ----------
    values : np.ndarray
        The ``class`` values appearing in the history, NaN included.

    Returns
    -------
    (dict, dict)
        Color per value and display name per value, both keyed by the value
        itself (``None`` for NaN) so they can feed ``_draw_band``.
    """
    colors, names = {}, {}
    for value in values:
        key = None if np.isnan(value) else value
        kind = _label_kind(value)
        if kind in ("normal", "unknown"):
            colors[key] = LABEL_COLORS[kind]
        else:
            fault = int(value) - 100 if kind == "transient" else int(value)
            colors[key] = _tint(FAULT_COLORS[fault], 0.30 if kind == "active" else 0.15)
        names[key] = _label_name(value)
    return colors, names


def load_well_history(
    well_id: int, raw_dir: Path = RAW_DATA_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join every real instance of one well into a single chronological timeline.

    All ``WELL-000{well_id}_*.parquet`` files across the fault-class folders
    belong to the same physical well and are stitched into one series ordered
    by timestamp. The instances overlap heavily — they are sliding windows
    over a continuous recording, so roughly a fifth of the timestamps appear
    in two files — and duplicates are resolved by keeping, per timestamp, the
    record with the fewest missing values. That is safe because overlapping
    instances never disagree on a label where both are labeled: the only
    conflict is a missing value against a real one.

    Parameters
    ----------
    well_id : int
        Well number, as it appears in the filename (``5`` for ``WELL-00005``).
    raw_dir : Path
        Root of the 3W dataset.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        The joined timeline, timestamp-indexed with the ``class`` and
        ``state`` columns preserved, and one row per source instance with its
        ``file``, ``fault_class``, ``start``, ``end`` and ``n_samples``.
    """
    frames, spans = [], []
    for fault_class in FAULT_CLASSES:
        for filepath in list_instances(fault_class, "real", raw_dir):
            if parse_well_id(filepath.name) != str(well_id):
                continue
            df = pd.read_parquet(filepath)
            sensors = [c for c in df.columns if c not in ("class", "state")]
            df[sensors] = df[sensors].astype("float32")  # halve the memory of long wells
            frames.append(df)
            spans.append(
                {
                    "file": filepath.name,
                    "fault_class": fault_class,
                    "start": df.index.min(),
                    "end": df.index.max(),
                    "n_samples": len(df),
                }
            )
    if not frames:
        raise FileNotFoundError(f"No real instances of WELL-{well_id:05d} under {raw_dir}")

    joined = pd.concat(frames)
    del frames

    # Per timestamp keep the most complete record: ordering by timestamp first
    # and by descending completeness second makes the survivor the first row.
    completeness = joined.notna().sum(axis=1).to_numpy()
    joined = joined.iloc[np.lexsort((-completeness, joined.index.to_numpy()))]
    joined = joined[~joined.index.duplicated(keep="first")]

    spans = pd.DataFrame(spans).sort_values("start").reset_index(drop=True)
    return joined, spans


def _compress_gaps(env: pd.DataFrame, factor: float = 20.0, gap_slots: float = 5.0) -> pd.DataFrame:
    """Lay the buckets on a gap-compressed axis, one slot per bucket.

    A well is recorded in bursts: well 1 spans three and a half years but
    holds only weeks of samples, so on a true calendar axis the recordings
    shrink to invisible slivers. Plotting against bucket position instead
    gives every bucket the same width, and each long silence collapses to a
    fixed narrow blank marked by a separator row. Chronological order and the
    real timestamps are preserved — the axis ticks still carry real dates —
    only the empty stretches lose their proportional width.

    Parameters
    ----------
    env : pd.DataFrame
        Envelope from ``_envelope``, timestamp-indexed.
    factor : float
        A silence counts as a gap when it exceeds ``factor`` times the median
        spacing between buckets.
    gap_slots : float
        Width of a collapsed gap, in bucket slots.

    Returns
    -------
    pd.DataFrame
        The envelope with a ``position`` column for the x axis, a
        ``timestamp`` column keeping the real time of each bucket, and a
        boolean ``separator`` column flagging the inserted gap rows (whose
        values are all missing, so lines break across them).
    """
    frame = env.copy()
    frame["timestamp"] = frame.index
    frame["separator"] = False

    if len(frame) < 3:
        frame["position"] = np.arange(len(frame), dtype=float)
        return frame

    seconds = np.diff(env.index.to_numpy()).astype("timedelta64[s]").astype(float)
    positive = seconds[seconds > 0]
    gaps = (
        np.flatnonzero(seconds > factor * np.median(positive))
        if len(positive)
        else np.array([], int)
    )

    if not len(gaps):
        frame["position"] = np.arange(len(frame), dtype=float)
        return frame

    # One separator row per gap, placed between the buckets it separates.
    separators = pd.DataFrame(np.nan, index=range(len(gaps)), columns=frame.columns)
    separators["separator"] = True
    separators["timestamp"] = pd.NaT
    separators["_order"] = gaps + 0.5

    frame["_order"] = np.arange(len(frame), dtype=float)
    combined = pd.concat([frame, separators]).sort_values("_order").reset_index(drop=True)

    # Each real bucket advances one slot; each separator advances gap_slots.
    steps = np.where(combined["separator"].to_numpy(), gap_slots, 1.0)
    combined["position"] = np.concatenate(([0.0], np.cumsum(steps)[:-1]))
    return combined.drop(columns="_order")


def _plot_well_sensor(
    env: pd.DataFrame,
    sensor: str,
    well_id: int,
    unit: str,
    subtitle: str,
    class_style: tuple[dict, dict],
    coverage: tuple[int, int],
):
    """Build the one-page figure of a single sensor across a well's history.

    Parameters
    ----------
    env : pd.DataFrame
        Gap-compressed envelope from ``_compress_gaps``.
    sensor : str
        Sensor whose page this is.
    well_id : int
        Well number shown in the title.
    unit : str
        Physical unit of the sensor (see ``load_sensor_units``).
    subtitle : str
        History summary line shown under the title.
    class_style : (dict, dict)
        Color and name per ``class`` value, from ``_class_palette``.
    coverage : (int, int)
        Samples carrying a reading and total samples of the joined timeline,
        reported on the page. A sensor the well never recorded gets an empty
        panel instead of a signal.

    Returns
    -------
    matplotlib.figure.Figure
        The finished figure, ready to save.
    """
    x = env["position"].to_numpy(dtype=float)
    separator = env["separator"].to_numpy(dtype=bool)
    class_values = env["class"].to_numpy(dtype=float)
    state_values = env["state"].to_numpy(dtype=float)
    class_colors, class_names = class_style
    state_colors = STATE_COLORS | {None: "#ffffff"}  # gaps read as blank, not grey

    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(13, 5.4),
        gridspec_kw={"height_ratios": [0.13, 0.13, 1.0]},
    )
    fig.suptitle(
        f"WELL-{well_id:05d} | Joined history | {sensor}{f' [{unit}]' if unit else ''}",
        fontsize=11,
    )

    # Shade each run of constant label over exactly the slots it occupies, so a
    # collapsed gap stays blank instead of being bridged by its neighbours.
    def shade(ax, values, colors):
        for start, end, value in _segments(values):
            if separator[start]:
                continue
            key = None if np.isnan(value) else value
            ax.axvspan(
                x[start],
                x[end - 1] + 1.0,
                color=colors.get(key, LABEL_COLORS["unknown"]),
                lw=0,
                zorder=0,
            )

    for ax, values, colors, label in (
        (axes[0], state_values, state_colors, "state"),
        (axes[1], class_values, class_colors, "class"),
    ):
        shade(ax, values, colors)
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=7, rotation=90)
        ax.set_ylim(0, 1)

    ax = axes[2]
    shade(ax, class_values, class_colors)

    if f"{sensor}_min" in env.columns:
        low = env[f"{sensor}_min"].to_numpy(dtype=float)
        high = env[f"{sensor}_max"].to_numpy(dtype=float)
    else:  # the well's files do not even carry the column
        low = high = np.full(len(x), np.nan)
    ax.fill_between(x, low, high, color="#1f4e79", lw=0.4, edgecolor="#1f4e79", zorder=2)

    for position in x[separator]:  # mark where a silence was collapsed
        for band in axes:
            band.axvline(position, color="#9a9a9a", lw=0.6, ls=(0, (2, 2)), zorder=1)

    ax.set_ylabel(f"{sensor} [{unit}]" if unit else sensor, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(x[0], x[-1] + 1.0)
    ax.set_xlabel("Time (dashed lines mark collapsed silences)", fontsize=9)

    real = np.flatnonzero(~separator)
    ticks = real[np.linspace(0, len(real) - 1, min(9, len(real))).astype(int)]
    stamps = [pd.Timestamp(env["timestamp"].iloc[i]) for i in ticks]
    # Instances can be hours long, so a date-only tick would repeat itself;
    # add the clock time whenever consecutive ticks fall on the same day.
    fmt = (
        "%Y-%m-%d %H:%M"
        if len(stamps) > 1 and min(np.diff(stamps)) < pd.Timedelta(days=1)
        else "%Y-%m-%d"
    )
    ax.set_xticks(x[ticks])
    ax.set_xticklabels([s.strftime(fmt) for s in stamps], rotation=30, ha="right")

    valid_high = high[~np.isnan(high)]
    valid_low = low[~np.isnan(low)]
    n_valid, n_total = coverage
    share = 100 * n_valid / n_total if n_total else 0.0
    report = f"coverage {share:.1f}% ({n_valid:,} of {n_total:,} samples)"
    if len(valid_high) and len(valid_low):
        report += f"\nΔ = {valid_high.max() - valid_low.min():.3g} {unit}".rstrip()
    else:
        ax.text(
            0.5,
            0.5,
            "No data recorded for this sensor",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#8a8a8a",
            zorder=3,
        )
    ax.text(
        0.995,
        0.96,
        report,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        linespacing=1.5,
        bbox={"facecolor": "white", "edgecolor": "black", "lw": 0.4, "pad": 2},
        zorder=3,
    )

    # One legend per band, each titled with the column it explains, so the
    # operational status colors are named as well as the label colors.
    def legend_handles(values, colors, names, missing_label):
        real_values = values[~separator]
        handles = [
            Patch(color=colors[v], label=names.get(v, str(v)))
            for v in sorted({v for v in real_values if not np.isnan(v)})
        ]
        if np.isnan(real_values).any():
            handles.append(
                Patch(color=LABEL_COLORS["unknown"], ec="#bbbbbb", lw=0.4, label=missing_label)
            )
        return handles

    class_handles = legend_handles(class_values, class_colors, class_names, "Unlabeled")
    state_handles = legend_handles(state_values, state_colors, dict(WELL_STATES), "Unknown status")

    fig.legend(
        handles=class_handles,
        title="class — label",
        loc="lower left",
        bbox_to_anchor=(0.055, 0.0),
        ncol=min(4, max(1, len(class_handles))),
        fontsize=7,
        title_fontsize=7,
        alignment="left",
        frameon=False,
    )
    fig.legend(
        handles=state_handles,
        title="state — well operational status",
        loc="lower right",
        bbox_to_anchor=(0.99, 0.0),
        ncol=min(3, max(1, len(state_handles))),
        fontsize=7,
        title_fontsize=7,
        alignment="left",
        frameon=False,
    )

    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=7.5, color="#444444")
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0.115, 1, 0.93))
    return fig


def plot_well_history(
    well_id: int,
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = WELL_HISTORY_FIGURES_DIR,
    max_points: int = 20_000,
    verbose: bool = False,
) -> None:
    """Plot the whole recorded history of one well into a multi-page PDF.

    Every real instance of the well, across all fault-class folders, is joined
    into a single chronological timeline (see ``load_well_history``) and drawn
    one time series per page, so the pages show how the recordings tile the
    well's lifetime and where the faults sit. Long silences between recordings
    are collapsed to a narrow marked blank (see ``_compress_gaps``), because a
    well's samples cover only a small fraction of its calendar span; ticks
    still carry real dates.

    Coloring follows the ``plot_fault`` scheme, extended because a whole
    history usually carries several different faults: each page shows a band
    of the well operational status (``state``), a band of the label
    (``class``), and the sensor itself over a background shaded by label.
    Normal operation stays light green and unlabeled stretches grey, while
    every fault gets a pale tint of its own color — stronger for the active
    period, fainter for the preceding transient. Both bands get their own
    named legend. Sensors are drawn as a min/max envelope over
    position-bucketed samples, which keeps spikes visible instead of letting
    subsampling drop them.

    Every well gets a page for every sensor the dataset declares, in dataset
    order, including the ones this well never recorded: those pages are drawn
    empty. Each page reports how much of the timeline the sensor actually
    covers, so the missing data is visible rather than silently absent, and
    the PDFs are directly comparable page by page across wells.

    Parameters
    ----------
    well_id : int
        Well number, as it appears in the filename (``5`` for ``WELL-00005``).
    raw_dir : Path
        Root of the 3W dataset.
    out_dir : Path
        Destination directory of the PDF (named ``well_<id>_history.pdf``).
    max_points : int
        Envelope resolution in buckets per page; lower it for smaller PDFs.
    verbose : bool
        Print per-sensor progress and the instance overlap report (default off).
    """
    timeline, spans = load_well_history(well_id, raw_dir)

    overlapping = int(overlapping_mask(spans["start"].to_numpy(), spans["end"].to_numpy()).sum())
    duplicated = int(spans["n_samples"].sum() - len(timeline))
    faults = sorted(spans["fault_class"].unique())
    subtitle = (
        f"{len(spans)} instances ({overlapping} overlapping) | "
        f"{len(timeline):,} unique timestamps ({duplicated:,} deduplicated) | "
        f"{spans['start'].min():%Y-%m-%d} to {spans['end'].max():%Y-%m-%d} | "
        f"fault folders: {', '.join(FAULT_CLASSES[f] for f in faults)}"
    )
    if verbose:
        print(f"  WELL-{well_id:05d}: {subtitle}")

    units = load_sensor_units(raw_dir)
    sensors = load_sensor_names(raw_dir)

    # Coverage must be counted before decimation: a bucket's min/max is present
    # as soon as one sample in it is, which would overstate a sparse sensor.
    n_total = len(timeline)
    coverage = {
        sensor: (int(timeline[sensor].notna().sum()) if sensor in timeline.columns else 0, n_total)
        for sensor in sensors
    }

    env = _compress_gaps(_envelope(timeline, max_points))
    del timeline

    class_style = _class_palette(np.unique(env["class"].to_numpy(dtype=float)))
    n_empty = sum(1 for sensor in sensors if coverage[sensor][0] == 0)

    out_path = out_dir / f"well_{well_id}_history.pdf"

    def pages():
        for i, sensor in enumerate(sensors, start=1):
            if verbose:
                share = 100 * coverage[sensor][0] / n_total if n_total else 0.0
                print(f"  [{i}/{len(sensors)}] {sensor} ({share:.1f}% coverage)")
            yield _plot_well_sensor(
                env,
                sensor,
                well_id,
                units.get(sensor, ""),
                subtitle,
                class_style,
                coverage[sensor],
            )

    _write_pdf(out_path, pages())
    print(
        f"  Saved: {out_path} ({len(sensors)} time series, {n_empty} of them empty, "
        f"{len(spans)} instances joined)"
    )


def plot_wells_histories(
    well_ids: Iterable[int] | None = None,
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = WELL_HISTORY_FIGURES_DIR,
    max_points: int = 20_000,
    verbose: bool = False,
) -> None:
    """Plot the joined history of several wells, one multi-page PDF each.

    Wells are processed in ascending order. A well that fails — unreadable
    files, say — is reported and skipped rather than aborting the batch,
    since a full run covers every well in the dataset and takes a while; the
    failures are listed again at the end.

    Parameters
    ----------
    well_ids : Iterable[int] | None
        Well numbers to plot; ``None`` plots every real well in the dataset
        (see ``list_well_ids``).
    raw_dir : Path
        Root of the 3W dataset.
    out_dir : Path
        Destination directory of the PDFs.
    max_points : int
        Envelope resolution in buckets per page; lower it for smaller PDFs.
    verbose : bool
        Print per-sensor progress of each well (default off).
    """
    wells = sorted(set(list_well_ids(raw_dir) if well_ids is None else well_ids))
    if not wells:
        raise ValueError(f"No wells to plot under {raw_dir}")

    failures: list[tuple[int, str]] = []
    for i, well_id in enumerate(wells, start=1):
        print(f"\n[{i}/{len(wells)}] WELL-{well_id:05d} history...")
        try:
            plot_well_history(well_id, raw_dir, out_dir, max_points, verbose)
        except Exception as error:  # noqa: BLE001 - one bad well must not stop the batch
            failures.append((well_id, f"{type(error).__name__}: {error}"))
            print(f"  FAILED: {failures[-1][1]}")

    print(f"\n{len(wells) - len(failures)} of {len(wells)} wells plotted into {out_dir}")
    for well_id, reason in failures:
        print(f"  WELL-{well_id:05d} failed — {reason}")
