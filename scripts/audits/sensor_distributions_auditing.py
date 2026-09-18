"""Audit — distribution of pressure and temperature readings per well and per instance source.

Draws, for every pressure and temperature sensor of the 3W dataset, the
distribution of its raw readings in each real well, side by side with the same
distribution over the simulated and the hand-drawn instances. Every row also
carries its tallies: how many readings, what share is exactly zero (a sensor
frozen at zero — the one defect the default masking deliberately leaves in),
what share the default masking removes (outside the plausible band of
``preprocessing.mask_extreme_values``: negative pressures, magnitudes beyond
``EXTREME_VALUE_LIMIT``, temperatures outside ``TEMPERATURE_LIMITS``) and what
share is missing.

Why: the distilled trees of stage 5 split on readings such as ``P-TPT_max <=
63 Pa`` — a wellhead pressure of vacuum — which points at frozen sensors being
used as fault predictors, and the simulated instances that prop up the rare
classes need not share the real wells' operating levels at all. This audit
shows where every well's and every source's readings actually sit. It changes
nothing about the masking rules; that decision comes after looking.

The whole dataset is read (only the audited columns of each file), instance by
instance, into fixed-grid histograms inside each band, so memory stays flat and
the quartiles and percentiles are read off the histograms (0.1 bar and 0.03 °C
resolution). Overlapping instances are all kept: the question is what a well's
sensors read, and duplicated samples do not move a distribution.

Usage
-----
    uv run scripts/audits/sensor_distributions_auditing.py [--raw-dir PATH]
        [--max-instances N] [--sensors P-TPT,T-TPT,...] [--output-dir DIR] [--verbose]

Outputs
-------
    results/audits/sensor_distributions.pdf                 one page per sensor
    results/audits/sensor_distributions_<timestamp>.txt     the tallies behind the pages
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from flowml.config import (
    EXTREME_VALUE_LIMIT,
    FAULT_CLASSES,
    PRESSURE_MIN,
    PRESSURE_SENSORS,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TEMPERATURE_LIMITS,
    TEMPERATURE_SENSORS,
)
from flowml.preprocessing import list_raw_instances, parse_source_type, parse_well_id
from flowml.visualization.common import load_sensor_units

AUDITS_DIR = RESULTS_DIR / "audits"

# The plausible band per physical quantity: exactly the readings the default
# masking keeps. Below or above it is what ``mask_extreme_values`` turns into
# NaN — negative pressures and magnitudes beyond the limit, temperatures
# outside the band — so "outside the band" here is "masked" in the pipeline.
BANDS: dict[str, tuple[float, float]] = {
    **{sensor: (PRESSURE_MIN, EXTREME_VALUE_LIMIT) for sensor in PRESSURE_SENSORS},
    **{sensor: TEMPERATURE_LIMITS for sensor in TEMPERATURE_SENSORS},
}
N_BINS = 10_000  # 0.1 bar per bin for pressures, 0.03 °C for temperatures
PRESSURE_SCALE = 1e5  # Pa per bar, for the axis only — 3W stores Pa

# The three instance sources take the first three categorical slots of the
# project's chart palette (validated colorblind-safe as a set, all pairs) and
# every row is also named, so color groups the rows and never identifies one.
SOURCE_COLORS = {"WELL": "#2a78d6", "SIMULATED": "#eb6834", "DRAWN": "#1baf7a"}
SOURCE_NAMES = {
    "WELL": "real wells",
    "SIMULATED": "simulated instances",
    "DRAWN": "hand-drawn instances",
}
INK, INK_SECONDARY, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
QUANTILES = (0.01, 0.25, 0.50, 0.75, 0.99)  # whiskers, box, median, box, whiskers


@dataclass
class Tally:
    """Running counts of one sensor in one row (a well or a synthetic source)."""

    counts: np.ndarray = field(default_factory=lambda: np.zeros(N_BINS, dtype=np.int64))
    n_total: int = 0
    n_missing: int = 0
    n_zero: int = 0
    n_below: int = 0
    n_above: int = 0
    n_instances: int = 0

    def add(self, values: np.ndarray, band: tuple[float, float]) -> None:
        """Fold one instance's readings in."""
        missing = np.isnan(values)
        present = values[~missing]
        low, high = band
        below, above = present < low, present > high
        self.n_total += len(values)
        self.n_missing += int(missing.sum())
        self.n_zero += int((present == 0).sum())
        self.n_below += int(below.sum())
        self.n_above += int(above.sum())
        self.counts += np.histogram(present[~(below | above)], bins=N_BINS, range=band)[0]
        self.n_instances += 1

    def quantiles(self, band: tuple[float, float], qs=QUANTILES) -> list[float] | None:
        """Read quantiles off the in-band histogram; ``None`` when it is empty."""
        total = self.counts.sum()
        if total == 0:
            return None
        edges = np.linspace(band[0], band[1], N_BINS + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        cumulative = np.cumsum(self.counts) / total
        return [
            float(centers[min(int(np.searchsorted(cumulative, q, side="left")), N_BINS - 1)])
            for q in qs
        ]


def row_key(filename: str) -> tuple[str, str]:
    """``("WELL", "26")`` for a real instance, ``("SIMULATED", "")`` / ``("DRAWN", "")`` otherwise."""
    source = parse_source_type(filename)
    return (source, parse_well_id(filename) if source == "WELL" else "")


def tally_dataset(
    raw_dir: Path, sensors: list[str], max_instances: int | None, verbose: bool
) -> dict[tuple[str, str], dict[str, Tally]]:
    """Read every instance once, audited columns only, and tally per row and sensor."""
    entries = list_raw_instances(raw_dir, list(FAULT_CLASSES), max_instances)
    tallies: dict[tuple[str, str], dict[str, Tally]] = {}
    current_class = None
    for fault_class, path in entries:
        if fault_class != current_class:
            current_class = fault_class
            n_class = sum(1 for c, _ in entries if c == fault_class)
            print(f"  Class {fault_class} {FAULT_CLASSES[fault_class]}: {n_class} instances")
        present = [s for s in sensors if s in pq.read_schema(path).names]
        frame = pd.read_parquet(path, columns=present) if present else None
        key = row_key(path.name)
        per_sensor = tallies.setdefault(key, {})
        for sensor in sensors:
            tally = per_sensor.setdefault(sensor, Tally())
            values = (
                frame[sensor].to_numpy(dtype=float)
                if frame is not None and sensor in frame
                else np.full(len(frame) if frame is not None else 0, np.nan)
            )
            tally.add(values, BANDS[sensor])
        if verbose:
            print(f"    {path.name}: {len(frame) if frame is not None else 0:,} samples")
    return tallies


def ordered_rows(tallies: dict[tuple[str, str], dict[str, Tally]]) -> list[tuple[str, str]]:
    """Real wells in numeric order, then the synthetic sources."""
    wells = sorted((k for k in tallies if k[0] == "WELL"), key=lambda k: int(k[1]))
    synthetic = [k for k in (("SIMULATED", ""), ("DRAWN", "")) if k in tallies]
    return wells + synthetic


def row_label(key: tuple[str, str]) -> str:
    return f"well {key[1]}" if key[0] == "WELL" else key[0].lower()


def percent(part: int, whole: int) -> float:
    """``part`` as a percentage of ``whole``; 0 when there is nothing to count."""
    return 100 * part / whole if whole else 0.0


def tint(color: str, strength: float = 0.35) -> tuple[float, float, float]:
    """Mix a color toward white; ``strength`` 1.0 keeps it untouched."""
    base = np.array(mcolors.to_rgb(color))
    return tuple(1.0 - (1.0 - base) * strength)


def draw_sensor_page(
    sensor: str,
    rows: list[tuple[str, str]],
    tallies: dict[tuple[str, str], dict[str, Tally]],
    unit: str,
) -> plt.Figure:
    """One page: a horizontal box per row, its tallies in the right margin."""
    band = BANDS[sensor]
    is_pressure = sensor in PRESSURE_SENSORS
    scale = PRESSURE_SCALE if is_pressure else 1.0
    axis_unit = "bar" if is_pressure else unit or "°C"

    n_rows = len(rows)
    fig_h = 0.28 * n_rows + 2.6
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    fig.subplots_adjust(left=0.11, right=0.60, top=1 - 1.45 / fig_h, bottom=1.0 / fig_h)

    drawn_sources: dict[str, Patch] = {}
    span_low, span_high = [], []
    for position, key in enumerate(rows):
        y = n_rows - position  # first row on top
        color = SOURCE_COLORS[key[0]]
        tally = tallies[key][sensor]
        qs = tally.quantiles(band)
        if qs is None:
            ax.text(
                0.005,
                y,
                "not recorded" if tally.n_total == tally.n_missing else "no in-band reading",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=6.5,
                color=INK_MUTED,
            )
        else:
            p1, q1, med, q3, p99 = (q / scale for q in qs)
            span_low.append(p1)
            span_high.append(p99)
            ax.bxp(
                [{"whislo": p1, "q1": q1, "med": med, "q3": q3, "whishi": p99}],
                positions=[y],
                orientation="horizontal",
                widths=0.62,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": tint(color), "edgecolor": color, "linewidth": 1.0},
                medianprops={"color": INK, "linewidth": 1.3},
                whiskerprops={"color": color, "linewidth": 1.0},
                capprops={"color": color, "linewidth": 1.0},
            )
            drawn_sources.setdefault(
                key[0],
                Patch(
                    facecolor=tint(color),
                    edgecolor=color,
                    linewidth=1.0,
                    label=SOURCE_NAMES[key[0]],
                ),
            )
        n = tally.n_total
        ax.text(
            1.01,
            y,
            f"{tally.n_instances:>4} inst · {n / 1e6:6.2f}M · "
            f"zero {percent(tally.n_zero, n):5.1f}% · "
            f"masked {percent(tally.n_below + tally.n_above, n):5.1f}% · "
            f"missing {percent(tally.n_missing, n):5.1f}%",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=6.5,
            color=INK_SECONDARY,
            family="monospace",
        )

    ax.set_yticks(range(n_rows, 0, -1), [row_label(k) for k in rows], fontsize=7, color=INK)
    ax.set_ylim(0.3, n_rows + 0.7)
    if span_low:
        low, high = min(span_low), max(span_high)
        pad = 0.03 * (high - low or 1.0)
        ax.set_xlim(min(low, 0.0) - pad if is_pressure else low - pad, high + pad)
    if is_pressure:
        ax.axvline(0, color=INK_MUTED, linewidth=0.8, zorder=0)
    if len(rows) > 1 and rows[-1][0] != "WELL":
        first_synthetic = next(i for i, k in enumerate(rows) if k[0] != "WELL")
        ax.axhline(
            n_rows - first_synthetic + 0.5, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3))
        )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(axis="x", labelsize=7, colors=INK_SECONDARY)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(
        f"{sensor} [{axis_unit}]" + ("  (3W stores Pa)" if is_pressure else ""),
        fontsize=8,
        color=INK_SECONDARY,
    )
    low_text = f"{band[0] / scale:g}" if is_pressure else f"{band[0]:g}"
    high_text = f"{band[1] / scale:g}" if is_pressure else f"{band[1]:g}"
    fig.suptitle(
        f"{sensor} — raw readings per real well vs simulated and hand-drawn instances",
        x=0.02,
        ha="left",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.02,
        1 - 0.55 / fig_h,
        "box: quartiles · line: median · whiskers: 1st and 99th percentile of the readings inside the "
        f"plausible band [{low_text}, {high_text}] {axis_unit}\n"
        "right margin: instances, readings, share exactly 0 (frozen sensor — kept by the default masking), "
        "share outside the band (what the default masking removes), share missing",
        fontsize=7,
        color=INK_SECONDARY,
        va="top",
    )
    if drawn_sources:
        # Above the axes, so no box of a wide-ranging row can run under it.
        ax.legend(
            handles=[drawn_sources[s] for s in SOURCE_COLORS if s in drawn_sources],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=3,
            fontsize=7,
            frameon=False,
        )
    return fig


def write_summary(
    out: Path, sensors: list[str], rows: list[tuple[str, str]], tallies, units: dict[str, str]
) -> None:
    """The tallies and quantiles of every page as one text table per sensor."""
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"Sensor distributions audit — {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}\n")
        fh.write(
            "Per sensor and row: instances, readings, share missing, share exactly zero, share "
            "below / above the plausible band (masked by default), and the 1/25/50/75/99th "
            "percentiles of the in-band readings in the sensor's stored unit.\n"
        )
        for sensor in sensors:
            band = BANDS[sensor]
            records = []
            for key in rows:
                t = tallies[key][sensor]
                n = t.n_total or 1
                qs = t.quantiles(band)
                records.append(
                    {
                        "row": row_label(key),
                        "instances": t.n_instances,
                        "readings": t.n_total,
                        "missing_%": round(100 * t.n_missing / n, 2),
                        "zero_%": round(100 * t.n_zero / n, 2),
                        "below_band_%": round(100 * t.n_below / n, 3),
                        "above_band_%": round(100 * t.n_above / n, 3),
                        **{
                            f"p{int(q * 100)}": (None if qs is None else round(v, 2))
                            for q, v in zip(QUANTILES, qs or [None] * 5)
                        },
                    }
                )
            fh.write(f"\n{'=' * 100}\n{sensor} [{units.get(sensor, '')}] — band {band}\n")
            fh.write(pd.DataFrame(records).to_string(index=False))
            fh.write("\n")


def main() -> None:
    """Parse arguments, tally the dataset, and write the PDF and the summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="root of the 3W dataset (default: FLOWML_RAW_DATA_DIR or config)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="cap instances per class, for a quick look (default: all)",
    )
    parser.add_argument(
        "--sensors",
        default=",".join(BANDS),
        help="comma-separated sensors to audit (default: every pressure and temperature sensor)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AUDITS_DIR,
        help="where the PDF and the summary go (default: results/audits)",
    )
    parser.add_argument("--verbose", action="store_true", help="print one line per instance")
    args = parser.parse_args()

    sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
    unknown = [s for s in sensors if s not in BANDS]
    if unknown:
        parser.error(f"not a pressure or temperature sensor: {unknown} (known: {list(BANDS)})")

    print(f"Sensor distributions audit of {args.raw_dir}")
    print(f"Started {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")
    tallies = tally_dataset(args.raw_dir, sensors, args.max_instances, args.verbose)
    rows = ordered_rows(tallies)
    units = load_sensor_units(args.raw_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "sensor_distributions.pdf"
    with PdfPages(pdf_path) as pdf:
        for sensor in sensors:
            fig = draw_sensor_page(sensor, rows, tallies, units.get(sensor, ""))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"  Saved: {pdf_path}")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    summary_path = args.output_dir / f"sensor_distributions_{stamp}.txt"
    write_summary(summary_path, sensors, rows, tallies, units)
    print(f"  Saved: {summary_path}")
    print(f"Done {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
