"""Stage 0 (optional) — Visualize the raw 3W dataset.

Standalone of the modeling pipeline. Renders, in order, every real instance of
every fault (one multi-page PDF per fault), the timeline of the instances of
every well (one page per well), and the joined history of every well (one
multi-page PDF per well). Simulated and hand-drawn instances are never
plotted.

``--well`` narrows the well histories to a few wells, and ``--skip-faults``
drops the two per-fault stages, which together make inspecting one well quick.

Usage
-----
    uv run scripts/00_visualize_dataset.py [--raw-dir PATH] [--verbose]
    uv run scripts/00_visualize_dataset.py --well 1 13 --skip-faults

Outputs
-------
    results/figures/fault_<n>_real_instances.pdf   one per fault class
    results/figures/faults_per_well.pdf            one page per well
    results/figures/well_<id>_history.pdf          one per well
"""

import argparse
from datetime import datetime
from pathlib import Path

from flowml.config import FAULT_CLASSES, RAW_DATA_DIR
from flowml.visualization import plot_fault, plot_faults_per_well, plot_wells_histories


def main() -> None:
    """Parse arguments and render the requested dataset plots."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="root of the 3W dataset (default: FLOWML_RAW_DATA_DIR or config)",
    )
    parser.add_argument(
        "--well",
        type=int,
        nargs="+",
        default=None,
        help="wells whose joined history to plot (default: every well in the dataset)",
    )
    parser.add_argument(
        "--skip-faults",
        action="store_true",
        help="plot only the well histories, skipping the per-fault stages",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20_000,
        help="envelope resolution of a well history, in buckets (default: 20000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print per-instance progress",
    )
    args = parser.parse_args()

    print(f"Visualizing dataset at {args.raw_dir}")
    print(f"Started {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")

    if not args.skip_faults:
        for fault_class, fault_name in FAULT_CLASSES.items():
            print(f"\n[{fault_class + 1}/{len(FAULT_CLASSES) + 1}] {fault_name}...")
            plot_fault(str(fault_class), raw_dir=args.raw_dir, verbose=args.verbose)

        print(f"\n[{len(FAULT_CLASSES) + 1}/{len(FAULT_CLASSES) + 1}] Faults per well...")
        plot_faults_per_well(raw_dir=args.raw_dir, verbose=args.verbose)

    print("\nWell histories...")
    plot_wells_histories(
        args.well,
        raw_dir=args.raw_dir,
        max_points=args.max_points,
        verbose=args.verbose,
    )

    print(f"\nDone {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
