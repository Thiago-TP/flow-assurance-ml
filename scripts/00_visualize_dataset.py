"""Stage 0 (optional) — Visualize the raw 3W dataset.

Standalone of the modeling pipeline. Renders, in order, every real instance of
every fault (one multi-page PDF per fault), the signature of every fault that
has one (one multi-page PDF per fault and instance source), the timeline of the
instances of every well (one page per well), and the joined history of every
well (one multi-page PDF per well).

The plots keyed to a physical well cover real instances only, since simulated
and hand-drawn ones have no well; the signatures cover all three sources.

``--well`` narrows the well histories to a few wells, and ``--skip-faults``
drops the two per-fault stages, which together make inspecting one well quick.

Usage
-----
    uv run scripts/00_visualize_dataset.py [--raw-dir PATH] [--verbose]
    uv run scripts/00_visualize_dataset.py --well 1 13 --skip-faults

Outputs
-------
    results/figures/instances_per_fault/fault_<n>_real_instances.pdf
    results/figures/fault_signatures/<source>/fault_<n>_<source>_signatures.pdf
    results/figures/well_histories/faults_per_well.pdf
    results/figures/well_histories/well_<id>_history.pdf
"""

import argparse
from datetime import datetime
from pathlib import Path

from flowml.config import FAULT_CLASSES, RAW_DATA_DIR
from flowml.visualization import (
    plot_faults_per_well,
)


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
        stages = len(FAULT_CLASSES) + 2
        # for fault_class, fault_name in FAULT_CLASSES.items():
        #     print(f"\n[{fault_class + 1}/{stages}] {fault_name}...")
        #     plot_fault(str(fault_class), raw_dir=args.raw_dir, verbose=args.verbose)

        # print(f"\n[{stages - 1}/{stages}] Fault signatures...")
        # plot_all_fault_signatures(raw_dir=args.raw_dir, verbose=args.verbose)

        print(f"\n[{stages}/{stages}] Faults per well...")
        plot_faults_per_well(raw_dir=args.raw_dir, verbose=args.verbose)

    # print("\nWell histories...")
    # plot_wells_histories(
    #     args.well,
    #     raw_dir=args.raw_dir,
    #     max_points=args.max_points,
    #     verbose=args.verbose,
    # )

    print(f"\nDone {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
