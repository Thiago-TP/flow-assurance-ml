"""Stage 0 (optional) — Visualize the raw 3W dataset.

Standalone of the modeling pipeline: plots every real instance of every fault
(one multi-page PDF per fault) and the faults-per-well timeline. Simulated
and hand-drawn instances are never plotted.

Usage
-----
    uv run scripts/00_visualize_dataset.py [--raw-dir PATH] [--verbose]

Outputs
-------
    results/figures/fault_<n>_real_instances.pdf   one per fault class
    results/figures/faults_per_well.pdf
"""

import argparse
from datetime import datetime
from pathlib import Path

from flowml.config import FAULT_CLASSES, RAW_DATA_DIR
from flowml.visualization import plot_fault, plot_faults_per_well


def main() -> None:
    """Parse arguments and render every dataset plot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="root of the 3W dataset (default: FLOWML_RAW_DATA_DIR or config)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print per-instance progress",
    )
    args = parser.parse_args()

    print(f"Visualizing dataset at {args.raw_dir}")
    print(f"Started {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")

    for fault_class, fault_name in FAULT_CLASSES.items():
        print(f"\n[{fault_class + 1}/{len(FAULT_CLASSES) + 1}] {fault_name}...")
        plot_fault(str(fault_class), raw_dir=args.raw_dir, verbose=args.verbose)

    print(f"\n[{len(FAULT_CLASSES) + 1}/{len(FAULT_CLASSES) + 1}] Faults per well...")
    plot_faults_per_well(raw_dir=args.raw_dir, verbose=args.verbose)

    print(f"\nDone {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
