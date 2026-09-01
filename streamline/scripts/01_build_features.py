"""Stage 1 — Build the windowed features parquet from the raw 3W dataset.

Reads the raw per-instance parquet files, cleans each instance (bounded
forward-fill + critical-sensor quality gate), z-scores it, and extracts the
88 statistical features per 300 s window. Each row carries both task labels
(``window_label`` and ``fault_class``), so one parquet per filter serves both
the detection and the prediction task.

Usage
-----
    uv run scripts/01_build_features.py [--filter {gaussian,statistical,none}]
                                        [--max-instances N] [--raw-dir PATH]

Output
------
    data/features_<filter>.parquet
"""

from pathlib import Path

from flowml.cli import run_parser
from flowml.config import RAW_DATA_DIR, features_path
from flowml.features import build_features


def main() -> None:
    """Parse arguments and run the raw -> features pass."""
    parser = run_parser(__doc__.splitlines()[0], with_model=False)
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="cap instances per class for a quick smoke test (default: all)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="root of the 3W dataset (default: FLOWML_RAW_DATA_DIR or config)",
    )
    args = parser.parse_args()

    output_path = features_path(args.filter_type)
    print(f"Building features | filter={args.filter_type} | raw={args.raw_dir}")
    build_features(
        output_path=output_path,
        filter_type=args.filter_type,
        raw_dir=args.raw_dir,
        max_instances_per_class=args.max_instances,
    )


if __name__ == "__main__":
    main()
