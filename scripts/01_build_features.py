"""Stage 1 — Build the windowed features parquet from the raw 3W dataset.

Reads the raw per-instance parquet files, cleans each instance (bounded
forward-fill + critical-sensor quality gate), z-scores it, and extracts the
88 statistical features per 300 s window. Each row carries both task labels
(``window_label`` and ``fault_class``), so one parquet serves both
the detection and the prediction task.

Usage
-----
    uv run scripts/01_build_features.py [--max-instances N] [--raw-dir PATH]
                                        [--no-normalization] [--verbose]

Output
------
    data/features_zscore.parquet (or features_raw.parquet with --no-normalization)
"""

from datetime import datetime
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

    normalize = not args.no_normalization
    output_path = features_path(normalize)
    print(f"Building features from {args.raw_dir}")
    print(f"Started {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")
    build_features(
        output_path=output_path,
        raw_dir=args.raw_dir,
        max_instances_per_class=args.max_instances,
        normalize=normalize,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
