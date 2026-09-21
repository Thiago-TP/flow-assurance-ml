"""Stage 1 — Build the windowed features parquet from the raw 3W dataset.

Reads the raw per-instance parquet files, drops the real instances that
overlap another recording of the same well (unless ``--allow-overlap``),
cleans each remaining instance (bounded forward-fill + critical-sensor quality
gate), and extracts the 88 statistical features per 300 s window. Each row
carries both task labels (``window_label`` and ``fault_class``), so one
parquet serves both the detection and the prediction task.

The features are always **raw**. Normalization is no longer a property of the
dataset: each row additionally carries the (mu, sigma) of every reference a
run might z-score against, so choosing one costs a training run rather than a
rebuild of this file (see ``flowml.normalization`` and ``--normalization`` on
the later stages).

Usage
-----
    uv run scripts/01_build_features.py [--max-instances N] [--raw-dir PATH]
                                        [--allow-overlap] [--keep-extreme-values]
                                        [--verbose]

Output
------
    data/features.parquet
    (``_overlap`` is appended with --allow-overlap and ``_extremes`` with
    --keep-extreme-values, e.g. features_overlap_extremes.parquet)
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

    output_path = features_path(args.allow_overlap, args.keep_extreme_values)
    print(f"Building features from {args.raw_dir}")
    print(f"Started {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")
    build_features(
        output_path=output_path,
        raw_dir=args.raw_dir,
        max_instances_per_class=args.max_instances,
        allow_overlap=args.allow_overlap,
        keep_extreme_values=args.keep_extreme_values,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
