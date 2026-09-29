"""Build the feature dataset the new way: merged recordings first, then windows of a chosen length.

Prototype of the module that replaces ``01_build_features.py`` (see
``ideas/2026-09-28_new_feature_dataset.md``); the numbered stage keeps
working untouched until the pipeline overhaul retires it. Two steps:

1. ``flowml.dataset.reconstruction`` merges the overlapping real instances of
   every well into recordings, relabels the short unlabelled gaps, splits the
   one recording that spans two events, and writes the result in 3W's layout
   under ``data/merged/`` with a manifest (events and provenance included).
   The step is skipped when a reconstruction with the same parameters is
   already there; ``--rebuild-merged`` forces it.
2. ``flowml.dataset.extraction`` cuts the recordings and the untouched
   synthetic instances into non-overlapping windows and writes
   ``data/features_w<L>_o<R>.parquet`` with its manifest, one per length.

Usage
-----
    uv run scripts/build_dataset.py [--max-instances N] [--raw-dir PATH] [--source {merged,original}]
        [--window-length L [L ...]] [--window-overlap R] [--gap-relabel-max SECONDS]
        [--sensors P-TPT,T-TPT,...] [--keep-extreme-values] [--output-dir DIR]
        [--rebuild-merged] [--skip-extraction] [--verbose]

Outputs
-------
    data/merged[_n<N>]/<folder>/WELL-<well>_<start>.parquet + manifest.json
    data/features_w<L>_o<R>[_n<N>].parquet + .json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from flowml.config import DATA_DIR, KEY_SENSORS, RAW_DATA_DIR
from flowml.dataset.extraction import SOURCES, WINDOW_LENGTH, WINDOW_OVERLAP, build_features
from flowml.dataset.manifest import read_manifest
from flowml.dataset.reconstruction import (
    GAP_RELABEL_MAX_SECONDS,
    MANIFEST_NAME,
    build_merged_dataset,
    manifest_matches,
    merged_dir_name,
)


def main() -> None:
    """Parse the switches and run the two steps."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="root of the 3W dataset (default: FLOWML_RAW_DATA_DIR or config)",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="merged",
        help="real instances to featurize: the reconstruction, or 3W's files as they are (default: merged)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="cap instances per fault folder, for a smoke test (default: all)",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        nargs="+",
        default=[WINDOW_LENGTH],
        help=f"window length(s) in seconds, one parquet each (default: {WINDOW_LENGTH})",
    )
    parser.add_argument(
        "--window-overlap",
        type=int,
        default=WINDOW_OVERLAP,
        help=f"overlap of consecutive windows, percent in [0, 100) (default: {WINDOW_OVERLAP})",
    )
    parser.add_argument(
        "--gap-relabel-max",
        type=int,
        default=GAP_RELABEL_MAX_SECONDS,
        help=(
            "longest unlabelled gap that inherits its flanks' common class, in seconds; "
            f"0 disables the rule, -1 removes the bound (default: {GAP_RELABEL_MAX_SECONDS})"
        ),
    )
    parser.add_argument(
        "--sensors",
        default=",".join(KEY_SENSORS),
        help="comma-separated sensors to featurize (default: the eight key sensors)",
    )
    parser.add_argument(
        "--keep-extreme-values",
        action="store_true",
        help="keep the readings that cannot be measurements instead of masking them as missing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="where the reconstruction, the parquets and the manifests go (default: data/)",
    )
    parser.add_argument(
        "--rebuild-merged", action="store_true", help="rebuild the reconstruction even if present"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="stop after the reconstruction (step 1); no parquet is written",
    )
    parser.add_argument("--verbose", action="store_true", help="print one line per recording")
    args = parser.parse_args()

    gap_max = None if args.gap_relabel_max < 0 else args.gap_relabel_max
    sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
    print(
        f"Building the dataset from {args.raw_dir} · started {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}"
    )

    merged_dir = args.output_dir / merged_dir_name(args.max_instances)
    if args.source == "merged":
        manifest_path = merged_dir / MANIFEST_NAME
        reuse = (
            not args.rebuild_merged
            and manifest_path.exists()
            and manifest_matches(
                read_manifest(manifest_path), args.raw_dir, args.max_instances, gap_max
            )
        )
        if reuse:
            print(f"Step 1 · reconstruction already present: {manifest_path}")
        else:
            print(f"Step 1 · reconstructing the real instances into {merged_dir}")
            manifest_path = build_merged_dataset(
                args.raw_dir, merged_dir, args.max_instances, gap_max, args.verbose
            )
        totals = read_manifest(manifest_path)["totals"]
        print("  " + json.dumps(totals))

    if args.skip_extraction:
        print(f"Step 2 skipped · finished {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")
        return
    print(f"Step 2 · extracting windows of {args.window_length} s, overlap {args.window_overlap} %")
    manifests = build_features(
        source=args.source,
        raw_dir=args.raw_dir,
        merged_dir=merged_dir if args.source == "merged" else None,
        lengths=args.window_length,
        overlap=args.window_overlap,
        sensors=sensors,
        keep_extreme_values=args.keep_extreme_values,
        max_instances_per_class=args.max_instances,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    for manifest in manifests:
        output = read_manifest(manifest)["output"]
        print(
            f"  {output['parquet']}: {output['windows']:,} windows, "
            f"{output['size_bytes'] / 1e6:.1f} MB, by source {output['windows_by_source']}"
        )
        print(f"  manifest: {manifest}")
    print(f"Finished {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
