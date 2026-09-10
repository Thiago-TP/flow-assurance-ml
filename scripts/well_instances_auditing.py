"""This script audits parquet files in the 3W dataset related to a given set of wells.

It collects all parquet files (instances) for the specified well IDs and prints their amount.
Verbosely, it prints their names along with the start and end timestamps of the data they contain,
also checking for overlaps in time.

The script is intended to be ran from the command line,
and is not dependent/does not depend on the main pipeline stages
(that is, the scripts whose names start with numbers).

However, it is useful for auditing the dataset before running the main pipeline,
as overlapping instances of a same well may lead to train-test leakage.

Usage
-----
    uv run scripts/well_instances_auditing.py [--dataset-path PATH]
                                              [--output-file PATH]
                                              [--well-ids {all,1,2,3}]
                                              [--verbose]
"""

import argparse
from pathlib import Path

import pandas as pd

# Update this path to the actual dataset location
DATASET_PATH = Path.joinpath(Path(__file__).parent, "..", "..", "3W", "dataset")

# By default, all well IDs in the dataset, i.e., 1-42 except 17 and 18, are considered available.
ALL_IDS: set[int] = set(range(1, 42)) - {17, 18}


def _dataset_path_is_correct(dataset_path: Path) -> bool:
    """Check if the dataset path is correct by looking for a known file."""
    return (dataset_path / "dataset.ini").exists()


def find_parquet_files(well_id: int, dataset_path: Path) -> set[Path]:
    pattern = f"WELL-000{well_id}_*.parquet" if well_id > 9 else f"WELL-0000{well_id}_*.parquet"
    return {file for file in dataset_path.rglob(pattern)}


def _get_timestamps(file_name: Path, engine: str = "pyarrow") -> tuple[str, str]:
    data = pd.read_parquet(file_name, engine=engine)
    return (data.index[0], data.index[-1])  # Very important these be ordered


def pretty_print_parquet_files(
    parquet_files: set[Path], output_file, strformat: str = "%Y-%m-%d %H:%M:%S"
) -> None:

    if not parquet_files:
        print("  - No parquet files found.", file=output_file)
        return

    n_overlaps = 0
    timestamps: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for file in sorted(parquet_files):
        timestamp_start, timestamp_end = _get_timestamps(file)
        start_as_string = timestamp_start.strftime(strformat)
        end_as_string = timestamp_end.strftime(strformat)
        print(f"  - {file.name}    {start_as_string} - {end_as_string}", file=output_file)
        timestamps.append((timestamp_start, timestamp_end))

    for i in range(len(timestamps)):
        start_i, end_i = timestamps[i]
        for j in range(i + 1, len(timestamps)):
            start_j, end_j = timestamps[j]
            if start_i < end_j and end_i > start_j:
                n_overlaps += 1

    if n_overlaps > 0:
        ratio_files = (n_overlaps + 1) / len(parquet_files)
        msg = f"Warning: {n_overlaps + 1} ({ratio_files:.2%}) instances overlap in time."
        len_msg = len(msg)
        print("-" * len_msg, file=output_file)
        print(msg, file=output_file)
        print("-" * len_msg, file=output_file)
    else:
        print("------------------", file=output_file)
        print("No overlaps found.", file=output_file)
        print("------------------", file=output_file)
    print(file=output_file)  # Add a newline for better readability

    return


def run_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DATASET_PATH,
        help="Path to the 3W dataset (default: scripts/../../3W/dataset).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to the output file where the results will be saved.",
    )
    parser.add_argument(
        "--well-ids",
        type=str,
        default="all",
        help="Comma-separated list of well IDs to analyze (e.g., 1,2,3) or 'all' for all wells.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    return parser


def main(
    available_ids: set[int] = ALL_IDS,
) -> None:
    parser = run_parser(__doc__)
    args = parser.parse_args()

    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = DATASET_PATH

    assert _dataset_path_is_correct(dataset_path), "Check the dataset path."

    if args.output_file:
        output_file = args.output_file
    else:
        current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path.joinpath(
            Path("results"), "audits", f"well_instances_audit_{current_time}.txt"
        )

    if args.well_ids:
        if args.well_ids.lower() == "all":
            well_ids = available_ids
        else:
            well_ids = {int(well_id.strip()) for well_id in args.well_ids.split(",")}

    if args.verbose:
        verbose = args.verbose
    else:
        verbose = False

    output_file.parent.mkdir(parents=True, exist_ok=True)

    for well_id in well_ids:
        if well_id not in available_ids:
            print(f"Well ID {well_id} was set as not available in the dataset.", file=output_file)
            continue

        well_files = find_parquet_files(well_id, dataset_path)
        with open(output_file, "a") as f:
            n_files = len(well_files)
            msg = f"Well {well_id} has {n_files} instance" + "s" * (n_files > 1)
            print(msg, file=f)
            if verbose:
                pretty_print_parquet_files(well_files, f)
            f.close()


if __name__ == "__main__":
    main()
