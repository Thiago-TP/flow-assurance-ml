"""Rebuild the real part of 3W as non-overlapping recordings, in 3W's own layout.

3W stores a well's history as instances that mostly overlap in time: 986 of
the 1,119 real instances of 2.0.0 share samples with a neighbour, usually as a
chain of sliding windows over one long recording. The well-instances audit
(``scripts/audits/well_instances_auditing.py``) established the three facts
this module relies on:

- on a timestamp shared by two instances the readings are identical and the
  label is set in exactly one of them (never two different labels), so merging
  is "take the non-NaN label";
- the instances join into 208 recordings, of which exactly one draws from two
  fault-class folders (well 6: an Abrupt BSW Increase, then a PCK Scaling);
- the merged timelines contain unlabelled runs whose two flanks carry the same
  class — the seams of the manual labelling — which inherit that class when
  they are short enough (``GAP_RELABEL_MAX_SECONDS``, 30 minutes: the only
  longer gap with a genuine change of dynamics inside it was the 56-minute one
  of well 33).

The result is written as ``<out_dir>/<folder>/WELL-<id>_<start>.parquet``,
one file per recording, labels as nullable ``Int16`` like 3W, sensors as
``float64``, every recording reindexed to a complete 1 Hz grid. A recording
whose instances came from two folders is split at the unlabelled stretch
after the first event, each part going to its own folder. ``manifest.json``
next to the folders lists every recording with its events (the runs of its
labels), its provenance (the 3W files it came from) and what was relabelled.
Simulated and hand-drawn instances have no well and are not copied: the
extraction reads them from the raw dataset as they are.
"""

import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from flowml.config import DATA_DIR, FAULT_CLASSES, SOURCE_TYPES
from flowml.dataset.manifest import code_state, dataset_version, read_manifest, write_manifest
from flowml.preprocessing import list_raw_instances, parse_source_type, parse_well_id

MERGE_TOLERANCE_SECONDS = 1  # instances this close in time are one recording
GAP_RELABEL_MAX_SECONDS = 1800  # decided 2026-09-28 after inspecting the gaps beyond the p90
TRANSIENT_OFFSET = 100  # 3W labels a transient as 100 + its fault class
LABEL_COLUMNS = ("class", "state")
MANIFEST_KIND = "3w-merged"
MANIFEST_NAME = "manifest.json"


def merged_dir_name(max_instances_per_class: int | None) -> str:
    """``merged`` for the whole dataset, ``merged_n<N>`` when capped, so the two never mix."""
    return "merged" if max_instances_per_class is None else f"merged_n{max_instances_per_class}"


@dataclass
class Plan:
    """The instances of one well that will become one recording, in time order."""

    well: int
    start: pd.Timestamp
    end: pd.Timestamp
    files: list[Path] = field(default_factory=list)
    folders: list[int] = field(default_factory=list)
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)


def merge_spans(
    instances: list[tuple[pd.Timestamp, pd.Timestamp, int, Path]],
    well: int,
    tolerance_seconds: int = MERGE_TOLERANCE_SECONDS,
) -> list[Plan]:
    """Join the instances of one well that overlap or touch into plans.

    Parameters
    ----------
    instances : list[(start, end, folder, path)]
        One entry per real instance of the well, in any order.
    well : int
        The well's id.
    tolerance_seconds : int
        Two instances whose gap is at most this many seconds are joined.

    Returns
    -------
    list[Plan]
        In time order; every instance belongs to exactly one plan.
    """
    tolerance = pd.Timedelta(seconds=tolerance_seconds)
    plans: list[Plan] = []
    for start, end, folder, path in sorted(instances, key=lambda item: (item[0], item[1])):
        if plans and start <= plans[-1].end + tolerance:
            plan = plans[-1]
            plan.end = max(plan.end, end)
        else:
            plan = Plan(well, start, end)
            plans.append(plan)
        plan.files.append(path)
        plan.folders.append(folder)
        plan.spans.append((start, end))
    return plans


def list_instances_by_source(
    raw_dir: Path, max_instances_per_class: int | None = None
) -> list[tuple[int, Path, str]]:
    """Every instance file with its folder and source, capped per folder *and* source.

    3W sorts a folder's files by name, which puts the hand-drawn and simulated
    instances before the real ones; a cap taken over the folder as a whole
    would therefore drop every real instance of the folders with many
    synthetic files (1, 5, 6 and 9). The cap counts each source on its own.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset.
    max_instances_per_class : int | None
        Cap per fault folder and source; ``None`` lists everything.

    Returns
    -------
    list[(int, Path, str)]
        Folder, path and source type (a value of ``SOURCE_TYPES``) per file.
    """
    taken: Counter[tuple[int, str]] = Counter()
    entries = []
    for folder, path in list_raw_instances(raw_dir, list(FAULT_CLASSES)):
        kind = parse_source_type(path.name)
        if max_instances_per_class is not None and taken[(folder, kind)] >= max_instances_per_class:
            continue
        taken[(folder, kind)] += 1
        entries.append((folder, path, kind))
    return entries


def plan_recordings(raw_dir: Path, max_instances_per_class: int | None = None) -> list[Plan]:
    """List the real instances of the dataset and group them into recordings, well by well.

    Only each file's index is read here, so the pass takes seconds.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset.
    max_instances_per_class : int | None
        Cap on real instances per fault folder (the synthetic ones are counted
        apart), for a smoke test; the cap applies before merging, so a capped
        build may cut a chain of instances short.

    Returns
    -------
    list[Plan]
        Wells in ascending order, plans in time order within a well.
    """
    by_well: dict[int, list] = defaultdict(list)
    for folder, path, kind in list_instances_by_source(raw_dir, max_instances_per_class):
        if kind != SOURCE_TYPES["real"]:
            continue
        index = pd.read_parquet(path, columns=["class"]).index
        by_well[int(parse_well_id(path.name))].append((index.min(), index.max(), folder, path))
    plans: list[Plan] = []
    for well in sorted(by_well):
        plans.extend(merge_spans(by_well[well], well))
    return plans


def assemble(plan: Plan) -> tuple[pd.DataFrame, int]:
    """Read a plan's instances and merge them into one recording on a complete 1 Hz grid.

    Parameters
    ----------
    plan : Plan
        The instances to join.

    Returns
    -------
    (pd.DataFrame, int)
        The recording — every 3W column, ``float64``, index ``timestamp`` —
        and how many seconds of the grid had no sample in any instance
        (inserted as all-NaN rows).
    """
    frames = [pd.read_parquet(path) for path in plan.files]
    df = pd.concat(frames).sort_index(kind="stable")
    # ``first`` keeps the first non-null value of every column per timestamp: the
    # readings agree, and the label is set in one instance only.
    df = df.groupby(level=0, sort=True).first()
    grid = pd.date_range(df.index.min(), df.index.max(), freq="s", name="timestamp")
    missing_seconds = len(grid) - len(df)
    return df.reindex(grid).astype("float64"), missing_seconds


def relabel_gaps(
    labels: np.ndarray, max_seconds: int | None = GAP_RELABEL_MAX_SECONDS
) -> tuple[np.ndarray, dict[int, dict[str, int]]]:
    """Give the unlabelled runs flanked by one and the same class that class.

    A run at the edge of the array has one flank only and is left alone; so is
    a run whose two flanks differ, whatever the classes.

    Parameters
    ----------
    labels : np.ndarray
        The ``class`` column as ``float64``, NaN where unlabelled.
    max_seconds : int | None
        Longest run that is relabelled; ``None`` means no bound, ``0``
        disables the rule.

    Returns
    -------
    (np.ndarray, dict)
        The filled labels (a copy) and, per class relabelled, how many gaps
        and how many seconds.
    """
    filled = labels.copy()
    stats: dict[int, dict[str, int]] = {}
    isnan = np.isnan(labels)
    if max_seconds == 0 or not isnan.any() or isnan.all():
        return filled, stats
    edges = np.diff(np.concatenate([[0], isnan.astype(int), [0]]))
    for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if a == 0 or b == len(labels):
            continue
        left, right = labels[a - 1], labels[b]
        if left != right or (max_seconds is not None and b - a > max_seconds):
            continue
        filled[a:b] = left
        entry = stats.setdefault(int(left), {"gaps": 0, "seconds": 0})
        entry["gaps"] += 1
        entry["seconds"] += int(b - a)
    return filled, stats


def event_class(labels: np.ndarray) -> np.ndarray:
    """Fold the transient labels (100 + fault) onto their fault class; NaN stays NaN."""
    return np.where(labels >= TRANSIENT_OFFSET, labels - TRANSIENT_OFFSET, labels)


def split_indices(labels: np.ndarray, folders: list[int]) -> list[tuple[int, int]]:
    """Where a recording drawn from several fault folders is cut, and which folder each part gets.

    The events are ordered by their first labelled sample; between two
    consecutive ones the cut falls on the first labelled sample after the
    earlier event's last sample, so the unlabelled stretch that separates them
    stays with the earlier part and the later part starts on its own prefix.

    Parameters
    ----------
    labels : np.ndarray
        The recording's ``class`` column, ``float64``.
    folders : list[int]
        The fault folders of the instances the recording came from.

    Returns
    -------
    list[(int, int)]
        ``(start_index, folder)`` per part, in order; a single part when the
        recording draws from one fault folder (or from Normal only).
    """
    faults = [f for f in dict.fromkeys(folders) if f != 0]
    if len(faults) < 2:
        return [(0, faults[0] if faults else 0)]
    events = event_class(labels)
    present = [f for f in faults if (events == f).any()]
    ordered = sorted(present, key=lambda f: int(np.argmax(events == f)))
    if len(ordered) < 2:
        return [(0, ordered[0] if ordered else faults[0])]
    parts = [(0, ordered[0])]
    for earlier, later in pairwise(ordered):
        last = int(np.where(events == earlier)[0].max())
        labelled_after = np.where(~np.isnan(labels[last + 1 :]))[0]
        cut = last + 1 + int(labelled_after[0]) if len(labelled_after) else last + 1
        parts.append((cut, later))
    return parts


def label_events(labels: np.ndarray, index: pd.DatetimeIndex) -> list[dict]:
    """The runs of one label in a recording, unlabelled stretches skipped.

    Parameters
    ----------
    labels : np.ndarray
        The ``class`` column, ``float64``.
    index : pd.DatetimeIndex
        The recording's timestamps.

    Returns
    -------
    list[dict]
        ``class``, ``start``, ``end`` (ISO strings) and ``seconds`` per run.
    """
    keys = np.where(np.isnan(labels), -1.0, labels)
    change = np.concatenate([[True], keys[1:] != keys[:-1]])
    starts = np.where(change)[0]
    ends = np.append(starts[1:], len(labels)) - 1
    return [
        {
            "class": int(labels[a]),
            "start": index[a].isoformat(),
            "end": index[b].isoformat(),
            "seconds": int(b - a + 1),
        }
        for a, b in zip(starts, ends)
        if not np.isnan(labels[a])
    ]


def recording_name(well: int, start: pd.Timestamp) -> str:
    """3W's file-name convention for a real recording."""
    return f"WELL-{well:05d}_{start:%Y%m%d%H%M%S}.parquet"


def write_recording(df: pd.DataFrame, well: int, folder: int, out_dir: Path) -> Path:
    """Write one recording in 3W's layout: ``<out_dir>/<folder>/WELL-<id>_<start>.parquet``.

    Labels go back to nullable ``Int16`` as in 3W; the sensors stay ``float64``.
    """
    path = out_dir / str(folder) / recording_name(well, df.index[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for column in LABEL_COLUMNS:
        if column in out.columns:
            out[column] = out[column].round().astype("Int16")
    out.to_parquet(path, engine="pyarrow", compression="brotli")
    return path


def _clear_output(out_dir: Path) -> None:
    """Remove a previous reconstruction — and only a reconstruction — before rebuilding."""
    if not out_dir.exists():
        return
    manifest = out_dir / MANIFEST_NAME
    if manifest.exists() and read_manifest(manifest).get("kind") == MANIFEST_KIND:
        shutil.rmtree(out_dir)
    elif any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} exists and is not a reconstruction written by this module; not touching it"
        )


def build_merged_dataset(
    raw_dir: Path,
    out_dir: Path | None = None,
    max_instances_per_class: int | None = None,
    gap_relabel_max: int | None = GAP_RELABEL_MAX_SECONDS,
    verbose: bool = False,
) -> Path:
    """Reconstruct the real instances into recordings and write them with their manifest.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset.
    out_dir : Path | None
        Destination; defaults to ``data/merged`` (``data/merged_n<N>`` when
        capped). An earlier reconstruction there is replaced.
    max_instances_per_class : int | None
        Cap on instances per fault folder, for a smoke test.
    gap_relabel_max : int | None
        Longest unlabelled gap that inherits its flanks' common class
        (``None``: no bound; ``0``: never relabel).
    verbose : bool
        Print one line per recording.

    Returns
    -------
    Path
        The manifest written next to the class folders.
    """
    out_dir = out_dir or DATA_DIR / merged_dir_name(max_instances_per_class)
    _clear_output(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plans = plan_recordings(raw_dir, max_instances_per_class)
    records: list[dict] = []
    totals = {
        "source_instances": sum(len(p.files) for p in plans),
        "recordings": 0,
        "samples": 0,
        "missing_seconds_inserted": 0,
        "relabelled_gaps": 0,
        "relabelled_seconds": 0,
        "split_recordings": 0,
    }
    for plan in plans:
        df, missing = assemble(plan)
        labels, relabelled = relabel_gaps(df["class"].to_numpy(), gap_relabel_max)
        df["class"] = labels
        parts = split_indices(labels, plan.folders)
        if len(parts) > 1:
            totals["split_recordings"] += 1
        bounds = [start for start, _ in parts] + [len(df)]
        for (start, folder), stop in zip(parts, bounds[1:]):
            part = df.iloc[start:stop]
            path = write_recording(part, plan.well, folder, out_dir)
            span = (part.index[0], part.index[-1])
            sources = [
                file.name
                for file, (s, e) in zip(plan.files, plan.spans)
                if s <= span[1] and e >= span[0]
            ]
            records.append(
                {
                    "file": path.relative_to(out_dir).as_posix(),
                    "well": plan.well,
                    "folder": folder,
                    "start": span[0].isoformat(),
                    "end": span[1].isoformat(),
                    "samples": len(part),
                    "missing_seconds": int(missing) if len(parts) == 1 else None,
                    "source_files": sources,
                    "split_from": recording_name(plan.well, plan.start) if len(parts) > 1 else None,
                    "events": label_events(part["class"].to_numpy(), part.index),
                    "relabelled": relabelled if len(parts) == 1 else None,
                }
            )
            totals["recordings"] += 1
            totals["samples"] += len(part)
            if verbose:
                print(
                    f"  well {plan.well:>2} · folder {folder} · {span[0]:%Y-%m-%d %H:%M:%S} - "
                    f"{span[1]:%Y-%m-%d %H:%M:%S} · {len(part):,} samples · "
                    f"{len(sources)} source file{'s' * (len(sources) != 1)}",
                    flush=True,
                )
        totals["missing_seconds_inserted"] += int(missing)
        totals["relabelled_gaps"] += sum(v["gaps"] for v in relabelled.values())
        totals["relabelled_seconds"] += sum(v["seconds"] for v in relabelled.values())
        del df

    return write_manifest(
        out_dir / MANIFEST_NAME,
        {
            "kind": MANIFEST_KIND,
            "code": code_state(),
            "source": {"raw_dir": str(raw_dir), "dataset_version": dataset_version(raw_dir)},
            "parameters": {
                "max_instances_per_class": max_instances_per_class,
                "gap_relabel_max_seconds": gap_relabel_max,
                "merge_tolerance_seconds": MERGE_TOLERANCE_SECONDS,
            },
            "layout": {
                "files": "<folder>/WELL-<well>_<start>.parquet, one per recording",
                "labels": "class and state as nullable Int16, as in 3W",
                "sensors": "float64, every 3W column, complete 1 Hz grid",
                "synthetic_instances": "not copied; read from raw_dir at extraction",
            },
            "totals": totals,
            "recordings": records,
        },
    )


def manifest_matches(
    manifest: dict, raw_dir: Path, max_instances_per_class, gap_relabel_max
) -> bool:
    """Whether an existing reconstruction was built from the same source with the same parameters."""
    parameters = manifest.get("parameters", {})
    return (
        manifest.get("kind") == MANIFEST_KIND
        and manifest.get("source", {}).get("dataset_version") == dataset_version(raw_dir)
        and parameters.get("max_instances_per_class") == max_instances_per_class
        and parameters.get("gap_relabel_max_seconds") == gap_relabel_max
    )
