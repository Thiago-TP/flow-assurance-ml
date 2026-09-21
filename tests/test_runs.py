"""Tests of the run-directory machinery: naming, resolution, manifest, index."""

import json
from datetime import UTC, datetime

import pytest

from flowml import runs
from flowml.runs import (
    RUN_DIR_ENV,
    RunDir,
    append_index,
    create_run,
    latest_run,
    read_index,
    read_manifest,
    record_stage,
    resolve_run,
    run_id,
)

CLEAN = {"commit": "a" * 40, "short": "aaaaaaa", "branch": "main", "dirty": False}
DIRTY = {**CLEAN, "dirty": True}


def at(hour: int = 15, minute: int = 7, second: int = 33) -> datetime:
    """A fixed, timezone-aware moment, so the run ids in the assertions are stable."""
    return datetime(2026, 9, 21, hour, minute, second, tzinfo=UTC)


@pytest.fixture
def results(tmp_path, monkeypatch):
    """Point the run directory and the index at a temporary results tree."""
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runs, "RUN_INDEX_PATH", tmp_path / "index.jsonl")
    monkeypatch.delenv(RUN_DIR_ENV, raising=False)
    return tmp_path


def make_run(results, name: str, *artifacts: str) -> RunDir:
    """Create a run directory by name and touch the given relative artifacts."""
    run = RunDir(results / "runs" / name).mkdirs()
    for artifact in artifacts:
        (run.path / artifact).write_text("x", encoding="utf-8")
    return run


# -- Naming --------------------------------------------------------------------


def test_run_id_leads_with_the_timestamp_so_names_sort_chronologically():
    early = run_id(CLEAN, at())
    late = run_id(CLEAN, at(16, 0, 0))
    assert early == "20260921_150733_aaaaaaa"
    assert early < late


def test_run_id_marks_a_dirty_working_tree():
    assert run_id(DIRTY, at()).endswith("_aaaaaaa-dirty")


# -- Creation ------------------------------------------------------------------


def test_create_run_writes_a_manifest_and_the_three_artifact_folders(results, monkeypatch):
    monkeypatch.setattr(runs, "git_state", lambda: CLEAN)
    monkeypatch.setattr(runs, "dirty_files", list)

    run = create_run(at())

    assert run.id == "20260921_150733_aaaaaaa"
    assert run.models.is_dir() and run.metrics.is_dir() and run.figures.is_dir()
    manifest = read_manifest(run)
    assert manifest["run_id"] == run.id
    assert manifest["git"] == CLEAN
    assert manifest["stages"] == []


def test_create_run_saves_the_patch_of_a_dirty_tree(results, monkeypatch):
    monkeypatch.setattr(runs, "git_state", lambda: DIRTY)
    monkeypatch.setattr(runs, "dirty_files", lambda: [" M main.py"])
    monkeypatch.setattr(runs, "working_tree_diff", lambda: "--- a/main.py")

    run = create_run(at())

    assert (run.path / "changes.diff").read_text(encoding="utf-8").startswith("--- a/main.py")
    assert read_manifest(run)["dirty_files"] == [" M main.py"]


def test_create_run_reuses_a_directory_without_discarding_its_manifest(results, monkeypatch):
    monkeypatch.setattr(runs, "git_state", lambda: CLEAN)
    monkeypatch.setattr(runs, "dirty_files", list)
    started = at()
    first = create_run(started)
    record_stage(first, "02_train_val_test", "tag", started)

    again = create_run(started)

    assert again.path == first.path
    assert len(read_manifest(again)["stages"]) == 1


# -- Resolution ----------------------------------------------------------------


def test_latest_run_skips_the_runs_that_lack_the_wanted_artifact(results):
    make_run(results, "20260101_000000_aaaaaaa", "metrics/wanted.json")
    make_run(results, "20260202_000000_bbbbbbb", "metrics/other.json")

    assert latest_run("metrics/wanted.json").id == "20260101_000000_aaaaaaa"
    assert latest_run().id == "20260202_000000_bbbbbbb"


def test_latest_run_accepts_a_run_holding_any_one_of_several_artifacts(results):
    make_run(results, "20260101_000000_aaaaaaa", "metrics/standard.json")

    found = latest_run(["metrics/grouped.json", "metrics/standard.json"])

    assert found.id == "20260101_000000_aaaaaaa"


def test_resolve_run_prefers_the_explicit_id_over_the_environment(results, monkeypatch):
    asked = make_run(results, "20260101_000000_aaaaaaa")
    other = make_run(results, "20260202_000000_bbbbbbb")
    monkeypatch.setenv(RUN_DIR_ENV, str(other.path))

    assert resolve_run("20260101_000000_aaaaaaa").path == asked.path


def test_resolve_run_prefers_the_environment_over_creating_one(results, monkeypatch):
    named = make_run(results, "20260101_000000_aaaaaaa")
    monkeypatch.setenv(RUN_DIR_ENV, str(named.path))

    assert resolve_run(None, create=True).path == named.path


def test_resolve_run_creates_one_when_nothing_names_a_run(results, monkeypatch):
    monkeypatch.setattr(runs, "git_state", lambda: CLEAN)
    monkeypatch.setattr(runs, "dirty_files", list)

    run = resolve_run(None, create=True)

    assert run.path.parent == results / "runs"
    assert run.manifest.exists()


def test_resolve_run_falls_back_to_the_newest_run_holding_the_input(results):
    make_run(results, "20260101_000000_aaaaaaa", "models/m.joblib")
    make_run(results, "20260202_000000_bbbbbbb")

    assert resolve_run(None, must_contain="models/m.joblib").id == "20260101_000000_aaaaaaa"


def test_resolve_run_latest_reads_the_newest_matching_run(results):
    make_run(results, "20260101_000000_aaaaaaa", "models/m.joblib")
    make_run(results, "20260202_000000_bbbbbbb", "models/m.joblib")

    assert resolve_run("latest", must_contain="models/m.joblib").id == "20260202_000000_bbbbbbb"


def test_resolve_run_refuses_a_run_id_that_does_not_exist(results):
    with pytest.raises(SystemExit, match="no such run directory"):
        resolve_run("20260101_000000_aaaaaaa")


def test_resolve_run_names_the_missing_artifacts_when_nothing_matches(results):
    make_run(results, "20260101_000000_aaaaaaa")

    with pytest.raises(SystemExit, match="models/m.joblib"):
        resolve_run(None, must_contain="models/m.joblib")


# -- Manifest ------------------------------------------------------------------


def test_record_stage_appends_relative_outputs_and_the_extra_fields(results, monkeypatch):
    monkeypatch.setattr(runs, "git_state", lambda: CLEAN)
    monkeypatch.setattr(runs, "dirty_files", list)
    started = at()
    run = create_run(started)
    output = run.metrics / "tag_eval.parquet"
    output.write_text("x", encoding="utf-8")

    record_stage(run, "02_train_val_test", "tag", started, [output], test_f1_macro=0.5)

    (entry,) = read_manifest(run)["stages"]
    assert entry["stage"] == "02_train_val_test"
    assert entry["outputs"] == ["metrics/tag_eval.parquet"]
    assert entry["test_f1_macro"] == 0.5
    # The run's own checkout is not repeated on every stage, only a differing one.
    assert "git" not in entry


def test_record_stage_records_a_checkout_that_moved_since_the_run_started(results, monkeypatch):
    monkeypatch.setattr(runs, "git_state", lambda: CLEAN)
    monkeypatch.setattr(runs, "dirty_files", list)
    started = at()
    run = create_run(started)

    moved = {**CLEAN, "commit": "b" * 40, "short": "bbbbbbb"}
    monkeypatch.setattr(runs, "git_state", lambda: moved)
    record_stage(run, "03_evaluate", "tag", started)

    assert read_manifest(run)["stages"][0]["git"] == moved


def test_record_stage_survives_a_naive_start_time(results, monkeypatch):
    # The duration is the difference of two timestamps taken hours apart, and
    # the record is written at the very end of the stage: a naive start time
    # must not be what loses it.
    monkeypatch.setattr(runs, "git_state", lambda: CLEAN)
    monkeypatch.setattr(runs, "dirty_files", list)
    run = create_run(at())

    record_stage(run, "02_train_val_test", "tag", datetime(2026, 9, 21, 15, 7, 33))  # noqa: DTZ001

    assert read_manifest(run)["stages"][0]["duration_s"] >= 0


# -- Index ---------------------------------------------------------------------


def test_append_index_writes_one_line_per_stage(results, monkeypatch):
    monkeypatch.setattr(runs, "git_state", lambda: CLEAN)
    monkeypatch.setattr(runs, "dirty_files", list)
    run = create_run(at())

    append_index(run, "02_train_val_test", "tag", {"test_f1_macro": 0.5})
    append_index(run, "03_evaluate", "tag", {"full_f1_macro": 0.6})

    rows = read_index()
    assert [row["stage"] for row in rows] == ["02_train_val_test", "03_evaluate"]
    assert rows[0]["run_id"] == run.id
    assert rows[0]["commit"] == "aaaaaaa"
    assert rows[1]["full_f1_macro"] == 0.6


def test_read_index_drops_only_the_line_a_crash_truncated(results, monkeypatch):
    monkeypatch.setattr(runs, "git_state", lambda: CLEAN)
    monkeypatch.setattr(runs, "dirty_files", list)
    run = create_run(at())
    append_index(run, "02_train_val_test", "tag", {"test_f1_macro": 0.5})
    with open(runs.RUN_INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": "03_evaluate"})[:12])  # killed mid-write

    rows = read_index()

    assert len(rows) == 1
    assert rows[0]["stage"] == "02_train_val_test"
