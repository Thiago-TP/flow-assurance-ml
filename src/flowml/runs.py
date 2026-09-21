"""Per-experiment run directories and the append-only index over them.

Every artifact the modeling stages write belongs to one *run*: one invocation
of ``main.py``, or one stage started by hand. A run owns a directory under
``results/runs/`` named ``<timestamp>_<commit>`` — ``-dirty`` appended when the
working tree carried uncommitted changes — holding the ``models/``,
``metrics/`` and ``figures/`` subdirectories the stages write into::

    results/
      index.jsonl                          one line per finished stage
      runs/
        20260921_150733_3687aa2-dirty/
          run.json                         provenance of the run and its stages
          changes.diff                     `git diff HEAD`, only when dirty
          figures/ metrics/ models/

The timestamp comes first so a directory listing is chronological; the commit
follows because it is the one thing the artifact tags of ``cli.run_tag`` cannot
express. Neither is the real record, though: ``run.json`` is. It carries the
full argument vector of every stage, the resolved commit and branch, the list
of dirty files, the interpreter and lockfile the run used, and the outputs each
stage produced — everything needed to say what a number came from, which a
directory name can only hint at.

Resolving a run
---------------
Stages find their run directory in this order:

1. ``--run <id|path|latest>``, when given;
2. the ``FLOWML_RUN_DIR`` environment variable, which ``main.py`` sets once and
   every stage subprocess inherits, so one chain writes into one directory;
3. for stage 2, a freshly created run; for the later stages, the newest run
   holding the input they need.

The last rule is what keeps a stage runnable on its own: ``03_evaluate.py``
started by hand finds the most recent run that actually trained the tag it is
asked to score. Whichever rule fires, the chosen directory is printed.

The index
---------
``results/index.jsonl`` holds one JSON object per line, appended when a stage
finishes, with the run id, the stage, the tag and the run's headline scores. It
exists because nesting artifacts by run costs the one thing the old flat layout
gave for free: seeing every configuration at once. One ``pd.read_json(...,
lines=True)`` brings the whole history back as a table.

JSON Lines rather than a single JSON array because the file is only ever
appended to: a new record is one ``write`` of one line, where an array would
have to be parsed, extended and rewritten whole on every run. That also decides
what a crash costs — a killed run leaves at most one truncated last line, which
``read_index`` drops, whereas a half-written array is unparseable and takes the
entire history with it.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from flowml.config import PACKAGE_ROOT, RUN_INDEX_PATH, RUNS_DIR

RUN_DIR_ENV = "FLOWML_RUN_DIR"  # set by main.py, inherited by every stage
MANIFEST_NAME = "run.json"
DIFF_NAME = "changes.diff"
NO_GIT = "nogit"  # stands in for the short hash when git cannot answer


# -- Git ----------------------------------------------------------------------


def _git(*args: str) -> str | None:
    """Run a git command in the project root and return its stripped stdout.

    Parameters
    ----------
    *args : str
        Arguments after ``git``.

    Returns
    -------
    str | None
        The output, or ``None`` when git is absent or the command failed —
        a run outside a checkout still gets a directory, just an honest one.
    """
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=PACKAGE_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_state() -> dict:
    """Describe the checkout the current process is running from.

    Returns
    -------
    dict
        ``commit`` (full hash or ``None``), ``short`` (7 characters, or
        ``NO_GIT``), ``branch``, and ``dirty`` — whether the working tree
        had uncommitted changes to tracked or untracked files.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "short": commit[:7] if commit else NO_GIT,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def dirty_files() -> list[str]:
    """List the working tree's uncommitted changes, ``git status --porcelain`` style.

    Returns
    -------
    list[str]
        One entry per changed or untracked path, empty on a clean tree.
    """
    return (_git("status", "--porcelain") or "").splitlines()


def working_tree_diff() -> str | None:
    """The patch between ``HEAD`` and the working tree.

    Saved beside the artifacts of a dirty run, which is what makes such a run
    reproducible at all: the commit alone does not describe the code that ran.
    Untracked files are not in a diff — ``dirty_files`` names them instead.

    Returns
    -------
    str | None
        The patch, or ``None`` when there is none or git cannot answer.
    """
    return _git("diff", "HEAD") or None


def _file_sha256(path: Path) -> str | None:
    """Hex digest of a file, or ``None`` when it does not exist."""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aware(moment: datetime) -> datetime:
    """Attach the local timezone to a naive timestamp.

    Stage durations are the difference of two timestamps taken at opposite
    ends of a long run, and mixing a naive one with an aware one raises. The
    stages pass aware timestamps; this keeps a caller that does not from
    losing the whole record at the moment it is written.

    Parameters
    ----------
    moment : datetime
        Timestamp, aware or naive.

    Returns
    -------
    datetime
        The same moment, guaranteed timezone-aware.
    """
    return moment if moment.tzinfo else moment.astimezone()


# -- Run directories -----------------------------------------------------------


@dataclass(frozen=True)
class RunDir:
    """One experiment's directory, with the three artifact folders under it.

    Attributes
    ----------
    path : Path
        The run directory itself, ``results/runs/<run_id>``.
    """

    path: Path

    @property
    def id(self) -> str:
        """The run identifier, i.e. the directory name."""
        return self.path.name

    @property
    def models(self) -> Path:
        """Where fitted pipelines and label encoders are written."""
        return self.path / "models"

    @property
    def metrics(self) -> Path:
        """Where scores, predictions, rankings and tree rules are written."""
        return self.path / "metrics"

    @property
    def figures(self) -> Path:
        """Where plots are written."""
        return self.path / "figures"

    @property
    def manifest(self) -> Path:
        """The run's ``run.json``."""
        return self.path / MANIFEST_NAME

    def mkdirs(self) -> "RunDir":
        """Create the run directory and its three artifact folders.

        Returns
        -------
        RunDir
            ``self``, so the call can be chained onto a constructor.
        """
        for directory in (self.models, self.metrics, self.figures):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def run_id(git: dict, started: datetime) -> str:
    """Compose a run identifier from the checkout and the start time.

    Parameters
    ----------
    git : dict
        As returned by ``git_state``.
    started : datetime
        When the experiment started.

    Returns
    -------
    str
        E.g. ``"20260921_150733_3687aa2"``, or ``..._3687aa2-dirty`` when the
        working tree had uncommitted changes. The timestamp leads so that
        sorting the directory names sorts the runs chronologically.
    """
    return f"{started:%Y%m%d_%H%M%S}_{git['short']}" + ("-dirty" if git["dirty"] else "")


def create_run(started: datetime | None = None) -> RunDir:
    """Create a run directory and write its manifest.

    A dirty working tree additionally gets its patch saved as ``changes.diff``.
    Reusing an existing directory (two runs started within the same second on
    the same commit) leaves the manifest alone, so the stages of both are
    recorded in one run rather than one overwriting the other.

    Parameters
    ----------
    started : datetime | None
        Start time of the experiment; now, in local time, by default.

    Returns
    -------
    RunDir
        The created (or reused) directory, with its subfolders in place.
    """
    started = _aware(started) if started else datetime.now().astimezone()
    git = git_state()
    run = RunDir(RUNS_DIR / run_id(git, started)).mkdirs()
    if run.manifest.exists():
        return run

    write_manifest(
        run,
        {
            "run_id": run.id,
            "created_at": started.isoformat(),
            "git": git,
            "dirty_files": dirty_files(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "uv_lock_sha256": _file_sha256(PACKAGE_ROOT / "uv.lock"),
            "stages": [],
        },
    )
    if git["dirty"]:
        diff = working_tree_diff()
        if diff:
            (run.path / DIFF_NAME).write_text(diff + "\n", encoding="utf-8")
    return run


def all_runs() -> list[RunDir]:
    """Every run directory, newest first.

    Returns
    -------
    list[RunDir]
        Sorted by name descending, which is chronological because the
        identifier starts with the timestamp.
    """
    if not RUNS_DIR.is_dir():
        return []
    return [RunDir(p) for p in sorted(RUNS_DIR.iterdir(), reverse=True) if p.is_dir()]


def latest_run(must_contain: str | Iterable[str] | None = None) -> RunDir | None:
    """The newest run holding at least one of the given artifacts.

    Parameters
    ----------
    must_contain : str | Iterable[str] | None
        Glob patterns relative to a run directory, e.g.
        ``"metrics/xgb_prediction_zscore_eval.parquet"``. A run qualifies when
        *any* of them matches, since a stage that reads two label sets can
        still do half its work. ``None`` accepts any run.

    Returns
    -------
    RunDir | None
        The newest qualifying run, or ``None`` when there is none.
    """
    patterns = [must_contain] if isinstance(must_contain, str) else must_contain
    for run in all_runs():
        if patterns is None or any(next(run.path.glob(p), None) for p in patterns):
            return run
    return None


def resolve_run(
    explicit: str | None = None,
    must_contain: str | Iterable[str] | None = None,
    create: bool = False,
) -> RunDir:
    """Find the run directory a stage should read from and write into.

    The order is ``--run``, then ``FLOWML_RUN_DIR``, then either a new run
    (``create``, for the stage that starts a chain) or the newest run holding
    the required input. The choice and the reason for it are printed, so a
    stage started by hand never scores one run's model against another's
    predictions without saying so.

    Parameters
    ----------
    explicit : str | None
        Value of ``--run``: a run id, a path, or ``"latest"``.
    must_contain : str | Iterable[str] | None
        Artifacts the run must hold, as globs relative to it (see
        ``latest_run``). Only consulted when falling back to the newest run.
    create : bool
        Make a new run when neither ``--run`` nor the environment names one.
        Set by stage 2, which produces the artifacts the others consume.

    Returns
    -------
    RunDir
        An existing directory, with its subfolders ensured.

    Raises
    ------
    SystemExit
        When ``--run`` names a directory that does not exist, or when nothing
        can be resolved and no run holds the required input.
    """
    if explicit and explicit != "latest":
        candidate = Path(explicit)
        # A bare run id names a directory under RUNS_DIR; anything that exists
        # as given is taken as a path. Either way it is stored resolved, so the
        # manifest records outputs relative to the run however it was named.
        chosen = candidate if candidate.is_absolute() or candidate.exists() else RUNS_DIR / explicit
        run = RunDir(chosen.resolve())
        if not run.path.is_dir():
            raise SystemExit(f"--run {explicit}: no such run directory ({run.path}).")
        source = "--run"
    elif explicit == "latest":
        found = latest_run(must_contain)
        if found is None:
            raise SystemExit(_not_found_message(must_contain))
        run, source = found, "--run latest"
    elif os.environ.get(RUN_DIR_ENV):
        run, source = RunDir(Path(os.environ[RUN_DIR_ENV]).resolve()), f"${RUN_DIR_ENV}"
    elif create:
        run, source = create_run(), "new"
    else:
        found = latest_run(must_contain)
        if found is None:
            raise SystemExit(_not_found_message(must_contain))
        run, source = found, "latest matching run"

    run.mkdirs()
    print(f"Run {run.id} ({source}) -> {run.path}")
    return run


def _not_found_message(must_contain: str | Iterable[str] | None) -> str:
    """Explain that no run holds what a stage needs, naming the artifacts."""
    if must_contain is None:
        return f"No run directory found under {RUNS_DIR}. Train something first (stage 2)."
    patterns = [must_contain] if isinstance(must_contain, str) else list(must_contain)
    wanted = "\n".join(f"    {p}" for p in patterns)
    return (
        f"No run under {RUNS_DIR} holds any of:\n{wanted}\n"
        "Train this configuration first (stage 2), or point at a run with --run <id>."
    )


# -- Manifest ------------------------------------------------------------------


def read_manifest(run: RunDir) -> dict:
    """Read a run's ``run.json``, or a minimal stand-in when it has none."""
    if not run.manifest.exists():
        return {"run_id": run.id, "git": git_state(), "stages": []}
    with open(run.manifest, encoding="utf-8") as f:
        return json.load(f)


def write_manifest(run: RunDir, manifest: dict) -> None:
    """Write a run's ``run.json``."""
    run.path.mkdir(parents=True, exist_ok=True)
    with open(run.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def record_stage(
    run: RunDir,
    stage: str,
    tag: str,
    started: datetime,
    outputs: Iterable[Path] = (),
    **extra,
) -> None:
    """Append one stage's provenance to the run manifest.

    The stage's own checkout is recorded only when it differs from the run's,
    which is exactly the case worth seeing: a later stage rerun by hand after
    the code moved on writes into the same directory, and the manifest then
    shows that its numbers come from different code than the model's.

    Parameters
    ----------
    run : RunDir
        The run to record into.
    stage : str
        Script name of the stage, e.g. ``"02_train_val_test"``.
    tag : str
        The artifact tag the stage worked under.
    started : datetime
        When the stage began.
    outputs : Iterable[Path]
        Files the stage wrote; stored relative to the run directory.
    **extra
        Further fields to store on the entry, e.g. the headline scores.
    """
    started, finished = _aware(started), datetime.now().astimezone()
    manifest = read_manifest(run)
    entry = {
        "stage": stage,
        "tag": tag,
        "argv": sys.argv[1:],
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": round((finished - started).total_seconds(), 1),
        "outputs": sorted(_relative(run, p) for p in outputs),
        **extra,
    }
    current = git_state()
    if current != manifest.get("git"):
        entry["git"] = current
    manifest.setdefault("stages", []).append(entry)
    write_manifest(run, manifest)


def _relative(run: RunDir, path: Path) -> str:
    """Path as written in the manifest: relative to the run, POSIX separators."""
    path = Path(path)
    try:
        return path.relative_to(run.path).as_posix()
    except ValueError:
        return path.as_posix()


# -- Index ---------------------------------------------------------------------


def append_index(run: RunDir, stage: str, tag: str, summary: dict) -> None:
    """Append one line to ``results/index.jsonl`` for a finished stage.

    Parameters
    ----------
    run : RunDir
        The run the stage belongs to.
    stage : str
        Script name of the stage.
    tag : str
        The artifact tag the stage worked under.
    summary : dict
        Headline numbers of the stage, flattened onto the row.
    """
    git = read_manifest(run).get("git", {})
    row = {
        "run_id": run.id,
        "finished_at": datetime.now().astimezone().isoformat(),
        "stage": stage,
        "tag": tag,
        "commit": git.get("short"),
        "dirty": git.get("dirty"),
        **summary,
    }
    RUN_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_index() -> list[dict]:
    """Read ``results/index.jsonl``, skipping any line a crash left unfinished.

    Returns
    -------
    list[dict]
        One entry per recorded stage, oldest first; empty when the index does
        not exist yet.
    """
    if not RUN_INDEX_PATH.exists():
        return []
    rows = []
    for line in RUN_INDEX_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a truncated last line costs one record, not the file
    return rows
