"""Manifests: what a dataset artifact was built from, with what, by which code.

Every artifact the ``dataset`` package writes — the merged recordings, each
features parquet — gets a JSON sidecar. A run that reads the artifact records
the sidecar's hash, which is how a result stays tied to the exact dataset it
was computed on without a second repository.
"""

import configparser
import hashlib
import json
from datetime import datetime
from pathlib import Path

from flowml.runs import git_state


def dataset_version(raw_dir: Path) -> str | None:
    """The 3W dataset version declared in ``dataset.ini``, or ``None`` when absent.

    Parameters
    ----------
    raw_dir : Path
        Root of the 3W dataset.

    Returns
    -------
    str | None
        E.g. ``"2.0.0"``.
    """
    ini_path = raw_dir / "dataset.ini"
    if not ini_path.exists():
        return None
    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    return parser.get("VERSION", "DATASET", fallback=None)


def code_state() -> dict:
    """The checkout the artifact is being built from (``runs.git_state``)."""
    return git_state()


def write_manifest(path: Path, payload: dict) -> Path:
    """Write ``payload`` as indented JSON, stamping it with the time of writing.

    Parameters
    ----------
    path : Path
        Destination ``.json`` file; parent directories are created.
    payload : dict
        Anything JSON can hold; timestamps and paths are written as strings.

    Returns
    -------
    Path
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created": datetime.now().astimezone().isoformat(timespec="seconds"), **payload}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")
    return path


def read_manifest(path: Path) -> dict:
    """Read a manifest written by ``write_manifest``."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
