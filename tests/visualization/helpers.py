"""Shared fixtures for the ``visualization`` test suite: a miniature 3W dataset.

Every test file builds its own small ``raw_dir`` out of these building blocks
rather than reading the real (large, environment-specific) 3W dataset, so the
suite is fast and self-contained.
"""

import configparser
from pathlib import Path

import numpy as np
import pandas as pd

from flowml.config import FAULT_CLASSES


def make_all_fault_dirs(raw_dir: Path) -> None:
    """Create an empty folder per fault class (0-9) under ``raw_dir``.

    Several ``visualization`` functions (``list_well_ids``, ``load_well_history``,
    ``plot_faults_per_well``, ...) scan every fault-class folder unconditionally,
    exactly as a real 3W dataset provides all ten of them even where a given
    fault has no instance of a given source. Tests that build a partial
    dataset still need every folder to exist for that reason.
    """
    for fault_class in FAULT_CLASSES:
        (raw_dir / str(fault_class)).mkdir(parents=True, exist_ok=True)


# A small stand-in for the real ``dataset.ini``: enough variables to exercise
# unit parsing (a plain unit, a valve-state enum, an ASCII-encoded degree
# sign) without pulling in the full 3W sensor list.
DATASET_INI_VARIABLES = {
    "P-PDG": "Downhole pressure gauge [Pa]",
    "T-PDG": "Downhole temperature [oC]",
    "P-TPT": "Wet christmas tree pressure [Pa]",
    "T-TPT": "Wet christmas tree temperature [oC]",
    "P-MON-CKP": "Pressure upstream of the production choke [Pa]",
    "T-JUS-CKP": "Temperature downstream of the production choke [oC]",
    "ABER-CKP": "Production choke opening [0, 0.5, or 1]",
    "ESTADO-W1": "Well 1 state [0, 0.5, or 1]",
    "ESTADO-SDV-P": "Production SDV state [0, 0.5, or 1]",
}


def write_dataset_ini(raw_dir: Path, variables: dict[str, str] | None = None) -> None:
    """Write a minimal ``dataset.ini`` with a ``PARQUET_FILE_PROPERTIES`` section."""
    parser = configparser.ConfigParser()
    fields = {"timestamp": "Timestamp [s]"}
    fields.update(DATASET_INI_VARIABLES if variables is None else variables)
    fields["class"] = "Label [0, 1, ..., 9, 101, ..., 109]"
    fields["state"] = "Well operational status [0, 1, ..., 8]"
    parser["PARQUET_FILE_PROPERTIES"] = fields
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "dataset.ini").open("w", encoding="utf-8") as handle:
        parser.write(handle)


def make_instance(
    start: str,
    n: int = 20,
    freq: str = "1s",
    sensors: dict[str, np.ndarray] | None = None,
    class_values=None,
    state_values=None,
) -> pd.DataFrame:
    """Build one synthetic raw instance: a timestamp-indexed frame with labels.

    Parameters
    ----------
    start : str
        First timestamp, parsed by ``pd.date_range``.
    n : int
        Number of samples.
    freq : str
        Sampling period.
    sensors : dict[str, array-like] | None
        Sensor columns to include; defaults to a couple of pressure/temperature
        series so a plotted instance always has something to draw.
    class_values, state_values : array-like | None
        The ``class``/``state`` columns; default to all-zero (normal, open).

    Returns
    -------
    pd.DataFrame
        The synthetic instance, ready to be written to parquet.
    """
    index = pd.date_range(start, periods=n, freq=freq, name="timestamp")
    if sensors is None:
        sensors = {
            "P-PDG": np.linspace(1.0e7, 1.1e7, n),
            "T-TPT": np.linspace(80.0, 85.0, n),
        }
    data = {name: np.asarray(values, dtype=float) for name, values in sensors.items()}
    data["class"] = np.zeros(n) if class_values is None else np.asarray(class_values, dtype=float)
    data["state"] = np.zeros(n) if state_values is None else np.asarray(state_values, dtype=float)
    return pd.DataFrame(data, index=index)


def write_instance(path: Path, df: pd.DataFrame) -> None:
    """Write a synthetic instance frame to ``path``, creating its parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def real_name(well_id: int, stamp: str) -> str:
    """Filename of a real instance, matching the 3W ``WELL-000{id}_{stamp}`` pattern."""
    return f"WELL-{well_id:05d}_{stamp}.parquet"


def simulated_name(tag: str) -> str:
    """Filename of a simulated instance, matching the 3W ``SIMULATED_*`` pattern."""
    return f"SIMULATED_{tag}.parquet"


def drawn_name(tag: str) -> str:
    """Filename of a hand-drawn instance, matching the 3W ``DRAWN_*`` pattern."""
    return f"DRAWN_{tag}.parquet"
