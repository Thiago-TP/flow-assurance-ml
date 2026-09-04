"""Central configuration: paths, sensors, class maps, and pipeline constants.

Every tunable of the pipeline lives here so scripts never need editing.
The only machine-specific value is ``RAW_DATA_DIR``, which points at a local
copy of the 3W dataset and can be overridden with the ``FLOWML_RAW_DATA_DIR``
environment variable.
"""

import os
from pathlib import Path

# -- Paths --------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # the streamline/ directory

RAW_DATA_DIR = Path(os.environ.get("FLOWML_RAW_DATA_DIR", r"../3W/dataset"))

DATA_DIR = PACKAGE_ROOT / "data"
RESULTS_DIR = PACKAGE_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"


def norm_suffix(normalized: bool) -> str:
    """Artifact-name suffix for the normalization status of the features.

    Parameters
    ----------
    normalized : bool
        Whether the features were built from per-instance z-scored sensors.

    Returns
    -------
    str
        ``"zscore"`` or ``"raw"``.
    """
    return "zscore" if normalized else "raw"


def features_path(normalized: bool = True) -> Path:
    """Features parquet path for the given normalization status.

    Parameters
    ----------
    normalized : bool
        Whether the features were built from per-instance z-scored sensors.

    Returns
    -------
    Path
        ``data/features_zscore.parquet`` or ``data/features_raw.parquet``.
    """
    return DATA_DIR / f"features_{norm_suffix(normalized)}.parquet"


# -- 3W dataset classes -------------------------------------------------------

FAULT_CLASSES = {
    0: "Normal",
    1: "Abrupt BSW Increase",
    2: "Spurious DHSV Closure",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick PCK Restriction",
    7: "PCK Scaling",
    8: "Hydrate in Production Line",
    9: "Hydrate in Service Line",
}

# Well operational status codes of the 3W ``state`` column
# (Table 5 of the 3W Dataset 2.0.0 paper, arXiv:2507.01048).
WELL_STATES = {
    0: "Open",
    1: "Shut-In",
    2: "Flushing Diesel",
    3: "Flushing Gas",
    4: "Bullheading",
    5: "Closed With Diesel",
    6: "Closed With Gas",
    7: "Restart",
    8: "Depressurization",
}

# Window-state labels: 0 = normal, 1-9 = active event, 101-109 = transient.
# Classes 3 and 4 have no transient period in the 3W dataset.
_TRANSIENT_CAPABLE = {k: v for k, v in FAULT_CLASSES.items() if k not in {0, 3, 4}}
WINDOW_CLASSES = {
    0: "Normal",
    **{k: f"{v} (active)" for k, v in FAULT_CLASSES.items() if k != 0},
    **{100 + k: f"{v} (transient)" for k, v in _TRANSIENT_CAPABLE.items()},
}

# -- Label groupings ----------------------------------------------------------

# Coarse regrouping for the hydrate question: is the well heading for normal
# operation, a hydrate event, or some other flow-assurance problem? Hydrate
# covers faults 8 and 9 (production and service line); every other fault
# collapses into a single "Other Problem" group. Transient labels (101-109)
# join the group of their active counterpart.
CLASS_GROUPINGS = ("none", "hydrate", "custom")
HYDRATE_CLASS_GROUPING: dict[int, str] = {
    0: "Normal",
    1: "Other Problem",
    2: "Other Problem",
    3: "Other Problem",
    4: "Other Problem",
    5: "Other Problem",
    6: "Other Problem",
    7: "Other Problem",
    8: "Hydrate",
    9: "Hydrate",
}
CUSTOM_CLASS_GROUPING: dict[int, str] = {
    0: "",
    1: "",
    2: "",
    3: "",
    4: "",
    5: "",
    6: "",
    7: "",
    8: "",
    9: "",
}

# -- Sensors ------------------------------------------------------------------

KEY_SENSORS = [
    "P-PDG",  # downhole pressure gauge
    "T-PDG",  # downhole temperature
    "P-TPT",  # wet christmas tree pressure (critical sensor)
    "T-TPT",  # wet christmas tree temperature
    "P-MON-CKP",  # pressure upstream of the production choke
    "T-JUS-CKP",  # temperature downstream of the production choke
    "P-JUS-CKGL",  # pressure downstream of the gas-lift choke
    "QGL",  # gas-lift flow rate
]

# -- Cleaning -----------------------------------------------------------------

FFILL_LIMIT = 60  # forward-fill gaps up to 60 samples (60 s at 1 Hz)
CRITICAL_SENSOR = "P-TPT"  # instance is dropped when this sensor is too sparse
MAX_MISSING_RATIO = 0.50  # NaN threshold on the critical sensor

# -- Feature engineering ------------------------------------------------------

WINDOW_SIZE = 300  # window length in samples (300 s at 1 Hz)
STEP_SIZE = 150  # stride between windows (50 % overlap)
MIN_VALID_SAMPLES = 10  # windows with fewer valid samples yield NaN features
CONSTANT_THRESHOLD = 1e-10  # std below this means a flat (stuck/off) signal

FEATURE_STATS = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "iqr",
    "skewness",
    "kurtosis",
    "diff1_std",
    "diff2_std",
    "max_zscore",
]

META_COLS = [
    "instance_id",
    "well_id",
    "fault_class",
    "window_label",
    "source_type",
    "window_start",
]

# -- Modeling -----------------------------------------------------------------

RANDOM_STATE = 42
CV_GROUPINGS = ("instance_id", "well_id")  # columns GroupKFold can group by
CV_GROUPING = "instance_id"  # default GroupKFold grouping column
CV_SPLITS = 5  # GroupKFold folds
N_SPLITS_CV = max(2, CV_SPLITS)  # GroupKFold folds (hyperparameter search)
N_ITER_SEARCH = 10  # RandomizedSearchCV iterations

EVAL_MODES = ("holdout", "nested")
EVAL_MODE = "holdout"  # default evaluation protocol
TEST_SIZE = 0.2  # group fraction held out for testing (holdout evaluation)
N_SPLITS_OUTER = 5  # outer GroupKFold folds (nested evaluation)
N_JOBS = max(
    1,
    min(6, os.cpu_count() - 2),  # parallel workers (keep below core count to preserve RAM)
)

RF_PARAM_GRID = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 10, 20, 30],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2"],
}

XGB_PARAM_GRID = {
    "clf__n_estimators": [100, 200, 300, 500],
    "clf__max_depth": [3, 4, 6, 8],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "clf__subsample": [0.7, 0.8, 1.0],
    "clf__colsample_bytree": [0.7, 0.8, 1.0],
    "clf__min_child_weight": [1, 3, 5],
}

# -- Interpretation -----------------------------------------------------------

TOP_N_FEATURES = 15  # features shown in importance plots / rankings
SHAP_SAMPLE = 5_000  # windows sampled for SHAP
PERM_SAMPLE = 10_000  # windows sampled for permutation importance
PERM_REPEATS = 10  # shuffles per feature in permutation importance
