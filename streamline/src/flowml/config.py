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

RAW_DATA_DIR = Path(os.environ.get("FLOWML_RAW_DATA_DIR", r"H:\projetos\3W\dataset"))

DATA_DIR = PACKAGE_ROOT / "data"
RESULTS_DIR = PACKAGE_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"


def features_path(filter_type: str) -> Path:
    """Return the features-parquet path for a given signal filter.

    Parameters
    ----------
    filter_type : str
        One of ``"gaussian"``, ``"statistical"``, ``"none"``.

    Returns
    -------
    Path
        ``data/features_<filter_type>.parquet``.
    """
    return DATA_DIR / f"features_{filter_type}.parquet"


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

# Window-state labels: 0 = normal, 1-9 = active event, 101-109 = transient.
# Classes 3 and 4 have no transient period in the 3W dataset.
_TRANSIENT_CAPABLE = {k: v for k, v in FAULT_CLASSES.items() if k not in {0, 3, 4}}
WINDOW_CLASSES = {
    0: "Normal",
    **{k: f"{v} (active)" for k, v in FAULT_CLASSES.items() if k != 0},
    **{100 + k: f"{v} (transient)" for k, v in _TRANSIENT_CAPABLE.items()},
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
GAUSSIAN_SIGMA = 2.0  # sigma of the Gaussian smoothing filter
STATISTICAL_SIGMA = 0.5  # measurement-noise scale of the adaptive filter (z-units)
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
    "fault_class",
    "window_label",
    "source_type",
    "window_start",
]

# -- Modeling -----------------------------------------------------------------

RANDOM_STATE = 42
N_SPLITS_CV = 5  # GroupKFold folds (grouped by instance_id)
N_ITER_SEARCH = 20  # RandomizedSearchCV iterations
N_JOBS = 6  # parallel workers (keep below core count to preserve RAM)

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
