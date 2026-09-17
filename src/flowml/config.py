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
VISUALIZATION_DIR = PACKAGE_ROOT / "plots"

# The dataset plots of stage 0 (see the ``visualization`` package) come in hundreds of
# files, so each family gets its own directory instead of sharing the one the
# modeling stages write their figures into. The fault timeline lives with the
# well histories because it is read alongside them: both are per-well views.
INSTANCE_FIGURES_DIR = VISUALIZATION_DIR / "instances_per_fault"
WELL_HISTORY_FIGURES_DIR = VISUALIZATION_DIR / "well_histories"
SIGNATURE_FIGURES_DIR = VISUALIZATION_DIR / "fault_signatures"


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


def overlap_suffix(allow_overlap: bool) -> str:
    """Artifact-name suffix for the overlap rule the features were built under.

    Parameters
    ----------
    allow_overlap : bool
        Whether the real instances overlapping another of the same well were
        kept (see ``preprocessing.select_instances``).

    Returns
    -------
    str
        ``"_overlap"`` when they were kept, ``""`` for the default dataset.
    """
    return "_overlap" if allow_overlap else ""


def extreme_suffix(keep_extreme_values: bool) -> str:
    """Artifact-name suffix for the extreme-value rule the features were built under.

    Parameters
    ----------
    keep_extreme_values : bool
        Whether readings too large to be measurements were kept (see
        ``preprocessing.mask_extreme_values``).

    Returns
    -------
    str
        ``"_extremes"`` when they were kept, ``""`` for the default dataset.
    """
    return "_extremes" if keep_extreme_values else ""


def features_path(
    normalized: bool = True, allow_overlap: bool = False, keep_extreme_values: bool = False
) -> Path:
    """Features parquet path for the given normalization, overlap and extreme rules.

    Parameters
    ----------
    normalized : bool
        Whether the features were built from per-instance z-scored sensors.
    allow_overlap : bool
        Whether overlapping real instances were kept when building them.
    keep_extreme_values : bool
        Whether readings beyond ``EXTREME_VALUE_LIMIT`` were kept.

    Returns
    -------
    Path
        ``data/features_zscore.parquet`` by default; ``raw`` replaces
        ``zscore`` without normalization, ``_overlap`` is appended when
        overlapping instances were kept and ``_extremes`` when extreme
        readings were, e.g. ``features_raw_overlap_extremes.parquet``.
    """
    return DATA_DIR / (
        f"features_{norm_suffix(normalized)}"
        f"{overlap_suffix(allow_overlap)}{extreme_suffix(keep_extreme_values)}.parquet"
    )


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

# The three origins a 3W instance can have, mapped to the filename prefix that
# identifies one (see ``preprocessing.parse_source_type``): field recordings,
# OLGA simulations, and series hand-drawn by Petrobras experts. Not every
# fault has instances of every source — the hand-drawn ones exist only for
# faults 1 and 7, and no fault has simulated normal operation.
SOURCE_TYPES = {"real": "WELL", "simulated": "SIMULATED", "drawn": "DRAWN"}

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
CLASS_GROUPINGS = ("standard", "hydrate", "custom")
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

# What a reading must satisfy to be a measurement rather than instrument
# garbage. Readings that fail become NaN at loading time and are imputed with
# the rest of the missing data; ``--keep-extreme-values`` turns all three
# rules off at once. See ``preprocessing.mask_extreme_values``.
#
# 1. Magnitude. The 3W dataset carries sensors frozen at absurd levels (well 6
#    reports P-PDG = -1.2e42 Pa and T-PDG = -1.7e38 °C for whole instances)
#    and signals off by orders of magnitude (well 26 reports P-JUS-CKP around
#    1.4e9 Pa, i.e. 14,000 bar). Surveying every instance of 3W 2.0.0 shows a
#    clean gap around this limit: the largest varying reading below it is
#    4.9e7 Pa, the smallest value above it is 1.3e8, and everything from there
#    up is garbage — so the limit removes no real signal, which matters
#    because genuine spikes are fault signatures (see
#    ``features.window_features``).
EXTREME_VALUE_LIMIT = 1e8

# 2. Pressures are absolute (Table 3 of the 3W paper) and choke openings are
#    percentages, so a negative reading of either is impossible; the dataset
#    has 106 files with a negative pressure, usually for the whole recording —
#    a broken or mis-mapped tag, which the paper itself warns about — and one
#    well reporting a choke opening of -99.99 %, a sentinel. Zero is left
#    alone: it is the frozen-at-zero case, a different defect.
PRESSURE_MIN = 0.0
OPENING_MIN = 0.0

# 3. Temperatures live in a narrow physical band. The floor is below every
#    genuine reading of the dataset (T-TPT reaches -33.8 °C, which is real:
#    Joule-Thomson cooling during a blowdown is exactly the condition hydrates
#    form in), and the ceiling is twice the hottest one (127.7 °C). The band
#    catches the two sentinel values the PIMS leaks into the data, -999 and
#    -99.99, as well as T-PDG readings of 30,000 °C.
TEMPERATURE_LIMITS = (-50.0, 250.0)

# The variables each rule applies to, by physical quantity (Table 2 of the 3W
# paper). Everything else — valve states, flow rates — is subject to the
# magnitude rule only.
PRESSURE_SENSORS = (
    "P-ANULAR",
    "P-JUS-BS",
    "P-JUS-CKGL",
    "P-JUS-CKP",
    "P-MON-CKGL",
    "P-MON-CKP",
    "P-MON-SDV-P",
    "P-PDG",
    "P-TPT",
    "PT-P",
)
TEMPERATURE_SENSORS = ("T-JUS-CKP", "T-MON-CKP", "T-PDG", "T-TPT")
OPENING_SENSORS = ("ABER-CKGL", "ABER-CKP")  # gas-lift and production choke

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
CV_SPLITS = 2  # GroupKFold folds
N_SPLITS_CV = max(2, CV_SPLITS)  # GroupKFold folds (hyperparameter search)
N_ITER_SEARCH = 5  # RandomizedSearchCV iterations

EVAL_MODES = ("holdout", "nested")
EVAL_MODE = "holdout"  # default evaluation protocol
TEST_SIZE = 0.2  # group fraction held out for testing (holdout evaluation)
N_SPLITS_OUTER = 5  # outer GroupKFold folds (nested evaluation)
N_JOBS = max(
    1,
    min(6, os.cpu_count() - 2),  # parallel workers (keep below core count to preserve RAM)
)

DT_PARAM_GRID = {
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [3, 4, 5, 6, 8, 12, None],
    "clf__min_samples_leaf": [1, 2, 4, 8],
    "clf__min_samples_split": [2, 5, 10],
    "clf__max_features": [None, "sqrt"],
}

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

# Canvas of an exported tree figure (see ``interpretation.export_tree``). A
# tree is drawn one leaf per column and one level per row, so its width
# follows the *leaf count* rather than the depth: real trees are nowhere near
# full (the depth-12 tree of a `--model dt` run has 171 leaves, not 4096), and
# sizing by depth would ask for a canvas thousands of times too wide. The
# pixel cap keeps a large tree within what matplotlib can rasterize (it
# refuses past 2**16 px a side) by lowering the resolution rather than by
# giving up on the figure.
TREE_FIGURE_LEAF_WIDTH = 1.5  # inches of width per leaf
TREE_FIGURE_LEVEL_HEIGHT = 2.5  # inches of height per level
TREE_FIGURE_DPI = 150
TREE_FIGURE_MAX_PIXELS = 60_000
SHAP_SAMPLE = 5_000  # windows sampled for SHAP
PERM_SAMPLE = 10_000  # windows sampled for permutation importance
PERM_REPEATS = 10  # shuffles per feature in permutation importance
