"""Central configuration: paths, sensors, class maps, and pipeline constants.

Every tunable of the pipeline lives here so scripts never need editing.
The only machine-specific value is ``RAW_DATA_DIR``, which points at a local
copy of the 3W dataset and can be overridden with the ``FLOWML_RAW_DATA_DIR``
environment variable.
"""

import os
from collections.abc import Iterable
from pathlib import Path

# -- Paths --------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # the streamline/ directory

RAW_DATA_DIR = Path(os.environ.get("FLOWML_RAW_DATA_DIR", r"../3W/dataset"))

DATA_DIR = PACKAGE_ROOT / "data"
RESULTS_DIR = PACKAGE_ROOT / "results"
VISUALIZATION_DIR = PACKAGE_ROOT / "plots"

# Artifacts of the modeling stages are not written to ``results`` directly:
# each experiment gets a directory of its own under ``RUNS_DIR``, named after
# the commit and the time it started, holding the ``models``, ``metrics`` and
# ``figures`` folders the stages fill (see the ``runs`` module). The index is
# the flat view over them — one line per finished stage. The audit scripts and
# the written reports are not experiments and stay directly under ``results``.
RUNS_DIR = RESULTS_DIR / "runs"
RUN_INDEX_PATH = RESULTS_DIR / "index.jsonl"

# The dataset plots of stage 0 (see the ``visualization`` package) come in hundreds of
# files, so each family gets its own directory instead of sharing the one the
# modeling stages write their figures into. The fault timeline lives with the
# well histories because it is read alongside them: both are per-well views.
INSTANCE_FIGURES_DIR = VISUALIZATION_DIR / "instances_per_fault"
WELL_HISTORY_FIGURES_DIR = VISUALIZATION_DIR / "well_histories"
SIGNATURE_FIGURES_DIR = VISUALIZATION_DIR / "fault_signatures"


# -- Normalization ------------------------------------------------------------

# The reference a window's features are z-scored against, chosen per run
# rather than baked into the features parquet (see the ``normalization``
# module). Stage 1 always builds raw features and stores the statistics of
# every reference beside them, so switching costs a training run, not a
# dataset rebuild.
#
# ``none``
#     the raw features, unscaled. The default: for the axis-aligned tree models
#     here, per-column scaling is a no-op, and the only scaling that does change
#     the matrix — per instance — is the one that leaks.
# ``instance``
#     mean and standard deviation over the whole recording. This is what the
#     pipeline used to do at build time, kept so the old results stay
#     reproducible; it is also the leak, since the divisor of a
#     normal-operation window encodes how badly the well later failed (TODO
#     item 9).
# ``normal-operation-values``
#     the same statistics over the instance's normal-operation samples only.
#     Leak-free with respect to the fault period, and the honest choice for the
#     detection task; for prediction it is close to circular, since every
#     modeled window is already a normal-operation one.
NORMALIZATIONS = ("none", "instance", "normal-operation-values")
NORMALIZATION = "none"
NORMALIZATION_REFERENCES = tuple(n for n in NORMALIZATIONS if n != "none")
NORMAL_OPERATION_REFERENCE = "normal-operation-values"

# Columns holding those statistics. They live in the features parquet beside
# the window features and are never fed to a model.
REF_COL_PREFIX = "ref__"

# Storage keys of the references, deliberately short and held apart from the
# names above: the switch is user-facing prose and may be reworded, while these
# are written into every features parquet. Renaming a choice must not
# invalidate a 240 MB dataset, so only this mapping changes when one is.
REFERENCE_KEYS = {"instance": "instance", NORMAL_OPERATION_REFERENCE: "normal"}


def ref_col(sensor: str, stat: str, reference: str) -> str:
    """Name of the column holding one reference statistic of one sensor.

    Parameters
    ----------
    sensor : str
        Sensor name, e.g. ``"P-TPT"``.
    stat : str
        ``"mean"`` or ``"std"``.
    reference : str
        One of ``NORMALIZATION_REFERENCES``.

    Returns
    -------
    str
        E.g. ``"ref__P-TPT_mean_instance"``. The suffix is the reference's
        storage key (``REFERENCE_KEYS``), not its switch name.
    """
    if reference not in REFERENCE_KEYS:
        raise ValueError(f"Unknown reference: {reference!r} (expected {NORMALIZATION_REFERENCES})")
    return f"{REF_COL_PREFIX}{sensor}_{stat}_{REFERENCE_KEYS[reference]}"


# How each of ``FEATURE_STATS`` behaves when its sensor is z-scored by the
# reference (mu, sigma). Together these three groups plus ``max_zscore`` cover
# all eleven, which is what lets the z-scored features be derived exactly from
# the raw ones instead of rebuilt from the raw signals.
LOCATION_SCALE_STATS = ("mean", "min", "max", "median")  # (v - mu) / sigma
SCALE_STATS = ("std", "iqr", "diff1_std", "diff2_std")  # v / sigma
INVARIANT_STATS = ("skewness", "kurtosis")  # unchanged by an affine map


def sensors_with_features(columns: Iterable[str]) -> list[str]:
    """Sensors that a set of column names carries a complete feature set for.

    Feature columns are ``<sensor>_<stat>``, and both halves are free text, so
    a name cannot be split on a suffix alone: three entries of
    ``FEATURE_STATS`` end in ``std`` (``std``, ``diff1_std``, ``diff2_std``),
    which means matching ``_std`` turns ``P-TPT_diff1_std`` into a sensor
    called ``P-TPT_diff1``. Candidates are therefore taken from the one
    statistic that is not a suffix of another, ``mean``, and then *verified*
    by requiring every statistic to be present — so the answer does not
    silently depend on which names happen not to collide.

    Parameters
    ----------
    columns : Iterable[str]
        Column names of a features frame.

    Returns
    -------
    list[str]
        Sensor names, in the order the columns appear.
    """
    columns = list(columns)
    present = set(columns)
    candidates = (c[: -len("_mean")] for c in columns if c.endswith("_mean"))
    return [s for s in candidates if all(f"{s}_{stat}" in present for stat in FEATURE_STATS)]


# -- Sensor health ------------------------------------------------------------

# What to do about a sensor that never moves. A frozen sensor makes every
# dispersion statistic of its window exactly zero, and the trees split on that
# at the root — "is this standard deviation exactly zero?" — so instrument
# status reaches the model disguised as a physical quantity (TODO item 10).
# Like the normalization reference, this is a choice of the run rather than of
# the dataset: the window's own standard deviation is already in the parquet,
# so the policy is applied at load time (see the ``sensor_health`` module).
#
# ``keep``  leave the degenerate zeros in place. The pipeline's behaviour
#           before item 10, kept so the earlier results stay reproducible.
# ``flag``  every statistic of a frozen sensor's window becomes NaN, for the
#           per-fold imputer to fill, and an explicit ``<sensor>_frozen``
#           indicator is added. Instrument status still reaches the model, but
#           through one named feature instead of eleven fake measurements.
# ``drop``  discard whole instances whose ``CRITICAL_SENSOR`` never moves, and
#           otherwise leave the features alone.
#
# The two remedies are the two the TODO item proposes, kept as alternatives so
# a report can compare each against ``keep`` one change at a time.
FROZEN_MODES = ("keep", "flag", "drop")
FROZEN_MODE = "flag"

# Dispersion below which a sensor counts as frozen. Larger than
# ``CONSTANT_THRESHOLD``, which is a guard against catastrophic cancellation in
# scipy's moments rather than a statement about instruments; this one follows
# the 3W Toolkit's ``CleanSignals.absolute_std_threshold``. The toolkit pairs it
# with an IQR bound fitted across events, which is not reproduced here: the
# physical bounds above already reject out-of-range readings, and fitting a
# bound over the whole dataset before splitting is the contamination item 9
# just removed.
FROZEN_STD_THRESHOLD = 1e-6


def frozen_suffix(frozen_mode: str) -> str:
    """Artifact-name suffix for the frozen-sensor policy a run uses.

    Parameters
    ----------
    frozen_mode : str
        One of ``FROZEN_MODES``.

    Returns
    -------
    str
        ``""`` for the default, ``"_<name>"`` otherwise.
    """
    if frozen_mode not in FROZEN_MODES:
        raise ValueError(f"Unknown frozen-sensor mode: {frozen_mode!r} (expected {FROZEN_MODES})")
    return "" if frozen_mode == FROZEN_MODE else f"_{frozen_mode}"


def norm_suffix(normalization: str) -> str:
    """Artifact-name suffix for the normalization a run uses.

    Parameters
    ----------
    normalization : str
        One of ``NORMALIZATIONS``.

    Returns
    -------
    str
        ``""`` for ``"none"``, ``"_<name>"`` otherwise.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization: {normalization!r} (expected {NORMALIZATIONS})")
    return "" if normalization == "none" else f"_{normalization}"


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


def features_path(allow_overlap: bool = False, keep_extreme_values: bool = False) -> Path:
    """Features parquet path for the given overlap and extreme-value rules.

    Normalization is no longer part of the name: the parquet holds the raw
    features plus the statistics of every reference, and a run picks one at
    load time (see the ``normalization`` module). Only the two rules that
    change *which rows and readings exist* still produce separate datasets.

    Parameters
    ----------
    allow_overlap : bool
        Whether overlapping real instances were kept when building them.
    keep_extreme_values : bool
        Whether readings beyond ``EXTREME_VALUE_LIMIT`` were kept.

    Returns
    -------
    Path
        ``data/features.parquet`` by default; ``_overlap`` is appended when
        overlapping instances were kept and ``_extremes`` when extreme
        readings were, e.g. ``features_overlap_extremes.parquet``.
    """
    return DATA_DIR / (
        f"features{overlap_suffix(allow_overlap)}{extreme_suffix(keep_extreme_values)}.parquet"
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
# Example of a custom grouping that merges faults 1-7 into a single "Other" group.
CUSTOM_CLASS_GROUPING: dict[int, str] = {
    0: "Other",
    1: "Other",
    2: "Other",
    3: "Other",
    4: "Other",
    5: "Other",
    6: "Other",
    7: "Other",
    8: "Hydrate",
    9: "Hydrate",
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

EVAL_MODES = ("holdout", "nested", "leave-one-out")
# Default protocol per CV grouping. A single seeded holdout of 20 % of the
# groups is fine with a thousand instances, but with 33 wells it is one draw
# of a lopsided lottery: the 8 wells held out for prediction/well_id carry
# 52 % of the windows and 68 % of normal operation (see
# results/reports/2026-09-17_dt_from_xgb_cv_and_class_grouping.md). Leaving
# one well out at a time instead gives a score per well and never lets one
# draw decide, so it is the default whenever the groups are wells.
EVAL_MODE_DEFAULTS = {"instance_id": "holdout", "well_id": "leave-one-out"}
TEST_SIZE = 0.2  # group fraction held out for testing (holdout evaluation)
N_SPLITS_OUTER = 5  # outer GroupKFold folds (nested evaluation)


def default_eval_mode(cv_group: str) -> str:
    """Evaluation protocol a run falls back to when ``--eval`` is not given.

    Parameters
    ----------
    cv_group : str
        The run's grouping column, ``"instance_id"`` or ``"well_id"``.

    Returns
    -------
    str
        ``"holdout"`` for instance grouping, ``"leave-one-out"`` for well
        grouping (see ``EVAL_MODE_DEFAULTS``).
    """
    return EVAL_MODE_DEFAULTS[cv_group]


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

# Canvas of an exported tree figure (see ``interpretation.export_tree``). The
# canvas is sized from the tree itself: the nodes are drawn once on a probe
# figure, the widest and tallest node box are measured, and the figure is
# then made exactly as wide and tall as it takes for two neighbouring nodes
# — the closest pair on any level — to sit ``TREE_FIGURE_NODE_GAP`` apart, so
# no two boxes overlap however deep the tree gets. Real trees are nowhere near
# full (the depth-12 tree of a `--model dt` run has 171 leaves, not 4096), so
# the width follows the leaf count, not the depth. The pixel cap keeps a large
# tree within what matplotlib can rasterize (it refuses past 2**16 px a side)
# by lowering the resolution rather than by giving up on the figure; when it
# bites, a vector PDF is written next to the PNG so nothing is lost.
TREE_FIGURE_NODE_GAP = 0.4  # inches of clear space between neighbouring node boxes
TREE_FIGURE_MIN_WIDTH = 14.0  # inches; keeps a tiny tree's title legible
TREE_FIGURE_DPI = 150
TREE_FIGURE_MAX_PIXELS = 60_000
SHAP_SAMPLE = 5_000  # windows sampled for SHAP
PERM_SAMPLE = 10_000  # windows sampled for permutation importance
PERM_REPEATS = 10  # shuffles per feature in permutation importance
