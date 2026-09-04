"""Dataset assembly, model construction, hyperparameter search, and evaluation.

Two tasks share the same features parquet:

- ``detection`` — classify the *current* operational state of a window
  (``window_label``: 17 classes).
- ``prediction`` — from windows of *normal operation only*
  (``window_label == 0``), predict which fault the well will develop
  (``fault_class``: 8 classes; faults 3 and 4 have no recorded normal period).

Every split is grouped by ``instance_id`` — or by ``well_id`` with
``--cv-group well_id`` — so windows of the same time series (or well) never
appear on both sides of any split. Imputation lives inside the model pipeline
and is therefore refit on each training set — no information leaks across
splits.

Data used to *select* hyperparameters is kept separate from data used to
*evaluate* the selected model, in one of two ways:

- ``holdout`` (default) — a grouped test set is split off first and never
  touches the search; the search cross-validates on the remainder and the
  refit winner is scored once on the test set.
- ``nested`` — an outer grouped CV whose every fold runs its own inner search;
  the out-of-fold predictions of the per-fold winners form the evaluation.
  Unbiased and uses all data for evaluation, at roughly ``N_SPLITS_OUTER``
  times the cost.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from flowml.config import (
    CLASS_GROUPINGS,
    CUSTOM_CLASS_GROUPING,
    CV_GROUPING,
    CV_GROUPINGS,
    FAULT_CLASSES,
    HYDRATE_CLASS_GROUPING,
    META_COLS,
    N_ITER_SEARCH,
    N_JOBS,
    N_SPLITS_CV,
    N_SPLITS_OUTER,
    RANDOM_STATE,
    RF_PARAM_GRID,
    TEST_SIZE,
    WINDOW_CLASSES,
    XGB_PARAM_GRID,
    features_path,
)

TASKS = ("prediction", "detection")
MODEL_TYPES = ("rf", "xgb")

_GROUPING_MAPS = {
    "hydrate": HYDRATE_CLASS_GROUPING,
    "custom": CUSTOM_CLASS_GROUPING,
}


def _grouping_map(grouping: str) -> dict[int, str]:
    """Validate a grouping name and return its fault-class -> group-name map.

    Parameters
    ----------
    grouping : str
        ``"hydrate"`` or ``"custom"`` (``"none"`` has no map).

    Returns
    -------
    dict[int, str]
        The class-to-group-name mapping from ``config``.
    """
    if grouping not in CLASS_GROUPINGS:
        raise ValueError(f"Unknown grouping: {grouping!r} (expected {CLASS_GROUPINGS})")
    mapping = _GROUPING_MAPS[grouping]
    if grouping == "custom" and not all(mapping.values()):
        raise ValueError(
            "CUSTOM_CLASS_GROUPING has empty group names. "
            "Fill it in src/flowml/config.py before using --class-grouping custom."
        )
    return mapping


def grouping_label_map(grouping: str) -> dict[int, str]:
    """Return the label-value -> group-name map produced by ``group_labels``.

    Group numbers come from a ``LabelEncoder`` fit on the group names, so they
    follow the alphabetical order of the names (e.g. for the hydrate grouping:
    0 = Hydrate, 1 = Normal, 2 = Other Problem).

    Parameters
    ----------
    grouping : str
        ``"hydrate"`` or ``"custom"``.

    Returns
    -------
    dict[int, str]
        Human-readable name per grouped label value.
    """
    mapping = _grouping_map(grouping)
    encoder = LabelEncoder().fit(list(mapping.values()))
    return {i: str(name) for i, name in enumerate(encoder.classes_)}


def group_labels(y: np.ndarray, grouping: str) -> np.ndarray:
    """Collapse fine-grained fault labels onto a coarser set of groups.

    The ``"hydrate"`` grouping answers the operational triage question —
    normal operation, a hydrate event, or some other flow-assurance problem —
    while ``"custom"`` applies the user-defined ``CUSTOM_CLASS_GROUPING``.
    Transient labels (101-109) fall in the group of their active counterpart,
    so the mapping works for both tasks. Group numbers are assigned by a
    ``LabelEncoder`` fit on the full set of group names (alphabetical order),
    so they are stable regardless of which groups appear in ``y``; use
    ``grouping_label_map`` to translate them back to names.

    Parameters
    ----------
    y : np.ndarray
        Original integer labels.
    grouping : str
        ``"none"`` (returns ``y`` unchanged), ``"hydrate"`` or ``"custom"``.

    Returns
    -------
    np.ndarray
        Grouped labels, same shape as ``y``.
    """
    if grouping == "none":
        return y
    mapping = _grouping_map(grouping)

    base = np.where(y >= 100, y - 100, y)
    names = np.vectorize(mapping.get)(base)
    encoder = LabelEncoder().fit(list(mapping.values()))
    return encoder.transform(names)


@dataclass
class TaskData:
    """A modeling-ready dataset for one task.

    Attributes
    ----------
    X : np.ndarray
        Feature matrix, one row per window (may contain NaN; the model
        pipeline imputes).
    y : np.ndarray
        Original integer labels (non-contiguous for detection).
    groups : np.ndarray
        Group key per row (``instance_id`` or ``well_id``), for grouped
        cross-validation.
    feature_cols : list[str]
        Feature column names, aligned with ``X``.
    label_map : dict[int, str]
        Human-readable name per label value.
    n_windows : int
        Number of rows in ``X``.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_cols: list[str]
    label_map: dict[int, str]
    n_windows: int


def load_task_data(task: str, normalized: bool = True, cv_group: str = CV_GROUPING) -> TaskData:
    """Load the features parquet and assemble the dataset for one task.

    Parameters
    ----------
    task : str
        ``"prediction"`` or ``"detection"``.
    normalized : bool
        Load the features built from per-instance z-scored sensors (default)
        or the raw ones.
    cv_group : str
        Metadata column used as the grouping key of every split:
        ``"instance_id"`` (default) keeps windows of one recording together;
        ``"well_id"`` additionally keeps all recordings of one well together.
        With ``"well_id"``, simulated and hand-drawn instances are dropped —
        they have no physical well to group by.

    Returns
    -------
    TaskData
        Feature matrix, labels, groups, and label names for the task.
    """
    if cv_group not in CV_GROUPINGS:
        raise ValueError(f"Unknown cv_group: {cv_group!r} (expected {CV_GROUPINGS})")
    path = features_path(normalized)
    if not path.exists():
        flag = "" if normalized else " --no-normalization"
        raise FileNotFoundError(
            f"{path} not found. Build it first:\n  uv run scripts/01_build_features.py{flag}"
        )
    df = pd.read_parquet(path)
    if cv_group not in df.columns:
        raise ValueError(
            f"{path} has no {cv_group!r} column; it predates the well-id metadata. "
            "Rebuild it:\n  uv run scripts/01_build_features.py"
        )
    if cv_group == "well_id":
        df = df[df["source_type"] == "WELL"]

    if task == "prediction":
        df = df[df["window_label"] == 0]
        label_col, label_map = "fault_class", FAULT_CLASSES
    elif task == "detection":
        label_col, label_map = "window_label", WINDOW_CLASSES
    else:
        raise ValueError(f"Unknown task: {task!r} (expected {TASKS})")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    return TaskData(
        X=df[feature_cols].to_numpy(),
        y=df[label_col].to_numpy(),
        groups=df[cv_group].to_numpy(),
        feature_cols=feature_cols,
        label_map=label_map,
        n_windows=len(df),
    )


def make_pipeline(model_type: str, n_jobs: int = N_JOBS) -> tuple[Pipeline, dict]:
    """Build the imputer + classifier pipeline and its search space.

    RF balances classes through ``class_weight``; XGBoost has no such
    constructor option for the multiclass objective, so balanced sample
    weights are passed at fit time instead (see ``search_hyperparameters``).

    Parameters
    ----------
    model_type : str
        ``"rf"`` or ``"xgb"``.
    n_jobs : int
        Parallel workers for the classifier.

    Returns
    -------
    (Pipeline, dict)
        The sklearn pipeline and the hyperparameter grid for the search.
    """
    if model_type == "rf":
        clf = RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=n_jobs
        )
        grid = RF_PARAM_GRID
    elif model_type == "xgb":
        clf = XGBClassifier(
            objective="multi:softmax",
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbosity=0,
        )
        grid = XGB_PARAM_GRID
    else:
        raise ValueError(f"Unknown model type: {model_type!r} (expected {MODEL_TYPES})")

    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
    return pipe, grid


def search_hyperparameters(
    model_type: str,
    data: TaskData,
    encoder: LabelEncoder,
    n_jobs: int = N_JOBS,
    verbose: bool = False,
) -> RandomizedSearchCV:
    """Run a grouped randomized hyperparameter search optimizing F1-macro.

    Parameters
    ----------
    model_type : str
        ``"rf"`` or ``"xgb"``.
    data : TaskData
        Dataset returned by ``load_task_data``.
    encoder : LabelEncoder
        Fitted encoder mapping original labels to the contiguous range
        XGBoost requires (harmless for RF).
    n_jobs : int
        Parallel workers for the classifier and the search.
    verbose : bool
        Print per-candidate search progress (default off).

    Returns
    -------
    RandomizedSearchCV
        The fitted search object; ``best_estimator_`` is the refit pipeline.
    """
    pipe, grid = make_pipeline(model_type, n_jobs)
    y_enc = encoder.transform(data.y)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=grid,
        n_iter=N_ITER_SEARCH,
        cv=GroupKFold(n_splits=N_SPLITS_CV),
        scoring="f1_macro",
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        verbose=2 if verbose else 0,
        refit=True,
        return_train_score=True,
    )

    fit_params = {}
    if model_type == "xgb":
        fit_params["clf__sample_weight"] = compute_sample_weight("balanced", y_enc)

    search.fit(data.X, y_enc, groups=data.groups, **fit_params)
    return search


def subset_task_data(data: TaskData, idx: np.ndarray) -> TaskData:
    """Restrict a task dataset to the rows at ``idx``.

    Parameters
    ----------
    data : TaskData
        Dataset returned by ``load_task_data``.
    idx : np.ndarray
        Row indices to keep.

    Returns
    -------
    TaskData
        The restricted dataset (feature columns and label map unchanged).
    """
    return replace(
        data, X=data.X[idx], y=data.y[idx], groups=data.groups[idx], n_windows=len(idx)
    )


def holdout_split(data: TaskData) -> tuple[np.ndarray, np.ndarray]:
    """Split off the seeded grouped test set shared by every holdout consumer.

    The split depends only on the groups and ``RANDOM_STATE``, so every stage
    that calls it on the same dataset (same task, normalization, and CV
    grouping) holds out exactly the same groups — the ensemble and the
    distilled tree are judged on identical test data.

    Parameters
    ----------
    data : TaskData
        Dataset returned by ``load_task_data``.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Row indices of the train+val part and of the test part.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return next(splitter.split(data.X, data.y, data.groups))


def holdout_evaluation(
    model_type: str, data: TaskData, n_jobs: int = N_JOBS, verbose: bool = False
) -> tuple[RandomizedSearchCV, LabelEncoder, pd.DataFrame, dict]:
    """Grouped holdout: search on train+val, score the winner once on test.

    A grouped test set (``TEST_SIZE`` of the groups, seeded split) is carved
    off before anything else and never touches the hyperparameter search, so
    its score is an unbiased estimate of the selected model's generalization.
    The returned model is the search winner refit on all of train+val — the
    exact model the test score describes.

    Parameters
    ----------
    model_type : str
        ``"rf"`` or ``"xgb"``.
    data : TaskData
        Dataset returned by ``load_task_data``.
    n_jobs : int
        Parallel workers for the classifier and the search.
    verbose : bool
        Print per-candidate search progress (default off).

    Returns
    -------
    (RandomizedSearchCV, LabelEncoder, pd.DataFrame, dict)
        The fitted search (on train+val), its label encoder, the test
        predictions (``group``, ``fold`` = 1, ``y_true``, ``y_pred``), and a
        summary of the split sizes.
    """
    trainval_idx, test_idx = holdout_split(data)
    trainval = subset_task_data(data, trainval_idx)

    encoder = LabelEncoder().fit(trainval.y)
    search = search_hyperparameters(model_type, trainval, encoder, n_jobs, verbose)

    y_pred = encoder.inverse_transform(search.best_estimator_.predict(data.X[test_idx]))
    eval_frame = pd.DataFrame(
        {
            "group": data.groups[test_idx],
            "fold": 1,
            "y_true": data.y[test_idx],
            "y_pred": y_pred,
        }
    )
    info = {
        "test_size": TEST_SIZE,
        "n_trainval_groups": int(pd.Series(trainval.groups).nunique()),
        "n_test_groups": int(pd.Series(data.groups[test_idx]).nunique()),
        "n_trainval_windows": len(trainval_idx),
        "n_test_windows": len(test_idx),
    }
    return search, encoder, eval_frame, info


def nested_evaluation(
    model_type: str, data: TaskData, n_jobs: int = N_JOBS, verbose: bool = False
) -> tuple[pd.DataFrame, list[dict]]:
    """Nested grouped CV: every outer fold runs its own inner search.

    Each outer training set selects hyperparameters with its own inner
    ``GroupKFold`` search and predicts its held-out outer fold, so no window
    is ever predicted by a model whose hyperparameters saw it. Unbiased and
    uses all data for evaluation, at roughly ``N_SPLITS_OUTER`` times the
    cost of the holdout protocol. Note the selected hyperparameters may
    differ between folds — the evaluation describes the *procedure*, not one
    fixed configuration.

    Parameters
    ----------
    model_type : str
        ``"rf"`` or ``"xgb"``.
    data : TaskData
        Dataset returned by ``load_task_data``.
    n_jobs : int
        Parallel workers for the classifier and the searches.
    verbose : bool
        Print per-candidate search progress (default off).

    Returns
    -------
    (pd.DataFrame, list[dict])
        Outer-fold predictions over all data (``group``, ``fold``,
        ``y_true``, ``y_pred``) and one record per outer fold with the
        selected parameters and its validation/test F1-macro.
    """
    outer = GroupKFold(n_splits=N_SPLITS_OUTER)
    parts, records = [], []

    for fold, (train_idx, test_idx) in enumerate(
        outer.split(data.X, data.y, data.groups), start=1
    ):
        print(f"  Outer fold {fold}/{N_SPLITS_OUTER}: inner search...")
        train = subset_task_data(data, train_idx)
        encoder = LabelEncoder().fit(train.y)
        search = search_hyperparameters(model_type, train, encoder, n_jobs, verbose)

        y_pred = encoder.inverse_transform(search.best_estimator_.predict(data.X[test_idx]))
        y_true = data.y[test_idx]
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"    val F1-macro = {search.best_score_:.4f} | test F1-macro = {f1:.4f}")

        parts.append(
            pd.DataFrame(
                {
                    "group": data.groups[test_idx],
                    "fold": fold,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        )
        records.append(
            {
                "fold": fold,
                "best_params": {
                    k.removeprefix("clf__"): v for k, v in search.best_params_.items()
                },
                "val_f1_macro": round(float(search.best_score_), 4),
                "test_f1_macro": round(float(f1), 4),
                "n_test_windows": len(test_idx),
            }
        )

    return pd.concat(parts, ignore_index=True), records
