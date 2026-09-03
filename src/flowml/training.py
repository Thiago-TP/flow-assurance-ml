"""Dataset assembly, model construction, hyperparameter search, and OOF predictions.

Two tasks share the same features parquet:

- ``detection`` — classify the *current* operational state of a window
  (``window_label``: 17 classes).
- ``prediction`` — from windows of *normal operation only*
  (``window_label == 0``), predict which fault the well will develop
  (``fault_class``: 8 classes; faults 3 and 4 have no recorded normal period).

All validation is grouped by ``instance_id`` (GroupKFold), so windows of the
same time series never appear in train and test simultaneously. Imputation lives
inside the model pipeline and is therefore refit on each training fold —
no information leaks across folds.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from flowml.config import (
    CLASS_GROUPINGS,
    CUSTOM_GROUPING,
    FAULT_CLASSES,
    FEATURES_PATH,
    HYDRATE_GROUPING,
    META_COLS,
    N_ITER_SEARCH,
    N_JOBS,
    N_SPLITS_CV,
    RANDOM_STATE,
    RF_PARAM_GRID,
    WINDOW_CLASSES,
    XGB_PARAM_GRID,
)

TASKS = ("prediction", "detection")
MODEL_TYPES = ("rf", "xgb")


def group_labels(y: np.ndarray, grouping: str) -> np.ndarray:
    """Collapse fine-grained fault labels onto a coarser set of groups.

    The ``"hydrate"`` grouping answers the operational triage question —
    normal operation, a hydrate event, or some other flow-assurance problem —
    by mapping 0 to 0, faults 8 and 9 to 2, and every other fault to 1.
    Transient labels (101-109) fall in the group of their active counterpart,
    so the mapping works for both tasks.

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
    if grouping not in CLASS_GROUPINGS:
        raise ValueError(f"Unknown grouping: {grouping!r} (expected {CLASS_GROUPINGS})")
    if grouping == "none":
        return y
    if grouping == "hydrate":
        maping = HYDRATE_GROUPING
    if grouping == "custom":
        maping = CUSTOM_GROUPING

    base = np.where(y >= 100, y - 100, y)
    names = np.vectorize(maping.get)(base)
    numbers = LabelEncoder().fit(names).transform(names)
    return numbers


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
        ``instance_id`` per row, for grouped cross-validation.
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


def load_task_data(task: str) -> TaskData:
    """Load the features parquet and assemble the dataset for one task.

    Parameters
    ----------
    task : str
        ``"prediction"`` or ``"detection"``.

    Returns
    -------
    TaskData
        Feature matrix, labels, groups, and label names for the task.
    """
    path = FEATURES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:\n  uv run scripts/01_build_features.py "
        )
    df = pd.read_parquet(path)

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
        groups=df["instance_id"].to_numpy(),
        feature_cols=feature_cols,
        label_map=label_map,
        n_windows=len(df),
    )


def make_pipeline(model_type: str) -> tuple[Pipeline, dict]:
    """Build the imputer + classifier pipeline and its search space.

    RF balances classes through ``class_weight``; XGBoost has no such
    constructor option for the multiclass objective, so balanced sample
    weights are passed at fit time instead (see ``oof_predict``).

    Parameters
    ----------
    model_type : str
        ``"rf"`` or ``"xgb"``.

    Returns
    -------
    (Pipeline, dict)
        The sklearn pipeline and the hyperparameter grid for the search.
    """
    if model_type == "rf":
        clf = RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS
        )
        grid = RF_PARAM_GRID
    elif model_type == "xgb":
        clf = XGBClassifier(
            objective="multi:softmax",
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=0,
        )
        grid = XGB_PARAM_GRID
    else:
        raise ValueError(f"Unknown model type: {model_type!r} (expected {MODEL_TYPES})")

    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
    return pipe, grid


def search_hyperparameters(
    model_type: str, data: TaskData, encoder: LabelEncoder
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

    Returns
    -------
    RandomizedSearchCV
        The fitted search object; ``best_estimator_`` is the refit pipeline.
    """
    pipe, grid = make_pipeline(model_type)
    y_enc = encoder.transform(data.y)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=grid,
        n_iter=N_ITER_SEARCH,
        cv=GroupKFold(n_splits=N_SPLITS_CV),
        scoring="f1_macro",
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=2,
        refit=True,
        return_train_score=True,
    )

    fit_params = {}
    if model_type == "xgb":
        fit_params["clf__sample_weight"] = compute_sample_weight("balanced", y_enc)

    search.fit(data.X, y_enc, groups=data.groups, **fit_params)
    return search


def oof_predict(
    model_type: str,
    best_params: dict,
    data: TaskData,
    encoder: LabelEncoder,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate out-of-fold predictions with the best hyperparameters.

    Each fold refits a fresh pipeline on its training split (imputer included)
    and predicts the held-out instances. For XGBoost, balanced sample weights
    are computed on each training fold — not globally — so the class balance
    of the held-out fold never influences training.

    Parameters
    ----------
    model_type : str
        ``"rf"`` or ``"xgb"``.
    best_params : dict
        Pipeline-prefixed parameters (``clf__*``) from the search.
    data : TaskData
        Dataset returned by ``load_task_data``.
    encoder : LabelEncoder
        Fitted label encoder for the task.
    verbose : bool
        Print per-fold F1-macro.

    Returns
    -------
    pd.DataFrame
        One row per window: ``instance_id``, ``fold``, ``y_true``, ``y_pred``
        (original label values).
    """
    y_enc = encoder.transform(data.y)
    gkf = GroupKFold(n_splits=N_SPLITS_CV)
    parts = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(data.X, y_enc, data.groups), start=1):
        pipe, _ = make_pipeline(model_type)
        pipe.set_params(**best_params)

        fit_params = {}
        if model_type == "xgb":
            fit_params["clf__sample_weight"] = compute_sample_weight("balanced", y_enc[train_idx])
        pipe.fit(data.X[train_idx], y_enc[train_idx], **fit_params)

        y_pred = encoder.inverse_transform(pipe.predict(data.X[test_idx]))
        y_true = data.y[test_idx]
        parts.append(
            pd.DataFrame(
                {
                    "instance_id": data.groups[test_idx],
                    "fold": fold,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        )
        if verbose:
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            print(f"  Fold {fold}: F1-macro = {f1:.4f}")

    return pd.concat(parts, ignore_index=True)
