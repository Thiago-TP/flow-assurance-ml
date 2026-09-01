"""Feature-importance rankings and plots for trained tree models.

Four complementary views of what drives the predictions:

- MDI (RF only) — mean Gini-impurity decrease; fast but biased toward
  high-variance features.
- Gain / weight / cover (XGBoost only) — the booster's native split metrics.
- Permutation importance — F1-macro drop when a feature is shuffled;
  model-agnostic and robust to the MDI bias, but expensive.
- SHAP — mean |SHAP value| over samples and classes; the reference global
  ranking used to pick features for downstream compact decision trees.

Every method returns a full descending ranking (feature -> score) so the
consumer decides how many features to keep.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from flowml.config import (
    N_JOBS,
    PERM_REPEATS,
    PERM_SAMPLE,
    RANDOM_STATE,
    SHAP_SAMPLE,
    TOP_N_FEATURES,
)


def _subsample(X: np.ndarray, size: int) -> np.ndarray:
    """Draw a reproducible random row sample from ``X``.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix.
    size : int
        Sample size (capped at ``len(X)``).

    Returns
    -------
    np.ndarray
        The sampled rows.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(size, len(X)), replace=False)
    return X[idx], idx


def mdi_importance(rf, feature_cols: list[str]) -> pd.Series:
    """Rank features by the Random Forest's mean decrease in impurity.

    Parameters
    ----------
    rf : RandomForestClassifier
        Fitted forest (the bare classifier, not the pipeline).
    feature_cols : list[str]
        Feature names aligned with the training matrix.

    Returns
    -------
    pd.Series
        Importance per feature, descending.
    """
    return pd.Series(rf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )


def xgb_importance(xgb, feature_cols: list[str]) -> dict[str, pd.Series]:
    """Rank features by XGBoost's native gain, weight, and cover metrics.

    Parameters
    ----------
    xgb : XGBClassifier
        Fitted booster (the bare classifier, not the pipeline).
    feature_cols : list[str]
        Feature names aligned with the training matrix.

    Returns
    -------
    dict[str, pd.Series]
        ``{"gain": ..., "weight": ..., "cover": ...}``, each descending;
        features never used in a split score 0.
    """
    booster = xgb.get_booster()
    booster.feature_names = feature_cols
    return {
        imp_type: pd.Series(
            {
                f: booster.get_score(importance_type=imp_type).get(f, 0.0)
                for f in feature_cols
            }
        ).sort_values(ascending=False)
        for imp_type in ("gain", "weight", "cover")
    }


def permutation_ranking(
    clf, X_imputed: np.ndarray, y: np.ndarray, feature_cols: list[str]
) -> tuple[pd.Series, np.ndarray]:
    """Rank features by F1-macro drop under feature shuffling.

    Runs on a random subsample of ``PERM_SAMPLE`` windows with
    ``PERM_REPEATS`` shuffles per feature.

    Parameters
    ----------
    clf : estimator
        Fitted classifier expecting imputed input.
    X_imputed : np.ndarray
        Feature matrix after imputation (what the classifier was trained on).
    y : np.ndarray
        Labels in the encoding the classifier was fit with.
    feature_cols : list[str]
        Feature names aligned with the matrix.

    Returns
    -------
    (pd.Series, np.ndarray)
        Mean importance per feature (descending) and the raw
        ``(n_features, n_repeats)`` importance matrix in ``feature_cols`` order.
    """
    X_sample, idx = _subsample(X_imputed, PERM_SAMPLE)
    result = permutation_importance(
        clf,
        X_sample,
        y[idx],
        n_repeats=PERM_REPEATS,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    mean = pd.Series(result.importances_mean, index=feature_cols).sort_values(
        ascending=False
    )
    return mean, result.importances


def shap_ranking(clf, X_imputed: np.ndarray, feature_cols: list[str]) -> pd.Series:
    """Rank features by mean |SHAP value| over samples and classes.

    Uses ``shap.TreeExplainer`` on a random subsample of ``SHAP_SAMPLE``
    windows. Note that RF SHAP values live in probability space and XGBoost
    (``multi:softmax``) SHAP values in log-odds space: rankings are comparable
    across models, absolute magnitudes are not.

    Parameters
    ----------
    clf : estimator
        Fitted tree model (bare classifier).
    X_imputed : np.ndarray
        Imputed feature matrix.
    feature_cols : list[str]
        Feature names aligned with the matrix.

    Returns
    -------
    pd.Series
        Mean |SHAP| per feature, descending.
    """
    X_sample, _ = _subsample(X_imputed, SHAP_SAMPLE)
    explainer = shap.TreeExplainer(clf)
    values = np.abs(explainer(X_sample).values)  # (n_samples, n_features, n_classes)
    scores = values.mean(axis=(0, 2))
    return pd.Series(scores, index=feature_cols).sort_values(ascending=False)


def plot_ranking(
    ranking: pd.Series, title: str, xlabel: str, out_path, cmap="Blues_r"
) -> None:
    """Save a horizontal bar plot of the top-N features of a ranking.

    Parameters
    ----------
    ranking : pd.Series
        Descending feature ranking.
    title : str
        Figure title.
    xlabel : str
        X-axis label describing the score.
    out_path : Path
        Destination PNG.
    cmap : str
        Matplotlib colormap for the bars.
    """
    top = ranking.head(TOP_N_FEATURES)
    _, ax = plt.subplots(figsize=(9, 6))
    colors = plt.get_cmap(cmap)(np.linspace(0.25, 0.80, len(top)))
    top[::-1].plot(kind="barh", ax=ax, color=colors[::-1])

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_permutation_boxplot(
    importances: np.ndarray, feature_cols: list[str], title: str, out_path
) -> None:
    """Save a boxplot of permutation-importance spread for the top-N features.

    Boxes crossing zero flag features whose importance is unstable across
    shuffles and probably unreliable.

    Parameters
    ----------
    importances : np.ndarray
        Raw ``(n_features, n_repeats)`` matrix in ``feature_cols`` order.
    feature_cols : list[str]
        Feature names aligned with the matrix rows.
    title : str
        Figure title.
    out_path : Path
        Destination PNG.
    """
    mean = pd.Series(importances.mean(axis=1), index=feature_cols)
    top = mean.nlargest(TOP_N_FEATURES).index.tolist()
    idx = {f: i for i, f in enumerate(feature_cols)}
    matrix = importances[[idx[f] for f in top], :]

    _, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(
        matrix[::-1].T,
        vert=False,
        patch_artist=True,
        tick_labels=top[::-1],
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )
    colors = plt.get_cmap("Blues_r")(np.linspace(0.25, 0.80, len(top)))
    for patch, color in zip(bp["boxes"], colors[::-1]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("F1-macro drop after shuffling", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
