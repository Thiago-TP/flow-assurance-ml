"""Metric computation and evaluation plots from held-out predictions.

Works purely on the evaluation table produced by the train/val/test stage, so
evaluation never refits a model.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the headline classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    dict
        ``f1_macro``, ``f1_weighted``, and ``accuracy``, rounded to 4 decimals.
    """
    return {
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_map: dict[int, str]) -> dict:
    """Compute precision/recall/F1/support per class.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    label_map : dict[int, str]
        Human-readable name per label value.

    Returns
    -------
    dict
        ``{str(label): {name, precision, recall, f1, support}}`` for every
        label present in the data.
    """
    labels = sorted(set(y_true) | set(y_pred))
    names = [label_map.get(c, str(c)) for c in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    return {
        str(c): {
            "name": name,
            "precision": round(report[name]["precision"], 4),
            "recall": round(report[name]["recall"], 4),
            "f1": round(report[name]["f1-score"], 4),
            "support": int(report[name]["support"]),
        }
        for c, name in zip(labels, names)
    }


# A per-group score sheet is printed and stored whenever the evaluation table
# has at most this many groups — always the case with wells as groups (3W has
# 42), never with instances (a thousand). See ``per_group_metrics``.
PER_GROUP_MAX_GROUPS = 50


def per_group_metrics(preds: pd.DataFrame, max_groups: int = PER_GROUP_MAX_GROUPS) -> dict:
    """Score every group of the evaluation table on its own.

    A pooled score hides which groups fail: with wells as groups, one well
    holding half the windows decides the pooled number, and a well predicted
    entirely wrong is invisible if it is small. The sheet is only built when
    the groups are few enough to read (``max_groups``), which is the well
    case; with a thousand instances it returns empty.

    Parameters
    ----------
    preds : pd.DataFrame
        Held-out predictions with columns ``group``, ``y_true``, ``y_pred``.
    max_groups : int
        Skip the sheet when the table has more groups than this.

    Returns
    -------
    dict
        ``{str(group): {f1_macro, accuracy, n_windows, n_classes, classes}}``
        in group order (numeric when the groups are numbers), or ``{}``.
        ``classes`` are the true labels present, so a one-class group's
        macro F1 can be read for what it is.
    """
    if preds["group"].nunique() > max_groups:
        return {}

    def key(group) -> tuple:
        text = str(group)
        return (not text.isdigit(), int(text) if text.isdigit() else text)

    sheet = {}
    for group in sorted(preds["group"].unique(), key=key):
        g = preds[preds["group"] == group]
        sheet[str(group)] = {
            "f1_macro": round(
                float(f1_score(g["y_true"], g["y_pred"], average="macro", zero_division=0)), 4
            ),
            "accuracy": round(float(accuracy_score(g["y_true"], g["y_pred"])), 4),
            "n_windows": len(g),
            "n_classes": int(g["y_true"].nunique()),
            "classes": sorted(int(c) for c in g["y_true"].unique()),
        }
    return sheet


def per_fold_metrics(preds: pd.DataFrame) -> dict:
    """Compute F1-macro per fold from the evaluation table.

    The holdout protocol has a single fold; the nested protocol one per outer
    fold; leave-one-out one per group.

    Parameters
    ----------
    preds : pd.DataFrame
        Held-out predictions with columns ``fold``, ``y_true``, ``y_pred``.

    Returns
    -------
    dict
        ``{"fold_<k>": f1_macro}`` plus mean and standard deviation across folds.
    """
    scores = {
        f"fold_{fold}": round(
            float(f1_score(g["y_true"], g["y_pred"], average="macro", zero_division=0)),
            4,
        )
        for fold, g in preds.groupby("fold")
    }
    values = list(scores.values())
    scores["mean"] = round(float(np.mean(values)), 4)
    scores["std"] = round(float(np.std(values)), 4)
    return scores


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict[int, str],
    title: str,
    out_path,
) -> None:
    """Plot and save a row-normalized confusion matrix.

    Each cell shows the fraction of samples of the true class (row) predicted
    as the column class, i.e. the diagonal is per-class recall.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    label_map : dict[int, str]
        Human-readable name per label value.
    title : str
        Figure title.
    out_path : Path
        Destination PNG.
    """
    labels = sorted(set(y_true) | set(y_pred))
    names = [label_map.get(c, str(c)) for c in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(6, n * 0.8)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > 0.5 else "black",
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
