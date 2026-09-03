"""Metric computation and evaluation plots from out-of-fold predictions.

Works purely on the OOF prediction table produced by the training script, so
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
        "f1_macro": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }


def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_map: dict[int, str]
) -> dict:
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


def per_fold_metrics(oof: pd.DataFrame) -> dict:
    """Compute F1-macro per CV fold from the OOF table.

    Parameters
    ----------
    oof : pd.DataFrame
        OOF predictions with columns ``fold``, ``y_true``, ``y_pred``.

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
        for fold, g in oof.groupby("fold")
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
