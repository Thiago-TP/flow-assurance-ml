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

A single decision tree needs none of them: it *is* its own explanation, so
``export_tree`` writes it out directly as rules and a figure. Both the
distilled tree of stage 5 and the ``--model dt`` run of stage 2 export
themselves that way.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text, plot_tree
from sklearn.tree._tree import TREE_LEAF as _TREE_LEAF
from sklearn.tree._tree import TREE_UNDEFINED as _TREE_UNDEFINED

from flowml.config import (
    FIGURES_DIR,
    METRICS_DIR,
    N_JOBS,
    PERM_REPEATS,
    PERM_SAMPLE,
    RANDOM_STATE,
    SHAP_SAMPLE,
    TOP_N_FEATURES,
    TREE_FIGURE_DPI,
    TREE_FIGURE_MAX_PIXELS,
    TREE_FIGURE_MIN_WIDTH,
    TREE_FIGURE_NODE_GAP,
)

# How every tree is drawn; shared by the probe drawing that measures the nodes
# and the final one, so what is measured is what is drawn.
_TREE_STYLE = {"filled": True, "rounded": True, "impurity": False, "fontsize": 8}


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
    return pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)


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
            {f: booster.get_score(importance_type=imp_type).get(f, 0.0) for f in feature_cols}
        ).sort_values(ascending=False)
        for imp_type in ("gain", "weight", "cover")
    }


def permutation_ranking(
    clf,
    X_imputed: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    n_jobs: int = N_JOBS,
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
    n_jobs : int
        Parallel workers for the shuffles.

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
        n_jobs=n_jobs,
    )
    mean = pd.Series(result.importances_mean, index=feature_cols).sort_values(ascending=False)
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


def walk_tree(tree) -> list[tuple[int, int]]:
    """Every node reachable from the root, as ``(node id, depth)`` pairs.

    ``prune_redundant_splits`` detaches subtrees by turning their parent into
    a leaf; it does not shrink ``tree_``'s arrays, so the orphaned entries
    stay behind. sklearn's own ``get_depth`` and ``get_n_leaves`` read those
    arrays flat and would keep counting them, which is why everything here
    measures a tree by walking it instead.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        Fitted tree (the bare classifier, not the pipeline).

    Returns
    -------
    list[(int, int)]
        One pair per reachable node, root first.
    """
    left, right = tree.tree_.children_left, tree.tree_.children_right
    reached, stack = [], [(0, 0)]
    while stack:
        node, depth = stack.pop()
        reached.append((node, depth))
        if left[node] != _TREE_LEAF:
            stack.extend(((left[node], depth + 1), (right[node], depth + 1)))
    return reached


def tree_shape(tree) -> tuple[int, int]:
    """Effective ``(depth, leaf count)`` of a tree, counting reachable nodes only.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        Fitted tree (the bare classifier, not the pipeline).

    Returns
    -------
    (int, int)
        Depth of the deepest reachable leaf, and how many leaves there are.
    """
    left = tree.tree_.children_left
    reached = walk_tree(tree)
    return max(d for _, d in reached), sum(1 for n, _ in reached if left[n] == _TREE_LEAF)


def prune_redundant_splits(tree) -> int:
    """Collapse every split whose whole subtree predicts a single class.

    A depth-limited tree fit for purity keeps splitting as long as a split
    lowers impurity, even when both sides end up predicting the same class:
    the instance-grouped tree of the first exploration report has four such
    sibling pairs, three of them for Rapid Productivity Loss. Those splits
    read as decisions the model makes and are not — they change nothing about
    the prediction — so they inflate the published depth and the reader's
    sense of the model's complexity.

    Each one is collapsed into its parent, bottom up, so a whole degenerate
    subtree folds into one leaf rather than one level at a time. The
    prediction of the new leaf is the class its subtree already agreed on:
    ``tree_.value`` at an internal node is the class distribution of the
    training samples that reach it, and summing distributions whose argmax is
    all the same ``c`` keeps ``c`` the argmax — so predictions are provably
    unchanged. The one way that could fail is an exact tie at the parent,
    where sklearn's argmax would pick the lower class index, so a node whose
    own argmax disagrees is left alone.

    The detached nodes stay in ``tree_``'s arrays, unreachable; use
    ``tree_shape`` rather than sklearn's ``get_depth`` / ``get_n_leaves`` to
    measure the result.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        Fitted tree (the bare classifier, not the pipeline), pruned in place.

    Returns
    -------
    int
        How many splits were collapsed.
    """
    inner = tree.tree_
    left, right, feature, threshold = (
        inner.children_left,
        inner.children_right,
        inner.feature,
        inner.threshold,
    )
    removed = 0

    def agreed_class(node: int) -> int | None:
        """The class the whole subtree predicts, or ``None`` if it disagrees."""
        nonlocal removed
        own = int(np.argmax(inner.value[node]))
        if left[node] == _TREE_LEAF:
            return own
        # Both sides are visited first, so a subtree folds bottom up.
        below = {agreed_class(left[node]), agreed_class(right[node])}
        if len(below) > 1 or None in below or own not in below:
            return None
        left[node] = right[node] = _TREE_LEAF
        feature[node] = _TREE_UNDEFINED
        threshold[node] = _TREE_UNDEFINED
        removed += 1
        return own

    agreed_class(0)
    return removed


def tree_canvas_size(tree, feature_names: list[str], class_names: list[str]) -> tuple[float, float]:
    """Figure size, in inches, at which no two node boxes of the tree overlap.

    ``plot_tree`` places the nodes at fixed fractions of the axes — siblings
    one slot apart, levels one row apart — and writes each node's text at a
    fixed point size, so a canvas too small for its text makes neighbouring
    boxes collide. The tree is therefore drawn once on a probe canvas, the
    widest and tallest box are measured in inches (text size does not depend
    on the canvas), and so is the closest pair of nodes on any level and the
    distance between levels, in axes fractions. The canvas that puts
    ``TREE_FIGURE_NODE_GAP`` of air between that closest pair, and between the
    rows, is the one returned; a tree whose boxes are already far apart still
    gets ``TREE_FIGURE_MIN_WIDTH`` so its title stays legible.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        Fitted tree (the bare classifier, not the pipeline).
    feature_names : list[str]
        Feature names aligned with the matrix the tree was fit on.
    class_names : list[str]
        Class names in ``tree.classes_`` order.

    Returns
    -------
    (float, float)
        Width and height of the figure, in inches.
    """
    probe, ax = plt.subplots(figsize=(10, 10), dpi=TREE_FIGURE_DPI)
    try:
        nodes = plot_tree(
            tree, feature_names=feature_names, class_names=class_names, ax=ax, **_TREE_STYLE
        )
        probe.canvas.draw()
        renderer = probe.canvas.get_renderer()
        boxes = [(node.get_bbox_patch() or node).get_window_extent(renderer) for node in nodes]
        box_w = max(box.width for box in boxes) / probe.dpi
        box_h = max(box.height for box in boxes) / probe.dpi
        positions = np.array([node.xyann for node in nodes])  # axes fractions
    finally:
        plt.close(probe)

    levels = np.unique(np.round(positions[:, 1], 6))
    gaps = [np.diff(np.sort(positions[np.isclose(positions[:, 1], level), 0])) for level in levels]
    row_gaps = np.concatenate([g for g in gaps if g.size]) if any(g.size for g in gaps) else None
    min_dx = float(row_gaps.min()) if row_gaps is not None else None
    min_dy = float(np.diff(levels).min()) if len(levels) > 1 else None

    width = (box_w + TREE_FIGURE_NODE_GAP) / min_dx if min_dx else box_w + 2 * TREE_FIGURE_NODE_GAP
    height = (box_h + TREE_FIGURE_NODE_GAP) / min_dy if min_dy else box_h + 2 * TREE_FIGURE_NODE_GAP
    return max(TREE_FIGURE_MIN_WIDTH, width), height + 2.0  # room for the title


def export_tree(
    tree, feature_names: list[str], label_map: dict[int, str], name: str, prune: bool = True
) -> None:
    """Write a fitted decision tree as plain-text rules and a figure.

    The tree explains itself, so no ranking is computed: the exported rules
    and the drawing *are* the model. Both are keyed to the class values the
    tree itself outputs, so ``label_map`` must be in the tree's own label
    space — encoded values when the tree was fit on encoded labels, the fine
    fault classes when the tree was fit on them even if its predictions are
    collapsed onto groups afterwards.

    Splits whose whole subtree predicts one class are collapsed first
    (``prune_redundant_splits``), **in place**: they change no prediction, so
    the exported depth and leaf count describe what the model actually
    decides rather than how hard it worked to get there. Callers that save
    the model should save it after this, so that the artifact and the drawing
    agree.

    Every tree is drawn, however large, on a canvas sized so that no two node
    boxes overlap (``tree_canvas_size``). Where that canvas exceeds what
    matplotlib can rasterize, the PNG's resolution drops to fit and a vector
    PDF is written beside it, so the tree stays readable at any zoom.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        Fitted tree (the bare classifier, not the pipeline).
    feature_names : list[str]
        Feature names aligned with the matrix the tree was fit on.
    label_map : dict[int, str]
        Human-readable name per class value in ``tree.classes_``.
    name : str
        Base name of the exported artifacts.
    prune : bool
        Collapse the redundant splits before exporting (default on).
    """
    grown_depth, grown_leaves = tree_shape(tree)
    removed = prune_redundant_splits(tree) if prune else 0
    depth, leaves = tree_shape(tree)
    if removed:
        print(
            f"  Pruned {removed} split{'s' if removed > 1 else ''} whose whole subtree "
            f"predicted one class: depth {grown_depth} -> {depth}, "
            f"{grown_leaves} -> {leaves} leaves, predictions unchanged."
        )
    class_names = [label_map.get(c, str(c)) for c in tree.classes_]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rules_path = METRICS_DIR / f"{name}_rules.txt"
    rules_path.write_text(
        # export_text truncates past its own default of 10 levels, which the
        # depth sweep now reaches; ask for the whole tree.
        export_text(
            tree, feature_names=feature_names, class_names=class_names, max_depth=depth + 1
        ),
        encoding="utf-8",
    )
    print(f"  Saved: {rules_path}")

    width, height = tree_canvas_size(tree, feature_names, class_names)
    dpi = min(TREE_FIGURE_DPI, TREE_FIGURE_MAX_PIXELS / max(width, height))
    print(
        f"  Tree figure: {leaves} leaves over {depth} levels drawn on "
        f"{width:.0f}x{height:.0f} in"
        + (
            f"; the resolution drops to {dpi:.0f} dpi to stay within what matplotlib can "
            "rasterize, so a vector PDF is written as well."
            if dpi < TREE_FIGURE_DPI
            else "."
        )
    )

    fig, ax = plt.subplots(figsize=(width, height))
    plot_tree(tree, feature_names=feature_names, class_names=class_names, ax=ax, **_TREE_STYLE)
    title = f"Decision tree (depth {depth}" + (
        f", pruned from {grown_depth}) — {name}" if removed else f") — {name}"
    )
    ax.set_title(title, fontsize=13, pad=10)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tree_path = FIGURES_DIR / f"{name}_tree.png"
    plt.savefig(tree_path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {tree_path}")
    if dpi < TREE_FIGURE_DPI:
        pdf_path = tree_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"  Saved: {pdf_path}")
    plt.close(fig)


def plot_ranking(ranking: pd.Series, title: str, xlabel: str, out_path, cmap="Blues_r") -> None:
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
