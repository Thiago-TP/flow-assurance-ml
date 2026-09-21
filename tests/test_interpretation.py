"""Tests for the tree-export helpers: pruning, measuring, and canvas sizing."""

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from flowml.interpretation import (
    export_tree,
    prune_redundant_splits,
    tree_shape,
    walk_tree,
)


def degenerate_tree():
    """A tree with a split whose two leaves carry the same class.

    The grown tree is::

        a <= 0.5            -> class 0
        a >  0.5, b <= 0.5  -> class 1
        a >  0.5, b >  0.5, a <= 1.5 -> class 0
        a >  0.5, b >  0.5, a >  1.5 -> class 0

    The last split lowers impurity — it isolates the one class-1 sample that
    the branch cannot separate — without changing either side's majority, so
    it is exactly the kind of split ``prune_redundant_splits`` removes.

    Returns
    -------
    (DecisionTreeClassifier, np.ndarray)
        The fitted tree and the matrix it was fit on.
    """
    X = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    )
    y = np.array([1, 1, 0, 1, 1, 0, 0, 0])
    return DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y), X


def test_walk_tree_reaches_every_node_once():
    tree, _ = degenerate_tree()
    reached = walk_tree(tree)
    assert len(reached) == len({node for node, _ in reached})
    assert reached[0] == (0, 0)


def test_tree_shape_matches_sklearn_before_pruning():
    tree, _ = degenerate_tree()
    depth, leaves = tree_shape(tree)
    assert depth == tree.get_depth()
    assert leaves == tree.get_n_leaves()


def test_pruning_keeps_every_prediction():
    """The whole point: fewer splits, identical output.

    Label noise under a depth cap is what grows degenerate splits in the real
    runs — the tree keeps splitting to chase samples it cannot separate — so
    the fixture reproduces that rather than a clean problem.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(int)
    y[rng.choice(300, 40, replace=False)] ^= 1
    tree = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=0).fit(X, y)

    before = tree.predict(X)
    depth_before, leaves_before = tree_shape(tree)
    removed = prune_redundant_splits(tree)
    depth_after, leaves_after = tree_shape(tree)

    assert removed > 0
    assert np.array_equal(tree.predict(X), before)
    assert leaves_after == leaves_before - removed
    assert depth_after <= depth_before


def test_pruning_collapses_a_degenerate_split():
    tree, X = degenerate_tree()
    assert tree_shape(tree) == (3, 4)
    before = tree.predict(X)

    assert prune_redundant_splits(tree) == 1

    assert tree_shape(tree) == (2, 3)
    assert np.array_equal(tree.predict(X), before)
    assert np.array_equal(before, np.array([1, 1, 0, 1, 0, 0, 0, 0]))


def test_pruning_is_idempotent():
    tree, _ = degenerate_tree()
    prune_redundant_splits(tree)
    assert prune_redundant_splits(tree) == 0


def test_pruning_leaves_a_single_leaf_tree_alone():
    tree = DecisionTreeClassifier().fit(np.array([[0.0], [1.0]]), np.array([1, 1]))
    assert prune_redundant_splits(tree) == 0
    assert tree_shape(tree) == (0, 1)


@pytest.mark.parametrize("prune", [True, False])
def test_export_tree_writes_rules_and_figure(tmp_path, prune):
    tree, _ = degenerate_tree()

    written = export_tree(
        tree,
        ["a", "b"],
        {0: "Zero", 1: "One"},
        "t",
        metrics_dir=tmp_path / "metrics",
        figures_dir=tmp_path / "figures",
        prune=prune,
    )

    assert written == [tmp_path / "metrics" / "t_rules.txt", tmp_path / "figures" / "t_tree.png"]
    rules = (tmp_path / "metrics" / "t_rules.txt").read_text(encoding="utf-8")
    assert "Zero" in rules and "One" in rules
    assert (tmp_path / "figures" / "t_tree.png").exists()
    # Pruning folds the split whose two leaves are both "Zero" into one leaf.
    assert rules.count("class: Zero") == (2 if prune else 3)
    assert rules.count("class: One") == 1
