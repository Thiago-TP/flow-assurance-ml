import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure after each test so a failed assertion cannot leak them."""
    yield
    plt.close("all")


@pytest.fixture
def subplots():
    """Factory for a bare figure/axes pair, for tests that draw on an Axes directly."""

    def _make(nrows: int = 1):
        return plt.subplots(nrows, 1, squeeze=nrows == 1)

    return _make
