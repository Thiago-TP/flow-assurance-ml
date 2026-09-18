"""Streamlined flow-assurance ML pipeline for the Petrobras 3W dataset.

Modules
-------
config
    Paths, sensor lists, class maps, and pipeline constants.
preprocessing
    Raw-parquet loading, cleaning, per-instance normalization.
features
    Sliding-window statistical feature extraction and labeling.
train_val_test
    Dataset assembly per task, model factories, grouped CV search, held-out evaluation.
evaluation
    Metric computation, per-class reports, confusion-matrix plotting.
interpretation
    Feature-importance rankings (MDI, permutation, XGBoost gain, SHAP).
visualization
    Raw-data plots: per-fault instance histories.

Every figure this package draws is written to a file; nothing is ever shown
interactively. The matplotlib backend is therefore pinned to Agg here, once,
before any submodule imports ``pyplot``. Without it matplotlib picks an
interactive backend when a display is present, and a large canvas — the
depth-12 trees of the stage-5 sweep need tens of thousands of pixels a side
(see ``interpretation.tree_canvas_size``) — fails with ``X Error: BadAlloc``
when the X server cannot allocate the pixmap. Agg renders the same figure in
process memory instead. Callers that do want another backend can still set
one after importing ``flowml``.
"""

import matplotlib

matplotlib.use("Agg")
