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
"""
