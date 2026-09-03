"""Streamlined flow-assurance ML pipeline for the Petrobras 3W dataset.

Modules
-------
config
    Paths, sensor lists, class maps, and pipeline constants.
preprocessing
    Raw-parquet loading, cleaning, per-instance normalization, signal filters.
features
    Sliding-window statistical feature extraction and labeling.
training
    Dataset assembly per task, model factories, grouped CV search, OOF predictions.
evaluation
    Metric computation, per-class reports, confusion-matrix plotting.
interpretation
    Feature-importance rankings (MDI, permutation, XGBoost gain, SHAP).
"""
