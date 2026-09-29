"""The feature dataset, rebuilt: 3W's real instances merged into recordings, then windowed.

Two steps, each with a manifest of its own (see ``ideas/2026-09-28_new_feature_dataset.md``):

``reconstruction``
    joins the real instances of every well that overlap or touch in time into
    one recording, gives the short unlabelled gaps the class of their flanks,
    splits the recordings whose instances came from different fault folders,
    and writes the result in 3W's own layout under ``data/merged*/`` with a
    manifest that carries every recording's events and provenance.
``extraction``
    cuts the recordings (and the untouched simulated and hand-drawn instances)
    into windows of a configurable length and overlap and writes one features
    parquet per length, with the window statistics, the instrument flags, the
    causal running statistics that later normalizations are built from, and
    the label-derived metadata a model never sees.
``manifest``
    the small helpers both steps share.

This package is the prototype of the module that replaces
``scripts/01_build_features.py``; the current stage 1 keeps working untouched
until the pipeline overhaul retires it.
"""
