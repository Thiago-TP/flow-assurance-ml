"""Audit — how much of a normal-operation window is the well history, and how much is the coming fault.

The well-grouped protocol scores far below the instance-grouped one (see
``results/reports/2026-09-21_cv_grouping_and_well_generalization.md``). This
audit asks whether that is the features' doing, with four probes on the real
normal-operation windows of the prediction task and the features parquet as it
is — nothing is trained through the pipeline, and nothing is tuned:

1. **Can one window name its well?** A random forest guesses which well
   produced a single window, with the window's own recording held out
   (``GroupKFold`` by instance). Run on the pipeline's features under every normalization
   reference, on the dispersion statistics alone, on sensor-to-sensor level
   differences along the flow path, and on the instrument-availability flags
   alone (which sensors are frozen or absent). A feature set that generalizes
   across wells should score *low* here.
2. **Is the instance-grouped score a look-up?** Predict each recording's fault
   as the majority fault of the *other* recordings of its well — no features
   at all. This is what an instance-grouped model can achieve by recognizing
   the well.
3. **Is there a precursor at all?** Across wells a precursor is confounded
   with well identity; inside one well the identity is fixed. So, in the few
   wells that have both ordinary Normal recordings and recordings that end in
   a fault, a forest trained on the rest of the well tries to tell the
   held-out recording's normal-operation windows apart: ordinary normal, or
   normal before a fault? If it recognizes the pre-fault recordings, a
   precursor exists; running it on raw and on normalized features says
   whether the precursor lives in the levels or in the window shape.
4. **What transfers to a well never seen?** The same forest predicts the
   coming fault under 5-fold ``GroupKFold`` by well — the protocol of the
   pipeline's well grouping — once per feature set, against the chance
   baselines.

Note that a features parquet is required, so user is expected to have run
``scripts/01_build_features.py`` first (or else there is nothing to audit).

Usage
-----
    uv run scripts/audits/well_identifiability_auditing.py [--frozen-sensors {flag,keep,drop}]
        [--max-windows-per-well N] [--fault-max-windows-per-well N] [--n-estimators N]
        [--allow-overlap] [--keep-extreme-values] [--n-jobs N] [--output-file PATH] [--verbose]

Output
------
    results/audits/well_identifiability[_overlap][_extremes]_<timestamp>.txt
    (everything is printed to the terminal as well)
"""

import contextlib
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline

from flowml.cli import add_frozen_sensors_arg, run_parser
from flowml.config import (
    FAULT_CLASSES,
    KEY_SENSORS,
    META_COLS,
    NORMALIZATIONS,
    RANDOM_STATE,
    RESULTS_DIR,
    extreme_suffix,
    features_path,
    overlap_suffix,
)
from flowml.normalization import normalize_features, reference_columns
from flowml.sensor_health import apply_frozen_policy, frozen_windows

AUDITS_DIR = RESULTS_DIR / "audits"

# Statistics that carry a sensor's absolute level; the rest describe the window's shape.
LEVEL_SUFFIXES = ("_mean", "_min", "_max", "_median", "_max_zscore")
# Adjacent sensors along the flow path, upstream first: the differences are the pressure
# drop from the downhole gauge to the tree, from the tree to upstream of the production
# choke, and the temperature drops over the same stretches (the last one is the
# Joule-Thomson cooling across the choke).
FLOW_PATH_PAIRS = (
    ("P-PDG", "P-TPT"),
    ("P-TPT", "P-MON-CKP"),
    ("T-PDG", "T-TPT"),
    ("T-TPT", "T-JUS-CKP"),
)
DIFFERENCE_STATS = ("mean", "min", "max")
N_FOLDS = 5


class Tee:
    """Write everything to several streams, so the audit is both shown and saved."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def load_real_normal_windows(allow_overlap: bool, keep_extreme_values: bool) -> pd.DataFrame:
    """Read the features parquet and keep the prediction task's real windows.

    Parameters
    ----------
    allow_overlap : bool
        Read the ``_overlap`` parquet.
    keep_extreme_values : bool
        Read the ``_extremes`` parquet.

    Returns
    -------
    pd.DataFrame
        Raw window features plus metadata and reference columns, real
        (``WELL``) instances only, normal-operation windows only, index reset.
    """
    path = features_path(allow_overlap, keep_extreme_values)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first: uv run scripts/01_build_features.py"
        )
    df = pd.read_parquet(path)
    df = df[(df["source_type"] == "WELL") & (df["window_label"] == 0)]
    return df.reset_index(drop=True)


def pipeline_features(
    raw: pd.DataFrame, normalization: str, frozen_mode: str
) -> tuple[pd.DataFrame, list[str]]:
    """Apply one normalization reference and the frozen-sensor policy, as ``load_task_data`` does.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of ``load_real_normal_windows``.
    normalization : str
        One of ``NORMALIZATIONS``.
    frozen_mode : str
        One of ``FROZEN_MODES``. ``drop`` removes rows, which would put the
        probes on different windows, so it is applied as ``keep`` here.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The transformed frame (metadata kept) and its feature column names.
    """
    df = normalize_features(raw.copy(), normalization)
    df = apply_frozen_policy(df, "keep" if frozen_mode == "drop" else frozen_mode)
    excluded = set(META_COLS) | set(reference_columns(df))
    return df, [c for c in df.columns if c not in excluded]


def availability_flags(raw: pd.DataFrame) -> pd.DataFrame:
    """Binary instrument-status features: one frozen and one missing flag per key sensor.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw window features.

    Returns
    -------
    pd.DataFrame
        ``<sensor>_frozen`` (window standard deviation below the frozen
        threshold) and ``<sensor>_missing`` (no valid samples in the window).
    """
    flags = {}
    for sensor in KEY_SENSORS:
        flags[f"{sensor}_frozen"] = frozen_windows(raw, sensor).astype(float)
        flags[f"{sensor}_missing"] = raw[f"{sensor}_mean"].isna().astype(float)
    return pd.DataFrame(flags, index=raw.index)


def flow_path_differences(raw: pd.DataFrame) -> pd.DataFrame:
    """Sensor-to-sensor level differences along the flow path.

    Built from the per-sensor window statistics already in the parquet, so
    they are the differences of the statistics, which equal the statistics of
    the difference only when both sensors are valid on the same samples.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw window features.

    Returns
    -------
    pd.DataFrame
        One column per pair in ``FLOW_PATH_PAIRS`` and statistic in
        ``DIFFERENCE_STATS``, named ``d[<up>-<down>]_<stat>``.
    """
    cols = {}
    for up, down in FLOW_PATH_PAIRS:
        for stat in DIFFERENCE_STATS:
            cols[f"d[{up}-{down}]_{stat}"] = raw[f"{up}_{stat}"] - raw[f"{down}_{stat}"]
    return pd.DataFrame(cols, index=raw.index)


def forest(n_estimators: int, n_jobs: int) -> Pipeline:
    """The untuned classifier every probe uses: median imputation, then a balanced forest."""
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        ),
    )


def out_of_fold(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_estimators: int, n_jobs: int
) -> np.ndarray:
    """Grouped out-of-fold predictions of the probe forest.

    Parameters
    ----------
    X, y, groups : np.ndarray
        Features, target and grouping key, one row per window.
    n_estimators, n_jobs : int
        Forest size and parallel workers.

    Returns
    -------
    np.ndarray
        A prediction per row, each made by a forest that never saw the row's group.
    """
    pred = np.empty_like(y)
    for train, test in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        pred[test] = forest(n_estimators, n_jobs).fit(X[train], y[train]).predict(X[test])
    return pred


def chance_baselines(y: np.ndarray) -> dict[str, float]:
    """F1-macro of guessing, for the class mix of ``y``.

    Sampling from the prior gives every class precision and recall equal to
    its prior, so F1-macro is the mean prior, ``1 / n_classes``; guessing
    uniformly gives precision ``p_c`` and recall ``1 / K``; the majority class
    scores its own F1 and zero elsewhere.

    Parameters
    ----------
    y : np.ndarray
        Target labels.

    Returns
    -------
    dict[str, float]
        ``uniform``, ``prior`` and ``majority`` F1-macro.
    """
    priors = pd.Series(y).value_counts(normalize=True)
    k = len(priors)
    uniform = float(np.mean([2 * p * (1 / k) / (p + 1 / k) for p in priors]))
    majority = float(2 * priors.iloc[0] / (1 + priors.iloc[0]) / k)
    return {"uniform": uniform, "prior": float(priors.mean()), "majority": majority}


def cap_per_well(df: pd.DataFrame, cap: int | None, rng: np.random.Generator) -> pd.DataFrame:
    """Keep at most ``cap`` windows of every well, drawn at random.

    Parameters
    ----------
    df : pd.DataFrame
        Windows with a ``well_id`` column.
    cap : int or None
        Maximum windows per well; ``None`` keeps everything.
    rng : np.random.Generator
        Seeded generator, so the sample is reproducible.

    Returns
    -------
    pd.DataFrame
        The sampled rows, original index kept.
    """
    if cap is None:
        return df
    idx = np.concatenate(
        [
            rng.choice(part.index.to_numpy(), size=min(cap, len(part)), replace=False)
            for _, part in df.groupby("well_id")
        ]
    )
    return df.loc[np.sort(idx)]


def probe_well_identity(raw: pd.DataFrame, args, rng: np.random.Generator) -> None:
    """Section 1: can one window name its well?"""
    print(
        f"\n=== 1. Can one window name its well? (target = well_id, {N_FOLDS}-fold GroupKFold by "
        f"instance, at most {args.max_windows_per_well} windows per well) ==="
    )
    sample = cap_per_well(raw, args.max_windows_per_well, rng)
    y, groups = sample["well_id"].to_numpy(), sample["instance_id"].to_numpy()
    single = sample.groupby("well_id")["instance_id"].nunique().eq(1).sum()
    chance = chance_baselines(y)
    print(
        f"  {len(sample):,} windows | {len(set(y))} wells | {single} wells have a single "
        f"recording and cannot be identified by construction | chance F1-macro: "
        f"uniform {chance['uniform']:.3f}, prior {chance['prior']:.3f}"
    )

    flags = availability_flags(raw)
    diffs = flow_path_differences(raw)
    sets: list[tuple[str, pd.DataFrame]] = []
    for normalization in NORMALIZATIONS:
        df, cols = pipeline_features(raw, normalization, args.frozen_mode)
        sets.append((f"pipeline features, {normalization}", df[cols]))
        if normalization == "none":
            shape_cols = [c for c in cols if not c.endswith(LEVEL_SUFFIXES)]
            sets.append(("dispersion and derivative statistics only", df[shape_cols]))
    sets.append(("flow-path level differences", diffs))
    sets.append(("frozen flags only", flags[[c for c in flags if c.endswith("_frozen")]]))
    sets.append(("frozen + missing flags", flags))

    for name, features in sets:
        X = features.loc[sample.index].to_numpy()
        pred = out_of_fold(X, y, groups, args.n_estimators, args.n_jobs)
        print(
            f"  {name:44s} {X.shape[1]:3d} features | accuracy {accuracy_score(y, pred):.3f} | "
            f"F1-macro {f1_score(y, pred, average='macro'):.3f}"
        )


def well_lookup_baseline(raw: pd.DataFrame) -> None:
    """Section 2: is the instance-grouped score a look-up of the well?"""
    print(
        "\n=== 2. Is the instance-grouped score a look-up? (no features: the majority fault "
        "of the well's other recordings) ==="
    )
    instances = raw.groupby(["instance_id", "well_id"])["fault_class"].first().reset_index()
    lookup = {}
    for row in instances.itertuples(index=False):
        others = instances[
            (instances.well_id == row.well_id) & (instances.instance_id != row.instance_id)
        ]
        if others.empty:
            lookup[row.instance_id] = 0  # the well has no other recording: guess Normal
        else:
            counts = others["fault_class"].value_counts()
            lookup[row.instance_id] = int(counts[counts == counts.max()].index.min())
    y = raw["fault_class"].to_numpy()
    pred = raw["instance_id"].map(lookup).to_numpy()
    chance = chance_baselines(y)
    print(
        f"  window-level accuracy {accuracy_score(y, pred):.3f} | F1-macro "
        f"{f1_score(y, pred, average='macro'):.3f} | chance F1-macro: uniform "
        f"{chance['uniform']:.3f}, prior {chance['prior']:.3f}, majority {chance['majority']:.3f}"
    )
    print("  per-class recall:")
    for c in sorted(set(y)):
        mask = y == c
        print(
            f"    {c:>3} {FAULT_CLASSES[c]:<28} {np.mean(pred[mask] == c):.3f}  (n={mask.sum():,})"
        )


def mixed_wells(raw: pd.DataFrame) -> pd.DataFrame:
    """Recordings per class of the wells that have both ordinary Normal and pre-fault recordings.

    Only these wells can pose section 3's question: one well, two kinds of
    normal operation. A fault class needs at least two recordings in the well
    for the held-out one to have an example of its class in training, so the
    usable classes are a subset of each row.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of ``load_real_normal_windows``.

    Returns
    -------
    pd.DataFrame
        One row per qualifying well, one column per fault class, the number of
        recordings as values.
    """
    counts = raw.groupby(["well_id", "fault_class"])["instance_id"].nunique().unstack(fill_value=0)
    if 0 not in counts.columns:
        return counts.iloc[0:0]
    has_normal = counts[0] > 0
    has_fault = (counts.drop(columns=0) > 0).any(axis=1)
    return counts[has_normal & has_fault]


def recording_folds(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> list:
    """The held-out sets of section 3: every pre-fault recording alone, the Normal ones in chunks.

    A well has two or three recordings per fault class, so each is held out on
    its own (leave-one-recording-out). It has dozens to hundreds of Normal
    recordings, so those are held out in at most ``N_FOLDS`` chunks, which
    keeps the number of forests small without ever training on the recording
    being predicted.
    """
    fault_recordings = np.unique(groups[y != 0])
    normal_recordings = rng.permutation(np.unique(groups[y == 0]))
    chunks = np.array_split(normal_recordings, min(N_FOLDS, len(normal_recordings)))
    return [np.array([g]) for g in fault_recordings] + list(chunks)


def predict_held_out(X, y, groups, folds, args) -> np.ndarray:
    """Predict every window with a forest trained on the rest of its well."""
    pred = np.empty_like(y)
    for held in folds:
        test = np.isin(groups, held)
        if len(set(y[~test])) < 2:  # nothing to tell apart: predict the one class left
            pred[test] = y[~test][0]
            continue
        clf = forest(args.n_estimators, args.n_jobs).fit(X[~test], y[~test])
        pred[test] = clf.predict(X[test])
    return pred


def recording_verdicts(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """One verdict per recording: its true class and the majority of its windows' predictions."""
    return (
        pd.DataFrame({"recording": groups, "y": y, "pred": pred})
        .groupby("recording")
        .agg(y=("y", "first"), pred=("pred", lambda s: s.value_counts().idxmax()))
    )


def within_well_control(raw: pd.DataFrame, args, rng: np.random.Generator) -> None:
    """Section 3: inside one well, does normal operation before a fault look different?

    Across wells, any difference between pre-fault and ordinary normal windows
    is confounded with which well produced them. Inside one well the well is
    fixed, so a difference the forest can learn there is a precursor (or a
    drift of the well over time, which this test cannot tell apart). The steps:

    1. keep the wells that have both kinds of recording (``mixed_wells``);
    2. take the well's normal-operation windows, labelled with the class of the
       recording they come from: 0 for an ordinary Normal recording, the fault
       class for the normal prefix of a recording that ends in that fault;
    3. hold each recording out in turn and predict its windows with a forest
       trained on the rest of the well (``recording_folds``,
       ``predict_held_out``);
    4. give every recording the majority verdict of its windows
       (``recording_verdicts``) and count how many pre-fault recordings were
       recognized as such — the headline number.

    Steps 2 to 4 run twice: on raw features, where the absolute levels are
    available, and on features normalized by each recording's normal
    operation, where they are not. A precursor found only in the first run
    lives in the levels.
    """
    print(
        "\n=== 3. Is there a precursor at all? (inside one well: ordinary Normal recordings "
        "against the normal prefixes of recordings that end in a fault; each recording held out "
        "in turn) ==="
    )
    counts = mixed_wells(raw)
    if counts.empty:
        print("  no well has both ordinary Normal and pre-fault recordings; nothing to compare")
        return
    print("  recordings per (well, class):")
    print("  " + counts.to_string().replace("\n", "\n  "))
    print(
        "  a class needs at least two recordings in the well, so the held-out one has a "
        "training example"
    )

    for normalization in ("none", "normal-operation-values"):
        df, cols = pipeline_features(raw, normalization, args.frozen_mode)
        print(f"\n  normalization = {normalization}")
        for well in counts.index:
            usable = counts.loc[well][counts.loc[well] >= 2].index
            part = df[(df["well_id"] == well) & (df["fault_class"].isin(usable))]
            if part["fault_class"].nunique() < 2:
                print(f"    well {well:>3}: only one class has two or more recordings, skipped")
                continue
            X = part[cols].to_numpy()
            y, groups = part["fault_class"].to_numpy(), part["instance_id"].to_numpy()
            pred = predict_held_out(X, y, groups, recording_folds(y, groups, rng), args)
            verdicts = recording_verdicts(y, pred, groups)
            pre_fault = verdicts[verdicts.y != 0]
            recognized = int((pre_fault.y == pre_fault.pred).sum())
            prior = pd.Series(y).value_counts(normalize=True).iloc[0]
            classes = [int(c) for c in sorted(set(y))]
            print(
                f"    well {well:>3}: classes {classes} | pre-fault recordings recognized "
                f"{recognized}/{len(pre_fault)} | all recordings right "
                f"{np.mean(verdicts.y == verdicts.pred):.3f} of {len(verdicts)} | windows: "
                f"F1-macro {f1_score(y, pred, average='macro'):.3f}, accuracy "
                f"{accuracy_score(y, pred):.3f} against a majority prior of {prior:.3f}"
            )


def probe_fault_across_wells(raw: pd.DataFrame, args, rng: np.random.Generator) -> None:
    """Section 4: what transfers to a well the forest has never seen?"""
    cap = args.fault_max_windows_per_well
    sample = cap_per_well(raw, cap, rng)
    print(
        f"\n=== 4. What transfers to a well never seen? (target = fault_class, {N_FOLDS}-fold "
        f"GroupKFold by well, "
        f"{'at most ' + str(cap) + ' windows per well' if cap else 'all windows'}, pooled out-of-fold) ==="
    )
    y, groups = sample["fault_class"].to_numpy(), sample["well_id"].to_numpy()
    chance = chance_baselines(y)
    print(
        f"  {len(sample):,} windows | chance F1-macro: uniform {chance['uniform']:.3f}, prior "
        f"{chance['prior']:.3f}, majority {chance['majority']:.3f} | a class held out with all its "
        "wells cannot be predicted in that fold"
    )
    df, cols = pipeline_features(raw, "none", args.frozen_mode)
    shape_cols = [c for c in cols if not c.endswith(LEVEL_SUFFIXES)]
    diffs = flow_path_differences(raw)
    sets = [
        ("pipeline features, none", df[cols]),
        ("dispersion and derivative statistics only", df[shape_cols]),
        ("flow-path level differences", diffs),
        ("pipeline features + differences", pd.concat([df[cols], diffs], axis=1)),
    ]
    for name, features in sets:
        X = features.loc[sample.index].to_numpy()
        pred = out_of_fold(X, y, groups, args.n_estimators, args.n_jobs)
        recalls = " ".join(f"{c}:{np.mean(pred[y == c] == c):.2f}" for c in sorted(set(y)))
        print(
            f"  {name:44s} {X.shape[1]:3d} features | accuracy {accuracy_score(y, pred):.3f} | "
            f"F1-macro {f1_score(y, pred, average='macro'):.3f} | recall per class {recalls}"
        )


def main() -> None:
    """Parse the shared switches and run the four probes."""
    parser = run_parser(__doc__.splitlines()[0], with_model=False)
    add_frozen_sensors_arg(parser)
    parser.add_argument(
        "--max-windows-per-well",
        type=int,
        default=800,
        help="windows sampled per well for the well-identifiability probe (default: 800)",
    )
    parser.add_argument(
        "--fault-max-windows-per-well",
        type=int,
        default=None,
        help="windows sampled per well for the well-grouped fault probe (default: all)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=300, help="trees in every probe forest (default: 300)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="where to save the audit (default: results/audits/well_identifiability_<...>.txt)",
    )
    args = parser.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out = args.output_file or AUDITS_DIR / (
        f"well_identifiability{overlap_suffix(args.allow_overlap)}"
        f"{extreme_suffix(args.keep_extreme_values)}_{stamp}.txt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    with (
        open(out, "w", encoding="utf-8") as fh,
        contextlib.redirect_stdout(Tee(sys.stdout, fh)),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        print(f"Well identifiability audit — {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}")
        print(
            f"features {features_path(args.allow_overlap, args.keep_extreme_values).name} | "
            f"frozen sensors {args.frozen_mode} | forests of {args.n_estimators} trees, untuned"
        )
        raw = load_real_normal_windows(args.allow_overlap, args.keep_extreme_values)
        print(
            f"real normal-operation windows: {len(raw):,} | recordings "
            f"{raw['instance_id'].nunique()} | wells {raw['well_id'].nunique()}"
        )
        probe_well_identity(raw, args, rng)
        well_lookup_baseline(raw)
        within_well_control(raw, args, rng)
        probe_fault_across_wells(raw, args, rng)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
