- [X] Remove filtering of data and its related flag.
- [X] Include a custom class grouping flag that allows user to apply his desired class grouping (variable `CUSTOM_CLASS_GROUPING` in `config.py`).
- [X] Add verbose flag that affects all functions that have a verbose (or verbose-like) argument (off by default).
- [X] Add a no-normalization flag that does not apply z-score normalization to the features (off by default).
- [X] Add an option for the number of jobs (value of N_JOBS in `config.py`) to be set by the user (default is 6 or number of cores minus 2, whichever is smaller).
- [X] Change the setting of the max_z variable in `features.py` so that it doesn't re-normalize the window data.
- [X] Add a group cross-validation option for the group types.
  By default, groups are the instance ID, but they should be able to be changed to the well ID.
  Instance ID is the name of the parquet file, and well ID is inside the instance ID as follows:
  `instance_id = WELL-000{well_id}_Y`,
  e.g. file WELL-00026_20170608230000.parquet has well ID 26.
- [X] Add a data visualization module that can be used to visualize the data in the parquet files.
  For starters, it should have two functions:
  - [X] `plot_fault(fault: str) -> None`:
    Plots all time series in each instance for all instances of a given fault.
    In short, it plots a folder in 3W/dataset.
    The result is a PDF file with one page per instance, and multiple subplots per page.
    Plotted time series are colored by their label, i.e.,
    light green for normal, light yellow for transients, and light red for active problems.
    Labels outside this pattern are grey.
  - [X] `plot_faults_per_well() -> None`:
    Plots which faults occur in each well across time.
    Each well is represented by a horizontal line, and the faults are shown as colored segments along the line.
    The length of each segment represents the duration of the fault.
    Faults are color-coded by their label (normal, flow instability, hydrate in produciton line, etc.).
    The duration of each fault is represented by a horizontal bar, and the time axis is shared across all wells.
    Since durations tend to be short, the bars look like "nicks" in the plot.
- [X] Confirm that a last OOF validation (step 3/3 in `02_training.py`) is valid after the hyperparameter search.
  Whatever the case, it does not seem the case that data dedicated to group CV
  (i.e., training and validation) is kept separate from data used for evaluation (i.e., testing),
  so there could be leakage.
  (Audited: fold mechanics were leak-free, but the OOF folds were identical to the search folds,
  so reported scores were selection-biased. Fixed with `--eval holdout` (grouped test split, default)
  and `--eval nested` (nested grouped CV).)
- [X] Since training scripts (`02_training.py`, `training.py`) do both validation and training,
  it would be good to rename them. E.g., `02_train_val_test.py` and `train_val_test.py`.
- [X] Fix stage 5's depth-selection reporting: the depth sweep still reports the OOF score
  of the best depth on the same folds that selected it (same validation-as-test pattern
  fixed in stage 2).
  (Fixed: the sweep now validates on train+val only and the chosen depth is scored once
  on the same seeded grouped test set stage 2's holdout evaluation uses.)
- [X] Add a fault-signature visualization: `plot_fault_signatures(fault: int, source: str) -> None`
  plots, for every instance of a fault folder, the variables whose joint behavior identifies
  that fault (the ones the 3W Dataset 2.0.0 paper puts on its figures 3 to 7), all on one
  time axis with a y axis of its own per variable, one page per instance.
  `plot_all_fault_signatures()` covers every fault that has a signature, once per instance
  source (real, simulated, hand-drawn), and stage 0 calls it.
  (Note: hand-drawn instances exist only for faults 1 and 7, neither of which has a published
  signature, so the `drawn` directory comes out empty until those two faults get an entry
  in `FAULT_SIGNATURES`.)
- [X] Split the stage-0 output into one directory per plot family under `results/figures/`:
  `instances_per_fault/`, `well_histories/` (the per-well histories and the fault timeline)
  and `fault_signatures/{real,simulated,drawn}/`.
- [X] Instances from a given well may overlap in time (see `results/figures/well_histories/faults_per_well.pdf`), making train-test leakage possible, and, more importantly, assigning different labels to the same data point/vector (e.g., a pressure at the end of an instance is labeled "faulty steady state" in the earlier instance, and "normal in the next"). As such, it must be implemented that default behavior discards overlapping instances (i.e., those instances in stack level 2 or higher in `faults_per_well.pdf`) *before* building the feature dataset. Conversely, a new flag  `--allow-overlap` would include all instances regardless of overlap.
  (Done: stage 1 stacks the real instances of every well exactly as `faults_per_well.pdf` does
  (`preprocessing.pack_lanes`, shared with the plot) and drops those on stack level 2 or higher;
  `--allow-overlap` keeps them and, like `--no-normalization`, is a shared switch that tags the
  features parquet and every downstream artifact with `_overlap` so both datasets coexist.
  `--verbose` reports how many instances overlap and how many were removed, per well.)
- [X] the `scr/visualize.py` backend may be bloated, a review and, if necessary, a refactor, are in order.
  (Done: the length was mostly docstrings and four plot families in one module; the actual bloat
  was duplication, folded into single helpers (`dataset.ini` reader, `list_instances`, label
  bands/shading, tint, envelope, `overlapping_mask`, `TextToPath` widths). Then a shared PDF writer
  and the flat-signal guard on instance pages, and finally a verbatim split into the
  `flowml.visualization` package: `common`, `instances`, `signatures`, `wells`, `timeline`, with the
  public API re-exported so the stage-0 script's imports did not change.)
- [X] some real instances present extreme values of sensor data. For example, well 6 has P-PDG and T-PDG values of magnitude greater than 10^32. This breaks classifiers when the `--no-normalization` flag is up. Proposal: make the default behavior of the pipeline to, at data loading time, swap extreme values with NaN, which will be subject to later imputation. This can be deactivated with flag `--keep-extreme-values`.
  (Done: `preprocessing.mask_extreme_values` replaces with NaN, at loading time and before
  cleaning, every reading that cannot be a measurement, so the forward-fill and the model's
  imputer handle them like any other missing value. Three rules, all bounded by a survey of
  every instance of 3W 2.0.0 so that no genuine spike is at risk: magnitude beyond
  `EXTREME_VALUE_LIMIT` (1e8; largest varying reading below it 4.9e7, smallest above it 1.3e8),
  pressures below `PRESSURE_MIN` (0 — 3W pressures are absolute; zero is left for the frozen
  sensor item), and temperatures outside `TEMPERATURE_LIMITS` (-50 to 250 C, which keeps the
  genuine -33.8 C of Joule-Thomson cooling and catches the -999/-99.99 sentinels and the
  30,000 C readings). Together they mask 6,582,455 readings in 72 instances and, via the
  quality gate, cost 10 instances whose `P-TPT` was garbage throughout; wells 31 and 34 lose
  all their instances but no fault class loses well coverage.
  `--keep-extreme-values` is a shared switch like `--allow-overlap`, turning all the rules
  off and tagging the features parquet and downstream artifacts with `_extremes`; `--verbose`
  reports the masking per rule, per sensor and per instance. A fourth rule followed: negative
  choke openings (`OPENING_SENSORS`, `OPENING_MIN`), which the dataset has in exactly one
  file — well 30's `ABER-CKP` at -99.99 % for all 161,538 samples.)
- [X] the Severe Slugging class (default number: 3) is effectively monopolized by well 14: 31 of the 32 real instances (96.875%) from this fault class come from it, with the remaining one coming from well 1. Therefore, train/validation/testing splits may lead to a configuration where no Severe Slugging is seen during training, breaking classifier evaluation and, more importantly, making it wholly unable of predicting the "missing" fault. Proposal: at run time, it must be ensured that training and test splits present all classes. If they don't, then one instance (`--cv-group instance_id`) or well (`--cv-group well_id`) is picked at random and moved into the empty split. Then the splits are checked again, and if valid, the pipeline is carried out as normal; if not, the splits are reset and a different instance/well is picked at random again. If after all instances/wells are picked no configuration is valid, then the pipeline breaks. User should be notified of what's happening through `--verbose` logs.
  (Done, as proposed and a step further. `train_val_test.repair_holdout` fixes the seeded holdout
  split that stages 2 and 5 share: for each class missing from a side, a carrier group on the other
  side is picked at random (seeded) and moved, accepted only if the donor keeps every class it has,
  otherwise another carrier is tried; a class living in a single group stops the run with a message
  naming it. Today that repair moves one instance in detection/instance_id (class 7), four wells in
  prediction/well_id, and stops detection/well_id, where Quick PCK Restriction is one well. The step
  further: XGBoost 3.x refuses to fit a fold that lacks a class, so the search folds and the nested
  outer folds get the same repair (`coverage_folds`), plus pinning — a class carried by a single
  group stays in every training fold and is never validated on. `--verbose` prints every move and
  every pinned group. Note that moving whole wells shifts the split: prediction/well_id ends with
  11 of 35 wells and 55 % of the windows in test — 8 of 33 wells and 52 % once the extreme-value
  masking of the item above dropped wells 31 and 34; see the report and the next-steps items below.)
- [X] Include a decision tree classifier as one of the model options passed to `--model`. 
    > [!TIP]
    > Since this model is already white-box and interpretable, 
    > the interpretation (4th) and decision tree (5th) stages are automatically skipped.
  (Done: `--model dt` fits a `DecisionTreeClassifier` (class-weight balanced, `DT_PARAM_GRID`
  in `config.py`) through the same grouped search and the same holdout/nested evaluation as
  `rf` and `xgb`. `WHITE_BOX_MODELS` in `train_val_test.py` drives the skipping: `main.py`
  does not call stages 4 and 5, and running either directly says why and exits 0
  (`cli.skip_if_white_box`). So that the option still yields something readable, stage 2
  exports the fitted tree itself — `<tag>_rules.txt` and `<tag>_tree.png` — through
  `interpretation.export_tree`, which stage 5 now shares. Every tree is drawn, however large:
  the canvas is one `TREE_FIGURE_LEAF_WIDTH` per *leaf* by one `TREE_FIGURE_LEVEL_HEIGHT` per
  level (sizing by depth would ask for a canvas thousands of times too wide, since real trees
  are nowhere near full), and a tree beyond what matplotlib can rasterize loses resolution
  rather than the figure. The search is free to pick deep trees: on prediction/zscore_overlap
  it picks depth 12 with 171 leaves — a 30,000 x 3,900 px poster — scoring F1-macro 0.8740 on
  the held-out test set.)
Next steps from `results/reports/2026-09-17_dt_from_xgb_cv_and_class_grouping.md` (the whitening of
the XGBoost predictor under both CV groupings and both class groupings), in the order to take them.
Each is tagged **feature**, **documentational** or **bugfix**.

- [X] **1 · feature** — Fix the well-level evaluation before drawing any well-generalization
  conclusion: the seeded 20 %-of-groups holdout gives 8 of 33 wells 52 % of the windows and 68 % of
  normal operation. Make leave-one-group-out an evaluation protocol, `leave-one-out` in `--eval`,
  usable with both CV groupings and the default under `--cv-group well_id`.
  (Done: `--eval leave-one-out` is the nested protocol with one group per outer fold
  (`train_val_test.outer_folds`, `n_outer_splits`): every fold runs its own inner search and predicts
  its held-out group, so every well gets a score of its own — 33 searches for the well grouping, a
  thousand for instances, stated as a fit count before starting. `--eval` left unset now follows the
  grouping (`EVAL_MODE_DEFAULTS`, resolved by `cli.RunParser`): `holdout` with `instance_id`,
  `leave-one-out` with `well_id`. Artifacts carry `_loo`. Stage 3 adds a per-group score sheet
  (`evaluation.per_group_metrics`) whenever the table has at most 50 groups. Stage 5 follows the
  run's protocol instead of always using the holdout: under `nested` / `leave-one-out` each outer
  fold runs its own depth sweep and predicts its held-out part, the pooled predictions are scored,
  and the exported tree comes from a final sweep on all data — tree and ensemble are judged on
  identical held-out rows under every protocol.)
- [X] **2 · feature** — Report the class composition of both sides of every split in the run logs.
  A train-val prior of 49 % Normal against a test prior of 95 % should not have to be reconstructed
  from the parquet. Cannot be too detailed.
  (Done: `train_val_test.split_composition` prints, for every split, windows and groups per side
  and, per class, the count, the share of the class and the prior within the side, listing the
  groups when they are few. Always on for the holdout split; every outer fold logs what it holds
  out (group, windows, classes) and its full table under `--verbose`, as do the inner search folds.
  The holdout summary JSON stores the per-side class counts, the outer-fold records store theirs.)
- [X] **3 · feature** — Extend `--depths` past 6: four of six sweeps picked 6, the top of the grid,
  and the 8-class sweep was still climbing (0.6197 → 0.7597 from depth 5 to 6).
  (Done by hand: the default is `2,…,12` in stage 5 and `main.py`. Reviewed: the sweep still scores
  on grouped validation folds only, and with the deeper trees the figures needed item 7b.)
- [X] **4 · feature + documentational** — Wire `--class-grouping` into stages 2 and 4, so that a
  natively trained grouped ensemble and a grouped SHAP ranking exist. Today the grouping reaches
  stages 3 and 5 only: the DT-native tree (0.9553) has no XGBoost-native counterpart to be compared
  with, and it distills a ranking built for the 8-class question. Grouped runs need their own tag
  suffix so they coexist with the standard ones, and the README class-groupings section — its
  "scoring only" warning, the switch table, the artifact diagram, the strategy table — must be
  rewritten with care, as must the stage docstrings.
  (Done. `--class-grouping` now reaches every stage but feature building, which it cannot change.
  Stage 2 fits on the grouped labels, stage 4 ranks that model, and `run_tag` carries the grouping
  (`cli.grouping_suffix`), so a grouped run coexists with the standard one. The key invariant that
  makes them comparable: `TaskData` carries both label spaces (`y` modeled, `fine_y` the dataset's
  own) and **every split is built from `fine_y`** — covering every fine class covers every group of
  them, and a split independent of the label set puts both runs on identical rows, which the logs
  confirm window for window. Stage 3 now reports `native` (the grouped run) beside `collapse` (the
  standard run's predictions mapped onto the groups), the way stage 5 always did, and names the
  command for whichever run is missing. Stage 5 goes further and pairs each strategy with the
  ranking of the ensemble trained on the same labels it is — the standard run's for `collapse`, the
  grouped run's for `native` — so the native tree is no longer handed features chosen for the
  8-class question. `main.py` therefore runs the training stages once per label set. README: the
  switch table, the artifact diagram, the class-groupings section (its "scoring only" warning
  replaced by the split invariant) and the strategy table were rewritten.)
- [X] **5 · documentational** — Re-run the four configurations (instance/well grouping × standard/
  hydrate grouping) on the normalized features (`features_zscore_overlap.parquet`), with the new
  protocol defaults and depth grid, and write a second report under `results/reports/` on the effect
  of the groupings on normalized data. Per-instance z-scoring removes the absolute pressure and
  temperature levels that make a well identifiable; if the instance-grouped scores hold up, the
  well-identity hypothesis of the first report weakens, if they drop toward the well-grouped ones it
  is confirmed.
  (Done: `results/reports/2026-09-17_normalization_and_groupings.md`. The scores rose everywhere, and
  the predicted test turned out to be confounded, because normalization does two opposite things at
  once. The genuine half: it removes the between-well level offsets, so the well-grouped ensemble
  goes from 1.5 % to 77.6 % accuracy and Normal recall from 0.0004 to 0.793 — which *confirms* the
  well-identity hypothesis and strengthens it, since raw levels do not merely offer a shortcut, they
  actively block cross-well transfer. The defect half is item 9 below. Also found: the collapse /
  native ordering flips with the feature quality (native wins on raw, collapse wins on z-scored well
  grouping, and the native model's failure mode is over-predicting minorities on a 94.9 %-Normal
  test set), and the z-scored trees root on tests of whether a sensor's standard deviation is
  exactly zero — instrument health, not flow physics. The raw baselines were re-run on the same
  2-to-12 depth grid first, so every comparison in the report is like-for-like; that moved the raw
  artifacts off report 1's grid, which report 1 §10.6 records.)
- [ ] **9 · bugfix** — Per-instance z-scoring leaks the coming fault into the prediction task's
  features. `features.extract_instance_features` calls `preprocessing.normalize_instance` on the
  *whole* cleaned instance, so every sensor is centred and scaled by statistics of the entire
  recording — the fault period included — and only afterwards are the windows cut and the faulty
  ones dropped. A normal-operation window is therefore divided by a number that encodes how badly
  the well later failed. `scripts/audits/normalization_leakage_auditing.py` measures it on real
  instances: the divisor is 1.04x the leak-free one for Normal (the control, no fault period to
  inflate it) but 10x for Abrupt BSW Increase, 29x for Hydrate in Production Line and 372x for
  Spurious DHSV Closure; the per-class recall gain from z-scoring is monotone in that ratio across
  the four classes with headroom (Spearman 1.000; 0.833 over all eight, p = 0.010). Fix: compute
  each instance's statistics from its normal-operation prefix, or a fixed leading window, not the
  whole recording — which keeps everything normalization is for and drops what it should never
  have carried. Then regenerate every z-scored artifact and rewrite the affected sections of report
  2. Note this is a *deployment* problem in the detection task too (the statistics are not
  available online), though not a label leak there.
- [ ] **10 · feature** — Decide what to do about frozen sensors, now that they sit at the root of
  the z-scored trees ("is this standard deviation exactly zero?"). `sensor_distributions_auditing.py`
  shows `P-TPT` frozen at exactly 0 for 100 % of the readings of wells 35, 36 and 40 and of every
  hand-drawn instance, and constant at 817 bar for well 29. Either drop instances whose critical
  sensors never move, or add an explicit sensor-health feature so the model states that it is using
  instrument status instead of smuggling it through a dispersion statistic.
- [X] **6 · feature** — Audit the sensor-dropout thresholds the trees split on (`P-TPT_max <= 63 Pa`
  is a wellhead at vacuum): an audit script that plots the distribution of the pressure and
  temperature readings of every real well and compares it with the same distributions over the
  simulated instances, without touching the current frozen / extreme-value masking.
  (Done: `scripts/audits/sensor_distributions_auditing.py` reads the whole raw dataset, audited
  columns only, into fixed-grid histograms and writes `results/audits/sensor_distributions.pdf` —
  one page per pressure and temperature sensor, a box per real well and per synthetic source, and
  per row the share of readings exactly at zero, the share the default masking removes and the
  share missing — plus a text summary with the percentiles. The masking rules are unchanged.)
- [X] **7a · feature** — Prune degenerate splits before export: sibling leaves that share a label
  (four such pairs in the instance-grouped 8-class tree) inflate the apparent depth without changing
  a decision. Collapsing them makes the published trees smaller and honest about their effective
  complexity; the metrics must not change, since the predictions do not.
  (Done: `interpretation.prune_redundant_splits` collapses every split whose *whole subtree* agrees
  on one class — the general form of the sibling-pair case, so a degenerate subtree folds into a
  single leaf rather than one level per pass — working bottom up. Predictions are provably
  unchanged: `tree_.value` at a node is the class distribution of the samples reaching it, and
  summing distributions that share an argmax keeps that argmax, so the collapsed leaf predicts what
  its subtree already did; the one case where an exact tie could flip the answer (sklearn's argmax
  takes the lower index) is detected and left alone. `export_tree` prunes before writing and reports
  what it removed, and stages 2 and 5 now export *before* dumping the model, so the saved `.joblib`
  is the pruned tree the rules and the drawing describe. Since the detached nodes stay in `tree_`'s
  arrays, sklearn's `get_depth` / `get_n_leaves` would keep counting them, so `walk_tree` and
  `tree_shape` measure a tree by traversal instead. Two related fixes fell out: `export_text`
  truncates past 10 levels by default, which the deeper sweep now reaches, so the full depth is
  requested; and the figure title states the pruned depth and the grown one it came from.
  `tests/test_interpretation.py` covers the invariants, including that pruning never changes a
  prediction.)
- [X] **7b · feature** — With the deeper sweep, let tree figures be as big as they need to be so
  that no two leaves overlap.
  (Done: `interpretation.tree_canvas_size` draws the tree once on a probe canvas, measures the
  widest and tallest node box and the closest pair of nodes on any level, and sizes the figure so
  the pair sits `TREE_FIGURE_NODE_GAP` apart; `TREE_FIGURE_LEAF_WIDTH` and `TREE_FIGURE_LEVEL_HEIGHT`
  are gone. Past `TREE_FIGURE_MAX_PIXELS` the PNG loses resolution as before and a vector
  `<tag>_tree.pdf` is written beside it.)
- [X] **8 · bugfix** — The `collapse` strategy of stage 5 exported its rules and figure with the
  grouped label map although the tree was fit on the fine classes, so `class: Normal` meant Abrupt
  BSW Increase, `class: Hydrate` meant Normal, and faults 5–9 appeared as bare integers. Metrics and
  confusion matrices were unaffected.
  (Fixed: stage 5 passes `export_tree` the map of the label space each tree was fit on — the fine
  classes for `full` and `collapse`, the groups for `native` — and the four affected exports of the
  report's runs were regenerated with the report's own depth grid, so their metrics are unchanged.)
