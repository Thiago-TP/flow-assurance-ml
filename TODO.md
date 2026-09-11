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
- [ ] some real instances present extreme values of sensor data. For example, well 6 has P-PDG and T-PDG values of magnitude greater than 10^32. This breaks classifiers when the `--no-normalization` flag is up. Proposal: make the default behavior of the pipeline to, at data loading time, swap extreme values with NaN, which will be subject to later imputation. This can be deactivated with flag `--keep-extreme-values`.
- [ ] the Severe Slugging class (default number: 3) is effectively monopolized by well 14: 31 of the 32 real instances (96.875%) from this fault class come from it, with the remaining one coming from well 1. Therefore, train/validation/testing splits may lead to a configuration where no Severe Slugging is seen during training, breaking classifier evaluation and, more importantly, making it wholly unable of predicting the "missing" fault. Proposal: at run time, it must be ensured that training and test splits present all classes. If they don't, then one instance (`--cv-group instance_id`) or well (`--cv-group well_id`) is picked at random and moved into the empty split. Then the splits are checked again, and if valid, the pipeline is carried out as normal; if not, the splits are reset and a different instance/well is picked at random again. If after all instances/wells are picked no configuration is valid, then the pipeline breaks. User should be notified of what's happening through `--verbose` logs.
