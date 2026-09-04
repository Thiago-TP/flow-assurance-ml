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
- [x] Add a data visualization module that can be used to visualize the data in the parquet files.
  For starters, it should have two functions:
  - [x] `plot_fault(fault: str) -> None`:
    Plots all time series in each instance for all instances of a given fault.
    In short, it plots a folder in 3W/dataset.
    The result is a PDF file with one page per instance, and multiple subplots per page.
    Plotted time series are colored by their label, i.e.,
    light green for normal, light yellow for transients, and light red for active problems.
    Labels outside this pattern are grey.
  - [x] `plot_faults_per_well() -> None`:
    Plots which faults occur in each well across time.
    Each well is represented by a horizontal line, and the faults are shown as colored segments along the line.
    The length of each segment represents the duration of the fault.
    Faults are color-coded by their label (normal, flow instability, hydrate in produciton line, etc.).
    The duration of each fault is represented by a horizontal bar, and the time axis is shared across all wells.
    Since durations tend to be short, the bars look like "nicks" in the plot.
- [x] Confirm that a last OOF validation (step 3/3 in `02_training.py`) is valid after the hyperparameter search.
  Whatever the case, it does not seem the case that data dedicated to group CV
  (i.e., training and validation) is kept separate from data used for evaluation (i.e., testing),
  so there could be leakage.
  (Audited: fold mechanics were leak-free, but the OOF folds were identical to the search folds,
  so reported scores were selection-biased. Fixed with `--eval holdout` (grouped test split, default)
  and `--eval nested` (nested grouped CV).)
- [x] Since training scripts (`02_training.py`, `training.py`) do both validation and training,
  it would be good to rename them. E.g., `02_train_val_test.py` and `train_val_test.py`.
- [x] Fix stage 5's depth-selection reporting: the depth sweep still reports the OOF score
  of the best depth on the same folds that selected it (same validation-as-test pattern
  fixed in stage 2).
  (Fixed: the sweep now validates on train+val only and the chosen depth is scored once
  on the same seeded grouped test set stage 2's holdout evaluation uses.)
