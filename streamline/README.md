# flowml — streamlined 3W pipeline

Self-contained, script-based rewrite of the flow-assurance ML pipeline for the
[Petrobras 3W dataset](https://github.com/petrobras/3W). Two tasks, two tree
models (Random Forest, XGBoost), four stages, one features parquet.

## Pipeline

```mermaid
flowchart LR
    subgraph S1["01_build_features.py"]
        A[("3W raw parquets<br/>1 file = 1 well instance")] --> B["clean<br/>ffill ≤ 60 s · quality gate"]
        B --> C["z-score<br/>per instance"]
        C --> D["filter<br/>gaussian | statistical | none"]
        D --> E["window 300 s / step 150 s<br/>11 stats × 8 sensors = 88 features"]
    end
    E --> F[("data/features_&lt;filter&gt;.parquet<br/>labels: window_label + fault_class")]
    F --> G["02_train.py<br/>GroupKFold search + OOF"]
    G --> H["03_evaluate.py<br/>metrics + confusion matrix"]
    G --> I["04_interpret.py<br/>MDI · gain · permutation · SHAP"]
    I --> J["05_decision_tree.py<br/>compact tree on top SHAP features"]
```

`main.py` orchestrates all five stages in order (stage 1 is skipped when its
parquet already exists).

## The two tasks

Every window row carries **both** labels, so one parquet serves both tasks:

```mermaid
flowchart TD
    P[("features_&lt;filter&gt;.parquet")] --> DET["task = detection<br/><i>what is happening now?</i>"]
    P --> PRED["task = prediction<br/><i>what fault is coming?</i>"]
    DET --> DL["label = <b>window_label</b><br/>17 classes: 0 normal · 1-9 active · 101-109 transient<br/>all windows used"]
    PRED --> PL["label = <b>fault_class</b><br/>8 classes: the fault the instance later develops<br/>only windows with window_label == 0"]
```

> Prediction has 8 classes (not 10) because faults 3 (severe slugging) and
> 4 (flow instability) have no recorded normal-operation period in 3W.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and a local copy of the 3W dataset.

```bash
cd streamline
uv sync

# point at the 3W dataset (or edit src/flowml/config.py)
export FLOWML_RAW_DATA_DIR="/path/to/3w/dataset"      # PowerShell: $env:FLOWML_RAW_DATA_DIR = "..."

# everything at once (defaults: --model xgb --task prediction --filter none)
uv run main.py                      # add --max-instances 3 for a quick smoke test

# or stage by stage
uv run scripts/01_build_features.py --filter none
uv run scripts/02_train.py
uv run scripts/03_evaluate.py
uv run scripts/04_interpret.py
uv run scripts/05_decision_tree.py  # compact tree on the top SHAP features
```

Every stage takes the same switches, and each combination writes its artifacts
under a unique tag:

| Switch | Choices | Default | Stages |
|--------|---------|---------|--------|
| `--model` | `rf`, `xgb` | `xgb` | 2–5 |
| `--task` | `prediction`, `detection` | `prediction` | 2–5 |
| `--filter` | `gaussian`, `statistical`, `none` | `none` | 1–5 |
| `--grouping` | `none`, `hydrate` | `none` | 3, 5 |

## Class groupings

`--grouping hydrate` collapses the classes onto the operational triage
question — is the well heading for normal operation, a hydrate event, or some
other flow-assurance problem? Faults 8 and 9 become **Hydrate**, every other
fault becomes **Other Problem**, and transients follow their active
counterpart:

```mermaid
flowchart LR
    subgraph F["fault_class / window_label"]
        N["0"]
        O["1 · 2 · 5 · 6 · 7<br/>(+3 · 4 in detection)"]
        H["8 · 9"]
    end
    N --> GN["0 Normal"]
    O --> GO["1 Other Problem"]
    H --> GH["2 Hydrate"]
```

Grouping affects **scoring only** — features and the ensemble are always built
on the full class set. Stage 5 then compares two ways of reaching the grouped
labels, both scored against the same truth:

| Strategy | Trains on | Answers |
|----------|-----------|---------|
| `collapse` | full class set, predictions collapsed afterwards | how well does the existing tree already serve the triage question? |
| `native` | the 3 grouped labels directly | what does spending the whole depth budget on this distinction buy? |

## Artifacts

```mermaid
flowchart LR
    T["02_train.py<br/>tag = model_task_filter"] --> M["results/models/<br/>&lt;tag&gt;.joblib · &lt;tag&gt;_label_encoder.joblib"]
    T --> O["results/metrics/<br/>&lt;tag&gt;_oof.parquet · &lt;tag&gt;_search.json · &lt;tag&gt;_cv_results.csv"]
    O --> EV["03_evaluate.py"] --> EM["results/metrics/&lt;tag&gt;_metrics.json<br/>results/figures/&lt;tag&gt;_confusion_matrix.png"]
    M --> IN["04_interpret.py"] --> IM["results/metrics/&lt;tag&gt;_importance.json<br/>results/figures/&lt;tag&gt;_{mdi|gain,permutation,shap}.png"]
    IM --> DT["05_decision_tree.py<br/>dtag = dt_task_filter_from_model"] --> DM["results/models/&lt;dtag&gt;.joblib<br/>results/metrics/&lt;dtag&gt;_{metrics.json,rules.txt}<br/>results/figures/&lt;dtag&gt;_{tree,confusion_matrix}.png"]
```

## Layout

```
streamline/
├── pyproject.toml            uv project (flowml package, src layout)
├── src/flowml/
│   ├── config.py             paths · sensors · class maps · constants
│   ├── preprocessing.py      loading · cleaning · z-score · signal filters
│   ├── features.py           windowing · 88 features · labeling
│   ├── training.py           task datasets · pipelines · CV search · OOF
│   ├── evaluation.py         metrics · confusion matrix
│   ├── interpretation.py     MDI · gain · permutation · SHAP
│   └── cli.py                shared argparse
├── main.py                   runs all stages in order
├── scripts/                  the 5 pipeline stages (thin CLIs)
├── data/                     generated features (git-ignored)
└── results/                  models · metrics · figures (git-ignored)
```

## Methodology notes

- **GroupKFold by `instance_id`** everywhere — windows of one well never split
  across train/test.
- **Imputation inside the model pipeline** (`SimpleImputer(median)`), so it is
  refit per fold — no leakage. This replaces the two divergent imputation
  strategies of the original repo.
- **Balanced classes**: RF via `class_weight="balanced"`; XGBoost via sample
  weights computed **per training fold** (the original computed them globally).
- **Outliers preserved**: pressure spikes are fault signatures; `max_zscore`
  captures them instead of removing them.
- Deep-learning branches (CNN-1D, CNN-LSTM) of the original repo were dropped
  deliberately: the end goal is interpretable tree models built on the top
  SHAP features.
