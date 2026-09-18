# Whitening the XGBoost predictor: effects of CV grouping and class grouping on the distilled decision tree

**Date:** 2026-09-17
**Task:** `prediction` — from windows of normal operation only, predict which fault the well will develop.
**Dataset flags (all runs):** `--no-normalization --allow-overlap`, extreme values masked (default).
**Question:** how much of a tuned XGBoost classifier survives distillation into a single depth-limited tree built on its top-10 SHAP features, and how that survival depends on (a) the cross-validation grouping key and (b) the granularity of the label set.

---

## 1. What was run

Four pipeline configurations, each a full stage 2 → 4 → 5 chain:

| # | CV grouping | Class grouping | Stage-5 tag |
|---|---|---|---|
| A | `instance_id` | `standard` (8 classes) | `dt_prediction_raw_overlap_from_xgb` |
| B | `instance_id` | `hydrate` (3 groups) | `dt_prediction_raw_overlap_from_xgb_hydrate` |
| C | `well_id` | `standard` (8 classes) | `dt_prediction_raw_overlap_from_xgb_wellcv` |
| D | `well_id` | `hydrate` (3 groups) | `dt_prediction_raw_overlap_from_xgb_wellcv_hydrate` |

Commands:

```sh
# A / C differ only in --cv-group; B / D add --class-grouping hydrate to stage 5.
uv run scripts/02_train_val_test.py --model xgb --task prediction \
    --no-normalization --allow-overlap [--cv-group well_id]
uv run scripts/04_interpret.py       --model xgb --task prediction \
    --no-normalization --allow-overlap [--cv-group well_id]
uv run scripts/05_decision_tree.py   --model xgb --task prediction \
    --no-normalization --allow-overlap [--cv-group well_id] \
    [--class-grouping hydrate]
```

Every tree is a `SimpleImputer(median)` + `DecisionTreeClassifier(class_weight="balanced")`
restricted to the top 10 SHAP features of the corresponding XGBoost run. Depth is chosen
by a sweep over `2,3,4,5,6,7,8,9,10,11,12` on grouped out-of-fold folds of train+val only; the winning
depth is refit on all of train+val and scored once on the same seeded grouped holdout the
XGBoost run was scored on.

Under `--class-grouping hydrate`, stage 5 runs two strategies against the same grouped truth:

- **collapse** — train on all 8 classes, then map the test predictions onto
  Normal / Other Problem / Hydrate.
- **native** — train directly on the 3 grouped labels.

The XGBoost reference under the hydrate grouping is a *collapse* of the 8-class model's stored
predictions (stage 3 has no native-training path), so it is comparable to the DT collapse
strategy and not to the DT native strategy. See §7.

---

## 2. The splits are not comparable across CV groupings — read this before the metrics

Two things change when `--cv-group` goes from `instance_id` to `well_id`, not one.

### 2.1 Well grouping silently discards the synthetic data

`load_task_data` drops `SIMULATED` and `DRAWN` instances under `well_id` — they have no
physical well to group by. That removes 18,069 of 100,184 windows, and those windows are
precisely the ones propping up the rare classes:

| Class | All sources | Real only | Simulated | Hand-drawn |
|---|---:|---:|---:|---:|
| 0 Normal | 59,797 | 59,797 | 0 | 0 |
| 1 Abrupt BSW Increase | 6,109 | 984 | 4,560 | 565 |
| 2 Spurious DHSV Closure | 938 | 554 | 384 | 0 |
| 5 Rapid Productivity Loss | 1,686 | **369** | 1,317 | 0 |
| 6 Quick PCK Restriction | 2,819 | **243** | 2,576 | 0 |
| 7 PCK Scaling | 7,442 | 5,665 | 0 | 1,777 |
| 8 Hydrate in Production Line | 4,307 | 3,335 | 972 | 0 |
| 9 Hydrate in Service Line | 17,086 | 11,168 | 5,918 | 0 |

Classes 5 and 6 lose 78 % and 91 % of their windows. Any "well grouping is harder" statement
below conflates the grouping change with this data loss.

### 2.2 The well-grouped holdout is pathologically lopsided

`holdout_split` holds out 20 % of the *groups*; with 33 wells that is 8 wells, and those 8
wells turn out to hold **52.3 % of the windows**.

| | windows | groups | share of windows |
|---|---:|---:|---:|
| instance grouping, train+val | 81,180 | 885 | 81.0 % |
| instance grouping, test | 19,004 | 222 | **19.0 %** |
| well grouping, train+val | 39,187 | 25 | 47.7 % |
| well grouping, test | 42,928 | 8 | **52.3 %** |

Per class, the well-grouped split is worse than lopsided — it is inverted:

| Class | train+val | test | % of class in test |
|---|---:|---:|---:|
| 0 Normal | 19,038 | 40,759 | **68.2 %** |
| 1 Abrupt BSW Increase | 294 | 690 | 70.1 % |
| 2 Spurious DHSV Closure | 420 | 134 | 24.2 % |
| 5 Rapid Productivity Loss | **28** | 341 | 92.4 % |
| 6 Quick PCK Restriction | **38** | 205 | 84.4 % |
| 7 PCK Scaling | 5,544 | 121 | 2.1 % |
| 8 Hydrate in Production Line | 2,801 | 534 | 16.0 % |
| 9 Hydrate in Service Line | 11,024 | 144 | **1.3 %** |

Compare the instance-grouped split, where every class lands between 10.2 % and 29.4 % in test.

The cause is concentration: only 8 of 33 real wells record any normal operation at all, and
wells **2** (24,166 Normal windows) and **6** (13,421) alone hold 62.9 % of all Normal windows. Both
landed in the test set (`test wells: 2, 3, 6, 7, 20, 28, 32, 40`). `repair_holdout` guarantees
that every class *appears* on both sides; it says nothing about proportion.

The consequence for training: the trainval prior is 48.6 % Normal, while the test prior is
**94.9 % Normal**. Both models are then explicitly weighted *against* the majority class —
the tree through `class_weight="balanced"`, XGBoost through
`compute_sample_weight("balanced")`, both computed on the trainval prior. A model trained to
avoid Normal, evaluated on a set that is almost entirely Normal, is what produces the
sub-random accuracies in §3.

---

## 3. Headline metrics (held-out test)

| Run | Grouping | Strategy | depth | F1-macro | F1-weighted | Accuracy |
|---|---|---|---:|---:|---:|---:|
| XGB (reference) | instance | — | — | **0.8175** | 0.9202 | 0.9268 |
| **A** DT | instance | full (8 classes) | 6 | **0.6996** | 0.8330 | 0.7888 |
| XGB (reference) | instance | hydrate, collapsed | — | **0.9447** | 0.9562 | 0.9571 |
| **B** DT | instance | hydrate, collapse | 6 | **0.8170** | 0.8414 | 0.8321 |
| **B** DT | instance | hydrate, native | 6 | **0.9553** | 0.9660 | 0.9664 |
| XGB (reference) | well | — | — | 0.0918 | 0.0046 | 0.0153 |
| **C** DT | well | full (8 classes) | 6 | 0.0815 | 0.2380 | 0.1447 |
| XGB (reference) | well | hydrate, collapsed | — | 0.1405 | 0.0084 | 0.0290 |
| **D** DT | well | hydrate, collapse | 5 | 0.3425 | 0.2505 | 0.1632 |
| **D** DT | well | hydrate, native | 4 | 0.1651 | 0.2606 | 0.1648 |

Distillation retention (DT ÷ XGB, F1-macro), like-for-like strategies only:

- instance / 8 classes: 0.6996 / 0.8175 = **85.6 %**
- instance / hydrate, both collapsed: 0.8170 / 0.9447 = **86.5 %**
- well / 8 classes: 0.0815 / 0.0918 = 88.8 % — of nothing.

### Depth sweeps (validation F1-macro on train+val folds)

| Run / strategy | d=2 | d=3 | d=4 | d=5 | d=6 | picked | test |
|---|---:|---:|---:|---:|---:|---:|---:|
| A full | 0.2794 | 0.4565 | 0.6098 | 0.6197 | **0.7597** | 6 | 0.6996 |
| B collapse | 0.4225 | 0.4397 | 0.7024 | 0.7208 | **0.8490** | 6 | 0.8170 |
| B native | 0.8743 | 0.9286 | 0.8916 | 0.9306 | **0.9564** | 6 | 0.9553 |
| C full | 0.0376 | 0.0582 | 0.1621 | 0.2438 | **0.2512** | 6 | 0.0815 |
| D collapse | 0.1188 | 0.1969 | 0.4083 | **0.6428** | 0.4968 | 5 | 0.3425 |
| D native | 0.3003 | 0.2987 | **0.3068** | 0.2694 | 0.2694 | 4 | 0.1651 |

Two things stand out. **The depth budget binds**: four of six sweeps pick 6, the top of the
swept grid, so the sweep is reporting "as deep as allowed", not an optimum. And the
**validation-to-test gap tracks the grouping**: instance runs lose 0.001–0.060 F1-macro going
from validation to test, well runs lose 0.14–0.30. Under well grouping the grouped CV folds
are not predictive of the holdout — unsurprising, since folds over 25 wells and a holdout of
8 different wells are drawn from very different well mixes.

### Per-class detail, instance grouping (run A, 8 classes)

| Class | Precision | Recall | F1 | n |
|---|---:|---:|---:|---:|
| 0 Normal | 0.919 | 0.832 | 0.874 | 12,125 |
| 1 Abrupt BSW Increase | 0.945 | 0.521 | 0.671 | 1,799 |
| 2 Spurious DHSV Closure | 0.010 | 0.243 | **0.019** | 107 |
| 5 Rapid Productivity Loss | 0.823 | 1.000 | 0.903 | 298 |
| 6 Quick PCK Restriction | 0.939 | 0.942 | 0.940 | 583 |
| 7 PCK Scaling | 1.000 | 0.675 | 0.806 | 757 |
| 8 Hydrate in Production Line | 0.870 | 0.300 | **0.446** | 1,071 |
| 9 Hydrate in Service Line | 0.886 | 0.997 | 0.938 | 2,264 |

Class 2 collapses (precision 0.010): the balanced weighting over 8 classes makes the tree
spray Spurious DHSV Closure over thousands of windows to capture 107. XGBoost handles the
same class with precision 1.000 / recall 0.327 — it declines to guess. Class 8 is the other
weak spot in both models (DT F1 0.446, XGB 0.640).

### Per-class detail, instance grouping under the hydrate grouping (run B)

| Group | collapse P / R / F1 | native P / R / F1 | n |
|---|---|---|---:|
| Hydrate | 0.984 / 0.861 / 0.918 | 0.970 / 0.918 / **0.943** | 3,335 |
| Normal | 0.919 / 0.832 / 0.874 | 0.961 / 0.997 / **0.979** | 12,125 |
| Other Problem | 0.558 / 0.805 / 0.659 | 0.985 / 0.906 / **0.944** | 3,544 |

The whole gap between the two strategies sits in *Other Problem*: F1 0.659 → 0.944. The
collapse tree spends its depth budget separating five heterogeneous faults from each other
and then throws that distinction away, and its over-prediction of the rare fault classes
shows up as Other-Problem precision of 0.558.

### Per-class detail, well grouping under the hydrate grouping (run D)

| Group | collapse P / R / F1 | native P / R / F1 | n |
|---|---|---|---:|
| Hydrate | 0.608 / **0.999** / 0.756 | 0.198 / 0.212 / 0.205 | 678 |
| Normal | 0.886 / 0.146 / 0.251 | 0.834 / 0.161 / 0.270 | 40,759 |
| Other Problem | 0.011 / 0.256 / 0.021 | 0.010 / 0.237 / 0.020 | 1,491 |

The collapse strategy's apparently respectable 0.3425 F1-macro is an averaging artifact: it
recovers 99.9 % of Hydrate because it calls almost everything *not Normal* (Normal recall
0.146), and Hydrate is 1.6 % of the test set. Its accuracy (0.163) is indistinguishable from
the native strategy's (0.165). Neither model is usable.

---

## 4. Which features the two rankings select

Stage 5 distils from the top 10 SHAP features of the matching XGBoost run. The two rankings
barely overlap — only `T-JUS-CKP_mean` and `T-TPT_max` are common.

| Rank | instance grouping | share | well grouping | share |
|---:|---|---:|---|---:|
| 1 | T-TPT_min | 9.7 % | T-JUS-CKP_mean | 13.5 % |
| 2 | T-TPT_max | 5.4 % | P-TPT_diff2_std | 7.4 % |
| 3 | P-TPT_max | 4.3 % | T-TPT_mean | 5.9 % |
| 4 | P-TPT_min | 4.0 % | T-TPT_max | 5.9 % |
| 5 | P-MON-CKP_max | 4.0 % | P-MON-CKP_median | 5.8 % |
| 6 | P-JUS-CKGL_max | 3.4 % | T-TPT_median | 4.9 % |
| 7 | P-MON-CKP_min | 3.4 % | T-PDG_mean | 3.3 % |
| 8 | QGL_median | 3.4 % | T-JUS-CKP_median | 3.1 % |
| 9 | T-JUS-CKP_mean | 3.2 % | P-PDG_max | 3.1 % |
| 10 | P-TPT_median | 3.0 % | P-MON-CKP_mean | 2.8 % |
| | **top-10 total** | 43.8 % | **top-10 total** | 55.7 % |

The character of the two lists differs. The instance ranking is dominated by *absolute
level* statistics of pressure (`P-TPT` min/max/median, `P-MON-CKP` min/max). The well
ranking leans on temperatures and, notably, brings in `P-TPT_diff2_std` at rank 2 — a
second-difference dispersion, i.e. a *dynamics* feature that is largely insensitive to a
well's static operating point. This is what one would expect if the instance-grouped model
is partly keying on well identity: with `--no-normalization`, an absolute wellhead pressure
is close to a well fingerprint, and it stops being usable the moment the well is unseen.

---

## 5. Quantifying the well-identity risk in the instance-grouped runs

The concern that motivates `--cv-group well_id` is directly measurable on the
instance-grouped holdout:

- The test set spans 20 wells, **17 of which also appear in train+val**.
- **97.1 % of test windows (18,445 / 19,004) belong to a well the model saw in training.**

Per class, share of test windows whose well is also in training:

| Class | share seen | n |
|---|---:|---:|
| 0 Normal | 100.0 % | 12,125 |
| 1 Abrupt BSW Increase | 100.0 % | 1,799 |
| 2 Spurious DHSV Closure | 76.6 % | 107 |
| 5 Rapid Productivity Loss | 100.0 % | 298 |
| 6 Quick PCK Restriction | 100.0 % | 583 |
| 7 PCK Scaling | 100.0 % | 757 |
| 8 Hydrate in Production Line | 50.1 % | 1,071 |
| 9 Hydrate in Service Line | 100.0 % | 2,264 |

Only classes 2 and 8 carry meaningful unseen-well content — and those are exactly the two
classes where run A performs worst (F1 0.019 and 0.446). That coincidence is suggestive
rather than conclusive, but it is the sharpest evidence in this set of runs that the
instance-grouped scores are inflated by well familiarity.

Well concentration explains why an honest well-level split is so hard to construct:

| Class (real instances) | wells | top well | top-2 wells |
|---|---:|---:|---:|
| 0 Normal | 8 | 40 % | 63 % |
| 1 Abrupt BSW Increase | 3 | 67 % | 97 % |
| 2 Spurious DHSV Closure | 6 | 53 % | 73 % |
| 5 Rapid Productivity Loss | 2 | 92 % | 100 % |
| 6 Quick PCK Restriction | 2 | 84 % | 100 % |
| 7 PCK Scaling | 5 | 51 % | 89 % |
| 8 Hydrate in Production Line | 7 | 39 % | 56 % |
| 9 Hydrate in Service Line | 12 | 58 % | 73 % |

Two classes live in two wells each. With `GroupKFold`/`GroupShuffleSplit` over wells, those
classes can only ever be "one well to train, one well to test" — there is no split in which
they are both well-represented and honestly evaluated.

---

## 6. The trees

### 6.1 Instance grouping, 8 classes (run A, depth 6)

![Tree — instance grouping, 8 classes](../figures/dt_prediction_raw_overlap_from_xgb_tree.png)
![Confusion matrix — instance grouping, 8 classes](../figures/dt_prediction_raw_overlap_from_xgb_confusion_matrix.png)

Full rules: [`dt_prediction_raw_overlap_from_xgb_rules.txt`](../metrics/dt_prediction_raw_overlap_from_xgb_rules.txt).

The root is `QGL_median <= 0.59` — gas lift essentially off versus on. This is an
*operational mode* split, not a fault-physics split, and it is the single most legible thing
the whitening exercise produced: the model first asks what the well is doing, then what is
going wrong with it.

The gas-lift-on branch is short and physically coherent:

```
QGL_median >  0.59
├── T-TPT_max <= 56.03
│   ├── T-JUS-CKP_mean <= 51.07, T-TPT_min <= 53.81 → Hydrate in Service Line
│   └── T-JUS-CKP_mean >  64.15, T-TPT_max > -10.70 → Hydrate in Production Line
└── T-TPT_max >  56.03                              → Hydrate in Service Line
```

Gas lift active plus a cold wet-christmas-tree reading routes straight to hydrate — the
service line *is* the gas-lift line, and hydrate formation in it is exactly a cold-gas
phenomenon. The catch is the third branch: `T-TPT_max > 56.03` alone, with gas lift on,
returns Hydrate in Service Line unconditionally, which is the tree exploiting the fact that
class 9 is the dominant gas-lift-on class (17,086 windows, 5,918 of them simulated) rather
than a statement about hydrates.

The gas-lift-off branch is where the tree becomes hard to defend:

- **Degenerate splits.** Four sibling pairs carry the same label on both sides — three of
  them for Rapid Productivity Loss (`T-TPT_max <= 72.58`, `T-TPT_min <= 126.60`,
  `P-TPT_max <= 20,190,461`) and one for Normal (`T-TPT_min <= 55.76`). These are
  purity-gain splits that do not change a decision; the effective tree is smaller than
  depth 6 suggests.
- **Thresholds on physically impossible readings.** `P-TPT_max <= 63.14` is 63 Pa absolute
  at the wet christmas tree — vacuum, on a live well. `T-TPT_max <= -10.70` and
  `T-TPT_max <= 56.03` on a branch that also contains `T-TPT_min > 126.60` span an
  implausible range for one sensor. The extreme-value masking in `config.py` deliberately
  leaves frozen-at-zero signals alone (`PRESSURE_MIN = 0.0`), so these are very likely
  instrument dropout being used as a fault predictor. Worth confirming directly before
  trusting any rule below these nodes.

### 6.2 Instance grouping, hydrate — native (run B, depth 6)

![Tree — hydrate native](../figures/dt_prediction_raw_overlap_from_xgb_hydrate_native_tree.png)
![Confusion matrix — hydrate native](../figures/dt_prediction_raw_overlap_from_xgb_hydrate_native_confusion_matrix.png)

Full rules: [`dt_prediction_raw_overlap_from_xgb_hydrate_native_rules.txt`](../metrics/dt_prediction_raw_overlap_from_xgb_hydrate_native_rules.txt).

Same root (`QGL_median <= 0.59`), but the gas-lift-on branch now reads as a single
mechanism:

```
QGL_median > 0.59
└── P-JUS-CKGL_max >  13,264,140 Pa (≈133 bar)
    └── T-TPT_max  > -12.44 °C
        ├── QGL_median <= 1.58, P-TPT_max <= 12.98 MPa → Hydrate
        └── QGL_median >  1.58, P-TPT_median <= 22.5 MPa → Hydrate
```

Gas lift flowing, gas-lift line pressure high, tree pressure comparatively low — a
restriction building in the gas-lift line. That is the textbook picture of a service-line
hydrate plug, and it is the clearest candidate rule this exploration has produced. The
gas-lift-off branch handles Normal vs Other Problem and reaches Hydrate only once, via
`P-MON-CKP_max > 2.53 MPa` and `T-TPT_min > 124.18 °C`, which is a hot, choked well —
plausible as a *production-line* hydrate precursor but resting on a single leaf.

This is the best tree in the set: 0.9553 F1-macro, three classes, depth 6, ten features, and
a readable hydrate rule.

### 6.3 Instance grouping, hydrate — collapse (run B, depth 6)

![Tree — hydrate collapse](../figures/dt_prediction_raw_overlap_from_xgb_hydrate_collapse_tree.png)
![Confusion matrix — hydrate collapse](../figures/dt_prediction_raw_overlap_from_xgb_hydrate_collapse_confusion_matrix.png)

Structurally this is **the same tree as run A** — same features, same depth, same seed, same
thresholds — because the collapse strategy trains on the full class set and only its
*predictions* are grouped. It adds nothing to run A's tree except the confusion matrix.

⚠️ **The exported rules and figure for this strategy are mislabeled.** See §8.

### 6.4 Well grouping (runs C and D)

![Tree — well grouping, 8 classes](../figures/dt_prediction_raw_overlap_from_xgb_wellcv_tree.png)
![Tree — well grouping, hydrate native](../figures/dt_prediction_raw_overlap_from_xgb_wellcv_hydrate_native_tree.png)

Rules: [`..._wellcv_rules.txt`](../metrics/dt_prediction_raw_overlap_from_xgb_wellcv_rules.txt),
[`..._wellcv_hydrate_native_rules.txt`](../metrics/dt_prediction_raw_overlap_from_xgb_wellcv_hydrate_native_rules.txt),
[`..._wellcv_hydrate_collapse_rules.txt`](../metrics/dt_prediction_raw_overlap_from_xgb_wellcv_hydrate_collapse_rules.txt).
Confusion matrices: [8 classes](../figures/dt_prediction_raw_overlap_from_xgb_wellcv_confusion_matrix.png),
[native](../figures/dt_prediction_raw_overlap_from_xgb_wellcv_hydrate_native_confusion_matrix.png),
[collapse](../figures/dt_prediction_raw_overlap_from_xgb_wellcv_hydrate_collapse_confusion_matrix.png).

The numbers make these trees indefensible as classifiers, but the depth-4 native tree is
worth reading anyway, because its root is *more* physical than the instance-grouped one:

```
T-JUS-CKP_mean <= 65.25 °C          (cold downstream of the production choke)
├── P-TPT_diff2_std <= 82.67, T-TPT_median <= 106.37, T-TPT_mean > -12.47 → Hydrate
└── P-TPT_diff2_std >  82.67, T-JUS-CKP_mean <= 51.02                     → Hydrate
```

Temperature downstream of the production choke is the canonical hydrate-formation variable —
Joule-Thomson cooling across the choke is the mechanism — and the tree splits on it at the
root, then refines with a pressure-transient measure. When well identity is removed as a
shortcut, both the SHAP ranking (§4) and the tree move toward dynamics and toward the choke
thermodynamics, and away from absolute pressure levels. That is a real signal about *what*
the instance-grouped model is leaning on, even though the well-grouped model itself cannot
be evaluated on this split.

---

## 7. Discussion

**The whitening works — under instance grouping, and only there.** A depth-6 tree on 10
features keeps ~86 % of the tuned ensemble's F1-macro on both the 8-class problem
(0.6996 vs 0.8175) and the collapsed 3-class problem (0.8170 vs 0.9447). That is a
reasonable price for a model a flow-assurance engineer can read in full. Under well
grouping the exercise is vacuous: the tree retains 89 % of an XGBoost that itself scores
0.0918 F1-macro at 1.5 % accuracy. You cannot whiten a black box that has nothing in it.

**The well-grouped collapse is a split-geometry failure first, a generalization failure
second.** A 1.5 % accuracy on a test set that is 94.9 % one class is not "the model learned
wells instead of faults" — it is a model trained with balanced weights on a 48.6 %-Normal
prior and then scored on a 94.9 %-Normal set, on a split where 5 of 8 classes have their
mass on the wrong side (§2.2). Before concluding anything about well-level generalization,
the split has to be fixed. The honest reading of runs C and D today is: *this configuration
did not produce an evaluable model*, not *fault prediction fails across wells*.

**The instance-grouped numbers are inflated, and we can now say by roughly what mechanism.**
97.1 % of test windows come from a well seen in training, rising to 100 % for six of eight
classes. Combined with `--no-normalization` — which leaves absolute pressures and
temperatures, i.e. near-fingerprints of a well's depth and operating point, as the model's
inputs — and with the instance-grouped SHAP ranking being dominated by exactly those absolute
levels, the instance-grouped scores should be read as an upper bound on same-well
performance, not as fault-prediction skill. The two classes with genuine unseen-well content
(2 and 8) are the two the tree handles worst.

**Class grouping helps, and native training is where the help comes from.** Collapsing an
8-class tree's predictions buys +0.117 F1-macro over the 8-class score (0.6996 → 0.8170);
training natively on three labels buys a further +0.138 (0.8170 → 0.9553). The mechanism is
visible per class: *Other Problem* goes from F1 0.659 to 0.944, because the collapse tree
burns depth separating five faults it is then not asked about, and — being class-balanced
over 8 classes — over-predicts the rare ones, costing precision on the merged group. If the
operational question really is triage (normal / hydrate / something else), train for it;
do not train for 8 classes and collapse.

**The native tree beats the XGBoost reference, but the comparison is not yet fair.** DT
native scores 0.9553 against XGB's collapsed 0.9447. This is *not* evidence that a depth-6
tree outperforms a tuned ensemble: the XGBoost was trained on 8 classes and collapsed, and
no natively-trained 3-class XGBoost exists, because `--class-grouping` is only wired into
stages 3 and 5, not 2 and 4. The fair like-for-like pair today is DT collapse (0.8170) vs
XGB collapse (0.9447), which the ensemble wins comfortably. There is a twist worth noting
in the tree's favour, though: the native tree is handicapped — its 10 features were selected
by SHAP on the *8-class* model, so they are optimized for a distinction it does not need —
and it still reaches 0.9553.

**The depth budget is binding.** Four of six sweeps select depth 6, the maximum offered, and
the two that do not are the well-grouped runs, where validation is uninformative anyway. The
run-A sweep is still climbing steeply at the top (0.6197 → 0.7597 from depth 5 to 6), so the
8-class tree is depth-starved rather than converged. The depth at which interpretability
stops being worth the accuracy is exactly the trade-off this exercise exists to locate, and
the current grid cannot locate it.

---

## 8. Known defect in the exported artifacts

The **collapse**-strategy rule files and tree figures are labeled with the *grouped* label
map although the tree was fit on the *fine* labels. In
[`scripts/05_decision_tree.py:335`](../../scripts/05_decision_tree.py#L335), `export_tree`
receives `label_map`, which is `grouping_label_map("hydrate")` = `{0: Hydrate, 1: Normal,
2: Other Problem}`, while `pipe.named_steps["clf"].classes_` is `[0, 1, 2, 5, 6, 7, 8, 9]`.
`export_tree` then maps by value, so the exports read:

| Tree really predicts | Exported as |
|---|---|
| 0 Normal | `Hydrate` |
| 1 Abrupt BSW Increase | `Normal` |
| 2 Spurious DHSV Closure | `Other Problem` |
| 5, 6, 7, 8, 9 | bare integers `5`…`9` |

Affected files: `dt_prediction_raw_overlap_from_xgb_hydrate_collapse_rules.txt` and
`_tree.png`, and the `_wellcv_hydrate_collapse_` pair. **Metrics and confusion matrices are
unaffected** — those are computed after collapsing the predictions, and the collapse there is
correct. The `native` exports are also correct (that tree really is fit on the grouped
labels). `export_tree`'s own docstring states the requirement the caller violates: *"label_map
must be in the tree's own label space"*. The fix is to pass the fine map (`FAULT_CLASSES` /
`WINDOW_CLASSES`) when `collapse` is true.

Until it is fixed, read the collapse tree from the standard-grouping export instead — for the
instance-grouped run they are the identical tree (§6.3).

**Status (2026-09-17, later the same day):** fixed — stage 5 now passes `export_tree` the map of
the label space each tree was fit on. The four runs above were regenerated with their original
`--depths 2,3,4,5,6` grid, so every metric in this report is unchanged and the collapse exports
now read correctly. The tree figures were also redrawn on the new self-sizing canvas
(`interpretation.tree_canvas_size`), so they are wider than when this report was first written.

---

## 9. Suggested next steps

Ordered by how much they would change the conclusions above. **All but 5 have since been done —
see §10 for what they changed**; they are kept here as written, with their status marked.

1. ✅ **Fix the well-level holdout before drawing any well-generalization conclusion.** Options:
   split wells so that the *window* share, not the group share, is near 20 %; or drop the
   single holdout for leave-one-well-out / repeated grouped CV, which gives a distribution of
   scores instead of one unlucky draw and makes per-well failure visible.
2. ✅ **Report the class priors of both splits in the run logs.** A trainval-vs-test prior shift
   of 48.6 % → 94.9 % should not be something a reader has to reconstruct from the parquet.
3. ✅ **Extend `--depths` past 6** (e.g. `2,3,4,5,6,8,10,12`) so the sweep can find an interior
   optimum and the interpretability/accuracy trade-off becomes visible.
4. ✅ **Wire `--class-grouping` into stages 2 and 4** so that a natively-trained 3-class XGBoost
   and a 3-class SHAP ranking exist. Without them the native tree has no fair reference and
   is distilling from a ranking built for a different question. *(§10.2, §10.3 — this one
   reversed a conclusion.)*
5. ✅ **Re-run instance grouping on `features_zscore_overlap.parquet`.** Per-instance z-scoring
   removes the absolute levels that make a well identifiable. If instance-grouped performance
   holds up under normalization, the well-identity hypothesis of §5 weakens considerably; if
   it drops toward the well-grouped numbers, it is confirmed.
   *(Done: [`2026-09-17_normalization_and_groupings.md`](2026-09-17_normalization_and_groupings.md).
   The scores rose everywhere and this test turned out to be confounded — normalization removes the
   between-well level offsets **and** leaks the coming fault into the normal-operation windows,
   because stage 1 scales each instance by statistics of its whole recording. The §5 hypothesis is
   confirmed by a different route: the one class with no leak, Normal, goes from 0.0004 to 0.793
   recall under well grouping, so raw absolute levels do not merely offer a shortcut — they block
   cross-well transfer outright.)*
6. ✅ **Audit the sensor-dropout thresholds.** Check how many training windows satisfy
   `P-TPT_max <= 63.14` and whether they are frozen-at-zero recordings. If they are, either
   mask them or accept explicitly that the model is partly a sensor-health classifier.
   *(Audited, masking deliberately left alone: `scripts/audits/sensor_distributions_auditing.py`
   shows `P-TPT` frozen at exactly 0 for 100 % of the readings of wells 35, 36 and 40 and of
   every hand-drawn instance — so `P-TPT_max <= 63.14 → Abrupt BSW Increase` in §6.1 is, in part,
   the tree detecting hand-drawn data. Well 29's `P-TPT` is a constant 817 bar, well 38's 380 bar.)*
7. ✅ **Prune degenerate splits before export.** Sibling leaves sharing a label (§6.1) inflate
   the apparent depth; collapsing them would make the published trees smaller and honest
   about their effective complexity. *(§10.4 — exactly the four this report counted.)*

---

## 10. Addendum (2026-09-17, later the same day) — what the follow-up work changed

Next steps 1, 2, 3, 4, 6, 7a and the §8 defect have since been implemented. Three of them change
numbers in this report, and one of them reverses a conclusion. Everything below was produced with
the same `--depths 2,3,4,5,6` grid this report used, so the differences are the method's, not the
sweep's.

### 10.1 What did *not* move

Re-run on this report's own `--depths 2,3,4,5,6`, every score in §3 that comes from training on the
dataset's own classes reproduces to four decimals: A `full` 0.6996, B `collapse` 0.8170, C `full`
0.0815, D `collapse` 0.3425. (The artifacts on disk were subsequently re-run on the deeper default
grid — see §10.5.) The split composition of §2 is unchanged as well: the splits are now built from
the fine classes whatever label set is modeled, which is exactly what they were built from before.
So §2, §4, §5, §6.1 and the split-geometry argument of §7 stand as written.

### 10.2 The missing XGBoost-native baseline exists now, and it wins

§7 flagged that DT-native's 0.9553 had no natively-trained ensemble to be compared with. It does
now (`xgb_prediction_raw_overlap_hydrate`), on the same held-out windows:

| Instance grouping, hydrate | F1-macro | Accuracy | Hydrate F1 | Normal F1 | Other Problem F1 |
|---|---:|---:|---:|---:|---:|
| XGB native | **0.9720** | 0.9787 | 0.991 | 0.986 | 0.940 |
| XGB collapse | 0.9447 | 0.9571 | 0.951 | 0.970 | 0.913 |
| DT native (depth 5) | 0.9105 | 0.9335 | 0.941 | 0.955 | 0.835 |
| DT collapse (depth 6) | 0.8170 | 0.8321 | 0.918 | 0.874 | 0.659 |

Two conclusions of §7 need correcting:

- **The tree does not beat the ensemble.** This report's "DT native 0.9553 > XGB collapse 0.9447"
  was the comparison §7 already warned was unfair; with the fair one in hand, the ensemble wins by
  0.06 F1-macro. What survives is the *retention* story, and it is better than before: the tree
  keeps 93.7 % of its ensemble under native training (0.9105 / 0.9720) against 86.5 % under
  collapsing (0.8170 / 0.9447).
- **Native training helps the ensemble too, not just the tree.** +0.027 F1-macro for XGBoost,
  where it was +0.138 for the tree. The gap is the point: a 300-tree ensemble can afford to learn
  eight classes and have the distinction collapsed afterwards, a depth-limited tree cannot. Native
  training buys most where capacity is scarcest, which is the case for the model you actually want
  to read.

### 10.3 Pairing each tree with its own ranking — and the surprise

Stage 5 now distils each strategy from the ranking of the ensemble trained on the same labels it
is: the standard run's for `collapse`, the grouped run's for `native`. This is the fix §7 asked
for, and it **lowers** the native tree's score:

| DT native, instance grouping | features from | depth | F1-macro |
|---|---|---:|---:|
| as reported in §3 | the 8-class ranking | 6 | 0.9553 |
| with the correct pairing | the 3-class ranking | 5 | 0.9105 |

That is worth stating plainly rather than burying: **the ranking built for the harder question is
the better feature set for a shallow tree on the easier one.** The likely reason is that SHAP ranks
features by their contribution *to the model that was fit*, not by standalone usefulness. The
3-class XGBoost reaches 0.972 using features that are individually weak but jointly sufficient for
300 boosted trees — its top-10 leans on `QGL_mean`, `P-PDG_std`, `P-MON-CKP_iqr` — whereas a depth-5
tree needs individually strong splits, which the 8-class ranking happens to supply. The fair
pairing is still the right default: it is the one whose provenance is defensible, and the 0.9105
tree is the one whose features a reader can trace to a model trained on the question being asked.
But "distil the ranking of the matching model" is a principle, not an optimisation — if the goal is
the best shallow tree, the feature set is worth treating as a hyperparameter of its own.

The well-grouped numbers move the same way in sign but stay unusable: D `native` goes from 0.1651
(8-class ranking, depth 4) to 0.3643 (3-class ranking, depth 5), with accuracy 0.2847 — still a
model that calls most of a 95 %-Normal test set something else. And the well-grouped ensemble
gains from native training as well (F1-macro 0.1405 → 0.2552), which changes nothing about §7's
verdict that this split cannot evaluate anything.

### 10.4 The four degenerate splits are gone

§6.1 counted four sibling pairs carrying the same label in the instance-grouped 8-class tree. The
pruning implemented since collapses every split whose *whole subtree* agrees on one class, and on
that tree it removes **exactly four**, taking it from 33 leaves to 29 at unchanged depth 6 and
unchanged predictions. The other trees of this report lose 3 (B native, D collapse, D native) and
6 (C full, 24 → 18 leaves). So the effective complexity of these trees was 10–25 % lower than §6
reported, and every tree figure and rules file listed in the appendix has been redrawn accordingly.

### 10.5 Consequences for the artifacts this report cites

⚠️ **The decision-tree artifacts on disk no longer match §3 and §6.** The follow-up report on the
z-scored features needed a like-for-like raw baseline, and its trees are swept to depth 12, so the
raw trees were re-run on the same `2,…,12` grid and the `dt_prediction_raw_overlap_*` artifacts now
hold *those*. Concretely:

| Run | §3 (grid `2…6`) | on disk now (grid `2…12`) |
|---|---|---|
| A `full` | depth 6, F1-macro 0.6996 | depth 12, F1-macro 0.7792 |
| B `collapse` | depth 6, 0.8170 | depth 12, **0.9688** |
| B `native` | depth 6, 0.9553 | depth 12, 0.9540 |
| C `full` | depth 6, 0.0815 | depth 8, 0.0420 |
| D `collapse` | depth 5, 0.3425 | depth 7, 0.1043 |
| D `native` | depth 4, 0.1651 | depth 5, 0.3643 |

The §3 tables stay as the record of what this report measured; the grid is stated in §1 and the
run is reproducible with `--depths 2,3,4,5,6`. Two of the deep-grid results are worth noting in
passing: the instance-grouped `collapse` tree at depth 12 reaches 0.9688, nearly the XGBoost-native
0.9720 of §10.2, and the well-grouped trees get *worse* with depth — more evidence that the split
of §2.2, not the model, is the constraint there.

The **ensemble** numbers of §3 are untouched by any of this: XGBoost has no depth sweep, and
`xgb_prediction_raw_overlap[_wellcv]_metrics.json` still reads 0.8175, 0.9447, 0.0918, 0.1405.

Other artifact changes:

- Every rules file and tree figure is **regenerated**: pruned (§10.4), and with the
  collapse-strategy label bug of §8 fixed.
- `dt_prediction_raw_overlap_from_xgb_hydrate_native_*` and `..._wellcv_hydrate_native_*` describe
  the newly paired trees (§10.3), not the ones §3 scored.
- `xgb_prediction_raw_overlap_hydrate_metrics.json` used to hold the *collapse* of the standard
  run; it now holds both strategies, keyed by name, with the collapse block carrying the same
  numbers as before.
- New artifacts: `xgb_prediction_raw_overlap[_wellcv]_hydrate_{search.json,importance.json,...}` —
  the natively-trained ensembles and their rankings.

### 10.6 On the note in §1

§1 states the sweep as `2,…,12` because that is the pipeline's default after next step 3 and what
the artifacts on disk now use; every number in §3 was produced with this report's original
`2,3,4,5,6`, and §10.5 gives both.

---

## Appendix — artifact index

| Run | metrics | rules | tree | confusion matrix |
|---|---|---|---|---|
| A instance / 8 | `dt_prediction_raw_overlap_from_xgb_metrics.json` | `..._from_xgb_rules.txt` | `..._from_xgb_tree.png` | `..._from_xgb_confusion_matrix.png` |
| B instance / hydrate | `..._from_xgb_hydrate_metrics.json` | `..._hydrate_{collapse,native}_rules.txt` | `..._hydrate_{collapse,native}_tree.png` | `..._hydrate_{collapse,native}_confusion_matrix.png` |
| C well / 8 | `..._from_xgb_wellcv_metrics.json` | `..._wellcv_rules.txt` | `..._wellcv_tree.png` | `..._wellcv_confusion_matrix.png` |
| D well / hydrate | `..._from_xgb_wellcv_hydrate_metrics.json` | `..._wellcv_hydrate_{collapse,native}_rules.txt` | `..._wellcv_hydrate_{collapse,native}_tree.png` | `..._wellcv_hydrate_{collapse,native}_confusion_matrix.png` |

XGBoost references: `xgb_prediction_raw_overlap[_wellcv][_hydrate]_metrics.json`,
`xgb_prediction_raw_overlap[_wellcv]_search.json`,
`xgb_prediction_raw_overlap[_wellcv]_importance.json`.

XGBoost hyperparameters selected (`_search.json`):

| | instance grouping | well grouping |
|---|---|---|
| n_estimators | 300 | 100 |
| max_depth | 3 | 8 |
| learning_rate | 0.1 | 0.2 |
| subsample | 0.8 | 1.0 |
| colsample_bytree | 1.0 | 0.8 |
| min_child_weight | 3 | 3 |
| best CV F1-macro | 0.8759 | 0.3630 |
