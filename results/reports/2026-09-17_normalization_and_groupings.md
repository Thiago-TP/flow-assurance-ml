# Per-instance normalization under both groupings: one switch, two opposite effects

**Date:** 2026-09-17
**Task:** `prediction` — from windows of normal operation only, predict which fault the well will develop.
**Dataset flags:** `--allow-overlap`, extreme values masked (default). The only variable against the
first report is normalization: these runs drop `--no-normalization`, so stage 1 z-scores every
sensor per instance.
**Companion to:** [`2026-09-17_dt_from_xgb_cv_and_class_grouping.md`](2026-09-17_dt_from_xgb_cv_and_class_grouping.md),
whose §9 next step 5 asked for exactly this comparison.
**Question as posed:** report 1 argued that with raw features the model can read a well's identity
off its absolute pressure and temperature levels, and predicted that per-instance z-scoring —
which removes those levels — would make the instance-grouped scores *fall* toward the well-grouped
ones if the well-identity hypothesis was right.

**Short answer:** the scores rise, everywhere, and the prediction was wrong for an interesting
reason. Normalization does two things at once, and they pull in opposite directions:

1. it removes the between-well level offsets, which is what a cross-well model needs — the
   well-grouped ensemble goes from 1.5 % accuracy to 77.6 %;
2. it **leaks the coming fault into the normal-operation windows**, because stage 1 divides every
   instance by statistics computed over the *whole recording, fault period included*, and only
   afterwards keeps the normal windows.

The second effect is a defect, it is large, and it inflates every z-scored number on the
prediction task. §3 quantifies it and §4 separates the two effects with a control class.

---

## 1. What was run

Eight configurations: two CV groupings × two class groupings × {XGBoost, distilled tree}, on
`features_zscore_overlap.parquet`, with the pipeline as it stands after the follow-up work of
report 1 §10 — class grouping wired through every stage, depth sweep `2…12`, trees pruned, splits
reported.

```sh
# per CV grouping, per class grouping; --eval holdout for both groupings, see below
uv run scripts/02_train_val_test.py --model xgb --allow-overlap [--cv-group well_id] \
    --eval holdout --class-grouping {standard,hydrate}
uv run scripts/04_interpret.py     --model xgb --allow-overlap [--cv-group well_id] \
    --eval holdout --class-grouping {standard,hydrate}
uv run scripts/03_evaluate.py      --model xgb --allow-overlap [--cv-group well_id] \
    --eval holdout --class-grouping {standard,hydrate}
uv run scripts/05_decision_tree.py --model xgb --allow-overlap [--cv-group well_id] \
    --eval holdout --class-grouping {standard,hydrate}
```

Two deliberate choices about protocol:

- **`--eval holdout` for the well grouping**, although leave-one-well-out is now its default. This
  report exists to isolate *normalization*, so it holds the protocol fixed at what report 1 used.
  Leave-one-well-out on these features is running separately; §8 says what it will add.
- **The raw baselines were re-run on the same `2…12` depth grid**, since report 1's trees were
  swept only to depth 6 and the sweep was still climbing there. Every raw-versus-normalized
  comparison below is therefore like-for-like. The raw *ensemble* numbers are unaffected by the
  grid and match report 1 exactly.

---

## 2. Headline metrics

XGBoost, held-out test. The grouping columns are the same eight test wells / 222 test instances in
both halves of the table — splits are built from the dataset's own classes whatever is modeled, so
every run here is scored on identical windows.

| CV grouping | Class grouping | | raw F1-macro | raw accuracy | **z-scored F1-macro** | **z-scored accuracy** |
|---|---|---|---:|---:|---:|---:|
| instance | standard (8 classes) | | 0.8175 | 0.9268 | **0.9437** | **0.9609** |
| instance | hydrate, native | | 0.9720 | 0.9787 | 0.9680 | 0.9737 |
| instance | hydrate, collapse | | 0.9447 | 0.9571 | **0.9668** | **0.9726** |
| well | standard (8 classes) | | 0.0918 | 0.0153 | **0.3282** | **0.7761** |
| well | hydrate, native | | 0.2552 | 0.0384 | **0.4484** | 0.3920 |
| well | hydrate, collapse | | 0.1405 | 0.0290 | **0.5997** | **0.7847** |

Distilled trees, depth swept `2…12`, same held-out data:

| CV grouping | Class grouping | raw depth | raw F1-macro | z-scored depth | z-scored F1-macro |
|---|---|---:|---:|---:|---:|
| instance | standard | 12 | 0.7792 | 12 | **0.8468** |
| instance | hydrate, collapse | 12 | **0.9688** | 12 | 0.9343 |
| instance | hydrate, native | 12 | 0.9540 | 10 | 0.9349 |
| well | standard | 8 | 0.0420 | 12 | **0.1508** |
| well | hydrate, collapse | 7 | 0.1043 | 12 | **0.3984** |
| well | hydrate, native | 5 | **0.3643** | 7 | 0.2253 |

The single most striking number in this report is not a macro F1. It is the well-grouped
ensemble's **accuracy: 0.0153 raw, 0.7761 normalized** — from far below "guess the majority class"
to a model that is actually doing something on eight wells it has never seen.

---

## 3. The leak: stage 1 divides by the future

### 3.1 The mechanism

`features.extract_instance_features` calls `preprocessing.normalize_instance` on the **whole
cleaned instance**, then cuts windows, and only later does `train_val_test.load_task_data` keep
the rows with `window_label == 0` for the prediction task. So a normal-operation window's features
are

```
z = (x − mean(whole recording)) / std(whole recording)
```

and both the centre and the divisor are computed over samples that include the fault the model is
being asked to predict. A window recorded before anything went wrong is scaled by how badly it
later went wrong.

This is a label leak specific to the *prediction* task. (For `detection` the same statistics are
still unavailable at inference time — you do not have the rest of the recording — so it remains a
deployment problem there, but it is not a leak of the label.)

### 3.2 How big it is

`scripts/audits/normalization_leakage_auditing.py` measures the channel directly: per instance and
sensor it compares the divisor stage 1 uses (whole recording) with the one a leak-free pipeline
would use (the normal-operation samples alone). A ratio of 1 means the fault period does not change
the divisor. Over the 1,693 instances that have both a normal period and a labelled fault period:

| Fault class (real instances only) | median std ratio | p90 | median normal share |
|---|---:|---:|---:|
| 0 Normal | **1.037** | 1.41 | 83 % |
| 9 Hydrate in Service Line | 1.168 | 5.58 | 77 % |
| 6 Quick PCK Restriction | 1.899 | 12.3 | 36 % |
| 7 PCK Scaling | 1.975 | 8.84 | 12 % |
| 5 Rapid Productivity Loss | 4.614 | 38.8 | 10 % |
| 1 Abrupt BSW Increase | 10.012 | 44.6 | 71 % |
| 8 Hydrate in Production Line | 29.098 | 175 | 19 % |
| 2 Spurious DHSV Closure | **372.350** | 2,192 | 34 % |

Class 0 is the control: a Normal instance has no fault period, so its ratio is 1.04 — the method
returns "no leak" exactly where there is none. Every fault class sits above it, and they sit at
very different heights. A normal-operation window of a Spurious DHSV instance is divided by a
number ~360× larger than the one a leak-free pipeline would use; a window of a Hydrate-in-Service
instance by ~1.17×. **The divisor carries the label.**

Simulated instances are worse by orders of magnitude (medians 815 to 56,195) because they start
from a steady state, so their normal period is nearly flat and the ratio explodes. They are
reported separately in the audit for that reason, and they are 16 % of the prediction dataset.

### 3.3 The leak predicts which classes normalization helps

If the leak is what lifts the instance-grouped scores, then the classes with the largest ratios
should gain the most recall when switching from raw to z-scored features. They do:

| Class | leak (median std ratio) | raw recall | z-scored recall | Δ recall |
|---|---:|---:|---:|---:|
| 0 Normal | 1.04 | 1.000 | 0.995 | −0.005 |
| 9 Hydrate in Service Line | 1.17 | 0.999 | 0.996 | −0.003 |
| 6 Quick PCK Restriction | 1.90 | 0.986 | 0.966 | −0.021 |
| 7 PCK Scaling | 1.98 | 0.744 | 0.840 | **+0.096** |
| 5 Rapid Productivity Loss | 4.61 | 1.000 | 0.997 | −0.003 |
| 1 Abrupt BSW Increase | 10.01 | 0.676 | 0.838 | **+0.161** |
| 8 Hydrate in Production Line | 29.10 | 0.501 | 0.779 | **+0.277** |
| 2 Spurious DHSV Closure | 372.35 | 0.327 | 0.991 | **+0.664** |

Spearman ρ between the leak and the recall gain, over all eight classes: **0.833 (p = 0.010)**.
Four classes are already at ≥ 0.986 recall on raw features and cannot gain; among the four with
headroom the relationship is **perfectly monotone (ρ = 1.000)** — PCK Scaling (leak 2.0) gains
0.096, Abrupt BSW (10.0) gains 0.161, Hydrate in Production (29.1) gains 0.277, Spurious DHSV
(372) gains 0.664.

Four points is not a proof, and the classes differ in more than their leak. But a mechanism
derived from the source code, a control class that reads exactly 1.0, and a monotone
dose-response across the classes that have room to move is as much as an observational argument
can offer. The instance-grouped z-scored numbers of §2 should not be treated as fault-prediction
skill.

---

## 4. The transfer: the control class isolates the genuine half

If normalization only leaked, nothing should improve where the leak is absent. Class 0 (Normal) is
that case — leak ratio 1.037, the lowest of any class, and under instance grouping it indeed gains
nothing (recall 1.000 → 0.995).

Under **well** grouping, the same class:

| Normal, well-grouped holdout (n = 40,759) | precision | recall | F1 |
|---|---:|---:|---:|
| raw | 0.027 | **0.0004** | 0.001 |
| z-scored | 0.975 | **0.7934** | 0.875 |

A class with no leak goes from 4 correct windows in 10,000 to 79 in 100, on wells the model has
never seen. That gain cannot be the leak. It is the other half of what normalization does: it
removes the absolute operating point, so a model trained on 25 wells is no longer being asked to
extrapolate a pressure scale it has never observed.

This **confirms report 1 §5's hypothesis, and strengthens it**. Report 1 argued that raw absolute
levels are close to a well fingerprint. The well-grouped results show they are worse than a
shortcut: with raw features the between-well level shift *actively destroys* cross-well
generalization, and removing it is the single change that makes a well-grouped model exist at all.

Report 1's *predicted test* — "if instance-grouped performance drops toward the well-grouped
numbers, the hypothesis is confirmed" — was simply confounded. Both effects fire at once, and on
the instance grouping the leak dominates.

Per well, the well-grouped 8-class ensemble:

| well | windows | raw F1-macro / accuracy | z-scored F1-macro / accuracy |
|---|---:|---|---|
| 2 | 24,429 | 0.001 / 0.001 | 0.168 / **0.713** |
| 6 | 14,200 | 0.027 / 0.027 | 0.294 / **0.869** |
| 3 | 3,186 | 0.020 / 0.031 | 0.499 / **0.995** |
| 20 | 366 | 0.185 / 0.068 | 0.167 / 0.068 |
| 28 | 292 | 0.000 / 0.000 | 0.000 / 0.000 |
| 32 | 242 | 0.000 / 0.000 | **1.000 / 1.000** |
| 40 | 119 | 1.000 / 1.000 | 1.000 / 1.000 |
| 7 | 94 | 0.000 / 0.000 | 0.014 / 0.021 |

The three large wells — 89 % of the test windows between them — go from nothing to 0.71–0.99
accuracy. The small ones stay all-or-nothing, which is what a single well carrying a single class
looks like. Wells 28 and 32 both hold only Hydrate in Production Line and land on opposite
extremes; that is the variance a single holdout cannot show and leave-one-well-out is meant to
replace.

---

## 5. Class grouping: the collapse/native ordering flips

| | native | collapse | native − collapse |
|---|---:|---:|---:|
| raw, instance | **0.9720** | 0.9447 | +0.027 |
| z-scored, instance | 0.9680 | 0.9668 | +0.001 |
| raw, well | **0.2552** | 0.1405 | +0.115 |
| z-scored, well | 0.4484 | **0.5997** | **−0.151** |

Native training — fitting the ensemble on Normal / Other Problem / Hydrate directly — is worth a
lot when the features are weak and nothing when they are strong, and under well grouping with
z-scored features it becomes actively harmful.

The reading that fits all four cells: native training concentrates limited capacity on the question
being asked, which is worth most when there is little signal to go round. When the features are
good, the 8-class model learns finer structure that survives collapsing, and an accurate fine model
collapsed beats a coarse model trained directly. The z-scored well case shows the cost of the
coarse model plainly — native reaches 0.4484 F1-macro at **0.3920 accuracy** against collapse's
0.5997 at **0.7847**: with three balanced classes over a test set that is 94.9 % Normal, the native
model spends its weight on the minorities and stops predicting the majority.

The same flip appears in the trees (instance: raw collapse 0.9688 > native 0.9540; z-scored
collapse 0.9343 ≈ native 0.9349), so it is not an artefact of ensemble capacity.

**Practical consequence:** "train for the question you will be asked" is good advice only when the
question is harder than your features. With z-scored features and the triage question, train on all
eight classes and collapse.

---

## 6. What the features and the trees say

Top-10 SHAP features, z-scored runs (compare report 1 §4, where the instance ranking was dominated
by absolute `P-TPT` and `P-MON-CKP` levels):

| Run | top-10 |
|---|---|
| instance, 8 classes | `T-TPT_diff2_std`, `P-TPT_min`, `P-TPT_max`, `T-TPT_min`, `QGL_max_zscore`, `P-MON-CKP_min`, `T-TPT_std`, `T-TPT_max`, `T-JUS-CKP_std`, `T-JUS-CKP_diff2_std` |
| instance, hydrate | `QGL_max_zscore`, `T-TPT_diff2_std`, `P-TPT_max`, `P-MON-CKP_diff2_std`, `QGL_max`, `T-JUS-CKP_std`, `QGL_std`, `P-TPT_std`, `T-JUS-CKP_diff2_std`, `QGL_iqr` |
| well, 8 classes | `T-TPT_diff2_std`, `P-MON-CKP_std`, `QGL_median`, `T-TPT_std`, `QGL_min`, `P-TPT_diff2_std`, `P-MON-CKP_diff1_std`, `T-TPT_diff1_std`, `QGL_diff2_std`, `T-JUS-CKP_diff2_std` |
| well, hydrate | `T-TPT_diff2_std`, `T-PDG_max_zscore`, `QGL_diff1_std`, `P-TPT_std`, `QGL_kurtosis`, `QGL_diff2_std`, `T-JUS-CKP_std`, `P-JUS-CKGL_diff2_std`, `QGL_median`, `P-TPT_max_zscore` |

Two observations, one reassuring and one not.

**Reassuring:** the well-grouped ranking is now *entirely* dispersion and derivative statistics —
`_std`, `_diff1_std`, `_diff2_std`, `_kurtosis`. When a model cannot lean on levels and cannot
memorize wells, what is left is the shape of the signal, which is what a physical fault signature
is. This is the same shift report 1 §4 noticed in miniature, now complete.

**Not reassuring:** the instance-grouped ranking keeps `P-TPT_min`, `P-TPT_max`, `T-TPT_min`,
`P-MON-CKP_min` — which, in z-scored units, *are* the leaky quantities: `(x − instance mean) /
instance std`, both terms contaminated. The instance ranking is partly a ranking of how well each
sensor's extremes encode the divisor.

The trees make it concrete. The z-scored well-grouped tree opens:

```
P-MON-CKP_std <= 0.00
├── QGL_diff2_std <= 0.00
│   ├── T-TPT_diff1_std <= 0.02 → Abrupt BSW Increase
│   └── T-TPT_diff1_std >  0.02 → Normal
└── QGL_diff2_std >  0.00
    ├── QGL_min <= 0.91, T-JUS-CKP_diff2_std <= 0.00 → Hydrate in Service Line
    └── QGL_min >  0.91                              → PCK Scaling
```

Every split at the top is a test of whether a standard deviation is **zero** — that is, whether a
sensor is *frozen*. After normalization the most informative thing about a window is which of its
sensors are flat. That is the instrument-health hypothesis of report 1 §6.1 arriving at the root of
the tree rather than in a corner of it, and it is consistent with
`scripts/audits/sensor_distributions_auditing.py`, which found `P-TPT` frozen at exactly 0 for
100 % of the readings of wells 35, 36 and 40 and of every hand-drawn instance.

The z-scored instance hydrate-native tree (depth 10, 355 rules) opens the same way —
`QGL_max_zscore <= 0.11`, then `P-MON-CKP_diff2_std <= 0.00`, then `T-JUS-CKP_std <= 0.00` — and
then spends most of its depth on thresholds of `P-TPT_max` in z-score units, i.e. on the leak.

Neither tree is a flow-physics model. Report 1's best artefact — the gas-lift-on / cold-WCT →
service-line hydrate rule — has no counterpart here, because the variables it was built from no
longer carry levels.

---

## 7. Discussion

**Normalization is not a knob with a sign.** Report 1 treated `--no-normalization` as a switch
between "absolute levels visible" and "absolute levels hidden", and asked which scored better. The
honest answer is that the two settings differ in *three* ways: levels visible or not, cross-well
transfer possible or not, and — only for the z-scored side, and only for the prediction task — a
label leak of up to two orders of magnitude. Any single score comparing them sums all three.

**The best numbers in either report are the least trustworthy.** The z-scored instance-grouped
ensemble at 0.9437 F1-macro / 0.9609 accuracy is the best model in this project so far and it
should not be reported as a fault predictor: §3 shows its largest per-class gains are exactly where
the leak is largest. By the same token report 1's headline — the raw instance-grouped runs — is
inflated by well familiarity (97.1 % of test windows from a well seen in training). The only
configuration free of *both* problems is **well grouping on z-scored features**, and it scores
0.3282 F1-macro at 0.7761 accuracy, with two of eight classes never predicted at all. That is the
number to quote as the current state of cross-well fault prediction.

**The leak is fixable and worth fixing before anything else.** Normalizing by statistics of the
normal-operation prefix only — or of a fixed leading window — keeps everything normalization is
for (removing the well's operating point, enabling transfer) and removes what it should never have
brought (the fault's magnitude). It is a change to one function, `preprocessing.normalize_instance`,
plus the decision of which samples define the baseline. Every z-scored number in this report should
be regenerated afterwards, and the expectation is that the instance-grouped scores fall — toward
the well-grouped ones, which is what report 1 predicted for the wrong reason.

**The depth budget is no longer the binding constraint.** With the `2…12` grid, two of six z-scored
sweeps pick an interior optimum (instance hydrate-native at 10, well hydrate-native at 7) and three
of six raw ones do (well standard at 8, well hydrate collapse at 7, native at 5). Report 1's
"four of six at the boundary" is resolved; where 12 is still chosen, the sweep has flattened
(instance standard gains 0.0027 F1-macro from depth 11 to 12).

**Pruning changed the published trees more than it changed the numbers** — by construction it
changed the numbers not at all:

| depth-swept tree | grown leaves | after pruning | removed |
|---|---:|---:|---:|
| z-scored, instance, 8 classes (d12) | 258 | 206 | 52 (20 %) |
| z-scored, instance, hydrate native (d10) | 150 | 119 | 31 (21 %) |
| z-scored, well, 8 classes (d12) | 204 | 174 | 30 (15 %) |
| z-scored, well, hydrate native (d7) | 73 | 62 | 11 (15 %) |
| raw, instance, 8 classes (d12) | 86 | 67 | 19 (22 %) |
| raw, well, 8 classes (d8) | 31 | 25 | 6 (19 %) |

So 15–22 % of every published tree was splits that decided nothing — a consistent fraction across
normalizations, groupings and depths, which suggests it is a property of purity-driven growth under
a depth cap rather than of any one dataset.

---

## 8. Not done here, and next steps

**Leave-one-well-out on these features was started and stopped.** The well results of §2 and §4
rest on one draw of eight wells, and §4's per-well table already shows two wells carrying the same
single class landing at 0.000 and 1.000 — exactly the variance a single draw cannot expose.
Leave-one-well-out replaces it with 33 scores and is what the pipeline now defaults to for wells.
Measured cost on this machine: **~6 minutes per outer fold**, so ≈ 3.5 h per label set and ≈ 7 h for
both, before stage 5. That is a deliberate decision to leave to the reader rather than to spend
unattended; no partial artifacts were left behind. To run it:

```sh
uv run scripts/02_train_val_test.py --model xgb --allow-overlap --cv-group well_id \
    --class-grouping {standard,hydrate}          # --eval leave-one-out is the default here
uv run scripts/03_evaluate.py      --model xgb --allow-overlap --cv-group well_id \
    --class-grouping hydrate                      # per-well sheet + both strategies
```

Expect it to lower the headline of §2 and widen the spread: the holdout draw gives wells 32 and 40
(pure Hydrate, 361 windows between them) a free 1.000, which 33 folds will not.

Ordered by how much they would change the conclusions above:

1. **Fix the normalization leak** (§3, §7): compute each instance's z-score statistics from its
   normal-operation prefix, not the whole recording. Then re-run everything in this report. This is
   the single change that determines whether the z-scored prediction numbers mean anything.
2. **Re-run report 1's raw configurations against the fixed normalization**, so the
   raw-versus-normalized comparison finally isolates one variable.
3. **Decide what to do about frozen sensors** (§6). The z-scored trees root on "is this sensor
   flat", and `sensor_distributions_auditing.py` shows entire wells with `P-TPT` frozen at zero.
   Either drop instances whose critical sensors never move, or add an explicit
   *sensor-health* feature so the model states that it is using instrument status rather than
   smuggling it through a standard deviation.
4. **Report the triage question binary** (Hydrate / Not Hydrate) alongside the three-group one.
   §5 shows the three-class native model failing by over-predicting minorities on a 94.9 %-Normal
   test set; a binary target with a tunable threshold addresses that directly, and
   `CUSTOM_CLASS_GROUPING` already supports it with no code change.
5. **Give the well grouping more wells.** Classes 5 and 6 are never predicted under well grouping
   in either normalization, and they have 2 wells each (report 1 §5). No split or protocol fixes a
   class that exists in two wells; either accept that those two classes are out of scope for
   cross-well work, or fold them into *Other Problem* and say so.

---

## Appendix — artifact index

Tags follow `<model>_<task>_<norm>[_overlap][_wellcv][_loo]_[<class-grouping>]`.

| Run | metrics | tree / rules |
|---|---|---|
| instance, 8 classes | `xgb_prediction_zscore_overlap_metrics.json` | `dt_prediction_zscore_overlap_from_xgb_{rules.txt,tree.png}` |
| instance, hydrate | `xgb_prediction_zscore_overlap_hydrate_metrics.json` | `dt_prediction_zscore_overlap_from_xgb_hydrate_{collapse,native}_*` |
| well, 8 classes | `xgb_prediction_zscore_overlap_wellcv_metrics.json` | `dt_prediction_zscore_overlap_from_xgb_wellcv_*` |
| well, hydrate | `xgb_prediction_zscore_overlap_wellcv_hydrate_metrics.json` | `dt_prediction_zscore_overlap_from_xgb_wellcv_hydrate_{collapse,native}_*` |

Rankings: `xgb_prediction_zscore_overlap[_wellcv][_hydrate]_importance.json` and the matching
`_shap.png` / `_gain.png` / `_permutation.png`.

Audits behind §3 and §6:

- `results/audits/normalization_leakage_20260917_175347.txt` — the leak, per class, source and sensor.
- `results/audits/sensor_distributions.pdf` — frozen and out-of-band readings per well and source.
- `results/audits/well_leakage_prediction_raw_overlap_*.txt` — the 97.1 % figure of report 1 §5.

XGBoost, z-scored runs (full parameter sets in the matching `_search.json`):

| | instance / standard | instance / hydrate | well / standard | well / hydrate |
|---|---:|---:|---:|---:|
| best CV F1-macro (validation) | 0.8503 | 0.9633 | 0.2805 | 0.3934 |
| held-out F1-macro | 0.9437 | 0.9680 | 0.3282 | 0.4484 |
| n_estimators | 300 | 300 | 300 | 300 |
| max_depth | 3 | 3 | 8 | 3 |
| learning_rate | 0.1 | 0.1 | 0.05 | 0.1 |
| subsample | 0.8 | 0.8 | 1.0 | 0.8 |
| colsample_bytree | 1.0 | 1.0 | 0.8 | 1.0 |
| min_child_weight | 3 | 3 | 5 | 3 |

Note that under well grouping the held-out score is *higher* than the validation score in both
label sets, the reverse of the instance runs and of report 1's raw well runs (validation 0.363,
test 0.092). With z-scored features the grouped validation folds over 25 wells are no longer wildly
optimistic about 8 unseen ones — another sign that the transfer problem, not the model, was what
raw features created.
