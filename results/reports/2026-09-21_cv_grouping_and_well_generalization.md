# CV grouping: the predictor does not generalize to unseen wells

**Exploration goal.** Every score this project reports depends on what a split keeps together.
`--cv-group instance_id` keeps the windows of one recording on one side; `--cv-group well_id`
additionally keeps every recording of one well together, so a held-out well is genuinely unseen.
Report 2 (`2026-09-17_dt_from_xgb_cv_and_class_grouping.md`) warned that the two are not
comparable and that the well-grouped holdout was a lopsided lottery. This report asks the question
underneath that warning:

**Does the fault predictor work on a well it has never seen?**

The answer is no, and the more interesting results are *why the usual summary statistics hide it*
and *why the obvious remedy does not work*. The per-well score sheet — the artifact the well
grouping exists to produce — turns out to be mostly a coin flip, because **25 of the 33 real wells
carry exactly one fault class**. And collapsing to the three-group triage question, which this
report's first draft predicted would rescue the task, raises the score and the chance baseline by
almost exactly the same amount (§6).

---

## 1. What was run

36 runs: 3 normalization references × 3 frozen-sensor policies × 2 CV groupings × 2 label sets.
This report reads the grouping axis; the other two have reports of their own
(`2026-09-21_normalization_after_the_leak_fix.md`,
`2026-09-21_frozen_sensors_and_instrument_status.md`).

```bash
uv run main.py --model xgb --task prediction --allow-overlap --skip-permutation \
  --cv-group {instance_id,well_id} [--eval nested] \
  --normalization {none,instance,normal-operation-values} \
  --frozen-sensors {keep,flag,drop}
```

- **Dataset**: `data/features_overlap.parquet`, commit `4a59209`, `--allow-overlap`.
- **Label sets**: the dataset's own 8 classes, and the `hydrate` grouping (Normal / Other Problem /
  Hydrate) as both `native` (trained on the three groups) and `collapse` (the 8-class model's
  predictions mapped onto them). §6 uses these to test this report's own prediction that collapsing
  might rescue the well-grouped task.

> [!IMPORTANT]
> **The two groupings are not comparable, for three independent reasons.** Any cross-grouping
> difference below mixes all three, and none of them is the grouping's effect on generalization:
>
> 1. **Different protocols.** Instance grouping uses `holdout` (its default); well grouping uses
>    `nested` with 5 outer folds. Leave-one-out is the documented default for well grouping but
>    costs ~33 inner searches per run — about 11 h of CPU for nine runs — so nested was substituted.
> 2. **Different data.** The well grouping drops all synthetic instances (§2), evaluating 82 % of
>    the windows the instance grouping uses.
> 3. **Different difficulty.** A held-out instance usually comes from a well that also contributed
>    training instances; a held-out well does not. That *is* the point of the comparison, but it
>    means the gap is not a measurement error to be corrected.

---

## 2. What the well grouping discards

`--cv-group well_id` keeps only real field recordings, because simulated and hand-drawn instances
have no well to group by:

| source          | windows           | instances       |
| --------------- | ----------------- | --------------- |
| `WELL` (real) | 82,115            | 658             |
| `SIMULATED`   | 15,727            | 439             |
| `DRAWN`       | 2,342             | 10              |
| **total** | **100,184** | **1,107** |

82 % of windows survive, but only 59 % of instances — the synthetic data is many short recordings.
The damage is very unevenly distributed across classes:

| class                        | windows, all sources | windows, real only | **wells carrying it** |
| ---------------------------- | -------------------- | ------------------ | --------------------------- |
| 0 Normal                     | 59,797               | 59,797             | **8**                 |
| 1 Abrupt BSW Increase        | 6,109                | 984 (−84 %)       | **3**                 |
| 2 Spurious DHSV Closure      | 938                  | 554 (−41 %)       | 6                           |
| 5 Rapid Productivity Loss    | 1,686                | 369 (−78 %)       | **2**                 |
| 6 Quick PCK Restriction      | 2,819                | 243 (−91 %)       | **2**                 |
| 7 PCK Scaling                | 7,442                | 5,665 (−24 %)     | 5                           |
| 8 Hydrate in Production Line | 4,307                | 3,335 (−23 %)     | 7                           |
| 9 Hydrate in Service Line    | 17,086               | 11,168 (−35 %)    | 12                          |

Three classes survive in **two or three wells**. Under 5-fold grouped CV a class living in two
wells can be evaluated in at most two folds, and the coverage repair
(`train_val_test.coverage_folds`) pins a class confined to a single group to the training side,
where it is never evaluated at all. Quick PCK Restriction is down to 243 windows across 2 wells —
a quarter of one percent of the data.

---

## 3. Headline metrics

F1-macro on held-out data.

| normalization               | frozen | instance (holdout) | well (nested)    | Δ                |
| --------------------------- | ------ | ------------------ | ---------------- | ----------------- |
| `none`                    | keep   | 0.8210             | 0.2816           | −0.539           |
| `none`                    | flag   | 0.8334             | 0.2249           | −0.609           |
| `none`                    | drop   | 0.8547             | 0.1938           | −0.661           |
| `normal-operation-values` | keep   | 0.8994             | 0.2965           | −0.603           |
| `normal-operation-values` | flag   | 0.8774             | 0.2368           | −0.641           |
| `normal-operation-values` | drop   | 0.9078             | 0.3414           | −0.566           |
| `instance`                | keep   | 0.9450             | 0.3905           | −0.555           |
| `instance`                | flag   | 0.9423             | 0.3781           | −0.564           |
| `instance`                | drop   | 0.9045             | 0.4063           | −0.498           |
| **mean**              |        | **0.8873**   | **0.3055** | **−0.582** |

**Every configuration loses about 0.6 F1-macro when the wells are held out.** The best well-grouped
run (0.4063) is well below the worst instance-grouped run (0.8210).

The right way to read 0.31 is against what guessing achieves on the same 82,115 rows:

| predictor                                | F1-macro                              |
| ---------------------------------------- | ------------------------------------- |
| uniform random over the 8 classes        | 0.068                                 |
| sampled from the class prior             | 0.125                                 |
| always the majority class (Normal, 73 %) | 0.105                                 |
| **observed, well-grouped**         | **0.194 – 0.406** (mean 0.306) |
| observed, instance-grouped               | 0.821 – 0.945                        |

So the well-grouped model is above chance, but the worst configuration (`none/drop`, 0.1938) beats
prior-sampled guessing by 0.07, and the mean beats it by 0.18. That is a working model in the
narrowest sense and not one anybody should deploy. It is also nowhere near the 0.89 that instance
grouping advertises.

Two secondary orderings survive the grouping change, which is mild evidence they are real:

- **Normalization**: `none` < `normal-operation-values` < `instance` under both groupings
  (0.233 / 0.292 / 0.392 for well; 0.836 / 0.895 / 0.931 for instance). See report 1.
- **Frozen sensors**: `flag` costs ~0.043 under well grouping and ~0 under instance grouping. See
  report 2, which argues this is the signature of genuine cross-well signal.

Note also that the well-grouped runs have real per-fold spread — standard deviations of 0.046 to
0.215 across the 5 outer folds — where the instance holdout has exactly one fold and therefore no
spread at all. A single number from a holdout hides variance that nested makes visible, and 0.215
is enormous relative to a mean of 0.38.

---

## 4. The per-well sheet is mostly a coin flip

The well grouping's real product is a score for every well, from a model that never saw it. Nested
still provides this: wells *are* the grouping, so the pooled out-of-fold predictions give each of
the 33 wells a score from a fold that excluded it.

| normalization | frozen | wells | scoring exactly 0 | median | mean  | p90   | max   |
| ------------- | ------ | ----- | ----------------- | ------ | ----- | ----- | ----- |
| `none`      | keep   | 33    | 8                 | 0.245  | 0.407 | 1.000 | 1.000 |
| `none`      | flag   | 33    | 5                 | 0.163  | 0.363 | 1.000 | 1.000 |
| `none`      | drop   | 27    | 5                 | 0.130  | 0.323 | 1.000 | 1.000 |
| `instance`  | keep   | 33    | 1                 | 0.331  | 0.422 | 1.000 | 1.000 |
| `instance`  | flag   | 33    | 0                 | 0.482  | 0.433 | 1.000 | 1.000 |
| `instance`  | drop   | 27    | 1                 | 0.321  | 0.419 | 1.000 | 1.000 |
| `normal-op` | keep   | 33    | 3                 | 0.237  | 0.341 | 1.000 | 1.000 |
| `normal-op` | flag   | 33    | 5                 | 0.158  | 0.199 | 0.496 | 1.000 |
| `normal-op` | drop   | 27    | 2                 | 0.266  | 0.345 | 1.000 | 1.000 |

Two things should stop the reader here. Every configuration has wells scoring **exactly 1.000**,
and most have wells scoring **exactly 0.000**. Those are not near-misses; they are the two
endpoints, and they are common.

The explanation is the class composition of the wells:

| distinct fault classes in the well | wells        | windows |
| ---------------------------------- | ------------ | ------- |
| 1                                  | **25** | 25,119  |
| 2                                  | 5            | 6,794   |
| 3                                  | 2            | 25,773  |
| 4                                  | 1            | 24,429  |

**Twenty-five of the thirty-three wells carry exactly one fault class.** For such a well,
F1-macro over its windows is a one-question exam: predict the class and score 1.000, miss it and
score 0.000. Splitting the `none/keep` sheet accordingly:

| wells        | n  | mean per-well F1 | exactly 0.000 or 1.000 |
| ------------ | -- | ---------------- | ---------------------- |
| single-class | 25 | **0.480**  | **17 of 25**     |
| multi-class  | 8  | **0.178**  | 0 (max 0.482)          |

The headline mean of 0.407 is therefore a blend of 25 coin flips averaging 0.480 and 8 genuine
multi-class problems averaging 0.178. **The wells that actually pose the prediction task score
0.178.** The five perfect wells in the `none/keep` sheet — 36, 37, 39, 40, 42 — all carry only
Hydrate in Service Line, the most common real fault; predicting it everywhere on those wells is
enough. The five worst — 4, 7, 9, 12, 13 — carry one class each too, and the model guessed wrong.

This is a defect of the *metric on this data*, not of the protocol. Per-well F1-macro was adopted
in report 2 to expose wells the pooled number hides, and it does — but on a fleet where three
quarters of the wells have one label, its mean is close to meaningless, and quoting the maximum
(1.000) would be actively misleading.

A further concentration compounds it: the three wells with 3 or 4 classes hold 50,202 of the
82,115 real windows — **61 % of the data in 9 % of the wells**. The pooled score is dominated by
whether those three wells happen to be predicted well.

---

## 5. What the model looks at when it cannot memorize wells

Top-10 SHAP under well grouping, `keep`:

| normalization               | top-10                                                                                                                                                                                                            |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `none`                    | `T-TPT_mean`, `T-TPT_min`, `T-JUS-CKP_min`, `P-TPT_mean`, `T-PDG_min`, `QGL_median`, `T-PDG_median`, `T-JUS-CKP_mean`, `P-TPT_max`, `P-MON-CKP_mean`                                          |
| `instance`                | `T-TPT_diff2_std`, `P-TPT_min`, `T-TPT_max`, `QGL_iqr`, `P-TPT_max`, `P-TPT_diff2_std`, `T-TPT_min`, `T-TPT_diff1_std`, `P-JUS-CKGL_std`, `T-TPT_std`                                         |
| `normal-operation-values` | `P-JUS-CKGL_diff2_std`, `T-TPT_diff2_std`, `QGL_diff1_std`, `P-JUS-CKGL_diff1_std`, `P-TPT_diff2_std`, `QGL_iqr`, `T-JUS-CKP_diff1_std`, `QGL_diff2_std`, `T-JUS-CKP_std`, `P-TPT_max_zscore` |

Report 2 found that well grouping alone shifted the ranking to pure dispersion and derivative
statistics, and read that as reassuring: "when a model cannot lean on levels and cannot memorize
wells, what is left is the shape of the signal". That holds for the normalized runs here. But
**`none` under well grouping is still entirely levels** — ten of ten. The grouping does not force
the model off absolute operating points; it only stops it memorising *which* well produced them.
With 33 wells and 8 carrying all the normal operation, absolute temperature and pressure remain
usable as a coarse well-type signature even across the split.

So the shift report 2 observed is attributable to the normalization, not to the grouping. Report 1
reaches the same conclusion from the other direction.

---

## 6. Does collapsing to three classes rescue it?

This report's first draft predicted that it would: "25 single-class wells become far less
pathological when there are 3 labels instead of 8". **That prediction was wrong, and the data says
so in two separate ways.**

Every configuration was re-run on the `hydrate` grouping — Normal / Other Problem / Hydrate — as
`native` (trained on the three groups) and `collapse` (the 8-class model's predictions mapped onto
them). Pooled F1-macro, mean over the nine configurations of each grouping:

|                         | 8 classes        | `native`       | `collapse`     |
| ----------------------- | ---------------- | ---------------- | ---------------- |
| instance grouping       | 0.8873           | 0.9538           | 0.9536           |
| **well grouping** | **0.3055** | **0.4778** | **0.5340** |

At face value that is a rescue: 0.306 → 0.534 under well grouping, a gain of 0.23.

**It is mostly mechanical.** Three classes are easier to guess than eight, and the baselines move
almost as far as the score:

| predictor (well-grouped evaluation set, 82,115 rows) | 8 classes        | 3 groups                   |
| ---------------------------------------------------- | ---------------- | -------------------------- |
| uniform random                                       | 0.068            | 0.279                      |
| always the majority class                            | 0.105            | 0.281                      |
| sampled from the class prior                         | 0.125            | 0.334                      |
| **observed (mean)**                            | **0.306**  | **0.534** (collapse) |
| **margin over prior-sampled guessing**         | **+0.180** | **+0.201**           |

The margin over chance improves by 0.021, not by 0.23. Nearly the whole apparent gain is the
question getting easier, not the model getting better. Any future report quoting a triage F1 must
quote the 3-class baseline beside it, or it will overstate the result by an order of magnitude.

**And the single-label-well problem is completely untouched.** Collapsing can only *merge* labels,
so a well carrying one fault class still carries exactly one group:

| distinct labels in the well | 8 classes    | 3 groups     |
| --------------------------- | ------------ | ------------ |
| 1                           | **25** | **25** |
| 2                           | 5            | 8            |
| 3                           | 2            | 0            |
| 4                           | 1            | 0            |

Twenty-five before, twenty-five after. The per-well sheet stays bimodal: under `none/keep` with
`collapse`, the 25 single-label wells average 0.656 with **17 of them at exactly 0.000 or 1.000**,
while the 8 multi-label wells average 0.344. The floor rises — guessing "Normal" is right more
often with three labels — but §4's objection to the metric survives intact.

### `collapse` beats `native`, except for the distilled tree

| grouping                 | `native − collapse` | configurations favouring`native` |
| ------------------------ | ---------------------- | ---------------------------------- |
| instance, ensemble       | +0.0002                | 6 of 9 (a tie)                     |
| well, ensemble           | **−0.0562**     | 2 of 9                             |
| instance, distilled tree | +0.0360                | 8 of 9                             |
| well, distilled tree     | **+0.0590**      | 6 of 9                             |

Report 1 §5 concluded "with z-scored features and the triage question, train on all eight classes
and collapse". That replicates and generalizes: under well grouping `collapse` wins for **every**
normalization reference (0.484 vs 0.403 for `none`, 0.601 vs 0.552 for `instance`, 0.517 vs 0.478
for `normal-operation-values`). Training on eight classes and collapsing afterwards keeps
distinctions the three-group target throws away, and those distinctions evidently transfer across
wells.

**The distilled tree reverses it**, in both groupings and most sharply where it matters (well,
+0.059). A depth-limited tree over ten features cannot afford to model eight classes when it is
scored on three; the `native` ranking selects features for the question actually being asked.

So the recommendation splits by deliverable, which is worth stating plainly because this project's
end goal is a compact tree:

- **If the artifact is the ensemble score**, train on 8 classes and collapse.
- **If the artifact is the compact decision tree** — which is what this repo is ultimately for —
  train `native` on the three groups.

---

## 7. Verdict

1. **The predictor does not transfer to unseen wells.** ~0.6 F1-macro is lost under every one of
   the nine configurations, and the multi-class wells — the only ones posing the real task — score
   0.178.
2. **Instance-grouped numbers should not be quoted as generalization.** 0.89 F1-macro describes
   performance on new recordings from known wells. That is a legitimate operational question, but
   it is not the one "will this predict a fault on a new well" asks.
3. **The per-well sheet needs a companion statistic.** Its mean is dominated by 25 single-class
   coin flips. Future reports should give the multi-class subset separately, as §4 does, and should
   never quote the maximum.
4. **The well-grouped experiment is underpowered by the data, not by the protocol.** Three classes
   live in two or three wells; 61 % of the windows are in three wells. No evaluation protocol fixes
   that — it is a property of 3W's real-well coverage. Switching to leave-one-out would refine the
   estimate without changing this conclusion.
5. **Report 2's "reassuring" feature shift is really a normalization effect**, not a grouping
   effect (§5).
6. **Collapsing to three classes does not rescue well generalization** (§6). It raises F1-macro
   from 0.306 to 0.534, but the chance baseline rises from 0.125 to 0.334 at the same time — a
   margin gain of 0.021, not 0.23 — and it leaves the 25 single-label wells exactly as they were.
   This report predicted otherwise in its first draft and was wrong.
7. **The best label strategy depends on the deliverable**: `collapse` for the ensemble, `native`
   for the distilled tree (§6). Since this repo exists to produce compact trees, that favours
   `native`.

---

## 8. Not done here

- **Leave-one-out**, for the cost reason in §1. It would give one model per well rather than five
  models total, and is the natural confirmation for whichever configuration is taken forward. It
  would not change §4's conclusion, which is about the wells' class composition.
- **The `custom` class grouping** (`CUSTOM_CLASS_GROUPING`) was not run; §6 covers `hydrate`,
  which is what reports 1 and 2 used. A 2-group variant would raise the chance baseline further
  still and should be read against it.
- **A grouping that helps the well problem.** §6 shows `hydrate` does not, because it merges
  labels rather than redistributing wells. What the well-grouped task needs is not fewer labels but
  more wells per label — which no relabelling can supply.
- **Stratifying or pooling wells.** Nothing here tries to compensate for the 8-well monopoly on
  normal operation.
- **Permutation importance** and the **detection task**.

---

## Appendix — artifact index

Commit `4a59209`, working tree dirty. The run-directory table for all 18 configurations is in the
appendix of `2026-09-21_normalization_after_the_leak_fix.md`; the nine well-grouped ones are the
rows whose tag carries `_wellcv_nested`, and each directory holds both its 8-class and its
`_hydrate` artifacts. Per-run provenance is in each directory's `run.json`; `results/index.jsonl`
holds one row per finished stage (72 rows).

Per-well sheets live in each run's `metrics/<tag>_metrics.json` under
`strategies.full.per_group` for the 8-class runs and under `strategies.native.per_group` /
`strategies.collapse.per_group` for the `..._hydrate` ones, which sit in the same directories. The
class-composition figures of §2, §4 and §6 come from `data/features_overlap.parquet` directly.
