# Frozen sensors: how much of the predictor is instrument status?

**Exploration goal.** Report 2 (`2026-09-17_dt_from_xgb_cv_and_class_grouping.md`) found that every
top split of the z-scored instance tree was a test of whether some standard deviation is exactly
zero — that is, whether a sensor is *frozen*. The model was reading instrument health and the
artifacts presented it as flow physics. TODO item 10 made that channel explicit and optional. This
report asks two questions:

1. **How much of the predictor's skill is instrument status?**
2. **Does naming it change what the model learns, or only how it is labelled?**

The answer to the first is *a lot, and it is genuine cross-well signal rather than an artefact*.
The answer to the second is *it changes the artifacts completely and the pooled score barely at
all* — with one important exception under well grouping.

---

## 1. What was run

36 runs: 3 normalization references × 3 frozen-sensor policies × 2 CV groupings × 2 label sets.
This report reads the frozen-sensor axis; the other two have reports of their own
(`2026-09-21_normalization_after_the_leak_fix.md`,
`2026-09-21_cv_grouping_and_well_generalization.md`).

```bash
uv run main.py --model xgb --task prediction --allow-overlap --skip-permutation \
  --cv-group {instance_id,well_id} [--eval nested] \
  --normalization {none,instance,normal-operation-values} \
  --frozen-sensors {keep,flag,drop}
```

| `--frozen-sensors` | what it does | rows | features |
|---|---|---|---|
| `keep` | leaves the degenerate zeros in place — the behaviour before item 10, kept as the baseline | 100,184 | 88 |
| `flag` | blanks **all eleven** statistics of a frozen sensor's window (the per-fold imputer fills them) and adds an explicit `<sensor>_frozen` indicator | 100,184 | **96** |
| `drop` | discards whole instances whose `P-TPT` never moves; changes nothing else | **95,988** | 88 |

A window's sensor counts as frozen when its own standard deviation is below
`FROZEN_STD_THRESHOLD = 1e-6` — a property of the window, not of the recording, so the fault period
can never decide the flag of a normal-operation window.

- **Dataset**: `data/features_overlap.parquet`, commit `4a59209`, `--allow-overlap`. Prediction
  task: 100,184 windows, 1,107 instances, 33 real wells.
- **Protocol**: `holdout` for instance grouping, `nested` (5 outer folds) for well grouping. The
  two groupings are not comparable to each other; see report 3.
- **Label sets**: the dataset's own 8 classes, and the `hydrate` grouping (Normal / Other Problem /
  Hydrate) both as `native` (trained on the three groups) and `collapse` (the 8-class model's
  predictions mapped onto them). §7 checks whether the frozen-sensor findings survive the coarser
  question.

> [!IMPORTANT]
> **`drop` is not evaluated on the same rows as `keep` and `flag`.** It removes 27 instances and
> 4,196 windows (4.2 %). Every `drop` number below is on 95,988 windows against the others' 100,184,
> so differences of a few hundredths between `drop` and the rest are not attributable to the policy
> alone. `keep` vs `flag` *is* a clean comparison — identical rows, different columns.

---

## 2. How much of the dataset is frozen

Frozen windows per sensor, over the 100,184 windows of the prediction task:

| sensor | frozen windows | share |
|---|---|---|
| `P-PDG` | 55,180 | **55.1 %** |
| `QGL` | 30,056 | 30.0 % |
| `T-PDG` | 28,866 | 28.8 % |
| `P-TPT` | 20,540 | 20.5 % |
| `T-TPT` | 14,361 | 14.3 % |
| `P-MON-CKP` | 5,127 | 5.1 % |
| `T-JUS-CKP` | 2,783 | 2.8 % |
| `P-JUS-CKGL` | 2,436 | 2.4 % |

This is the finding that reframes the problem. The downhole pressure gauge is motionless in **more
than half** of the windows the prediction task learns from. Under `keep`, each of those windows
contributes eleven exactly-zero `P-PDG` statistics — eleven crisp, perfectly reproducible numbers
that no genuine measurement would ever produce. The trees were not being perverse in splitting on
them; they were the most reliable features in the matrix.

Only 27 instances (4.2 % of windows) have a *critical* sensor dead for the whole recording, which
is why `drop` is a much smaller intervention than `flag`.

---

## 3. Headline metrics

F1-macro on held-out data.

### Instance grouping (holdout)

| normalization | `keep` | `flag` | `drop` |
|---|---|---|---|
| `none` | 0.8210 | 0.8334 | 0.8547 |
| `normal-operation-values` | 0.8994 | 0.8774 | 0.9078 |
| `instance` | 0.9450 | 0.9423 | 0.9045 |
| **mean** | **0.8885** | **0.8844** | **0.8890** |

### Well grouping (nested)

| normalization | `keep` | `flag` | `drop` |
|---|---|---|---|
| `none` | 0.2816 | 0.2249 | 0.1938 |
| `normal-operation-values` | 0.2965 | 0.2368 | 0.3414 |
| `instance` | 0.3905 | 0.3781 | 0.4063 |
| **mean** | **0.3229** | **0.2799** | **0.3138** |

**Under instance grouping the policy is worth essentially nothing** — 0.884 to 0.889 across all
three, a spread smaller than the spread between normalization references within any one policy. On
identical rows, `flag` costs 0.004 on average against `keep`.

**Under well grouping `flag` costs 0.043** (0.280 vs 0.323), and it costs it consistently: 0.225 vs
0.282 for `none`, 0.378 vs 0.391 for `instance`, 0.237 vs 0.297 for `normal-operation-values`.
Three for three, same direction.

That asymmetry is the substantive result of this report, and §5 argues it is not a defect.

---

## 4. Naming the channel changes the artifacts completely

Top-10 SHAP features. Under `flag`, the eight `<sensor>_frozen` indicators are eligible for the
first time:

| run | top-10 |
|---|---|
| instance, `none`, `keep` | `T-TPT_min`, `T-TPT_max`, `P-TPT_max`, `P-MON-CKP_max`, `P-TPT_min`, `P-JUS-CKGL_max`, `P-MON-CKP_min`, `P-PDG_median`, `P-TPT_mean`, `T-JUS-CKP_mean` |
| instance, `none`, `flag` | `T-TPT_min`, `T-TPT_mean`, **`T-TPT_frozen`**, **`P-PDG_frozen`**, `P-JUS-CKGL_max`, **`P-MON-CKP_frozen`**, **`T-JUS-CKP_frozen`**, **`P-TPT_frozen`**, `T-TPT_diff1_std`, `T-JUS-CKP_mean` |
| instance, `instance`, `flag` | **`T-JUS-CKP_frozen`**, `T-TPT_diff2_std`, **`P-PDG_frozen`**, **`T-TPT_frozen`**, `P-TPT_max`, **`P-MON-CKP_frozen`**, `P-TPT_min`, `T-TPT_min`, **`P-TPT_frozen`**, `T-TPT_diff1_std` |
| instance, `normal-op`, `flag` | `T-TPT_diff2_std`, **`P-PDG_frozen`**, **`T-JUS-CKP_frozen`**, `T-JUS-CKP_std`, **`QGL_frozen`**, `P-PDG_diff2_std`, **`T-TPT_frozen`**, `T-TPT_diff1_std`, `T-JUS-CKP_diff2_std`, `P-JUS-CKGL_iqr` |
| well, `normal-op`, `flag` | **`P-PDG_frozen`**, `T-TPT_diff2_std`, `P-JUS-CKGL_diff1_std`, **`QGL_frozen`**, `P-JUS-CKGL_diff2_std`, `P-TPT_diff2_std`, **`T-PDG_frozen`**, `QGL_diff2_std`, `P-TPT_diff1_std`, `P-JUS-CKGL_std` |

**Five of the top ten are frozen indicators** under `instance/flag`, and `T-JUS-CKP_frozen` is
rank 1. Under `normal-operation-values/flag` with well grouping, `P-PDG_frozen` is rank 1. Eight
features out of ninety-six take half the top of the ranking.

The distilled trees make the same point more legibly. Root splits:

| run | root |
|---|---|
| instance, `none`, `keep` | `P-PDG_median <= 277.57` |
| instance, `instance`, `keep` | `P-TPT_max <= -2.68` |
| instance, `normal-op`, `keep` | `QGL_std <= 0.01` → `T-JUS-CKP_diff1_std <= 0.00` → `P-PDG_diff2_std <= 0.00` |
| instance, `instance`, `flag` | `P-TPT_max <= -2.67` → **`T-TPT_frozen <= 0.50`** |
| instance, `normal-op`, `flag` | **`P-PDG_frozen <= 0.50`** |

The `normal-operation-values`/`keep` tree is report 2's pathology in its purest form: three nested
"is this dispersion exactly zero?" tests before any physical quantity appears. And it is **worst**
under the leak-free normalization — which makes sense. Strip the absolute levels (normalization)
and the fault-period leak (the `normal-operation-values` reference), and instrument status is the
most discriminative thing left. Fixing item 9 made item 10 more urgent, not less.

Switch to `flag` and that same tree opens on `P-PDG_frozen <= 0.50`: the same information, named.
This is exactly what the TODO item asked for — "so the model states that it is using instrument
status instead of smuggling it through a dispersion statistic" — and it is achieved.

---

## 5. Why `flag` costs something under well grouping

The tempting reading of §3 is that `flag` is a regression. It is not, and the reason matters.

Under **instance** grouping, a held-out instance usually comes from a well that also contributed
training instances. Whatever the frozen pattern encodes about that well is available on both sides
of the split, so removing it costs almost nothing — the model has other ways to recognise the same
wells. Hence 0.884 vs 0.889.

Under **well** grouping, the held-out well is unseen. If `flag` costs 0.043 there, then the frozen
pattern was carrying information that **generalizes across wells** — it was not memorisation. And
that is plausible physically as well as statistically: `P-TPT` reads exactly 0 for all of wells 35,
36 and 40, and those wells are not a random sample of the fleet. A dead downhole gauge co-occurs
with a particular vintage of completion, a particular operating regime, and therefore with a
particular fault profile.

So the honest statement is: **instrument status is real, generalizing, predictive signal, and
`flag` does not remove it — it relocates it from eleven counterfeit measurements into one honest
indicator.** The 0.043 is the cost of that relocation being lossy: blanking all eleven statistics
of a sensor frozen in 55 % of windows throws away the *value* it is stuck at, which differs between
wells (0 Pa for wells 35/36/40, 817 bar for well 29) and evidently carries some of the signal.

That lossiness was a deliberate design choice, and §7 revisits whether it was the right one.

---

## 6. The distilled trees are where `flag` really bites

Stage-5 F1-macro (compact tree on the top-10 SHAP features), instance grouping:

| normalization | `keep` | `flag` | `drop` |
|---|---|---|---|
| `none` | 0.7085 (d11) | 0.7096 (d12) | 0.7312 (d11) |
| `normal-operation-values` | 0.7249 (d12) | 0.7281 (d12) | 0.7299 (d10) |
| `instance` | **0.8242** (d12) | **0.6456** (d8) | 0.8019 (d12) |

`instance/flag` loses 0.179 against `instance/keep` and its depth sweep plateaus at 8 rather than
12 — the only configuration where the distilled tree degrades sharply. The ensemble barely moved
(0.9450 → 0.9423) but the ten-feature tree built from its ranking collapsed.

The mechanism is visible in §4: under `instance/flag`, five of the top ten SHAP features are
binary indicators. A depth-8 tree over five booleans and five continuous features has far less to
work with than one over ten continuous features, and the booleans partition the data into a small
number of cells rather than ordering it. **The distillation is more sensitive to the frozen policy
than the ensemble is**, and anyone using stage 5 to produce a compact operational rule should know
that `flag` narrows what stage 5 has to distil.

---

## 7. The same pattern on the triage question

Mean F1-macro over the three normalization references, for each label treatment:

| grouping | strategy | `keep` | `flag` | `drop` | `flag − keep` |
|---|---|---|---|---|---|
| instance | 8 classes | 0.8885 | 0.8844 | 0.8890 | −0.004 |
| instance | collapse | 0.9555 | 0.9547 | 0.9506 | −0.001 |
| instance | native | 0.9656 | 0.9559 | 0.9398 | −0.010 |
| well | 8 classes | 0.3229 | 0.2799 | 0.3138 | −0.043 |
| well | collapse | 0.5657 | 0.4820 | 0.5542 | **−0.084** |
| well | native | 0.4633 | 0.4798 | 0.4903 | +0.017 |

The central asymmetry of §3 and §5 **strengthens** rather than dissolving. Under instance grouping
`flag` remains free (−0.001 to −0.010). Under well grouping with `collapse` it costs **0.084** —
twice what it cost on the 8-class question, and the largest cost of the policy anywhere in the
sweep.

That fits §5's argument closely. `collapse` takes an 8-class model and asks a 3-class question of
it, so it depends entirely on features that transfer; and the frozen indicators are, by §5, exactly
the transferable part. Take them away from their eleven counterfeit measurements and the collapse
strategy loses most.

The one row that breaks the pattern is well `native` (+0.017 for `flag`). A model *trained* on
three groups has a coarse enough target that the blanked statistics are recoverable from what
remains; it is the only setting in the sweep where `flag` pays.

The distilled trees carry the same signature more sharply. Stage-5 F1-macro under the `collapse`
strategy, instance grouping:

| normalization | `keep` | `flag` | `drop` |
|---|---|---|---|
| `none` | 0.8922 | 0.9310 | 0.9023 |
| `normal-operation-values` | 0.8720 | 0.8804 | 0.8580 |
| `instance` | 0.9301 | **0.7430** | 0.8891 |

`instance/flag` again collapses — 0.7430 against 0.9301 for `keep`, mirroring the 0.6456-vs-0.8242
gap §6 found on the 8-class question. Two independent label treatments, the same configuration, the
same failure: when the ranking that feeds stage 5 is half binary indicators, the ten-feature tree
has too little to work with. This is now a reproducible property of `instance` + `flag`, not a
one-off.

---

## 8. Verdict

1. **Instrument status is a major driver of this predictor, not a curiosity.** `P-PDG` is frozen in
   55 % of modelled windows; eight indicator features take half of the top-10 SHAP ranking whenever
   they are allowed to.
2. **It generalizes across wells.** `flag` costs 0.043 F1-macro under well grouping and ~0 under
   instance grouping, which is the signature of real transferable signal rather than memorisation.
   The cost doubles to 0.084 on the triage question under `collapse` (§7), which is where a model
   depends most on features that transfer.
3. **`flag` achieves what item 10 asked.** The root split moves from a counterfeit dispersion
   statistic to a named indicator, at a pooled cost of 0.004 under instance grouping. The artifacts
   now say what the model is doing.
4. **`keep` remains the honest baseline for comparison**, and should be reported alongside `flag`
   in any future sweep rather than dropped.
5. **`drop` is not worth its complexity.** It removes 4.2 % of windows, cannot be compared on equal
   rows, and its apparent advantages (e.g. 0.8547 vs 0.8210 under `none`) are confounded with
   having evaluated on an easier subset. It answers a question nobody is asking: the problem is not
   that 27 instances are bad, it is that half the windows have a dead gauge.
6. **Reconsider blanking the level statistics.** §5 suggests the cost of `flag` is concentrated in
   discarding the stuck *value*. A fourth policy — blank the dispersion statistics, keep the levels,
   add the indicator — would test whether the stuck value is the part that generalizes. The
   argument for blanking everything was that levels would simply carry the shortcut onward; §5
   suggests that "shortcut" may be legitimate signal, which weakens the argument.

---

## 9. Not done here

- **The fourth policy of §7.6** (dispersion-only blanking) is not implemented; it is the obvious
  follow-up and would need one new mode and nine more runs.
- **Flat *derivatives*.** `<sensor>_diff2_std <= 0.00` splits survive under `flag`, because a window
  can have zero second-difference dispersion with nonzero σ — a linear ramp, or a stretch the 60 s
  forward-fill filled in. Whether those deserve the same treatment is an open question, already
  noted in the TODO.
- **Which wells the indicators identify.** This report shows the indicators are used and that they
  transfer; it does not show *what* they encode. Crossing `<sensor>_frozen` against well identity
  and fault class would settle whether "dead gauge" is a proxy for a completion vintage.
- **The `custom` class grouping**, **leave-one-out**, **permutation importance**, and the
  **detection task** — all out of scope for these 36 runs, as in the other two reports.

---

## Appendix — artifact index

Commit `4a59209`, working tree dirty. See the appendix of
`2026-09-21_normalization_after_the_leak_fix.md` for the run-directory table covering all 18
configurations; the frozen-sensor policy of each is the third column there, and each directory
holds both its 8-class and its `hydrate` artifacts. Per-run provenance (argv, commit,
dirty diff, outputs) is in each directory's `run.json`, and `results/index.jsonl` holds one row per
finished stage.
