# E14–E17 — Development on MVTec AD 2 after E13: what fails, and why

E13 left recipe v1 with near-chance image AUROC on four AD 2 categories and 5–7× its false-alarm
bound. This report takes the two failures apart on `test_public`, which is development data from
E13 on, and screens the remedies written down in [docs/11](../docs/11-experiment-plan.md)
(Stages H‴ and H⁗) before their runs. Every number is reproduced by a committed script named in
its section; the open-problem register is [docs/14](../docs/14-open-problems-and-remedies.md).

These are `test_public` numbers, not comparable with published AD 2 results (private splits,
MVTec's server). Dataset: MVTec AD 2 © MVTec Software GmbH, CC BY-NC-SA 4.0 — see
[ATTRIBUTION.md](../ATTRIBUTION.md). No AD 2 image or derived figure is published here.

## Summary

1. **`test_public` is five-sixths changed lighting by design.** Each object is shown as
   `regular` (the lighting of train and validation, which hold nothing else), `overexposed`,
   `underexposed` and `shift_1..k`. The false alarms that broke E13's operating point are the
   `shift` captures — a changed light direction and colour — and nothing else: 22.8% (v1) and
   31.2% (v2) of those normals against 2.6–5.7% for the other three conditions.
2. **The captures of one object are one sample.** With objects as the bootstrap unit, E13's one
   significant gain (`sheet_metal`) is no longer significant; the verdict stands.
3. **Global-exposure remedies fail, as the diagnosis predicts.** Per-image gain and
   standardization (E14b, E14c) raise pooled false alarms (47 → 73 and → 64) and collapse
   `walnuts` (−0.27); neither carries. An exposure-augmented bank (E14d) costs no category
   anything and gains on `fruit_jelly` and `rice` (+0.06, +0.09), but leaves the false alarms
   where they were (49) and falls short of the three-gain rule.
4. **Lighting-structure remedies fail too (E15).** Flat-field normalization raises the false
   alarms among `shift` normals from 41 to 60 of 187; a relit bank leaves them at 39, with no
   category significantly changed. `can`, whose `shift` captures alarm 35 times in 36, is
   untouched by all five remedies: a highlight that moves across a metal cylinder is neither a
   gain nor a smooth light field. Changed lighting is not removable by the preprocessing or the
   synthetic relighting tested here.
5. **Normal images under the new lighting do what synthesis did not (E16).** Adding every
   capture of 2–6 good parts per category to the bank halves the false alarms among `shift`
   normals (41 → 19 of 187; one part, 41 → 28), no category worse — the pre-registered reading is
   met, narrowly, and almost entirely on `can` (35 → 13 of 36). The price at the unchanged
   operating point is recall (macro 0.24 → 0.19), and **E17 shows the trade is not a gain**:
   at a threshold recalibrated on new-lighting normals, the adapted model's recall is no better
   than the unadapted one's (−0.014 [−0.034, +0.003]), and with 1–3 calibration objects even
   recalibration leaves false alarms above their bound (17 against 7.6 expected). On `can` the
   changed-lighting normals and the defects are not separable by these features.
6. **The image score is a separate problem.** It fails under the training lighting too (v1 on
   `regular` captures: `can` 0.57, `rice` 0.53, `sheet_metal` 0.53, `wallplugs` 0.46). No
   aggregation rule, object region or location normalization fixes it in general (E14a, R2); an
   image-level branch helps under the training lighting (`rice` 0.53 → 0.78) and fails under
   changed lighting (R4).
7. **Two defects found and fixed:** the 3σ pixel threshold (SegF1 0 on `fruit_jelly` v2 → 0.22
   at the validation quantile), and a regression that made the frozen E13 analysis match no run.

## 1. What `test_public` contains

`scripts/ad2_condition_study.py`, outputs in [studies/ad2-conditions/](studies/ad2-conditions/).
File names are `<object>_<condition>.png`; `fruit_jelly` has one shift, `vial` four, the others
three. Against the `regular` capture of the same object (three objects per category, 1/8-size
grey images): over- and under-exposure are a pure global gain — correlation ≥ 0.992, equal channel
gains 0.76–1.18 — while `shift_k` changes the structure of the light, correlation 0.82–0.99 (`vial`'s shifts
are near-identical to `regular` at this scale, 0.999–1.000), with
a warm cast on `rice`, `walnuts`, `fabric` (blue/red down to 0.78) and a 0.60–0.82 gain on
`wallplugs`.

E13's stored predictions by condition, mean over seeds 0–2 (the operating point is v1's own,
calibrated on `validation/good`):

| condition | normals | v1 false alarms | v2 false alarms | v1 macro AUROC | v2 macro AUROC |
|---|---|---|---|---|---|
| `regular` | 64 | 5.7% | 2.6% | 0.686 | **0.742** |
| `overexposed` | 64 | 5.7% | 3.6% | 0.694 | 0.734 |
| `underexposed` | 64 | 5.2% | 4.2% | 0.698 | 0.740 |
| `shift` | 187 | **22.8%** | **31.2%** | 0.684 | 0.659 |

The per-category bounds are 2.0–5.0%, so under the training lighting and under a global exposure
change the operating point is roughly where the rank rule puts it; the `shift` captures carry
the excess (`can`: 96% of its `shift` normals alarm under v1). v2 is the better recipe under
`regular` lighting (`sheet_metal` 0.867 vs 0.528, `vial` 0.947 vs 0.907) and the worse one under
`shift` (`vial` 0.696 vs 0.885): E13's verdict on v2 is about lighting robustness as much as
about the representation.

**Clustering.** E13 resampled images. Resampling objects (all captures together, class-stratified,
10,000 resamples, Holm over 8) widens the intervals 1.2–2.4× on seven categories:

| category | v2 − v1 | image bootstrap 95% CI, Holm p | object bootstrap 95% CI, Holm p | objects normal + anomalous |
|---|---|---|---|---|
| `sheet_metal` | +0.156 | [+0.057, +0.257], 0.022 * | [−0.033, +0.345], 0.74 | 4 + 15 |
| `vial` | −0.111 | [−0.182, −0.048], 0.002 * | [−0.165, −0.055], 0.002 * | 5 + 15 |
| others | −0.038 … +0.034 | n.s. | n.s. | 5–12 + 15 |

`sheet_metal` has four normal objects; nothing about its normal population is estimated
precisely. E13 stays *not confirmed*, with zero rather than one significant gain.

## 2. The operating point under changed lighting

### E14b–d: global exposure (pre-registered, `scripts/e14_analysis.py`)

Against E14-ref (v1 at implementation `54752852d47e`, which reproduces E13's seed-0 prediction
digests on all eight categories), seed 0, image bootstrap and Holm over 8 as registered:

| variant | pooled false alarms (of 379 normals; 9.7 expected) | significant gains | significant losses | macro AUROC | macro AU-PRO | carried |
|---|---|---|---|---|---|---|
| reference | 47 | – | – | 0.676 | 0.351 | – |
| E14b `normalize: gain` | 73 | 1 (`vial` +0.038) | 1 (`walnuts` −0.275) | 0.630 | 0.267 | no |
| E14c `normalize: standardize` | 64 | 1 (`rice` +0.190) | 2 (`fabric` −0.084, `walnuts` −0.266) | 0.647 | 0.319 | no |
| E14d `bank_exposures: [−0.5, 0.5]` | 49 | 2 (`fruit_jelly` +0.057, `rice` +0.088) | 0 | 0.688 | 0.362 | no |

Per-image normalization removes the absolute brightness and colour that `walnuts` defects are
recognized by, and it does nothing for the `shift` captures (`can`: 35 of 36 `shift` normals
alarm with or without gain normalization). E14d's first submission failed: the queued command set a key no config
had (`--set model.bank_exposures=…`), so it ran again from an overlay file at the same
implementation (docs/14 M8).

### E15: light direction and colour (pre-registered, `scripts/e15_analysis.py`)

Against E15-ref (v1 at implementation `67985a22598d`, which again reproduces E13's seed-0
prediction digests on all eight categories), seed 0, the object bootstrap and Holm over 8 as
registered:

| variant | false alarms among `shift` normals (of 187) | all false alarms (of 379) | significant gains / losses | macro AUROC all | `regular` | `shift` | macro AU-PRO | carried |
|---|---|---|---|---|---|---|---|---|
| reference | 41 | 47 | – | 0.676 | 0.674 | 0.678 | 0.351 | – |
| E15a `normalize: flatfield` | 60 | 77 | 0 / 0 | 0.652 | 0.670 | 0.634 | 0.313 | no |
| E15b `bank_relight: 2` | 39 | 52 | 0 / 0 | 0.675 | 0.687 | 0.662 | 0.365 | no |

Flat-field division removes shading the defects are seen against: `walnuts` −0.134 (object CI
[−0.261, −0.029], not significant after Holm), `fabric` −0.077, and the `shift` false alarms rise
on `fabric` (5 → 11) and `wallplugs` (1 → 9). The relit bank changes no category significantly;
its `can` `shift` false alarms stay at 35 of 36. Neither goes to seeds 1–2, and the combination
and the v2 screen the plan made conditional on a carry do not run.

All five lighting remedies leave the `can` failure where it was. What the `shift` captures change
on `can` is where the specular highlights fall on a curved metal surface; a gain, a per-channel
standardization, a smooth multiplicative field, or a bank of exposure and smooth-field variants
does not produce or remove a moved highlight. The remedies left are data (normal images under the
lighting to be tolerated, added to the bank and the calibration set), features that are
invariant to it (R8), or refusing such images, which §2's guards show needs a better detector
than global exposure or the lighting field.

### E16: a few normal images under the new lighting (pre-registered, `scripts/e16_adaptation.py`)

Per category the `test_public` objects are split into two folds (seeded, separately for normal
and anomalous objects). For each fold, v1 is fitted on `train/good` plus every capture of n good
objects of the *other* fold, and evaluated on its own fold's images; the operating point stays
calibrated on `validation/good`. Against E15-ref on the same images (the script with n = 0
reproduces its scores exactly), 32 fits from a clean worktree, object bootstrap and Holm over 8:

| adaptation | images added per fit | `shift` false alarms (of 187) | all false alarms (of 379) | significant gains / losses | macro AUROC all | `shift` | macro recall at OP | works? |
|---|---|---|---|---|---|---|---|---|
| none (E15-ref) | 0 | 41 | 47 | – | 0.676 | 0.678 | 0.239 | – |
| 1 object | 4–7 | 28 | 34 | 0 / 0 | 0.679 | 0.679 | 0.221 | no |
| one fold's objects (2–6) | 8–36 | **19** | 28 | 0 / 0 | 0.688 | 0.695 | 0.195 | **yes** |

The reduction is `can`'s: 35 → 26 → 13 of its 36 `shift` normals, with its image AUROC on `shift`
captures 0.484 → 0.631; elsewhere the `shift` false alarms were few and stay so (6 → 6). Real
captures under the new lighting give the bank the moved highlights that no synthetic variant
produced. The cost is at the operating point: the threshold still comes from training-lighting
validation normals while the adapted bank lowers anomaly scores too, so recall falls (`walnuts`
0.33 → 0.17, `can` 0.54 → 0.37) although no category's AUROC drops significantly. A line that
adapts its bank should recalibrate as well — with normals under the new lighting that the bank
did not see — which E16 did not do.

### E17: recalibration under the new lighting (pre-registered, `scripts/e17_recalibration.py`)

E16 kept the training-lighting threshold, which does not bound false alarms under the new
lighting. E17 compares the arms at a threshold that is meant to: same folds; the other fold's
normal objects are split, the first half into the adapted bank, the rest (every capture) into the
calibration set beside `validation/good`. 16 fits from a pinned worktree at E15-ref's
implementation.

| arm | false alarms (of 379) | expected (Σ bound × normals) | `shift` false alarms (of 187) | macro recall at OP |
|---|---|---|---|---|
| R — reference, validation threshold | 47 | 9.7 | 41 | 0.239 |
| R+cal — reference, recalibrated | 17 | 7.6 | 14 | 0.181 |
| A — half-fold bank, validation threshold | 36 | 9.7 | 27 | 0.208 |
| A+cal — half-fold bank, recalibrated | 18 | 7.6 | 12 | 0.167 |

Both readings fail. Recalibrating on 1–3 new-lighting objects cuts the false alarms by almost
two thirds but not to the bound (17 against 7.6 + 2√7.6 = 13.1): `can` 8 of 72, `fabric` 5 of 66,
`wallplugs` 4 of 60 — a handful of objects does not cover the object-to-object variation, so the
calibration maximum understates the tail. At the recalibrated threshold the adapted bank does not
buy recall: A+cal − R+cal = −0.014 [−0.034, +0.003] (object bootstrap over anomalous objects).
`can` shows why: once the threshold must clear its changed-lighting normals, recall falls to
0.09 (R+cal) and 0.04 (A+cal). E16's halving of the false alarms was a move along the same
trade-off curve, not a better curve; the `can` failure under changed lighting is one of
separability (AUROC ≈ 0.5–0.6), which a bank or a threshold cannot fix.

### Guards by condition (exploratory)

The deployed guards (global exposure and focus, fitted on `train/good`) against a
lighting-structure guard (R10: the gain-normalized 8-cell-grid lighting field, refused when
farther from every training field than the training fields are from each other); share of test
normals refused, and false alarms among the normals each accepts, v1, seeds 0–2.
`scripts/shading_guard_study.py`.

| condition | refused: exposure/focus | refused: lighting field | false alarms among accepted: exposure/focus | lighting field |
|---|---|---|---|---|
| `regular` | 8% | 0% | 6.0% | 5.0% |
| `overexposed` | 39% | 3% | 5.4% | 5.0% |
| `underexposed` | 56% | 0% | 4.2% | 3.9% |
| `shift` | 30% | 35% | 14.2% | 14.6% |

(Macro over categories.) The exposure guard spends most of its refusals on captures the model
handles (`rice` 100%, `fabric` 73–82%, `vial` 60–100% of over/under-exposed normals) and lets
61% of `can`'s `shift` normals through, 94% of which alarm. The lighting-field guard almost never
refuses a harmless capture — the specificity problem is solved — but the `shift` captures it
refuses (`sheet_metal` 100%, `rice` 95%, `can` 67%) are not the ones that alarm: false alarms
among accepted `shift` normals stay at 14–15%. It is a better guard for a deployment that wants
few needless refusals and no better as protection; the false alarms have to be removed in the
model (E15), not in front of it.

## 3. The image score

All on E13's stored maps, no refit.

**E14a — aggregation rule** (macro image AUROC over 8, mean of seeds 0–2; the E02 tooling, which
at the time ran 1,000 bootstrap resamples, so read effect sizes):

| rule | v1 | v2 |
|---|---|---|
| stored (re-weighted max) | 0.686 | 0.690 |
| max | 0.685 | 0.692 |
| top 0.1% mean | 0.689 | **0.707** |
| top 1% mean | **0.702** | 0.692 |
| top 5% mean | 0.701 | 0.669 |
| mean | 0.632 | 0.565 |

Top-k means help `fabric`, `vial` and `walnuts` by 0.03–0.06 and leave `can`, `rice` and
`wallplugs` at chance. No rule repairs the failing categories.

**Object-region scoring** (the E02b ROI, the top 1/3/6% of training-map energy as the object):
the defects are not where the training maps say the object is — only 28%, 46% and 58% of defect
regions lie fully inside the 1%, 3% and 6% regions. The gains, `can` v1 +0.05 and `rice` v2
+0.07, have intervals that include zero; the losses reach −0.11 on `fruit_jelly` v1 and −0.20 on
`sheet_metal` v2. Negative.

**R2 — location-conditional normalization** (each map location standardized by the validation
normals' mean and SD there, leave-one-out for the validation scores; seed 0,
`scripts/location_normalization_study.py`): `sheet_metal` v1 0.580 → 0.825, `rice` 0.515 → 0.586,
`fabric` 0.664 → 0.697, but `vial` 0.908 → 0.574 and `fruit_jelly` 0.871 → 0.775, and `sheet_metal`'s
false alarms go from 0 to half its normals. The objects are not registered — they sit at a
different place in each image — so a per-location statistic describes the background as often as
the part. Mixed; not adopted.

**R4 — an image-level branch** (`scripts/global_branch_study.py`; WRN50-2 layer2–4 mean-pooled at
640 px, fitted on `train/good`: a shrunk Gaussian per layer, or 5-NN; fused with v1's stored
score after z-scoring both on the validation normals):

| arm | macro AUROC all | `regular` | `shift` |
|---|---|---|---|
| patch (v1) | 0.686 | 0.686 | 0.684 |
| Gaussian alone | 0.591 | 0.713 | 0.597 |
| patch + Gaussian, sum | 0.626 | **0.730** | 0.640 |
| patch + 5-NN, sum | 0.641 | 0.682 | 0.665 |

Under the training lighting the global Gaussian sees what the patches miss on `rice`
(0.530 → 0.762) and `vial` (patch + Gaussian 0.996), but a pooled image descriptor moves with the
light: across all captures the Gaussian alone loses 0.22 on `vial` and 0.27 on `walnuts` against
the patch score (object bootstrap, significant after Holm over 8), and `wallplugs` — the missing-part category it was meant for — stays at chance
(0.46–0.53). R4 waits for a lighting-invariant representation.

## 4. Localization: the pixel threshold (docs/14 B3, fixed)

The evaluator now reports SegF1 at a second threshold, the validation pixel quantile at a 0.1%
normal-pixel rate, beside the AD 2 benchmark's 3σ (`inspector.evaluate`,
`scripts/pixel_threshold_study.py`, all 48 E13 runs re-evaluated from the registry; agreement with
the stored threshold-free metrics checked to 1e-3, the float16 storage error being ~1e-5):

| recipe | macro SegF1 3σ | macro SegF1 quantile | `fruit_jelly` | `vial` |
|---|---|---|---|---|
| v1 | 0.211 | 0.216 | 0.344 → 0.319 | 0.324 → 0.316 |
| v2 | 0.131 | 0.184 | **0.000 → 0.222** | **0.102 → 0.267** |

The quantile threshold never lands above every test pixel, which is what made 3σ return 0 on
bimodal `layer2` maps. On the normal pixels of the test set its false-positive rate is
0.15–1.5%, above the 0.1% it was set for — the same changed lighting, seen per pixel.

## 5. Data integrity

`scripts/near_duplicate_audit.py --layout mvtec_ad2 | mvtec_ad` (counts only for these
CC BY-NC-SA datasets): no validation or test image is pixel-identical to a training image in AD 2
or MPDD. The one cluster of near candidates, 19 of 150 `wallplugs` test images closer to a
training image than the training images' 1st-percentile neighbour distance, is two normal
objects across their captures plus six anomalous images, at a thumbnail RMSE of ~44 grey levels:
similar arrangements, not copies.

## 6. Defects found in the project itself

| defect | effect | fix |
|---|---|---|
| `PatchCore.hparams()` emitted `bank_exposures` unconditionally (`fc45fc3`) | the frozen ADR-13 analysis matched none of the 48 E13 runs to v1 or v2 | late settings omitted at their default; a test pins the default key set; the frozen analysis reproduces `confirmation.json` exactly (`d95cf0f`) |
| a queued command set a key that no config has | E14d failed at its turn, an hour after it was queued | rerun from an overlay; `inspector run --dry-run` resolves every planned run without fitting |
| 3σ pixel threshold on bimodal maps | SegF1 0.000 on `fruit_jelly` v2, all seeds | validation pixel quantile reported beside it (§4) |
| image bootstrap on clustered captures | E13 intervals too narrow | object bootstrap, used from E15 on |

## 7. What follows

- **Lighting** (B1). Synthetic photometric invariance is exhausted (five remedies, none carried).
  Real normals under the new lighting halve the false alarms at the old threshold (E16) but buy
  no recall at a recalibrated one (E17), and a few calibration objects do not restore the bound.
  For deployment: re-collect *many* normal parts (tens, not a handful) under each lighting before
  trusting the false-alarm figure, or fix the lighting. For research: features that separate
  `can`'s changed-lighting normals from its defects (E18, R8).
- **Image score** (A1, A3). Under the training lighting the failures persist; R1 tiling for
  `sheet_metal`/`fabric`/`rice`, and R8 stronger features, are the untested levers. R4 needs a
  lighting-robust descriptor first.
- **Held-out.** No variant earned a server submission beyond v1; submitting v1 to MVTec's server
  (`test_private`, `test_private_mixed`) measures the private lighting shift once (owner action).
- **Statistics.** On AD 2 the object is the resampling unit from here on.
