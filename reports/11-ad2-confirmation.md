# 11 · E13 — Confirmation v2 on MVTec AD 2: the candidate recipe is not confirmed

**Pre-registered** in [ADR-13](../docs/07-risks-and-decisions.md), committed and pushed (`e39e5fd`)
before any MVTec AD 2 archive was downloaded. Candidate recipe v2 (`layer2` features at 960 px,
from E8) against recipe v1 (`layer2+layer3` at 640 px), both unchanged, on all eight AD 2
categories, seeds 0–2, plus pixel-PCA and colour-histogram floors: **64 runs, none failed**, on
the laptop (RTX 3050), implementation `af89b5297f4d`, from a clean worktree. Each category is
fitted on `train/good`, calibrated on its own `validation/good`, and evaluated once on
`test_public`. The analysis is the committed script's output, unedited:
[studies/ad2-confirmation/confirmation.md](studies/ad2-confirmation/confirmation.md).

These are `test_public` numbers. They are **not comparable** with published AD 2 results, which
are computed on the private test sets by MVTec's server (DR-5).

Dataset: MVTec AD 2 © MVTec Software GmbH, CC BY-NC-SA 4.0 (Heckler-Kram et al., 2025) — see
[ATTRIBUTION.md](../ATTRIBUTION.md). No AD 2 image or derived figure is published here.

## Verdict: not confirmed

| hypothesis | rule (ADR-13) | result | holds? |
|---|---|---|---|
| **H3**, image AUROC v2 − v1, Holm over 8 | ≥ 5 categories significantly positive, none negative | 1 positive (`sheet_metal` +0.156), 1 negative (`vial` −0.111); macro +0.004 [−0.023, +0.032] | **no** |
| **H3b**, AU-PRO@0.05 v2 > v1 | ≥ 7 of 8 categories | 6 of 8 (sign test p = 0.14); macro 0.357 → 0.409 | **no** |

Recipe v1 stays the recommended recipe; candidate v2 is not adopted.

## Per category

Mean ± SD over seeds 0–2 (floors: seed 0).

| category | image AUROC v1 | v2 | AU-PRO@0.05 v1 | v2 | FPR@OP1 v1 / v2 (bound) | recall v1 / v2 | pixel PCA | histogram |
|---|---|---|---|---|---|---|---|---|
| `can` | 0.516 ± 0.013 | 0.504 ± 0.019 | 0.097 | 0.151 | 51% / 45% (2.1%) | 0.52 / 0.46 | 0.489 | 0.452 |
| `fabric` | 0.645 ± 0.017 | 0.607 ± 0.051 | 0.167 | 0.331 | 11% / 4% (2.3%) | 0.08 / 0.09 | 0.533 | 0.441 |
| `fruit_jelly` | 0.899 ± 0.028 | 0.891 ± 0.012 | 0.498 | 0.522 | 0% / 0% (2.6%) | 0.14 / 0.08 | 0.592 | 0.457 |
| `rice` | 0.565 ± 0.057 | 0.572 ± 0.046 | 0.262 | 0.302 | 0% / 6% (2.8%) | 0.14 / 0.19 | 0.568 | 0.574 |
| `sheet_metal` | 0.625 ± 0.051 | **0.781 ± 0.039** | 0.181 | 0.296 | 3% / 0% (5.0%) | 0.30 / 0.31 | 0.546 | 0.462 |
| `vial` | **0.897 ± 0.012** | 0.785 ± 0.005 | 0.762 | 0.737 | 0% / 14% (2.4%) | 0.34 / 0.39 | 0.599 | 0.476 |
| `wallplugs` | 0.484 ± 0.036 | 0.518 ± 0.018 | 0.272 | 0.342 | 14% / 29% (2.9%) | 0.13 / 0.24 | 0.616 | 0.418 |
| `walnuts` | 0.853 ± 0.012 | 0.861 ± 0.015 | 0.614 | 0.588 | 0% / 8% (2.0%) | 0.25 / 0.57 | 0.732 | 0.461 |
| **macro** | **0.685** | **0.690** | **0.357** | **0.409** | pooled false alarms: 160 / 195 vs 29.2 expected | 0.24 / 0.29 | 0.584 | 0.468 |

## Reading

- **AD 2 is a different regime from VisA.** The same v1 recipe that reached 0.981 macro image
  AUROC on nine held-out VisA categories (E12) reaches 0.685 here: at chance on `can` (0.52) and
  `wallplugs` (0.48), and below pixel PCA on `wallplugs` and `rice`. AD 2 was built to be hard — small defects, cluttered and
  transparent objects, lighting variation — and a recipe chosen on VisA's fixed rigs does not
  transfer. Localization is weaker still: AU-PRO@0.05 0.36 against 0.77 on VisA.
- **v2 changes what VisA said it would only half the way.** It localizes better on six of eight
  categories (AU-PRO@0.30: 0.58 → 0.67), with large gains where defects are small relative to the
  frame (`fabric` +0.16, `sheet_metal` +0.12), as on VisA. But it does not detect better: image
  AUROC moves by +0.004 on average, up on `sheet_metal`, down on `vial` (0.897 → 0.785), and
  its false alarms are worse. The image maximum over more, finer patches is set by more normal
  variation as often as by the defect.
- **The operating point fails, for both recipes.** At bounds of 2.0–5.0% (AD 2's 19–48
  validation normals cannot support 1%), the pooled false alarms are 160 (v1) and 195 (v2) against
  29.2 expected; on `can` half the test normals are flagged, and the exact exchangeability test
  rejects on `can` (both recipes) and `wallplugs` (v2). This is the VisA `pcb1` failure at scale:
  the validation normals do not represent the test normals.

## Exploratory (not pre-registered): the test normals are a shifted population

The input guards fitted on each category's training normals (exposure and focus ranges, E08b)
refuse 0–8% of the validation normals but **20–83% of the `test_public` normals** in six of eight
categories (`rice` 83%, `fabric` 41%, `vial` 34%, `sheet_metal` 33%, `can` 32%, `wallplugs` 20%;
`walnuts` 8%, `fruit_jelly` 5%). Median image intensity moves with them (`rice` 151 → 142,
`wallplugs` 52 → 41, `walnuts` 44 → 38 from validation to test). So `test_public` contains
capture conditions — exposure, focus, lighting — that the training and validation normals do not,
which answers open decision OD-1: lighting variation is not confined to `test_private_mixed`.
[studies/ad2-confirmation/guards_exploratory.csv](studies/ad2-confirmation/guards_exploratory.csv).

The guards do not remove the failure, though. Among the normals they accept, v1's false-alarm
rate is 0–8% on seven categories — about what it was on all normals — and still **49% on `can`**,
where the shift is not one global exposure and focus statistics see. The refusals also reach defective images (up to 90% on `rice`), so on AD 2
the guards would send most of some categories to manual review — honest, but not a detector.

## What follows

- **v1 remains the recipe; v2 is not adopted** (negative results #17).
- **AD 2 `test_public` is now development data.** Anything changed in response to these numbers
  is a development result; the private test sets behind MVTec's server are the held-out data left.
- The two failures it exposes — an operating point that breaks under a shifted normal population,
  and image scores dominated by normal variation — are the subject of the development study E14.
