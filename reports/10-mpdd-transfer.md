# 10 · H1 — MPDD transfer probe: the frozen recipe on a second dataset

**Pre-registered** in [ADR-12](../docs/07-risks-and-decisions.md) (commit `7810b7a`, 2026-10-02),
before any MPDD image was opened: the confirmation-v1 recipe (PatchCore, WRN50-2 `layer2+layer3`,
640 px, fixed 10k bank) applied unchanged to all six MPDD categories, seeds 0–2, with validation
carved from the training normals (15%, seed 0) exactly as for VisA; pixel PCA and the colour
histogram at 320 px as floors. ADR-12 attaches **no pass/fail rule**: the numbers are the result.

Ran on Kaggle (Tesla T4) at implementation `e53218f1371e` (commit `86300d2`), 30 runs, clean
checkout. Raw table: [studies/kaggle/mpdd.csv](studies/kaggle/mpdd.csv), from
`scripts/visa_k1_analysis.py`. MPDD (Jezek et al., ICUMT 2021) is CC BY-NC-SA 4.0; no MPDD image or
derived figure is published — see [ATTRIBUTION.md](../ATTRIBUTION.md).

## Results

| category | train / calib. / test (normal + defect) | I-AUROC (3 seeds) | AU-PRO@0.05 | FPR@OP1 (bound) | false alarms, observed / expected | recall@OP1 | pixel PCA | colour histogram | recipe − PCA, I-AUROC |
|---|---|---|---|---|---|---|---|---|---|
| `bracket_black` | 246 / 43 / 32 + 47 | 0.890 ± 0.017 | 0.919 | 1.0% (2.3%) | 1 / 2.2 | 0.64 | 0.604 | 0.281 | +0.286 [+0.133, +0.435] |
| `bracket_brown` | 157 / 28 / 26 + 51 | 0.927 ± 0.007 | 0.866 | 9.0% (3.4%) | 7 / 2.7 | 0.64 | 0.912 | 0.845 | +0.016 [-0.051, +0.096] |
| `bracket_white` | 94 / 16 / 30 + 30 | 0.883 ± 0.013 | 0.869 | 10.0% (5.9%) | 9 / 5.3 | 0.72 | 0.888 | 0.812 | -0.005 [-0.119, +0.111] |
| `connector` | 109 / 19 / 30 + 14 | 1.000 ± 0.000 | 0.811 | 0.0% (5.0%) | 0 / 4.5 | 1.00 | 1.000 | 0.974 | +0.000 [+0.000, +0.000] |
| `metal_plate` | 46 / 8 / 26 + 71 | 0.997 ± 0.002 | 0.856 | 17.9% (11.1%) | 14 / 8.7 | 1.00 | 0.552 | 0.733 | +0.445 [+0.332, +0.565] |
| `tubes` | 104 / 18 / 32 + 69 | 0.959 ± 0.009 | 0.846 | 0.0% (5.3%) | 0 / 5.1 | 0.89 | 0.418 | 0.576 | +0.541 [+0.434, +0.646] |

Recipe − PCA is the paired difference in seed-averaged image AUROC on the same test images
(class-stratified bootstrap, 10,000 resamples), descriptive only.

**Macro:** image AUROC **0.943**, AU-PRO@0.05 **0.861**; pixel PCA 0.729, colour histogram 0.703.
**The recipe beats pixel PCA on 4 of 6 categories** by point estimate — clearly (interval
excluding zero) on `bracket_black`, `metal_plate` and `tubes` — ties it on `connector`, where both
are perfect, and is level with it on `bracket_white` (−0.005, interval ±0.11).

## Reading

- **The recipe transfers without retuning.** Every category reaches image AUROC ≥ 0.88 and
  AU-PRO@0.05 ≥ 0.81 on a different camera, factory and annotation team, with non-uniform
  backgrounds and varying orientation — the conditions VisA's fixed rig does not test. The
  macro AUROC (0.943) is below VisA's confirmation result (0.981), and the three brackets carry
  the gap.
- **Where the floors keep up, the categories are easy globally.** On `connector`,
  `bracket_brown` and `bracket_white` a linear model on 64×64 pixels or a colour histogram
  already reaches 0.81–1.00. Their defect classes (parts mismatch, bends, defective painting)
  plausibly change the image as a whole — an inference from the class names and the floors, not
  from inspecting images. On `bracket_black`, `metal_plate` and `tubes` the floors fall to
  0.28–0.73 and the recipe is 0.29–0.54 ahead of pixel PCA. That split is what the floors exist
  to expose.
- **The 1% operating point cannot be set on MPDD.** Calibration sets of 8–43 normals make the
  smallest achievable rate 1/(n+1) = 2.3–11.1%; every category was relaxed to it and flagged
  (`target_met = false`, ADR-7/ADR-9). With the threshold at the single highest calibration
  score, the realized rate scatters widely: 0–18% per category (mean over seeds). Pooled over the 18 runs it is
  consistent with the bounds — **31 false alarms against 28.4 expected** — and no run's
  exceedance is significant under the exact exchangeability test (smallest p = 0.10, against
  p = 0.006 for the `pcb1` failure in E03b). The excess on `metal_plate`, `bracket_white` and
  `bracket_brown` is what 8–28 calibration images allow, not evidence of a capture-session effect
  like VisA's debris.
- **Recall at that operating point is 0.64–1.00**, the counterpart of the relaxed bounds: a
  threshold set by so few normals is loose, so it misses fewer defects and raises more alarms.

## What this does not establish

- **No comparison with published MPDD numbers.** The validation carve-out removes 15% of the
  training normals and the operating point is ours; the literature does not share a protocol.
- **Small test sets** (14–71 defective, 26–32 normal images per category): per-category intervals
  are wide, and a deployment-grade false-alarm rate would need far more calibration normals than
  MPDD provides.
- **MPDD is now used.** With the nine VisA confirmation categories (ADR-11), it has been spent on
  the frozen recipe; a recipe changed in response to the Kaggle sweep needs fresh data.
