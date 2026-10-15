# 06 · E09 — Data efficiency: how many normal images does the recipe need?

**Development study, exploratory.** Three development categories (`pcb1`, `macaroni2`,
`capsules`), the frozen recipe (PatchCore, WRN50-2 layer2+3, 640 px, fixed 10 k bank,
candidate fraction 0.1), fitted on nested random subsets of the training normals. Planned in
[docs/13 §7 E09](../docs/13-project-review-and-research-roadmap.md); 45 subset runs plus the 9
full-data runs, all at implementation `784f13e0bd4b` (commit `6858bfc`). Raw table:
[studies/e09/data_efficiency.csv](studies/e09/data_efficiency.csv).

## Design

- **Fit shots vary, calibration and evaluation do not.** `data.train_limit = n` keeps the first
  `n` images of a seeded permutation of the training split; for one subset seed the subsets are
  nested (10 ⊂ 25 ⊂ … ⊂ 200), so a curve over `n` changes the amount of data and nothing else.
  The validation normals that set the operating point (136 / 135 / 81) and the test set are the
  same in every run. "10 images" below therefore means **10 fitting images plus 81–136
  calibration normals**, not a 10-image end-to-end setup.
- **Three subset seeds** (0, 1, 2; the subset seed is the run seed), so the spread below covers
  the subset draw, the candidate sampling and the coreset together.
- **The bank is fixed at 10 k from `n = 25` up.** At `n = 10` the whole candidate pool is
  smaller than the bank (4,320–4,880 patches) and is stored entirely, so the first point also has
  a smaller bank.

## Results

Mean ± SD over three subset seeds; "worst" is the worst seed. Full data is 768 / 765 / 461
training normals.

| category | fit images | image AUROC | worst | AU-PRO@0.05 | Recall@OP1 | FPR@OP1 | fit s |
|---|---|---|---|---|---|---|---|
| `pcb1` | 10 | 0.890 ± 0.014 | 0.878 | 0.528 ± 0.061 | 0.28 | 0.7% | 2 |
| | 25 | 0.908 ± 0.003 | 0.906 | 0.731 ± 0.056 | 0.43 | 4.0% | 7 |
| | 50 | 0.946 ± 0.006 | 0.941 | 0.804 ± 0.008 | 0.51 | 5.0% | 12 |
| | 100 | 0.955 ± 0.008 | 0.947 | 0.826 ± 0.003 | 0.67 | 5.7% | 14 |
| | 200 | 0.960 ± 0.009 | 0.951 | 0.844 ± 0.002 | 0.76 | 5.7% | 32 |
| | all (768) | 0.979 ± 0.001 | 0.978 | 0.857 ± 0.008 | 0.85 | 4.7% | 118 |
| `macaroni2` | 10 | 0.723 ± 0.030 | 0.689 | 0.686 ± 0.015 | 0.20 | 4.0% | 2 |
| | 25 | 0.762 ± 0.014 | 0.747 | 0.742 ± 0.012 | 0.24 | 3.0% | 7 |
| | 50 | 0.800 ± 0.010 | 0.790 | 0.799 ± 0.009 | 0.20 | 2.7% | 9 |
| | 100 | 0.812 ± 0.004 | 0.808 | 0.826 ± 0.002 | 0.18 | 2.0% | 13 |
| | 200 | 0.830 ± 0.003 | 0.827 | 0.855 ± 0.003 | 0.23 | 1.7% | 31 |
| | all (765) | 0.884 ± 0.005 | 0.879 | 0.874 ± 0.007 | 0.39 | 1.7% | 107 |
| `capsules` | 10 | 0.775 ± 0.030 | 0.742 | 0.721 ± 0.014 | 0.06 | 0.0% | 2 |
| | 25 | 0.811 ± 0.013 | 0.799 | 0.758 ± 0.016 | 0.14 | 0.6% | 8 |
| | 50 | 0.830 ± 0.026 | 0.813 | 0.802 ± 0.004 | 0.22 | 0.0% | 9 |
| | 100 | 0.878 ± 0.007 | 0.870 | 0.827 ± 0.005 | 0.24 | 0.0% | 13 |
| | 200 | 0.899 ± 0.005 | 0.894 | 0.856 ± 0.003 | 0.35 | 0.6% | 35 |
| | all (461) | 0.919 ± 0.006 | 0.916 | 0.864 ± 0.012 | 0.54 | 2.8% | 68 |

## Reading

- **No plateau before the full training set on detection.** With 200 images the remaining
  image-AUROC gap to full data is 0.019 / 0.055 / 0.020; `macaroni2`, the category with the
  highest normal variance, is still climbing between 200 and 765 images. More normals keep
  paying on every category, slowly.
- **Localization saturates earlier than detection.** At 200 images AU-PRO@0.05 is within
  0.008–0.019 of full data and pixel AUROC reaches ≈ 0.99 by 25–100 images, while image AUROC
  still trails by up to 0.055. A small bank covers the typical normal patch; what extra images
  add is coverage of *rare* normal patches, and those matter for an image score that is the
  maximum over the image — one uncovered normal patch is enough for a false positive. This is
  the same mechanism as the development findings on aggregation (report §3).
- **Resolution is worth about an order of magnitude of data.** Ten fitting images at 640 px
  match PatchCore fitted on the *full* training set at 320 px on `macaroni2` (0.723 ± 0.030 vs
  0.708 ± 0.011) and exceed it on `capsules` (0.775 ± 0.030 vs 0.710 ± 0.044; 320 px: E01,
  development report §2, 3 seeds); `pcb1` needs about 50 (0.946 vs 0.944). This is a
  cross-experiment comparison on the same test images, not a paired test, and the `macaroni2`
  difference is within the seed spread.
- **The operating point stays honest because calibration was not shrunk.** The realized false
  alarm rate does not grow as fitting data shrinks: the threshold is re-set each time on the same
  81–136 validation normals, scored by the same small model as the test normals, so the two stay
  exchangeable. What small fits lose is **recall at the operating point** (0.06–0.28 at 10 images vs
  0.39–0.85 at full data). `pcb1`'s overshoot (4–6% at a 0.73% bound from 25 images up) is the
  background-debris failure documented in the development report §4, not a data-size effect.
- **Cost.** Fitting falls from 68–118 s on full data to 13–14 s at 100 images and ~2 s at 10, at
  90–95% (`pcb1`, `capsules`) and 81% (`macaroni2`) of the full-data above-chance AUROC with 100
  images.

## What this does not establish

- **Exploratory, development categories only.** The categories have been looked at throughout
  development; nothing here is confirmatory.
- **One method.** The comparison with other models at the same shots, and the planned contrast
  between *more distinct images* and *more patches per image* at the same bank budget
  (candidate fraction), were not run.
- **Calibration size was held fixed by design.** A genuinely few-shot deployment also has few
  calibration normals; the effect of that is E03 (development report §8), and the two together
  bound what a small-data setup can promise.
