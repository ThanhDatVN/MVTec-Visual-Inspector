# Kaggle development studies: ablation sweep (Stage E), autoencoder (Stage F), anomalib comparison (C2)

**Development evidence.** Three development categories (`pcb1`, `macaroni2`, `capsules`), 126 runs
on a Tesla T4 in three Kaggle sessions, all at implementation `e53218f1371e` (commit `86300d2`),
clean checkouts. Raw tables: [studies/kaggle/](studies/kaggle/), produced by
`python scripts/kaggle_studies.py --implementation e53218f1371e`. Image-AUROC differences are
paired on the same test images with a class-stratified bootstrap (10,000 resamples); **bold**
marks a difference that survives Holm correction within its family.

Dataset: VisA © Amazon, CC BY 4.0 — see [ATTRIBUTION.md](../ATTRIBUTION.md).

## Summary

1. **The pipeline reproduces exactly.** The nine reference runs, repeated independently in each
   session, produced bit-identical predictions; on different hardware (the laptop's RTX 3050)
   they agree within the seed spread.
2. **Two levers still pay beyond the frozen recipe: resolution and feature level.** 960 px adds
   +0.058 / +0.062 image AUROC on `macaroni2` / `capsules` for twice the cost; `layer2` alone
   adds +0.064 / +0.035 at no cost. Both are screening results, single seed, on the categories
   that shaped the recipe.
3. **Everything else is flat or worse.** A larger bank, k-NN averaging, the re-weighting and a
   smaller projection change nothing measurable; adding `layer4` or dropping `layer2` is
   catastrophic. A 512-d projection gives the same accuracy for 16% less latency.
4. **The autoencoder rung is closed as a negative result.** At 640 px and 150 epochs it is
   0.17–0.67 image AUROC behind PatchCore on every category and run, and below chance on `pcb1`.
5. **C2: our PatchCore is not distinguishable from anomalib v2.3.0 as shipped** (no difference
   survives Holm), at about a fifth of the host memory and fit time. The two deliberate
   differences — pipeline and backbone weights — each move image AUROC by up to ±0.06, in
   directions that change with the category.

## 1. Reproducibility

The PatchCore reference (640 px recipe, 3 seeds) ran first in every session. All nine run ids
recurred in all three sessions with identical prediction digests. Across hardware:

| category | laptop, E04 (RTX 3050) | Kaggle (T4) | 320 px, E01 laptop | 320 px, Kaggle (C2 "ours") |
|---|---|---|---|---|
| `pcb1` | 0.979 ± 0.001 | 0.979 ± 0.002 | 0.944 ± 0.003 | 0.943 ± 0.004 |
| `macaroni2` | 0.884 ± 0.005 | 0.887 ± 0.010 | 0.708 ± 0.011 | 0.707 ± 0.003 |
| `capsules` | 0.919 ± 0.006 | 0.914 ± 0.003 | 0.710 ± 0.044 | 0.717 ± 0.034 |

(Image AUROC, mean ± SD over 3 seeds; different hardware and code versions.)

## 2. Stage E — ablation sweep

One axis at a time from the frozen recipe (WRN50-2, `layer2+layer3`, 1024-d projection, fixed
10k bank, nearest-neighbour score with re-weighting, 640 px), seed 0, against the seed-0
reference. 51 comparisons; Holm over all of them. Δ AU-PRO is a point estimate — read it
against the reference's seed SD of 0.004 / 0.011 / 0.004. Cost columns are ratios to the
reference (fit 45–82 s, 90–102 ms per image end to end, 3.3–4.5 GB peak RSS).

| variant | Δ I-AUROC `pcb1` | `macaroni2` | `capsules` | mean Δ AU-PRO | fit time | latency | peak RSS (fit) |
|---|---|---|---|---|---|---|---|
| 960 px | +0.013 [+0.002, +0.026] | **+0.058** [+0.029, +0.091] | **+0.062** [+0.030, +0.098] | +0.037 | ×1.99 | ×2.00 | 7.4 GB |
| 1280 px | -0.000 [-0.015, +0.015] | **+0.076** [+0.043, +0.112] | **+0.068** [+0.031, +0.111] | +0.049 | ×3.41 | ×3.52 | 11.5 GB |
| layers `layer2` | +0.010 [-0.001, +0.024] | **+0.064** [+0.030, +0.100] | **+0.035** [+0.012, +0.060] | +0.032 | ×1.00 | ×0.98 | 4.7 GB |
| layers `layer3` | -0.015 [-0.033, +0.001] | **-0.233** [-0.295, -0.175] | **-0.123** [-0.176, -0.075] | -0.144 | ×0.72 | ×0.69 | 2.7 GB |
| layers `layer2+3+4` | **-0.097** [-0.142, -0.057] | **-0.465** [-0.543, -0.385] | **-0.315** [-0.409, -0.225] | -0.548 | ×1.04 | ×1.03 | 4.7 GB |
| backbone ResNet-50 | -0.004 [-0.016, +0.006] | +0.026 [-0.008, +0.062] | +0.012 [-0.017, +0.042] | -0.001 | ×0.82 | ×0.78 | 4.7 GB |
| backbone ResNet-18 | **-0.023** [-0.042, -0.008] | +0.016 [-0.017, +0.050] | +0.014 [-0.016, +0.045] | +0.003 | ×0.72 | ×0.64 | 4.7 GB |
| bank 2,500 | +0.003 [-0.002, +0.009] | -0.030 [-0.057, -0.005] | -0.020 [-0.051, +0.009] | -0.040 | ×0.81 | ×0.70 | 4.7 GB |
| bank 5,000 | +0.003 [-0.000, +0.008] | -0.013 [-0.032, +0.006] | +0.009 [-0.009, +0.027] | -0.026 | ×0.88 | ×0.79 | 4.7 GB |
| bank 20,000 | -0.001 [-0.003, +0.002] | +0.001 [-0.010, +0.014] | -0.003 [-0.018, +0.013] | +0.003 | ×1.29 | ×1.33 | 4.7 GB |
| bank 40,000 | -0.002 [-0.006, +0.002] | +0.004 [-0.009, +0.019] | -0.012 [-0.028, +0.004] | +0.003 | ×1.83 | ×2.11 | 4.6 GB |
| k = 3, mean | -0.002 [-0.007, +0.001] | -0.002 [-0.013, +0.010] | +0.009 [-0.006, +0.027] | +0.009 | ×1.03 | ×1.00 | 4.7 GB |
| k = 9, mean | -0.013 [-0.025, -0.003] | +0.001 [-0.018, +0.021] | -0.005 [-0.028, +0.018] | +0.004 | ×1.02 | ×1.00 | 4.7 GB |
| k = 3, k-th | -0.016 [-0.031, -0.004] | -0.015 [-0.038, +0.006] | -0.010 [-0.037, +0.017] | +0.003 | ×1.01 | ×1.00 | 4.7 GB |
| re-weighting off | +0.002 [+0.000, +0.005] | +0.002 [-0.003, +0.007] | +0.004 [-0.002, +0.010] | +0.000 | ×1.01 | ×0.99 | 4.7 GB |
| projection 512 | **+0.006** [+0.002, +0.011] | -0.003 [-0.023, +0.017] | +0.014 [-0.005, +0.034] | -0.002 | ×0.93 | ×0.84 | 3.4 GB |
| projection 256 | -0.005 [-0.012, +0.000] | +0.001 [-0.021, +0.024] | +0.018 [+0.000, +0.038] | +0.002 | ×0.91 | ×0.78 | 2.8 GB |

**Resolution.** The largest effect in E04 keeps paying past 640 px on the two categories whose
defects are small relative to the frame: +0.06 to +0.08 image AUROC and +0.03 to +0.06 AU-PRO on
`macaroni2` and `capsules`, with recall at the operating point up by 0.17–0.33. `pcb1` is
saturated (seed 0: 0.978 → 0.990 at 960, 0.978 at 1280). 1280 px buys little over 960 px for 1.7× more
cost. **Correction:** the host memory these need was estimated above 16 GB when the sweep was
planned; measured, it peaks at 7.4 GB (960) and 11.5 GB (1280), within the laptop's 16 GB.
(An earlier version of this report also quoted 4.1–5.1 GB of GPU memory at these resolutions.
Those records are not usable: until the fix described in [E8](E8-gate-e-followup.md), every
finished run left its backbone on the GPU until Python's cycle collector ran, so a run's recorded
peak includes earlier runs'. Clean, one-process-per-configuration measurements are in E8.)

**Feature level.** The strongest free lever. `layer2` alone beats `layer2+layer3` on every
category (significant on two) at identical cost, and `layer3` alone loses 0.12–0.23 on
`macaroni2`/`capsules` — the opposite of the "layer3 is enough" claim recorded as unverified in
docs/09. Adding `layer4` collapses `macaroni2` to 0.42: ImageNet-semantic features describe
*what* the object is, and every image here is the same object. Finer features help because the
defects are local texture.

**Backbone.** ResNet-50 is indistinguishable from WRN50-2 at 0.8× the cost; ResNet-18 loses
0.023 on `pcb1`. No reason to change.

**Bank size.** 10k is the knee: 2.5k and 5k lose 0.04–0.07 AU-PRO on `macaroni2` (the category
with the most normal variation), and 20k / 40k gain nothing while latency grows 1.3× / 2.1×. No
budget was capped.

**Neighbours, re-weighting.** Averaging over k neighbours or scoring by the k-th distance never
helps and costs up to 0.016 on `pcb1`; the reference re-weighting changes image AUROC by
≤ 0.004 and AU-PRO by nothing (it only rescales the image score), matching E02's stored-map
finding. Both stay as they are for reference fidelity.

**Projection.** 512-d (and 256-d) keep accuracy within noise while cutting latency by 16–22% and
peak fit memory by 18–39%. A cost reduction, not an accuracy change.

**Screening outcome (Gate E).** Variants worth a 3-seed rerun: **960 px**, **`layer2` only**, and
the joint re-vary E7 of the two (the two largest effects), plus 512-d as a free efficiency. Not
run yet. Adoption would also need evidence on data that shaped nothing: the nine VisA
confirmation categories are spent (ADR-11) and MPDD has now been used for the frozen recipe, so
an improved recipe needs MVTec AD 2 or another fresh dataset.

## 3. Stage F — the autoencoder at 640 px

The CAE (multiscale residual, latent 128) at the recipe's 640 px, 150 epochs with patience 20,
one seed per loss:

| loss | category | epochs run (max 150) | best val. loss | I-AUROC | AU-PRO@0.05 | P-AUROC | PatchCore − CAE, I-AUROC | fit |
|---|---|---|---|---|---|---|---|---|
| l2 | `pcb1` | 150 | 0.0220 | 0.314 | 0.058 | 0.465 | +0.665 [+0.586, +0.741] | 37 min |
| l2 | `macaroni2` | 150 | 0.0028 | 0.689 | 0.373 | 0.976 | +0.197 [+0.121, +0.276] | 31 min |
| l2 | `capsules` | 150 | 0.0149 | 0.749 | 0.199 | 0.876 | +0.165 [+0.078, +0.257] | 18 min |
| l2+ssim | `pcb1` | 150 | 0.1983 | 0.426 | 0.045 | 0.374 | +0.553 [+0.471, +0.632] | 48 min |
| l2+ssim | `macaroni2` | 150 | 0.0493 | 0.551 | 0.610 | 0.985 | +0.336 [+0.257, +0.417] | 41 min |
| l2+ssim | `capsules` | 150 | 0.1851 | 0.742 | 0.096 | 0.656 | +0.172 [+0.096, +0.254] | 25 min |
| ssim | `pcb1` | 150 | 0.1464 | 0.543 | 0.078 | 0.253 | +0.436 [+0.355, +0.521] | 48 min |
| ssim | `macaroni2` | 150 | 0.0436 | 0.560 | 0.600 | 0.984 | +0.326 [+0.246, +0.408] | 41 min |
| ssim | `capsules` | 150 | 0.1712 | 0.702 | 0.117 | 0.664 | +0.212 [+0.129, +0.298] | 24 min |

- **Not converged by its own rule, and it does not matter.** No run triggered early stopping:
  validation loss was still improving at epoch 150. But 2.5× the epochs and 2× the resolution of
  E01 (320 px, 60 epochs: 0.41–0.69) left detection where it was — some runs up, `pcb1`-L2 down
  to 0.31. Reconstruction keeps improving; telling defects from normal texture does not.
- **`pcb1` is inverted at both scales**, at image and pixel level (P-AUROC 0.25–0.47): the CAE
  reconstructs the smooth regions where `pcb1`'s defects lie better than the high-frequency
  normal components, the case-book mechanism (§3) at a higher resolution.
- **Good maps, bad image score again on `macaroni2`** — pixel AUROC 0.98 at image AUROC 0.55–0.69
  — as for pixel PCA in E01. An aggregation rule chosen on validation normals might recover some
  of it, but the CAE maps (~200 MB per run) were left out of the export and stay in session B's
  Kaggle output; that study was not run.
- **Gate F:** the ImageNet prior, as PatchCore uses it, is worth +0.17 to +0.67 image AUROC per
  category over the from-scratch CAE (every interval excludes zero). The CAE costs 18–48 GPU-
  minutes per fit against PatchCore's ~1 minute. The rung is closed; F2/F3 (latent size,
  residual) were not run, because nothing here suggests capacity or residual form is the limit.

## 4. C2 — our PatchCore against anomalib v2.3.0 on real images

anomalib's pipeline is vendored at its pinned commit and runs through the same runner,
evaluator and 320 px inputs (its all-patch store would need ~23 GB at 640 px). Three arms ×
3 seeds; the bank is 1% of the patches either way.

| category | arm | I-AUROC | AU-PRO@0.05 | bank × dim | fit | peak RSS (fit) |
|---|---|---|---|---|---|---|
| `pcb1` | ours (torchvision weights) | 0.943 ± 0.004 | 0.732 ± 0.004 | 9,523 × 1024 | 33 s | 2.6 GB |
| `pcb1` | anomalib (torchvision weights) | 0.972 ± 0.003 | 0.746 ± 0.006 | 9,523 × 1536 | 198 s | 13.1 GB |
| `pcb1` | anomalib (timm default weights) | 0.944 ± 0.007 | 0.719 ± 0.008 | 9,523 × 1536 | 201 s | 13.0 GB |
| `macaroni2` | ours (torchvision weights) | 0.707 ± 0.003 | 0.655 ± 0.030 | 8,262 × 1024 | 30 s | 2.5 GB |
| `macaroni2` | anomalib (torchvision weights) | 0.664 ± 0.014 | 0.591 ± 0.033 | 8,262 × 1536 | 148 s | 11.7 GB |
| `macaroni2` | anomalib (timm default weights) | 0.727 ± 0.005 | 0.660 ± 0.027 | 8,262 × 1536 | 150 s | 11.6 GB |
| `capsules` | ours (torchvision weights) | 0.717 ± 0.034 | 0.425 ± 0.011 | 4,979 × 1024 | 18 s | 2.3 GB |
| `capsules` | anomalib (torchvision weights) | 0.731 ± 0.005 | 0.446 ± 0.009 | 4,978 × 1536 | 64 s | 7.9 GB |
| `capsules` | anomalib (timm default weights) | 0.782 ± 0.021 | 0.516 ± 0.014 | 4,978 × 1536 | 65 s | 7.8 GB |

Paired differences in seed-averaged image AUROC (Holm over these 9):

| category | pipeline (anomalib-tv − ours) | weights (timm default − torchvision) | as shipped (anomalib default − ours) |
|---|---|---|---|
| `pcb1` | **+0.029** [+0.013, +0.049] | **-0.028** [-0.048, -0.012] | +0.001 [-0.023, +0.025] |
| `macaroni2` | -0.043 [-0.077, -0.009] | +0.063 [+0.014, +0.113] | +0.020 [-0.028, +0.067] |
| `capsules` | +0.014 [-0.032, +0.057] | +0.051 [+0.004, +0.099] | +0.065 [+0.012, +0.120] |

- **Verdict (Gate C):** as shipped, anomalib and our implementation are not distinguishable on
  any category after correction (+0.001, +0.020, +0.065); the largest gap, on `capsules`, comes
  from the weights rather than the pipeline. At 320 px, the PatchCore conclusions in this
  repository stand on an implementation that performs like the reference library.
- **Neither design difference is uniformly better.** anomalib's pipeline (1536-d descriptor,
  coreset over all patches) helps `pcb1` and hurts `macaroni2`; timm's default weights
  (`racm_in1k`, trained with RandAugment/CutMix/Mixup) hurt `pcb1` and help `macaroni2` and
  `capsules`. The effects are as large as many sweep axes, so **backbone weights are an
  unexplored axis** — worth a 640 px check before anyone concludes the backbone does not matter.
- **Cost:** per-image candidate sampling and the 1024-d pooling give our fit about a fifth of
  anomalib's host memory (2.3–2.6 GB vs 7.8–13.1 GB) and time, with the same inference latency.
  At 640 px that difference is what makes the recipe fit a 16 GB machine at all.

## What this does not establish

- **Development evidence on three categories.** The sweep is a single-seed screen; its winners
  need 3-seed reruns and new held-out data before any claim of improvement.
- **C2 at 320 px only.** The comparison at the recipe's 640 px needs anomalib's all-patch store
  (~23 GB) and was not run.
- **One CAE seed per loss**, no aggregation study for the CAE (maps not exported).
