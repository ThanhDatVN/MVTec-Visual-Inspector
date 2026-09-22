# P1 — Dataset Audit and Exploratory Analysis

Generated 2026-09-22 14:04 UTC by `inspector eda`. Do not edit by hand.

This report exists to answer one question before any model is fitted:
**what input resolution do these defects permit?** Everything else here is
supporting evidence for that decision (docs/04, task 1.7).

## Summary

| category | native | aspect | defect regions | median area | recommended |
|---|---|---|---|---|---|
| `pcb1` | 1404x1070 | 1.31:1 | 304 | 0.023% | 1404 (SAFE) |
| `macaroni2` | 1500x1000 | 1.50:1 | 378 | 0.002% | 1500 (MARGINAL) |
| `capsules` | 1500x1000 | 1.50:1 | 156 | 0.079% | 1500 (SAFE) |

## `pcb1`

### Splits

| split | images | normal | anomalous |
|---|---|---|---|
| test | 200 | 100 | 100 |
| train | 768 | 768 | 0 |
| validation | 136 | 136 | 0 |

Defect types: {'anomaly': 304}

### Defect size distribution

| statistic | p1 | p5 | median | max |
|---|---|---|---|---|
| area (px) | 1 | 8 | 348 | 95936 |
| area (% of image) | 0.0001% | 0.0005% | 0.0232% | 6.3860% |
| equivalent diameter (px) | 1.1 | 3.2 | 21.1 | - |
| thin dimension (px) | 1.0 | - | 18.0 | - |

Regions per anomalous image: 3.04. Local contrast against a 4px ring: median 58.2 grey levels, p5 4.9.

### Resolution impact — aspect-preserving

| long side | output | MP | scale | median diam. | p5 thin dim. | <1px | verdict |
|---|---|---|---|---|---|---|---|
| 256 | 256x195 | 0.05 | 0.182 | 3.8px | 0.55px | 19.4% | **DESTRUCTIVE** |
| 320 | 320x244 | 0.08 | 0.228 | 4.8px | 0.68px | 13.2% | **DESTRUCTIVE** |
| 448 | 448x341 | 0.15 | 0.319 | 6.7px | 0.96px | 8.6% | **DESTRUCTIVE** |
| 512 | 512x390 | 0.20 | 0.365 | 7.7px | 1.09px | 4.9% | **DESTRUCTIVE** |
| 1404 | 1404x1070 | 1.50 | 1.000 | 21.1px | 3.00px | 0.0% | **SAFE** |

`p5 thin dim.` is the 5th-percentile thin dimension of a defect after resizing, and `<1px` the share of regions whose thin dimension drops below one pixel. A region below one pixel cannot be detected by any model, so a resolution that destroys them measures the resize, not the method.

### Resolution impact — blind square resize

| long side | output | MP | scale | median diam. | p5 thin dim. | <1px | verdict |
|---|---|---|---|---|---|---|---|
| 256 | 256x256 | 0.07 | 0.209 | 4.4px | 0.55px | 19.4% | **DESTRUCTIVE** |
| 320 | 320x320 | 0.10 | 0.261 | 5.5px | 0.68px | 13.2% | **DESTRUCTIVE** |
| 448 | 448x448 | 0.20 | 0.366 | 7.7px | 0.96px | 8.6% | **DESTRUCTIVE** |
| 512 | 512x512 | 0.26 | 0.418 | 8.8px | 1.09px | 4.9% | **DESTRUCTIVE** |
| 1404 | 1404x1404 | 1.97 | 1.145 | 24.1px | 3.00px | 0.0% | **UPSCALED** |

Shown to quantify the cost of the reflexive `resize(256, 256)`. At an aspect ratio of 1.31:1 a square resize squashes one axis disproportionately, and the thin dimension of a defect is hit by whichever axis is squashed hardest.

### Decision

**Recommended: long side 1404 (1404x1070, 1.50 MP), verdict SAFE.**

cheapest resolution rated SAFE: p5 thin dimension 3.00px survives, 0.0% of regions lost, 1.50 MP within the 3.00 MP single-pass budget

Selection rule `smallest_safe` (docs/04): the cheapest resolution that is not LOSSY or DESTRUCTIVE. Compute is the scarce resource at 4 GB VRAM, so the rule is 'as small as the defects permit', not 'as large as fits'.

### Intensity shift relative to `train`

| split | mean delta | std ratio | Cohen's d |
|---|---|---|---|
| test | +1.40 | 0.990 | +0.363 |
| validation | +0.08 | 0.997 | +0.031 |

Largest shift: `test` at |d| = 0.363. A large value predicts threshold drift in Phase P7: a threshold fitted on validation lands somewhere else entirely on a shifted split, and the realized false-alarm rate moves even when AUROC does not.

## `macaroni2`

### Splits

| split | images | normal | anomalous |
|---|---|---|---|
| test | 200 | 100 | 100 |
| train | 765 | 765 | 0 |
| validation | 135 | 135 | 0 |

Defect types: {'anomaly': 378}

### Defect size distribution

| statistic | p1 | p5 | median | max |
|---|---|---|---|---|
| area (px) | 1 | 1 | 26 | 2854 |
| area (% of image) | 0.0001% | 0.0001% | 0.0017% | 0.1903% |
| equivalent diameter (px) | 1.1 | 1.1 | 5.7 | - |
| thin dimension (px) | 1.0 | - | 5.5 | - |

Regions per anomalous image: 3.78. Local contrast against a 4px ring: median 10.6 grey levels, p5 0.6.

### Resolution impact — aspect-preserving

| long side | output | MP | scale | median diam. | p5 thin dim. | <1px | verdict |
|---|---|---|---|---|---|---|---|
| 256 | 256x171 | 0.04 | 0.171 | 1.0px | 0.17px | 50.0% | **DESTRUCTIVE** |
| 320 | 320x213 | 0.07 | 0.213 | 1.2px | 0.21px | 43.9% | **DESTRUCTIVE** |
| 448 | 448x299 | 0.13 | 0.299 | 1.7px | 0.30px | 35.7% | **DESTRUCTIVE** |
| 512 | 512x341 | 0.17 | 0.341 | 1.9px | 0.34px | 27.5% | **DESTRUCTIVE** |
| 1500 | 1500x1000 | 1.50 | 1.000 | 5.7px | 1.00px | 0.0% | **MARGINAL** |

`p5 thin dim.` is the 5th-percentile thin dimension of a defect after resizing, and `<1px` the share of regions whose thin dimension drops below one pixel. A region below one pixel cannot be detected by any model, so a resolution that destroys them measures the resize, not the method.

### Resolution impact — blind square resize

| long side | output | MP | scale | median diam. | p5 thin dim. | <1px | verdict |
|---|---|---|---|---|---|---|---|
| 256 | 256x256 | 0.07 | 0.209 | 1.2px | 0.17px | 50.0% | **DESTRUCTIVE** |
| 320 | 320x320 | 0.10 | 0.261 | 1.5px | 0.21px | 43.9% | **DESTRUCTIVE** |
| 448 | 448x448 | 0.20 | 0.366 | 2.1px | 0.30px | 35.7% | **DESTRUCTIVE** |
| 512 | 512x512 | 0.26 | 0.418 | 2.4px | 0.34px | 27.5% | **DESTRUCTIVE** |
| 1500 | 1500x1500 | 2.25 | 1.225 | 7.0px | 1.00px | 0.0% | **UPSCALED** |

Shown to quantify the cost of the reflexive `resize(256, 256)`. At an aspect ratio of 1.50:1 a square resize squashes one axis disproportionately, and the thin dimension of a defect is hit by whichever axis is squashed hardest.

### Decision

**Recommended: long side 1500 (1500x1000, 1.50 MP), verdict MARGINAL.**

cheapest resolution rated MARGINAL: p5 thin dimension 1.00px survives, 0.0% of regions lost, 1.50 MP within the 3.00 MP single-pass budget

Selection rule `smallest_safe` (docs/04): the cheapest resolution that is not LOSSY or DESTRUCTIVE. Compute is the scarce resource at 4 GB VRAM, so the rule is 'as small as the defects permit', not 'as large as fits'.

### Intensity shift relative to `train`

| split | mean delta | std ratio | Cohen's d |
|---|---|---|---|
| test | -0.37 | 1.013 | -0.224 |
| validation | -0.15 | 1.008 | -0.110 |

Largest shift: `test` at |d| = 0.224. A large value predicts threshold drift in Phase P7: a threshold fitted on validation lands somewhere else entirely on a shifted split, and the realized false-alarm rate moves even when AUROC does not.

## `capsules`

### Splits

| split | images | normal | anomalous |
|---|---|---|---|
| test | 160 | 60 | 100 |
| train | 461 | 461 | 0 |
| validation | 81 | 81 | 0 |

Defect types: {'anomaly': 156}

### Defect size distribution

| statistic | p1 | p5 | median | max |
|---|---|---|---|---|
| area (px) | 4 | 9 | 1187 | 28667 |
| area (% of image) | 0.0002% | 0.0006% | 0.0791% | 1.9111% |
| equivalent diameter (px) | 2.1 | 3.4 | 38.9 | - |
| thin dimension (px) | 2.0 | - | 38.0 | - |

Regions per anomalous image: 1.56. Local contrast against a 4px ring: median 22.9 grey levels, p5 2.8.

### Resolution impact — aspect-preserving

| long side | output | MP | scale | median diam. | p5 thin dim. | <1px | verdict |
|---|---|---|---|---|---|---|---|
| 256 | 256x171 | 0.04 | 0.171 | 6.6px | 0.51px | 11.5% | **DESTRUCTIVE** |
| 320 | 320x213 | 0.07 | 0.213 | 8.3px | 0.64px | 9.6% | **DESTRUCTIVE** |
| 448 | 448x299 | 0.13 | 0.299 | 11.6px | 0.90px | 7.1% | **DESTRUCTIVE** |
| 512 | 512x341 | 0.17 | 0.341 | 13.3px | 1.02px | 4.5% | **DESTRUCTIVE** |
| 1500 | 1500x1000 | 1.50 | 1.000 | 38.9px | 3.00px | 0.0% | **SAFE** |

`p5 thin dim.` is the 5th-percentile thin dimension of a defect after resizing, and `<1px` the share of regions whose thin dimension drops below one pixel. A region below one pixel cannot be detected by any model, so a resolution that destroys them measures the resize, not the method.

### Resolution impact — blind square resize

| long side | output | MP | scale | median diam. | p5 thin dim. | <1px | verdict |
|---|---|---|---|---|---|---|---|
| 256 | 256x256 | 0.07 | 0.209 | 8.1px | 0.51px | 11.5% | **DESTRUCTIVE** |
| 320 | 320x320 | 0.10 | 0.261 | 10.2px | 0.64px | 9.6% | **DESTRUCTIVE** |
| 448 | 448x448 | 0.20 | 0.366 | 14.2px | 0.90px | 7.1% | **DESTRUCTIVE** |
| 512 | 512x512 | 0.26 | 0.418 | 16.3px | 1.02px | 4.5% | **DESTRUCTIVE** |
| 1500 | 1500x1500 | 2.25 | 1.225 | 47.6px | 3.00px | 0.0% | **UPSCALED** |

Shown to quantify the cost of the reflexive `resize(256, 256)`. At an aspect ratio of 1.50:1 a square resize squashes one axis disproportionately, and the thin dimension of a defect is hit by whichever axis is squashed hardest.

### Decision

**Recommended: long side 1500 (1500x1000, 1.50 MP), verdict SAFE.**

cheapest resolution rated SAFE: p5 thin dimension 3.00px survives, 0.0% of regions lost, 1.50 MP within the 3.00 MP single-pass budget

Selection rule `smallest_safe` (docs/04): the cheapest resolution that is not LOSSY or DESTRUCTIVE. Compute is the scarce resource at 4 GB VRAM, so the rule is 'as small as the defects permit', not 'as large as fits'.

### Intensity shift relative to `train`

| split | mean delta | std ratio | Cohen's d |
|---|---|---|---|
| test | -1.75 | 1.053 | -0.396 |
| validation | -1.06 | 1.022 | -0.255 |

Largest shift: `test` at |d| = 0.396. A large value predicts threshold drift in Phase P7: a threshold fitted on validation lands somewhere else entirely on a shifted split, and the realized false-alarm rate moves even when AUROC does not.


---

Data: MVTec AD / MVTec AD 2 (c) MVTec Software GmbH, CC BY-NC-SA 4.0. Non-commercial use only. See ATTRIBUTION.md.
