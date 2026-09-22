# P1 — Dataset Audit and Exploratory Analysis

Generated 2026-09-22 04:03 UTC by `inspector eda`. Do not edit by hand.

This report exists to answer one question before any model is fitted:
**what input resolution do these defects permit?** Everything else here is
supporting evidence for that decision (docs/04, task 1.7).

## Summary

| category | native | aspect | defect regions | median area | recommended |
|---|---|---|---|---|---|
| `synth_strip` | 256x64 | 4.00:1 | 9 | 0.183% | 256 (SAFE) |
| `synth_blob` | 128x128 | 1.00:1 | 6 | 1.843% | 128 (SAFE) |
| `synth_grain` | 128x128 | 1.00:1 | 6 | 0.952% | 128 (SAFE) |

## `synth_strip`

### Splits

| split | images | normal | anomalous |
|---|---|---|---|
| test_public | 11 | 5 | 6 |
| train | 12 | 12 | 0 |
| validation | 5 | 5 | 0 |

Defect types: {'pit': 6, 'scratch': 3}

### Defect size distribution

| statistic | p1 | p5 | median | max |
|---|---|---|---|---|
| area (px) | 13 | 16 | 30 | 86 |
| area (% of image) | 0.0776% | 0.0952% | 0.1831% | 0.5249% |
| equivalent diameter (px) | 4.0 | 4.4 | 6.2 | - |
| thin dimension (px) | 4.0 | - | 6.0 | - |

Regions per anomalous image: 1.50. Local contrast against a 4px ring: median 105.2 grey levels, p5 92.3.

### Resolution impact — aspect-preserving

| long side | output | MP | scale | median diam. | p5 thin dim. | <1px | verdict |
|---|---|---|---|---|---|---|---|
| 256 | 256x64 | 0.02 | 1.000 | 6.2px | 4.00px | 0.0% | **SAFE** |
| 320 | 320x80 | 0.03 | 1.250 | 7.7px | 5.00px | 0.0% | **UPSCALED** |
| 448 | 448x112 | 0.05 | 1.750 | 10.8px | 7.00px | 0.0% | **UPSCALED** |
| 512 | 512x128 | 0.07 | 2.000 | 12.4px | 8.00px | 0.0% | **UPSCALED** |
| 1024 | 1024x256 | 0.26 | 4.000 | 24.7px | 16.00px | 0.0% | **UPSCALED** |

`p5 thin dim.` is the 5th-percentile thin dimension of a defect after resizing, and `<1px` the share of regions whose thin dimension drops below one pixel. A region below one pixel cannot be detected by any model, so a resolution that destroys them measures the resize, not the method.

### Resolution impact — blind square resize

| long side | output | MP | scale | median diam. | p5 thin dim. | <1px | verdict |
|---|---|---|---|---|---|---|---|
| 256 | 256x256 | 0.07 | 2.000 | 12.4px | 4.00px | 0.0% | **UPSCALED** |
| 320 | 320x320 | 0.10 | 2.500 | 15.5px | 5.00px | 0.0% | **UPSCALED** |
| 448 | 448x448 | 0.20 | 3.500 | 21.6px | 7.00px | 0.0% | **UPSCALED** |
| 512 | 512x512 | 0.26 | 4.000 | 24.7px | 8.00px | 0.0% | **UPSCALED** |
| 1024 | 1024x1024 | 1.05 | 8.000 | 49.4px | 16.00px | 0.0% | **UPSCALED** |

Shown to quantify the cost of the reflexive `resize(256, 256)`. At an aspect ratio of 4.00:1 a square resize squashes one axis disproportionately, and the thin dimension of a defect is hit by whichever axis is squashed hardest.

### Decision

**Recommended: long side 256 (256x64, 0.02 MP), verdict SAFE.**

cheapest resolution rated SAFE: p5 thin dimension 4.00px survives, 0.0% of regions lost, 0.02 MP within the 1.00 MP single-pass budget

Selection rule `smallest_safe` (docs/04): the cheapest resolution that is not LOSSY or DESTRUCTIVE. Compute is the scarce resource at 4 GB VRAM, so the rule is 'as small as the defects permit', not 'as large as fits'.

### Intensity shift relative to `train`

| split | mean delta | std ratio | Cohen's d |
|---|---|---|---|
| test_public | +0.82 | 1.102 | +0.221 |
| validation | +1.82 | 1.060 | +0.399 |

Largest shift: `validation` at |d| = 0.399. A large value predicts threshold drift in Phase P7: a threshold fitted on validation lands somewhere else entirely on a shifted split, and the realized false-alarm rate moves even when AUROC does not.

## `synth_blob`

### Splits

| split | images | normal | anomalous |
|---|---|---|---|
| test_public | 11 | 5 | 6 |
| train | 12 | 12 | 0 |
| validation | 5 | 5 | 0 |

Defect types: {'bubble': 3, 'contamination': 3}

### Defect size distribution

| statistic | p1 | p5 | median | max |
|---|---|---|---|---|
| area (px) | 149 | 155 | 302 | 413 |
| area (% of image) | 0.9122% | 0.9476% | 1.8433% | 2.5208% |
| equivalent diameter (px) | 13.8 | 14.0 | 19.6 | - |
| thin dimension (px) | 14.1 | - | 19.5 | - |

Regions per anomalous image: 1.00. Local contrast against a 4px ring: median 56.0 grey levels, p5 19.0.

### Resolution impact — aspect-preserving

| long side | output | MP | scale | median diam. | p5 thin dim. | <1px | verdict |
|---|---|---|---|---|---|---|---|
| 128 | 128x128 | 0.02 | 1.000 | 19.6px | 14.25px | 0.0% | **SAFE** |
| 256 | 256x256 | 0.07 | 2.000 | 39.2px | 28.50px | 0.0% | **UPSCALED** |
| 320 | 320x320 | 0.10 | 2.500 | 49.0px | 35.62px | 0.0% | **UPSCALED** |
| 448 | 448x448 | 0.20 | 3.500 | 68.6px | 49.88px | 0.0% | **UPSCALED** |
| 512 | 512x512 | 0.26 | 4.000 | 78.4px | 57.00px | 0.0% | **UPSCALED** |
| 1024 | 1024x1024 | 1.05 | 8.000 | 156.9px | 114.00px | 0.0% | **UPSCALED** |

`p5 thin dim.` is the 5th-percentile thin dimension of a defect after resizing, and `<1px` the share of regions whose thin dimension drops below one pixel. A region below one pixel cannot be detected by any model, so a resolution that destroys them measures the resize, not the method.

### Resolution impact — blind square resize

| long side | output | MP | scale | median diam. | p5 thin dim. | <1px | verdict |
|---|---|---|---|---|---|---|---|
| 128 | 128x128 | 0.02 | 1.000 | 19.6px | 14.25px | 0.0% | **SAFE** |
| 256 | 256x256 | 0.07 | 2.000 | 39.2px | 28.50px | 0.0% | **UPSCALED** |
| 320 | 320x320 | 0.10 | 2.500 | 49.0px | 35.62px | 0.0% | **UPSCALED** |
| 448 | 448x448 | 0.20 | 3.500 | 68.6px | 49.88px | 0.0% | **UPSCALED** |
| 512 | 512x512 | 0.26 | 4.000 | 78.4px | 57.00px | 0.0% | **UPSCALED** |
| 1024 | 1024x1024 | 1.05 | 8.000 | 156.9px | 114.00px | 0.0% | **UPSCALED** |

Shown to quantify the cost of the reflexive `resize(256, 256)`. At an aspect ratio of 1.00:1 a square resize squashes one axis disproportionately, and the thin dimension of a defect is hit by whichever axis is squashed hardest.

### Decision

**Recommended: long side 128 (128x128, 0.02 MP), verdict SAFE.**

cheapest resolution rated SAFE: p5 thin dimension 14.25px survives, 0.0% of regions lost, 0.02 MP within the 1.00 MP single-pass budget

Selection rule `smallest_safe` (docs/04): the cheapest resolution that is not LOSSY or DESTRUCTIVE. Compute is the scarce resource at 4 GB VRAM, so the rule is 'as small as the defects permit', not 'as large as fits'.

### Intensity shift relative to `train`

| split | mean delta | std ratio | Cohen's d |
|---|---|---|---|
| test_public | -1.55 | 1.059 | -0.387 |
| validation | -1.87 | 1.020 | -0.388 |

Largest shift: `validation` at |d| = 0.388. A large value predicts threshold drift in Phase P7: a threshold fitted on validation lands somewhere else entirely on a shifted split, and the realized false-alarm rate moves even when AUROC does not.

## `synth_grain`

### Splits

| split | images | normal | anomalous |
|---|---|---|---|
| test_public | 11 | 5 | 6 |
| train | 12 | 12 | 0 |
| validation | 5 | 5 | 0 |

Defect types: {'chip': 3, 'crack': 3}

### Defect size distribution

| statistic | p1 | p5 | median | max |
|---|---|---|---|---|
| area (px) | 35 | 37 | 156 | 646 |
| area (% of image) | 0.2161% | 0.2258% | 0.9521% | 3.9429% |
| equivalent diameter (px) | 6.7 | 6.9 | 13.1 | - |
| thin dimension (px) | 6.1 | - | 15.0 | - |

Regions per anomalous image: 1.00. Local contrast against a 4px ring: median 101.9 grey levels, p5 94.3.

### Resolution impact — aspect-preserving

| long side | output | MP | scale | median diam. | p5 thin dim. | <1px | verdict |
|---|---|---|---|---|---|---|---|
| 128 | 128x128 | 0.02 | 1.000 | 13.1px | 6.50px | 0.0% | **SAFE** |
| 256 | 256x256 | 0.07 | 2.000 | 26.1px | 13.00px | 0.0% | **UPSCALED** |
| 320 | 320x320 | 0.10 | 2.500 | 32.6px | 16.25px | 0.0% | **UPSCALED** |
| 448 | 448x448 | 0.20 | 3.500 | 45.7px | 22.75px | 0.0% | **UPSCALED** |
| 512 | 512x512 | 0.26 | 4.000 | 52.2px | 26.00px | 0.0% | **UPSCALED** |
| 1024 | 1024x1024 | 1.05 | 8.000 | 104.4px | 52.00px | 0.0% | **UPSCALED** |

`p5 thin dim.` is the 5th-percentile thin dimension of a defect after resizing, and `<1px` the share of regions whose thin dimension drops below one pixel. A region below one pixel cannot be detected by any model, so a resolution that destroys them measures the resize, not the method.

### Resolution impact — blind square resize

| long side | output | MP | scale | median diam. | p5 thin dim. | <1px | verdict |
|---|---|---|---|---|---|---|---|
| 128 | 128x128 | 0.02 | 1.000 | 13.1px | 6.50px | 0.0% | **SAFE** |
| 256 | 256x256 | 0.07 | 2.000 | 26.1px | 13.00px | 0.0% | **UPSCALED** |
| 320 | 320x320 | 0.10 | 2.500 | 32.6px | 16.25px | 0.0% | **UPSCALED** |
| 448 | 448x448 | 0.20 | 3.500 | 45.7px | 22.75px | 0.0% | **UPSCALED** |
| 512 | 512x512 | 0.26 | 4.000 | 52.2px | 26.00px | 0.0% | **UPSCALED** |
| 1024 | 1024x1024 | 1.05 | 8.000 | 104.4px | 52.00px | 0.0% | **UPSCALED** |

Shown to quantify the cost of the reflexive `resize(256, 256)`. At an aspect ratio of 1.00:1 a square resize squashes one axis disproportionately, and the thin dimension of a defect is hit by whichever axis is squashed hardest.

### Decision

**Recommended: long side 128 (128x128, 0.02 MP), verdict SAFE.**

cheapest resolution rated SAFE: p5 thin dimension 6.50px survives, 0.0% of regions lost, 0.02 MP within the 1.00 MP single-pass budget

Selection rule `smallest_safe` (docs/04): the cheapest resolution that is not LOSSY or DESTRUCTIVE. Compute is the scarce resource at 4 GB VRAM, so the rule is 'as small as the defects permit', not 'as large as fits'.

### Intensity shift relative to `train`

| split | mean delta | std ratio | Cohen's d |
|---|---|---|---|
| test_public | -3.14 | 1.003 | -0.453 |
| validation | -2.63 | 1.004 | -0.413 |

Largest shift: `test_public` at |d| = 0.453. A large value predicts threshold drift in Phase P7: a threshold fitted on validation lands somewhere else entirely on a shifted split, and the realized false-alarm rate moves even when AUROC does not.


---

Data: MVTec AD / MVTec AD 2 (c) MVTec Software GmbH, CC BY-NC-SA 4.0. Non-commercial use only. See ATTRIBUTION.md.
