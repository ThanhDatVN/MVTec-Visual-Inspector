# Negative and null results

Every line of work that was tried and did not pay off, with what was run and why it stopped
(decision rule DR-3, docs/11 §5). A results table that shows only what worked hides most of what
the project learned. All entries are development evidence unless marked otherwise.

| # | What was tried | Evidence | Result | Decision |
|---|---|---|---|---|
| 1 | **Object-region image score as a default** (E02b): score only map cells inside a region fitted on training normals | 3 categories × 3 seeds, margins 1/3/6% declared in advance; at 320 and 640 px | +0.023 / +0.062 on `pcb1` / `macaroni2` but −0.021 on `capsules` at 320 px; at 640 px +0.006 / +0.035 / −0.012 | Fails DR-1's regression clause at both resolutions. Not adopted; kept as an option ([report](E00-E04-development-findings.md) §5, §7) |
| 2 | **A larger calibration pool for the `pcb1` threshold** (E03b): 5-fold cross-fitted out-of-fold training scores, 7× more calibration normals | 640 px recipe, seed 0, k = 9 at a 1% request | 5/100 false alarms against a 0.99% bound, tail p = 0.006 — unchanged | The failure is session-level non-exchangeability, not sample size. Line closed ([report](E00-E04-development-findings.md) §8) |
| 3 | **Higher resolution as a calibration fix** (E04) | 320 / 448 / 640 px | `pcb1` realized FPR 4% / 9% / 5% against 0.73% | Resolution sharpens debris as much as defects. Not a fix |
| 4 | **H1 — 640 px beats 320 px on held-out categories** (confirmation v1, pre-registered) | 9 VisA categories, paired bootstrap, Holm over 18 tests | 4/9 significant, rule required 6/9; macro +0.034 [+0.025, +0.043], none negative | **Not confirmed.** Development overstated the effect (+0.19/+0.25) by selecting hard-at-320 categories ([report](E12-confirmation-v1.md)) |
| 5 | **Validation normals as near-copies of training frames** (hypothesis for the `pcb1` failure) | model-free thumbnail proximity, 3 categories | AUROC 0.52 (p = 0.53) on `pcb1` | Rejected; the failure lies in the test split's upper tail |
| 6 | **64-bit pHash near-duplicate check** (rule L6 as first written) | all 12 VisA categories | distinct captures collide at Hamming 0 (`pcb4`: median within-train distance 0) | Unusable on fixed rigs; replaced by a calibrated thumbnail-RMSE criterion ([report](A4-near-duplicates.md)) |
| 7 | **From-scratch convolutional autoencoder** (Tier 1) at 60 epochs, map-maximum score | 320 px, L2 and SSIM losses, 3 seeds | image AUROC 0.41–0.69; no run triggered early stopping; SSIM's maximum lands on the `pcb1` transducer mesh in every image | Budget-limited, not measured at its best. A 150-epoch schedule at 640 px is queued on Kaggle ([case book](case-book/README.md) §3) |
| 8 | **Map smoothing as a localization lever** (E05) | stored 640 px maps, σ ∈ {0, 2, 4, 8} input px, 3 categories × 3 seeds | AU-PRO@0.05 moves by ≤ 0.01 across σ; SegF1 falls as σ grows | Null for AU-PRO; the default σ = 4 stays. Not a tuning axis |
| 9 | **anomalib's image-score re-weighting** (E02) | stored maps, 3 categories × 3 seeds | stored (re-weighted) vs raw maximum: ≤ 0.006 image AUROC | Nearly inert on these categories; kept for reference fidelity |
| 10 | **Mean of the top 5% of the map as the image score** (E02) | stored maps | +0.026 on `pcb1`, −0.067 on `macaroni2` | Trades small-defect sensitivity for debris robustness; not a default |
| 11 | **Input guards as a free lunch** (E08b) | 7 corruptions × 5 severities × 3 categories | every breaking corruption refused, but also strong resize round-trips and `capsules` sensor noise that do not break the operating point | Adopted with the cost stated: conservative refusals, 1/260 clean normals refused ([report](E08-robustness.md)) |
