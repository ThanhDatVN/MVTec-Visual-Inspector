# 14 — Open problems, measured errors and remedies

A register of everything that is still wrong, weak, uncertain or untested after E13, compiled on
2026-10-12 from the reports, from diagnostics on the run registry, and from an audit of the code.
Each entry gives the evidence, what is known about the cause (**established** by a measurement,
or a **hypothesis**), and the technique proposed to handle it. Section 4 orders the work.

## 1. Accuracy gaps

| # | Problem | Evidence | Cause | Remedy |
|---|---|---|---|---|
| A1 | **Image score near chance where the maps still localize** | MVTec AD 2, v1: pixel AUROC 0.89 → image AUROC 0.51 (`wallplugs`), 0.92 → 0.58 (`sheet_metal`), 0.82 → 0.52 (`rice`), 0.76 → 0.51 (`can`) ([E13](../reports/E13-confirmation-v2-mvtec-ad2.md)); the same pattern on VisA `macaroni2` (E02) | **Established:** the image score is the map maximum, and the maximum over a normal image is set by structured normal variation (edges, texture, glare) as high as by a defect | R2 location-conditional normalization of the map; R3 aggregation chosen on synthetic anomalies; object-region scoring (E02b) — E14a measures the existing rules on the stored maps |
| A2 | **Small or thin defects lost to resizing** | `sheet_metal` (4224×1056 px, scratches of a few pixels): v1 at 640 px 0.625, 960 px 0.781 image AUROC; the apparatus finding that at 512 px 79% of its defect regions fall below one pixel ([README](../README.md)); native resolution never evaluated | **Established:** the long side, not the defect, sets the scale; on a 4:1 image a 640 px long side leaves 160 px of height | R1 overlapping tiles at (near-)native resolution, with candidate sampling bounding host memory |
| A3 | **Logical and assembly anomalies** | AD 2 `wallplugs` (missing / mismatched parts) image AUROC 0.48, pixel PCA 0.62; `rice` 0.57; MPDD `connector` and `bracket_brown` are *parts mismatch* defects, where the global floors keep up with the recipe (H1) | **Hypothesis:** a patch-local nearest-neighbour score cannot see a wrong count or arrangement made of normal-looking patches | R4 a global branch (image-level descriptor k-NN) fused with the patch score on validation statistics |
| A4 | **Low recall at the operating point** | VisA development: `macaroni2` 0.39, `capsules` 0.54; AD 2: 0.08–0.57 | **Established:** a 1% false-alarm request with the threshold at the highest validation score; normal-variation peaks set it | Follows from A1/B1; for deployment, choose the operating point on the cost of an escape, and state it |
| A5 | **The best representation is dataset-dependent** | `layer2` alone: +0.07 on VisA `macaroni2`, but v2 (`layer2`, 960 px) −0.111 on AD 2 `vial`; timm `racm_in1k` weights change sign between 320 and 640 px (E8); C2 weights ±0.06 | **Established** that the ranking of representations does not transfer; why is open | R8 stronger, more general features (DINOv2-class ViTs) compared under one protocol; layer and weights chosen per category on synthetic anomalies (R3), never on test |
| A6 | **Localization weak on AD 2** | AU-PRO@0.05 0.36 (v1) against 0.77 on held-out VisA | Mix of A1, A2 and B1 (shifted normals light up whole maps; `can` SegF1 ≈ 0) | R1, R2, R6 |

## 2. Operating point and calibration

| # | Problem | Evidence | Cause | Remedy |
|---|---|---|---|---|
| B1 | **Validation normals do not represent test normals** | VisA `pcb1` 4.3% realized vs 0.73% bound, `pipe_fryum` (E12); AD 2 pooled false alarms 160 (v1) / 195 (v2) against 29.2 expected, `can` 51% of normals flagged; normal-score shift AUROC 0.79–0.83 on `can` | **Established:** session debris (VisA), and capture shifts in AD 2 `test_public` — the guards refuse 20–83% of its normals in six categories against 0–8% of validation normals | R5 shift-robust calibration; R6 photometric invariance (E14b–d, running); keep reporting the exact tail test so the failure is visible |
| B2 | **Calibration sets too small for 1%** | AD 2 19–48 and MPDD 8–43 validation normals: bounds relaxed to 2.0–11.1%; MPDD realized 0–18% per category | **Established:** a rank rule needs n ≥ 99 for 1% | More calibration normals when collecting; meanwhile report the relaxed bound (done) and pool across seeds; R5(b) can use the unlabeled stream |
| B3 | **Pixel threshold (3σ) breaks on skewed maps — a defect in the metric** | AD 2 `fruit_jelly` v2: validation pixel mean 0.99, SD 0.83, so the 3σ threshold 3.41 lies above every test map (max 3.27) and SegF1 = 0.000 on all seeds; `vial` v2 SegF1 0.05 | **Established:** mean + 3σ assumes near-Gaussian pixel scores; `layer2` maps are bimodal (object vs background) | R7: a validation pixel *quantile* (0.1% pixel false-positive rate) as the segmentation threshold, reported beside the 3σ one |
| B4 | **Input guards see only global exposure and focus** | Blind to a frame shift and to new content at the frame edge (G4, E08); on AD 2 `can`, 49% false alarms remain among accepted images; conservative refusals of strong resize / noise | **Established** by E08b, G4, E13 | Guards on feature-space statistics (distance of the image's mean embedding to the training distribution) and on spatial layout; a drift monitor on the score distribution of incoming images |

## 3. Robustness and real-world validity

| # | Problem | Evidence | Remedy |
|---|---|---|---|
| C1 | The operating point breaks at small optical changes | +¼–⅔ stop over-exposure or a 2–4 px blur → 45–100% false alarms while image AUROC drops 0.04 (E08) | R5, R6; guards stay as the safety net |
| C2 | Real lighting shift not measured on labelled data | AD 2 `test_private_mixed` is scored only by MVTec's server; no submission made | Submit v1 (and the best E14 variant) with `scripts/export_ad2_submission.py`; the owner's account |
| C3 | Synthetic robustness has not been validated against real shift (G5) | The E08 suite predicts *something*; whether it predicts real lighting is open | The C2 submission, private vs private-mixed, compared with E08's exposure curves |

## 4. Methodology and statistics

| # | Problem | Evidence | Remedy |
|---|---|---|---|
| D1 | **Development effects were overestimated twice** | Resolution: +0.19/+0.25 in development, +0.034 on held-out VisA (H1 not confirmed); v2: +0.05 to +0.09 in E8 (`capsules`, `macaroni2`), +0.004 macro on AD 2 (not confirmed) | **Established selection bias:** three development categories, chosen as hard, and winners picked on their test sets. Remedy R3: select on *synthetic* anomalies over validation normals, so the test set never takes part in selection; use more and more varied development categories |
| D2 | Bootstrap intervals ignore seed variance and image clustering | Image-level bootstrap per seed average; VisA has no session ids, though debris shows sessions exist | R9: a two-level bootstrap (seeds × images); cluster by session where it can be inferred (thumbnail clustering, the A4 tooling) |
| D3 | **`inspector study` defaulted to 1,000 bootstrap resamples — a defect, fixed** | The committed `reports/studies/studies.json` has min Holm p = 0.036, exactly the floor 18 × 2/1001; a family over ~25 could never reach significance. The E01 report itself used 10,000 | Default raised to 10,000; every comparison now records its Holm floor (`p_holm_floor`) |
| D4 | Test sets too small for precise false-alarm rates | AD 2 test normals 20–72 per category (one image = 1.4–5 points of FPR); VisA confirmation 9 × ~100 | Pool across seeds and categories (done); report counts, not only rates |
| D5 | Held-out data are spent | VisA confirmation (E12), MPDD (H1), AD 2 `test_public` (E13) | Remaining: AD 2 private splits (server, limited budget). Next confirmation set: a dataset never loaded here (e.g. MVTec AD classic, MVTec LOCO for A3), pre-registered as before |

## 5. Measurement and engineering

| # | Problem | Status |
|---|---|---|
| M1 | Per-run VRAM recorded before commit `93a82dd` includes earlier runs' backbones (reference cycle in `FeatureExtractor`) | **Fixed**; older VRAM columns are upper bounds; clean numbers from `scripts/bench_inference.py` |
| M2 | Kaggle T4 latency not measured with the O8 protocol | Open; batch-mode time per image recorded (90–102 ms at 640 px) |
| M3 | CI has no Docker smoke test, no golden-file test, and does not execute the notebooks | Open; notebooks are smoke-run by hand (`MVI_SMOKE=1`) |
| M4 | Nothing checks that a report's numbers still match the registry | Open; reports are generated from committed scripts, but by hand. A checker that re-runs each `scripts/*_analysis.py` and diffs the outputs would close it |
| M5 | Data acquisition is fragile (mydrive.ch drops long streams; SharePoint for MPDD) | **Fixed** with resumable downloads and SHA-256 manifests |
| M6 | Near-duplicate audit (A4) done for VisA only | Open for MPDD and AD 2; the A4 tooling applies unchanged |

## 6. Remedies in detail

- **R1 — Tiling.** Split each image into overlapping tiles at (near-)native resolution, embed each
  tile with the same backbone, pool all tiles' candidates into one bank, and stitch the per-tile
  maps back (max over overlaps). Candidate sampling keeps host memory bounded; the 4 GB card sees
  one tile at a time. First target: `sheet_metal`, `fabric`, `rice` (AD 2), `pcb1` (VisA).
- **R2 — Location-conditional normalization.** For categories whose images are registered (one
  object in a fixed pose), compute per-location mean and standard deviation of the anomaly map
  over normal images (training maps by leave-one-out, or validation maps) and score the
  standardized map. Structured normal hot spots (edges, glare) stop setting the maximum. It can be
  evaluated on stored maps without refitting.
- **R3 — Selection on synthetic anomalies.** Paste or synthesize defects (CutPaste, Perlin-noise
  masks, DRAEM-style blending) onto validation normals and choose the aggregation rule, the
  layers, the resolution and the pixel threshold on them. Test sets then never participate in a
  choice — the direct answer to D1.
- **R4 — Global branch.** An image-level descriptor (pooled deep features) scored by k-NN against
  the training normals, z-scored on validation normals and fused with the patch score by maximum
  or sum. Targets A3 (counts, missing parts, arrangement).
- **R5 — Shift-robust calibration.** (a) Calibrate on validation normals *plus* photometric
  variants of them, a conservative threshold under expected lighting changes; (b) adaptive
  conformal thresholds updated on the unlabeled stream of incoming images (valid under drift for
  a long-run error rate, assuming a low defect rate); (c) per-session calibration where sessions
  are identifiable.
- **R6 — Photometric invariance.** Per-image exposure normalization and an exposure-augmented
  memory bank: E14b–d, running now.
- **R7 — Robust pixel threshold.** The validation pixel score at a 0.1% pixel false-positive rate
  (a pixel-level analogue of the image rank rule), reported beside 3σ.
- **R8 — Stronger features.** DINOv2-class ViT features in the same PatchCore pipeline, under
  the same protocol and costs.
- **R9 — Hierarchical bootstrap.** Resample seeds, then images (and sessions when known), so
  intervals carry fitting and sampling variance together.

## 7. Order of work

1. **Now:** E14 screen (photometric, running) and E14a (aggregation and object region on stored
   maps, running); fix B3 (R7) once E14 is complete, so the two code versions do not mix.
2. **Next, cheap:** R2 on the stored AD 2 maps (no refit).
3. **Next, refit:** R1 tiling on the small-defect categories; R4 global branch on the logical
   categories.
4. **Methodology:** R3 for every later selection; R9 for every later interval.
5. **Held-out:** server submission of v1 and the best variant (owner); then a new, untouched
   dataset for any recipe change, pre-registered.
