# 14 — Open problems, measured errors and remedies

A register of everything that is still wrong, weak, uncertain or untested after E13, compiled on
2026-10-12 from the reports, from diagnostics on the run registry, and from an audit of the code;
revised on 2026-10-13 after the capture-condition analysis of AD 2 (§0) and E14–E15.
Each entry gives the evidence, what is known about the cause (**established** by a measurement,
or a **hypothesis**), and the technique proposed to handle it. Section 4 orders the work.

## 0. What MVTec AD 2 `test_public` is (found 2026-10-13)

Every object in `test_public` is photographed several times, `<object>_<condition>.png`:
`regular` (the lighting of train and validation, which hold nothing else), `overexposed`,
`underexposed`, and `shift_1`..`shift_k` (k = 1 on `fruit_jelly`, 4 on `vial`, 3 elsewhere).
Against the `regular` capture of the same object, over- and under-exposure are a global gain
(grey-level correlation ≥ 0.992, equal channel gains 0.76–1.18); `shift_k` moves highlights and
shading (correlation 0.82–0.99; `vial`'s shifts are near-identical at 1/8 scale) and tints the light (blue/red down to 0.78). Five sixths of the
test set is therefore changed lighting **by design**, and the 6–7 captures of an object are one
sampling unit, not 6–7. Several entries below change because of it (B1, B4, D2).
`scripts/ad2_condition_study.py`, `reports/studies/ad2-conditions/`.

## 1. Accuracy gaps

| # | Problem | Evidence | Cause | Remedy |
|---|---|---|---|---|
| A1 | **Image score near chance where the maps still localize** | MVTec AD 2, v1: pixel AUROC 0.89 → image AUROC 0.51 (`wallplugs`), 0.92 → 0.58 (`sheet_metal`), 0.82 → 0.52 (`rice`), 0.76 → 0.51 (`can`) ([E13](../reports/E13-confirmation-v2-mvtec-ad2.md)); the same under the training lighting alone (`regular` captures: 0.46–0.57); the same pattern on VisA `macaroni2` (E02) | **Established:** the image score is the map maximum, and the maximum over a normal image is set by structured normal variation (edges, texture, glare) as high as by a defect | **Measured, none general** ([E14–E15](../reports/E14-E15-ad2-development.md) §3): top-k aggregation +0.01–0.02 macro; object region negative; R2 mixed (needs registered parts). Open: R1 tiling, R3 selection on synthetic anomalies, R8 features |
| A2 | **Small or thin defects lost to resizing** | `sheet_metal` (4224×1056 px, scratches of a few pixels): v1 at 640 px 0.625, 960 px 0.781 image AUROC; the apparatus finding that at 512 px 79% of its defect regions fall below one pixel ([README](../README.md)); native resolution never evaluated | **Established:** the long side, not the defect, sets the scale; on a 4:1 image a 640 px long side leaves 160 px of height | **Measured (E19):** whole-image 1280 px lifts `sheet_metal` +0.23 image AUROC over 3 seeds (object bootstrap) and AU-PRO 0.18 → 0.35 (0.50 at 2816 px); `fabric` AU-PRO 0.15 → 0.52. As a global recipe (v3, 1280 px on all eight) it fails ADR-14's gate on `vial` (−0.14 under changed lighting), although macro AU-PRO rises 0.35 → 0.47. Next: a per-part resolution rule fixed from image geometry, pre-registered for a new held-out set; tiles beyond the host-memory limit |
| A3 | **Logical and assembly anomalies** | AD 2 `wallplugs` (missing / mismatched parts) image AUROC 0.48, pixel PCA 0.62; `rice` 0.57; MPDD `connector` and `bracket_brown` are *parts mismatch* defects, where the global floors keep up with the recipe (H1) | **Hypothesis:** a patch-local nearest-neighbour score cannot see a wrong count or arrangement made of normal-looking patches. R4 measured: a pooled-feature Gaussian lifts `rice` under the training lighting (0.53 → 0.76) but leaves `wallplugs` at chance and moves with the light | R4 with a lighting-robust descriptor; a count/arrangement model (part detection + layout statistics) for `wallplugs`-type parts; MVTec LOCO as the dataset that tests it |
| A4 | **Low recall at the operating point** | VisA development: `macaroni2` 0.39, `capsules` 0.54; AD 2: 0.08–0.57 | **Established:** a 1% false-alarm request with the threshold at the highest validation score; normal-variation peaks set it | Follows from A1/B1; for deployment, choose the operating point on the cost of an escape, and state it |
| A5 | **The best representation is dataset-dependent** | `layer2` alone: +0.07 on VisA `macaroni2`, but v2 (`layer2`, 960 px) −0.111 on AD 2 `vial`; timm `racm_in1k` weights change sign between 320 and 640 px (E8); C2 weights ±0.06 | **Established** that the ranking of representations does not transfer; why is open. R8 measured (E18): DINOv2 ViT-S/B lift AD 2 `rice` +0.27–0.33 but lose `vial` −0.17 and are more lighting-sensitive (`shift` false alarms 41 → 95) | Layer, backbone and weights chosen per category on synthetic anomalies (R3), never on test |
| A6 | **Localization weak on AD 2** | AU-PRO@0.05 0.36 (v1) against 0.77 on held-out VisA; server 37.6 / 35.1 (S1) | Mix of A1, A2 and B1 (shifted normals light up whole maps; `can` SegF1 ≈ 0) | **Addressed in development** ([E21](../reports/E21-ad2-localization.md)): tiled DINOv3-L fused with WRN50 at 1280 px, AU-PRO 0.351 → 0.564 and SegF1 0.205 → 0.571 on `test_public`; candidate v4 (ADR-15) awaits the server |

## 2. Operating point and calibration

| # | Problem | Evidence | Cause | Remedy |
|---|---|---|---|---|
| B1 | **Validation normals do not represent test normals** | VisA `pcb1` 4.3% realized vs 0.73% bound, `pipe_fryum` (E12); AD 2 pooled false alarms 160 (v1) / 195 (v2) against 29.2 expected, `can` 51% of normals flagged | **Established:** session debris (VisA). On AD 2 the excess is **changed light direction and colour** (§0): false alarms among `regular` / `overexposed` / `underexposed` / `shift` normals are 5.7 / 5.7 / 5.2 / **22.8%** for v1 and 2.6 / 3.6 / 4.2 / **31.2%** for v2 (`can` `shift`: 96%). The operating point holds under the training lighting and under a global exposure change | **Measured:** five photometric remedies (E14b–d, E15a–b) leave the `shift` false alarms where they were — `can`'s moved specular highlights are neither a gain nor a smooth field. **R11 is a trade, not a fix:** 2–6 good parts in the bank halve the `shift` false alarms at the old threshold (E16, 41 → 19 of 187, mostly `can`) at a recall cost; at a threshold recalibrated on new-lighting normals the adapted bank buys no recall (E17, −0.014 [−0.034, +0.003]), and 1–3 calibration objects leave false alarms above the bound (17 vs 7.6). On `can` it is a separability problem. Left: R8 features (E18); many more normals per lighting; the held-out test is AD 2 `test_private_mixed` (server) |
| B2 | **Calibration sets too small for 1%** | AD 2 19–48 and MPDD 8–43 validation normals: bounds relaxed to 2.0–11.1%; MPDD realized 0–18% per category | **Established:** a rank rule needs n ≥ 99 for 1% | More calibration normals when collecting; meanwhile report the relaxed bound (done) and pool across seeds; R5(b) can use the unlabeled stream |
| B3 | **Pixel threshold (3σ) breaks on skewed maps — a defect in the metric (fixed); SegF1 and ClassF1 coupled** | AD 2 `fruit_jelly` v2: validation pixel mean 0.99, SD 0.83, so the 3σ threshold 3.41 lies above every test map (max 3.27) and SegF1 = 0.000 on all seeds; `vial` v2 SegF1 0.05 | **Established:** mean + 3σ assumes near-Gaussian pixel scores; `layer2` maps are bimodal (object vs background) | **Fixed** (`d95cf0f`): SegF1 also at the validation pixel quantile (0.1%). E21: validation P95 × gain, morphological masks (`MaskRule`), and an image mark that sets the image decision apart from the pixel threshold, ClassF1 0.749 → 0.806 at no SegF1 cost ([E21](../reports/E21-ad2-localization.md)) |
| B4 | **Input guards watch the wrong variable** | Blind to a frame shift and to new content at the frame edge (G4, E08). On AD 2 they refuse the captures the model handles — over/under-exposed normals refused 100% on `rice`, 73–82% on `fabric`, 60–100% on `vial` — and accept most of those it does not: 61% of `can`'s `shift` normals pass, where 96% raise an alarm. They also refuse 20–25% of `regular` normals on `sheet_metal`, `vial`, `wallplugs` | **Established** by E08b, G4, E13 and the condition analysis: exposure is the condition the model tolerates, lighting structure the one it does not | R10: a lighting-structure guard (gain-normalized low-frequency field against the training fields); guards on feature-space statistics; a drift monitor on the score distribution of incoming images |

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
| D2 | **Bootstrap intervals ignore image clustering — established on AD 2** | AD 2 shows each object 6–7 times (§0). Resampling objects instead of images widens E13's intervals 1.2–2.4× on seven categories (`vial` 0.8×): `sheet_metal` v2 − v1 +0.156 goes from [+0.057, +0.257], Holm p 0.022, to [−0.033, +0.345], Holm p 0.74; `vial` −0.111 stays significant. E13's verdict (not confirmed) is unchanged; its count of significant positives drops from 1 to 0. VisA has no session ids, though debris shows sessions exist | The object bootstrap on AD 2 (from E15, pre-registered); R9 two-level bootstrap (seeds × objects); cluster by session where it can be inferred (thumbnail clustering, the A4 tooling) |
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
| M6 | Near-duplicate audit (A4) done for VisA only | `scripts/near_duplicate_audit.py --layout` now covers MPDD and AD 2 (results in the E14 report) |
| M7 | **A late setting broke a frozen analysis** — `PatchCore.hparams()` emitted `bank_exposures` unconditionally from `fc45fc3`, so `scripts/ad2_confirmation_analysis.py` matched none of the 48 E13 runs to v1 or v2 | **Fixed** (`d95cf0f`): late settings are omitted at their default; a test pins the default key set; the frozen analysis reproduces `confirmation.json` exactly again |
| M8 | **A queued run failed hours after it was planned** — `run_e14.sh` passed `--set model.bank_exposures=…`, a key no config has, so E14d failed at its turn | **Fixed:** rerun with a config overlay; `inspector run --dry-run` resolves every planned run (configs, overrides, models, run ids) without fitting, so a bad setting fails when it is queued |
| M9 | Maps are stored in float16, so a re-evaluation from the registry differs from the original by ~1e-5 in pixel metrics | Documented; re-evaluation scripts check agreement to 1e-3 |

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
- **R6 — Photometric invariance.** Per-image gain and standardization, an exposure-augmented
  bank (E14b–d), flat-field normalization and a relit bank (E15): none carried; see negative
  results 18–21.
- **R11 — Few-shot adaptation to a new lighting.** Add a handful of normal images captured
  under the new condition to the memory bank and to the calibration set — what a line does when
  its lighting is changed. E16 (disjoint objects for adaptation and evaluation): one part is not
  enough, 2–6 parts halve the changed-lighting false alarms at the old threshold. E17: at a
  recalibrated threshold the adapted bank buys no recall, and a few calibration objects do not
  restore the bound. A line needs many normal parts per lighting, or a fixed lighting.
- **R7 — Robust pixel threshold.** The validation pixel score at a 0.1% pixel false-positive rate
  (a pixel-level analogue of the image rank rule), reported beside 3σ.
- **R8 — Stronger features.** DINOv2-class ViT features in the same PatchCore pipeline, under
  the same protocol and costs.
- **R9 — Hierarchical bootstrap.** Resample seeds, then images (and sessions when known), so
  intervals carry fitting and sampling variance together. On AD 2 the object is the image-level
  unit (D2).
- **R10 — Lighting-structure guard.** Describe each image by its low-frequency lighting field,
  each channel divided by its mean so a global exposure change cancels, and refuse an image whose
  field is farther from every training field than the training fields are from each other.
  Exposure, which the model tolerates, stops causing refusals; a changed light direction, which
  it does not tolerate, starts to.

## 7. Order of work

Done on 2026-10-13: E14 and E15 (no variant carried), E14a, R2, R4 and R10 (exploratory), B3
fixed, the capture-condition and object-bootstrap analysis, M6–M9.

1. **Done:** E16–E17, few-shot adaptation and recalibration (R11): a trade along the same
   curve, not a gain. **Next:** E18, DINOv2 features (R8), the remaining lever for both A1 and B1.
2. **Done:** E18, DINOv2 features (R8): not carried. E19, resolution: `sheet_metal` +0.23
   (3 seeds), `fabric` AU-PRO ×3.6. E20: v3 (1280 px everywhere) fails ADR-14's gate on `vial`.
   **Owner:** upload S1 (v1). **Next candidate:** a per-part resolution rule from image geometry,
   pre-registered for a new held-out set.
3. **Methodology:** R3 for every later selection; the object bootstrap on AD 2 and R9 elsewhere.
4. **Held-out:** server submission of v1 (owner); a new, untouched dataset for any recipe change,
   pre-registered.
