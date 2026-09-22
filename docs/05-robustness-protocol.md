# 05 — Robustness Protocol

**Applied in Phase P7.** Robustness here means: the camera, the lamp, or the lens changed slightly,
and nobody re-trained the model. This is the normal condition of a deployed inspection system, not
an edge case — bulbs age, a technician nudges a lens, a maintenance swap changes a sensor.

---

## 1. Design principles

1. **Corrupt only the test images.** The model is fitted once on clean training data and never
   retrained or re-thresholded. Re-thresholding under corruption would answer a different (and
   much easier) question.
2. **Severity must be physically interpretable.** "Severity 3" is meaningless unless it maps to a
   documented physical quantity: a blur radius in pixels *relative to the median defect size*, an
   exposure change in stops, a downscale factor. Calibrate against the data, not against
   ImageNet-C's constants — our images are 20× larger than ImageNet's, so ImageNet-C kernel sizes
   are not transferable.
3. **Keep the ground truth aligned.** Geometric corruptions must transform the mask identically.
   Photometric corruptions leave the mask untouched. This is tested, not assumed.
4. **Always report the native control.** MVTec AD 2's lighting-shifted split is real distribution
   shift. Synthetic corruption is a model of it. Reporting both lets us check whether the model
   is any good.

---

## 2. Corruption suite

Six families × 5 severities. Implemented in `src/inspector/robustness/corruptions.py`, each as a
pure function `(image, mask, severity) -> (image', mask')`, each with a unit test.

| # | Family | Parameter swept | Severity 1 → 5 | Simulates |
|---|--------|-----------------|----------------|-----------|
| C1 | **Gaussian blur** | sigma, as a fraction of median defect diameter | 0.1 → 1.0 × d_med | Defocus, vibration, dirty lens |
| C2 | **Defocus blur** | disc-kernel radius | calibrated to d_med | True optical defocus (disc, not Gaussian — the difference matters for small-defect visibility) |
| C3 | **Illumination: global** | exposure in stops + gamma | ±0.25 → ±2.0 stops | Lamp ageing, exposure drift. **Run both directions and report separately** — over- and under-exposure fail differently. |
| C4 | **Illumination: spatial** | multiplicative gradient / simulated off-axis spotlight | 5% → 50% across-frame falloff | Lamp misalignment, added light source. Closest synthetic analogue to AD 2's real shift. |
| C5 | **Resize round-trip** | downscale then upscale back to native | 0.9× → 0.35× | Bandwidth-limited capture, wrong camera mode, thumbnail pipelines. **The most operationally common corruption and the one most likely to erase small defects.** |
| C6 | **Sensor noise + JPEG** | Gaussian + Poisson sigma; JPEG quality | sigma 1→10 / Q 95→40 | Higher gain at low light, lossy transport |

**Optional (report separately, not in the mean):** small rotation and translation (±1°, ±1%). On a
fixed industrial rig, large geometric shifts are out of distribution by design; reporting them in a
robustness mean would misrepresent the deployment condition. Report them as a separate
"registration sensitivity" line, because PaDiM in particular is position-dependent and this
predicts how much a fixture change would cost.

### Severity calibration procedure (task 7.2)

1. From P1's EDA, take `d_med` = median defect equivalent diameter in pixels, per category.
2. Set each family's severity anchors relative to `d_med`, per category.
3. Render one example of every (family, severity) for every category into a contact sheet and
   **look at it**. If severity 5 destroys the image so thoroughly that a human inspector would
   also fail, the scale is wrong — cap it there and say so.
4. Record the resolved parameters in `reports/robustness/calibration.md`. Severities are then
   frozen.

This step is what makes the robustness section evidence rather than decoration.

---

## 3. Evaluation grid

- **Models:** the top 3 by AU-PRO@0.05 from P5/P8, always including PatchCore and one Tier 1
  baseline (a robustness comparison between two strong models is less informative than one that
  spans the ladder — a reconstruction model and a memory model can fail in opposite directions).
- **Categories:** all three.
- **Grid:** 6 families × 5 severities × 3 categories × 3 models = 270 evaluations, plus 9 clean
  baselines. Inference only, on cached extractions where possible.

## 4. Reported quantities

For each cell:

| Quantity | Why |
|----------|-----|
| Image AUROC, AU-PRO@0.05, AUPIMO median | Absolute performance under corruption |
| **Relative degradation** `(clean − corrupt) / clean` | Comparable across models with different clean scores |
| **Realized FPR at the frozen `OP-FPR1` threshold** | **The deployment-critical number.** A model can hold its AUROC while its false-alarm rate goes from 1% to 30%, because the whole score distribution shifted. Ranking survives; the product does not. |
| **Recall at the frozen threshold** | The other half of the same story |
| SegF1 at frozen `tau_pix` | Whether localization is still usable |

### Summary statistics

- **mCE-style summary:** mean relative degradation over families and severities, normalized by a
  reference model's degradation, so a single robustness number per model exists — with the caveat
  that it hides the per-family structure, so it never appears without the per-family table.
- **Severity-at-which-it-breaks:** lowest severity at which realized FPR exceeds 3× its clean
  value. More actionable than an area-under-degradation number, because it translates to a
  tolerance a maintenance team can be given.

---

## 5. The native-lighting control (task 7.5) — the important part

MVTec AD 2 provides `test_private_mixed`, captured under a mixture of seen and unseen lighting.
Ground truth is withheld, so this requires either a leaderboard submission or the use of
`test_public` if it contains lighting variation (**verify which at P1** and record the answer).

Whichever is available, produce:

1. **Clean vs. lighting-shifted performance** for each model.
2. **Rank correlation (Spearman)** between model ranking under *synthetic* illumination
   corruptions (C3, C4) and under *real* lighting shift.

Then answer, in writing:

> Does synthetic illumination corruption predict real-world lighting robustness?

- **High correlation** → synthetic robustness testing is a validated, cheap proxy. That
  justifies its use for every other category and project.
- **Low correlation** → a finding of real interest to the field, since synthetic corruption
  benchmarks are widely used as evidence of deployment readiness on exactly this assumption.

Either result is worth writing up. This is the highest-value hour in Phase P7.

---

## 6. Tests for the robustness code itself

Corruption code that is subtly wrong invalidates the whole section, and its bugs are invisible —
a corrupted image still looks like a corrupted image.

| Test | Assertion |
|------|-----------|
| `test_severity_monotonic` | A scalar image-distance metric (e.g. LPIPS or RMSE vs clean) increases monotonically with severity, for every family and category |
| `test_severity_zero_is_identity` | Severity 0 returns bit-identical pixels |
| `test_mask_alignment_geometric` | For geometric corruptions, the transformed mask's centroid matches the transformed centroid of the original within 1 pixel |
| `test_mask_untouched_photometric` | Photometric corruptions return the mask unchanged (identity check) |
| `test_deterministic` | Same seed, same severity → bit-identical output |
| `test_dtype_and_range` | Output stays `uint8` in [0, 255]; no silent clipping that would itself act as a corruption |
| `test_no_defect_erasure_at_low_severity` | At severity 1, the defect region's mean contrast against its surroundings drops by less than 50%. Guards against a calibration error that quietly deletes the thing being detected. |
