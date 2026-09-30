# E08 — Robustness of the frozen recipe to optical and geometric change

Development categories `pcb1`, `macaroni2`, `capsules`; the ADR-11 recipe (PatchCore, 640 px,
fixed 10k bank), seed 0. The model is fitted once on clean training normals and **the operating
threshold is frozen on clean validation normals**; only the test split is corrupted — normal and
defective images alike, each with its own seed (a stable hash of its path) shared across
severities. Metrics come from the shared evaluator. 108 cells (7 corruptions × 5 severities × 3
categories, plus clean), `scripts/robustness_study.py`, raw output
[studies/e08/robustness.csv](studies/e08/robustness.csv).

The clean cells reproduce the E04 seed-0 runs exactly (image AUROC 0.979 / 0.887 / 0.916).

Dataset: VisA © Amazon, CC BY 4.0 — see [ATTRIBUTION.md](../ATTRIBUTION.md).

## Severity, in physical terms

Blur severities scale with each category's median defect diameter (`pcb1` 21.1 px, `macaroni2`
5.7 px, `capsules` 38.9 px, native pixels), so severity 1 is "a blur of a tenth of a typical
defect" everywhere — which is ~2 px on `pcb1`, ~4 px on `capsules`, but only ~0.6 px on
`macaroni2`. Exposure and the other corruptions are absolute.

| corruption | severity 1 | severity 5 |
|---|---|---|
| Gaussian blur | σ = 0.1 × defect diameter | σ = 1.0 × defect diameter |
| defocus blur | disc radius 0.1 × diameter | 0.8 × diameter |
| exposure up / down | ±0.25 stop (+ matching gamma) | ±2 stops |
| spatial light | 5% illumination falloff across the frame | 50% |
| resize round-trip | down to 0.9× and back | down to 0.35× and back |
| sensor noise + JPEG | σ = 1 grey level, quality 95 | σ = 10, quality 40 |

## Results

Mean change in image AUROC and AU-PRO@0.05 against clean (over the three categories), and the
**realized false-alarm rate at the frozen threshold** per category. Clean FPR: 5.0% / 2.0% / 1.7%
(bounds 0.73% / 0.74% / 1.22%).

| corruption | sev | Δ image AUROC | Δ AU-PRO@0.05 | FPR `pcb1` | FPR `macaroni2` | FPR `capsules` |
|---|---|---|---|---|---|---|
| Gaussian blur | 1 | −0.040 | −0.090 | 18% | 2% | **95%** |
| | 2 | −0.210 | −0.360 | 100% | 4% | 100% |
| | 3 | −0.224 | −0.439 | 100% | 28% | 100% |
| | 5 | −0.287 | −0.604 | 100% | 100% | 100% |
| defocus blur | 1 | −0.024 | −0.033 | 5% | 2% | 15% |
| | 2 | −0.075 | −0.174 | **100%** | 2% | 100% |
| | 5 | −0.223 | −0.402 | 100% | 9% | 100% |
| exposure up | 1 | −0.039 | −0.035 | **46%** | 2% | 2% |
| | 2 | −0.134 | −0.412 | 100% | 100% | 100% |
| | 5 | −0.450 | −0.671 | 100% | 100% | 100% |
| exposure down | 1 | +0.007 | −0.020 | 6% | 2% | 2% |
| | 2 | **+0.010** | −0.042 | **48%** | 3% | 2% |
| | 3 | +0.007 | −0.061 | 100% | 7% | 7% |
| | 4 | −0.033 | −0.098 | 100% | 58% | 83% |
| | 5 | −0.161 | −0.205 | 100% | 100% | 100% |
| spatial light | 1–4 | ≥ −0.011 | ≥ −0.017 | 3–5% | 2–4% | 2% |
| | 5 | −0.029 | −0.052 | 9% | 3% | 7% |
| resize round-trip | 1–4 | ≥ −0.013 | ≥ −0.027 | 5% | 2% | 2–3% |
| | 5 | −0.024 | −0.042 | 6% | 3% | 7% |
| noise + JPEG | 1–5 | ≥ −0.006 | ≥ −0.017 | 5% | 2–3% | 2–3% |

Severity at which the operating point **breaks** — the realized FPR exceeds both 3× its clean value
and 10%:

| corruption | `pcb1` | `macaroni2` | `capsules` |
|---|---|---|---|
| Gaussian blur | 1 (σ ≈ 2 px) | 3 (σ ≈ 3 px) | 1 (σ ≈ 4 px) |
| defocus blur | 2 | never | 1 |
| exposure up | 1 (+0.25 stop) | 2 (+0.69 stop) | 2 (+0.69 stop) |
| exposure down | 2 (−0.69 stop) | 4 (−1.56 stop) | 4 (−1.56 stop) |
| spatial light | never | never | never |
| resize round-trip | never | never | never |
| noise + JPEG | never | never | never |

## Reading

**The ranking survives far longer than the operating point.** Exposure down by 0.69 stop *raises*
mean image AUROC by 0.010 while `pcb1`'s false-alarm rate goes from 5% to 48%. Gaussian blur at
severity 1 costs 0.04 AUROC while `capsules` goes from 1.7% to 95% false alarms. A robustness
study that reported only AUROC would call these corruptions harmless; a line running at the frozen
threshold would stop. Recall "rises" under the breaking corruptions only because everything is
flagged.

**Three kinds of change are harmless across the whole tested range:** sensor noise with JPEG
compression down to quality 40, a resize round-trip down to 0.35×, and a smooth illumination
falloff of up to 50% across the frame. Realized FPR stays at its clean level.

**Two kinds break it almost at once:** global exposure — a quarter stop over-exposure breaks
`pcb1`, two-thirds of a stop breaks all three — and blur of about a tenth of a defect diameter,
2–4 native pixels. Both shift every patch descriptor at once, so the whole normal score
distribution moves past a threshold calibrated on its upper tail. `macaroni2` looks more
blur-tolerant only because its severities are physically smaller (its defects are 5.7 px).

**Deployment consequence.** The recipe needs input-side guards that the anomaly score cannot
provide: an exposure check (mean/histogram of the part region against the training range) and a
focus check (e.g. a Laplacian-variance floor), each with its own tolerance, and a recalibration
trigger when either drifts. Photometric normalization of the input against training statistics
is the obvious model-side experiment; it is not run here.

## E08b — input guards

The consequence above, implemented and measured (`src/inspector/api/guards.py`,
`scripts/guard_study.py`, output [studies/e08/guards.csv](studies/e08/guards.csv)). Two image
statistics of the downscaled grey image — **mean level** (exposure) and **Laplacian variance**
(focus) — each accept the [min, max] range seen on the category's training normals. For a clean
image exchangeable with them, falling outside that range has probability ≤ 2/(n+1) per guard
(0.26% for n = 768), the same rank argument as the operating threshold. An image outside the
range gets the decision **refused** instead of normal/anomalous: the threshold was not calibrated
for it. The guards never look at the model's score.

**Cost on clean test images:** 1 of 260 normals refused (0.4%; bound ≈ 0.5% for two guards). 10%
of `pcb1` and 11% of `capsules` *defective* images are refused too — their global brightness or
texture is outside the normal range (the same global shortcut the colour histogram exploits,
case book §6). A refused defect goes to manual review; it is not a miss.

Share of all test images refused, severities 1 → 5 (normals and defects together):

| corruption | breaks the operating point at | `pcb1` | `macaroni2` | `capsules` |
|---|---|---|---|---|
| Gaussian blur | 1 / 3 / 1 | 100% from 1 | 0% · 100% from 2 | 100% from 1 |
| defocus blur | 2 / never / 1 | 100% from 1 | 1% · 46% · 100% from 3 | 100% from 1 |
| exposure up | 1 / 2 / 2 | 100% from 1 | 100% from 1 | 99% · 100% |
| exposure down | 2 / 4 / 4 | 100% from 1 | 100% from 1 | 73% · 100% |
| spatial light | never | 4–6% | 0–1% | 7–9% |
| resize round-trip | never | 7% · 8% · 18% · 100% · 100% | 0% · 2% · 2% · 44% · 100% | 4% · 16% · 54% · 96% · 100% |
| noise + JPEG | never | 5–6% | 0–1% | 8% · 9% · 13% · 22% · 32% |

(Clean refusal: 5.0% / 0.5% / 6.9% of all test images, i.e. the defects noted above.)

- **Every corruption that breaks the operating point is refused at or before the severity at
  which it breaks,** on every category. There is no cell where the realized false-alarm rate
  exploded while the guards let the images through.
- **The guards are conservative:** they also refuse strong resize round-trips (0.5× and below
  lose enough detail to lower the Laplacian variance) and, on `capsules`, strong sensor noise
  (which raises it), although the operating point survives both. Refusing costs a manual review,
  not a wrong decision; tolerances wider than [min, max] would trade that cost against risk and
  need their own calibration.
- **Smooth illumination falloff passes, correctly** — it neither breaks the operating point nor
  moves the global statistics.

The service and the demo apply the guards whenever an artifact carries `guards.json`
(`inspector export` writes it), returning `decision: refused` with the reason.

## Limits

- Synthetic corruptions on three development categories, one seed. VisA has no real
  lighting-shifted split, so whether synthetic exposure predicts real lighting drift remains open
  (MVTec AD 2's `test_private_mixed` is the dataset that can answer it).
- The frozen threshold already exceeds its bound on clean `pcb1` (5% vs 0.73%, see the
  [development report](E00-E04-development-findings.md) §4); the breaking points above are
  relative to that clean level.
