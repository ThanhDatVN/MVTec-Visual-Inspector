# E21 — Localization on MVTec AD 2, rebuilt metric by metric (candidate v4, ADR-15)

S1 put recipe v1 on MVTec's server at AU-PRO@0.05 37.64 / 35.11, SegF1 26.42 / 23.01 and ClassF1
73.91 / 76.21 (`test_private` / `test_private_mixed`): above every baseline of the AD 2 paper, about
half of the challenge systems. This report takes each server metric apart, surveys what the
published systems do, rebuilds the pipeline where the evidence says the loss is, and freezes the
result as candidate v4 for the project's second and last submission
([ADR-15](../docs/07-risks-and-decisions.md)).

Everything below is development evidence on `test_public`, seed 0, scored as the server scores
(all pixels of a category pooled, normal images included; ClassF1 from mask emptiness), with
`scripts/mask_rule_study.py` and `scripts/fusion_study.py`. Dataset: MVTec AD 2 © MVTec Software
GmbH, CC BY-NC-SA 4.0. No AD 2 image or derived figure is published.

## 1. Why v1 scored what it did, metric by metric

| metric | v1 on the server | what limits it (evidence) |
|---|---|---|
| **AU-PRO@0.05** (continuous maps) | 37.6 / 35.1 | **Map resolution and features.** One WRN50 feature cell covers ~28 native px of a 2448 px image at a 640 px long side; thin scratches and small grains fall inside one cell (E19: 1280 px raised `sheet_metal` 0.18 → 0.35, `fabric` 0.15 → 0.52). |
| **SegF1** (binary masks) | 26.4 / 23.0 | **Map quality first, threshold second.** On v1's maps the best of 58 threshold/mask rules reaches 0.232 against 0.205 for 3σ: thresholding cannot separate what the map does not. Where maps are bimodal, 3σ fails outright (`fruit_jelly` at 1280 px: SegF1 0.001). |
| **ClassF1** (image rejected if its mask has a pixel) | 73.9 / 76.2 | **Coupled to the pixel threshold.** 65% of `test_public` images are anomalous, so *reject everything* scores 79.3; v1's 3σ masks reject 65% of normal images and sit below that floor. A threshold good for SegF1 leaves faint defects with empty masks. |
| lighting (private → mixed) | −2.5 AU-PRO, −3.4 SegF1 | Already the smallest drop of any published method; kept in view, not the main loss. |

## 2. What the published systems do

| system | SegF1 private / mixed | AU-PRO private / mixed | approach |
|---|---|---|---|
| SuperADD (VAND 4.0, 2026) | **57.42 / 54.35** | — | training-free: DINOv3 ViT-H+/16, 640 px overlapping tiles at 0.625 scale, per-block banks with density subsampling, brightness augmentation, threshold = P95 of held-out training maps × 1.42, line closing + hole filling |
| ISVL (VAND 3.0 winner) | 53.81 / 51.43 | — | INP-Former + CPR, tiling, morphological post-processing |
| RoBiS (VAND 3.0, 2nd) | 51.00 / 46.52 | 67.25 / 59.65 | INP-Former (trained, DINOv2-R ViT-B), 1024 px crops, lighting augmentation, MEBin + SAM masks |
| SuperAD (VAND 3.0, 4th) | 47.75 / 43.19 | 60.51 / 58.37 | training-free DINOv2-L kNN, 16 reference images, threshold tuned on `test_public` |
| v1 (S1, this project) | 26.42 / 23.01 | 37.64 / 35.11 | WRN50 PatchCore, whole image at 640 px, 3σ masks |

Sources: VAND 3.0 report (arXiv:2509.17615, Table 2), SuperADD (arXiv:2605.14808, Table 2),
RoBiS (arXiv:2505.21152, Table 2), SuperAD (arXiv:2505.19750). The lesson the VAND 3.0 report
draws — every valid entry used ≥ 448 px inputs or tiles, and post-processing of the maps mattered —
matches §1. SuperADD is closest to this project's own method (a memory bank, no training), so it
is the blueprint, reimplemented from the paper (its code is CC BY-NC 4.0; none is copied).

## 3. Maps: tiled DINOv3 (`src/inspector/models/tiled_knn.py`)

Two engineering findings before any number: DINOv3 overflows fp16 (NaN from the first blocks on a
real image), so the backbone runs with bf16 autocast over fp32 weights (0.4–1.5% feature error
against fp32, 150 ms per 640 px tile on the RTX 3050); and an axis only one token longer than a
tile (`sheet_metal`, 41 tokens high) was covered by two nearly identical windows — one window when
the axis fits in tile + 2 × overlap halved the cost. Final cost: ~2 s per image, 2.7 GB VRAM peak.

ViT-L against ViT-H+ (SuperADD's backbone) on `sheet_metal`, `vial`, `fruit_jelly`, `wallplugs`:
AU-PRO 0.595 vs 0.581, SegF1 at 3σ 0.463 vs 0.493 (registry, all captures). ADR-15 kept ViT-L on
those numbers — before the tuned-mask study of H+ had finished, which was a mistake: with tuned
masks H+ is better on SegF1 in all four (`vial` 0.519 → 0.610, `fruit_jelly` 0.509 → 0.550). §8
reports the eight-category comparison and the switch to H+ (ADR-16).

## 4. Masks and the image decision (`src/inspector/postproc/binarize.py`)

`MaskRule`: threshold, line closing at 16 orientations (radius 26 px at quarter resolution; the
closing dilates with a 3 px line and erodes with a 1 px one, so a 1 px scratch at an angle between
two sampled orientations still bridges), keep where the map exceeds 0.8 × the threshold, fill
holes, erode 1 px. Thresholds come from validation maps only (normal images):
`binarize.validation_thresholds`, shared by the study and the export.

`MaskRule.mark` separates ClassF1 from SegF1: an image whose mask came out empty but whose map
exceeds `mark` × the threshold gets one quarter-resolution pixel at its maximum. On the fused maps
`mark = 0.9` raises ClassF1 0.749 → 0.806, above the 0.793 floor, at no measurable SegF1 cost
(0.5710 → 0.5709); lower marks reject every image.

**The proxy is calibrated on the one server result we have** (v1, 3σ masks):

| | proxy on `test_public` | server |
|---|---|---|
| AU-PRO, `regular` ↔ private | 36.4 | 37.64 |
| AU-PRO, all ↔ mixed | 35.1 | 35.11 |
| SegF1, private / mixed | 23.2 / 20.5 | 26.42 / 23.01 |
| ClassF1, private / mixed | 77.0 / 77.7 | 73.91 / 76.21 |

## 5. Fusion and the per-category picture

`scripts/fusion_study.py`: each model's map divided by the 95th percentile of its own validation
pixels at quarter resolution, then the maximum. DINOv3 alone loses `can` (AU-PRO 0.000: its 25 px
tokens miss hair-thin scratches) and `fruit_jelly`; WRN50 at 1280 px (14 px cells) keeps them, and
the maximum costs DINOv3's strong categories at most 0.027 (`rice`; `sheet_metal` 0.013). Mean fusion was worse (AU-PRO 0.507 on the
first four categories against 0.535).

| category | AU-PRO v1 | DINOv3-L | **v4** | SegF1 v1 (3σ) | DINOv3-L (P95×1.3, morph) | **v4** | ClassF1 v1 | **v4** |
|---|---|---|---|---|---|---|---|---|
| `can` | 0.093 | 0.000 | 0.144 | 0.000 | 0.000 | 0.001 | 0.681 | 0.673 |
| `fabric` | 0.146 | 0.427 | 0.497 | 0.057 | 0.873 | 0.930 | 0.747 | 0.796 |
| `fruit_jelly` | 0.488 | 0.390 | 0.544 | 0.295 | 0.509 | 0.527 | 0.875 | 0.857 |
| `rice` | 0.265 | 0.672 | 0.645 | 0.062 | 0.707 | 0.693 | 0.811 | 0.808 |
| `sheet_metal` | 0.183 | 0.608 | 0.595 | 0.391 | 0.549 | 0.514 | 0.658 | 0.896 |
| `vial` | 0.755 | 0.859 | 0.855 | 0.322 | 0.519 | 0.499 | 0.891 | 0.857 |
| `wallplugs` | 0.270 | 0.523 | 0.559 | 0.067 | 0.733 | 0.641 | 0.759 | 0.753 |
| `walnuts` | 0.612 | 0.518 | 0.669 | 0.449 | 0.763 | 0.762 | 0.789 | 0.806 |
| **macro** | **0.351** | **0.500** | **0.564** | **0.205** | **0.582** | **0.571** | **0.776** | **0.806** |

(All captures — the mixed-lighting proxy. On `regular` captures, the private proxy, v4 reaches
AU-PRO 0.581 and SegF1 0.595.)

## 6. Candidate v4 and its held-out test

Frozen in [`candidate-v4-fusion.yaml`](../configs/recipes/candidate-v4-fusion.yaml) by ADR-15,
committed and pushed (`7401c9d`) before the export; one global setting, no per-category choice.
Pre-registered (H5): AU-PRO ≥ v1 + 10 and SegF1 ≥ v1 + 20 on both private splits, ClassF1 not below
v1. Server result: **pending the owner's upload**.

## 8. E21b — the remaining weaknesses, attacked one by one

After v4 was frozen, each open point was tested before the export, on `test_public`:

| question | test | result | decision |
|---|---|---|---|
| ViT-H+ instead of ViT-L? | all eight categories, fused with WRN50, tuned masks | SegF1 0.595 / 0.571 → **0.620 / 0.609**; AU-PRO 0.581 / 0.564 → 0.567 / 0.555 | **H+** (ADR-16): SegF1 is the weaker and the ranked metric |
| one threshold or one per category? | split-half over objects, 20 splits, per-image counts | per-category **−0.024** SegF1 out of sample, better in 32% of splits; global gain 1.3 chosen in 29/40 halves | **one global threshold** (SuperAD's per-category tuning on `test_public` would overfit) |
| shape of the mask? | connected-component filter (4/16/64 px), closing radius 13/40, lower 0.7/0.9, no erosion | all within ±0.01 SegF1 of the default | default `MaskRule` |
| fuse all three (H+, L, WRN50)? | max of three normalized maps | AU-PRO +1, SegF1 *unstable* (`regular` 0.540 at gain 1.3) | not adopted (+50% cost) |
| native resolution for thin defects? | H+ at scale 1.0 on `sheet_metal`, `can` | `sheet_metal` AU-PRO 0.594 → 0.634 alone but 0.573 → 0.583 fused, SegF1 0.551 → 0.533; `can` AU-PRO 0.000, pixel AUROC 0.48 | scale 0.625 kept (2.5× the cost for nothing after fusion) |
| `can`? | DINOv3 at two scales and two sizes | below chance at pixel level in every variant; only WRN50 carries it (0.14) | unresolved |

Published context: VAND 4.0 (arXiv:2610.04392) ranks SuperADD first (57.42 / 54.35) and ISVL++
second (58.31 / 53.75; INP-Former on DINOv3 with a segmentation head trained on synthetic
anomalies, which also set its thresholds); DL-SubspaceAD (DINOv3 ViT-7B at full resolution) fifth.
The common elements are DINOv3, tiling or high resolution, and morphological post-processing —
what v5 has. What it lacks is training: ISVL++'s synthetic-anomaly head is the one documented
lever left, beyond a 4 GB card for eight categories.

## 9. E22 — three post-processing ideas of our own, tested on v5's maps

Proposed for the failure mechanisms measured in E14–E21 and tested on the stored maps of the v5
components (`scripts/fusion_study.py`, all eight categories, `test_public`). AU-PRO is the
threshold-free test: a post-processing step that does not raise it does not separate defects
from normal surface any better.

| idea | rationale | AU-PRO `regular` / all | SegF1 all (best gain) | verdict |
|---|---|---|---|---|
| v5 as frozen | — | **0.567 / 0.555** | **0.609** | — |
| per-image background normalization (map − median, map / median) | a changed light raises the whole map of a normal image | 0.563 / 0.557, 0.562 / 0.554 | — / 0.595 | no gain: the fused maps already carry the lighting change locally, not as an offset |
| specular-highlight attenuation (white top-hat peaks, scores × 0.5 or × 0) | `can`'s false alarms sit on moving highlights | 0.506 / 0.497, 0.501 / 0.491 | 0.58 | **harmful:** on metal the defects *are* highlights — a scratch catches the light (`can` 0.109 → 0.054, `sheet_metal` 0.573 → 0.411) |
| robust z-score per component instead of P95 scaling | a noisy component (DINOv3 on `can`) should weigh less in the maximum | 0.541 / 0.526 | — | worse: the P95 scale balances the two components better |

All three are recorded as negative results (37–39). Together with E21b they leave the post-hoc
space largely explored for these maps: what remains is a better map, which in the published
record means training (§10).

## 10. Literature for what remains, and the one lever left

| weakness | what the literature does | applicable here? |
|---|---|---|
| `can`, thin defects on reflective metal | multi-light capture (photometric stereo; M2AD, arXiv:2505.10996), paired well-lit/low-light images | needs acquisition changes, not a model change |
| AU-PRO below trained systems | INP-Former reconstruction (RoBiS, ISVL), ISVL++'s segmentation head trained on synthetic anomalies over DINOv3 features (VAND 4.0, 58.31 SegF1) | yes, in a light form (below) |
| threshold choice | MEBin per-image binarization (AnomalyNCD), synthetic anomalies for thresholds (ISVL++, anomalib), TTT4AS per-image SVM | TTT4AS assumes every image is anomalous (its gains are on anomalous images only); per-category thresholds lost out of sample here (E21b) |
| transductive scoring | MuSc (mutual scoring among test images), test-time adaptation | outside this protocol (fit on training normals only) |
| bigger features | DINOv3 ViT-7B at full resolution (DL-SubspaceAD) | beyond a 4 GB card |

**Proposed (not run): a learned fusion head trained on synthetic anomalies.** The v5 map is a fixed
rule — mean of four block distances, maximum with WRN50, all equally weighted. A head of a few
hundred parameters (logistic regression or a two-layer MLP on the five per-token distances, the
four DINOv3 blocks and WRN50, plus their local 3×3 maxima) trained per category on synthetic
defects (Perlin-shaped regions filled with displaced texture from the same image, and thin
strokes) pasted on normal images held out of the bank would learn which blocks see which defects,
output a probability with a natural 0.5 threshold, and keep one procedure for all categories. It
is ISVL++'s idea without its reconstruction network. Cost on the RTX 3050: per-block maps for
held-out and synthetic images (~3 h with ViT-H+), then seconds to train; it is a new candidate and
would need its own development round before a submission.

## 7. What remains weak

- **`can`** — every system is near zero (SuperADD 11.6 / 0.4 SegF1, RoBiS 1.9 / 0.8); v4 0.001.
  Hair-thin scratches on a reflective cylinder under moving highlights; the remedy would be a
  native-resolution model for that category, which the global-setting rule excludes.
- **`vial`, `fruit_jelly`, `sheet_metal`** — 0.50 / 0.53 / 0.51 SegF1 against SuperADD's 64 / 68 / 56
  and RoBiS's 71 on `sheet_metal`. The four-category screen suggests ViT-H+ would add a few SegF1
  points on the transparent parts at twice the cost.
- **AU-PRO** stays below the trained systems (RoBiS 67): DINOv3 tokens at 0.625 scale are 25 native
  px. Native-scale tiles (2.5× the windows) and a trained reconstruction model (INP-Former) are the
  next levers, both beyond the laptop for eight categories in one session.
