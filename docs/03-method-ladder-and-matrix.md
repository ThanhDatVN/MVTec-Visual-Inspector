# 03 — Method Ladder and Experiment Matrix

The ladder runs from "a statistic you could compute in Excel" to techniques published in 2025–2026.
Each tier exists to answer a question the tier below it cannot, and each tier's cost is justified
by the answer it buys.

Ownership is explicit throughout:
**[OWN]** = we implement it from scratch in PyTorch · **[LIB]** = third-party implementation
(anomalib / official repo), used as a comparator and labelled as such in every table.

---

## Tier 0 — Trivial floors  `[OWN]`

**Question:** is this category separable by something embarrassingly simple, and is my evaluation
code correct?

| Model | Description | Cost |
|-------|-------------|------|
| `T0-mean` | Image mean and std of intensity, Gaussian likelihood | seconds |
| `T0-hist` | Chi-square distance of colour histogram to the training mean histogram | seconds |
| `T0-pca` | PCA on raw downscaled pixels, score = reconstruction error | ~1 min |
| `T0-random` | Uniform random scores | instant |

**Why this tier is not a waste of time.** `T0-random` must give AUROC ≈ 0.5 — if it does not, the
metric code is broken, and every number downstream is wrong. `T0-mean` catches a category where a
lighting offset alone separates the classes, which would make a deep model's success meaningless.
These four runs are the cheapest insurance in the project and they take an afternoon.

**Deliverable:** the floor row of every results table.

---

## Tier 1 — Custom baselines  `[OWN]` — *required by scope*

**Question:** what does a model that learns normality *from scratch on this data alone* achieve?

| Model | Architecture | Key detail |
|-------|--------------|------------|
| `T1-cae-l2` | Conv AE, 4 down / 4 up blocks, latent 64–256 | L2 reconstruction loss. The classic, and classically weak: L2 blurs, and a blurry reconstruction produces a high error everywhere, not at the defect. |
| `T1-cae-ssim` | Same, SSIM loss | SSIM compares local structure rather than per-pixel intensity, which is much closer to what "looks wrong" means. Expect a clear gain over L2; this is the ablation that teaches the most about loss design. |
| `T1-vae` | Variational AE | Score = reconstruction error, optionally + KL. Included mainly to show that the probabilistic framing does not rescue reconstruction on high-res industrial images. |
| `T1-dsvdd` | Deep SVDD | Trains a net to map normals into a minimum-volume hypersphere; score = distance to centre. Implement the centre-fixing and bias-free constraints that prevent hypersphere collapse — collapse is the classic failure and demonstrating you avoided it is worth a paragraph. |
| `T1-ocsvm` / `T1-kde` | OC-SVM / kernel density on a frozen feature vector | The bridge to Tier 2: one-class *learning* over pretrained *features*. |

**Scoring for reconstruction models.** Anomaly map = per-pixel residual, but compute it as a
multi-scale residual (residual at 1×, 1/2×, 1/4× then averaged) rather than raw per-pixel
difference; raw residuals are dominated by edge misalignment. Record both — the comparison is
instructive.

**Known outcome, and why we still do it.** Reconstruction methods are expected to lose badly to
Tier 3 on these categories. That is the point: the baseline establishes *how much* the pretrained
prior is worth, in points, on our own data. A project that starts at PatchCore cannot quantify
what PatchCore bought.

**Ablations:** latent dimension {32, 64, 128, 256} · loss {L2, SSIM, L2+SSIM} · resolution
{256², 320²} · residual scheme {raw, multi-scale}.

---

## Tier 2 — Pretrained features, no training  `[OWN]`

**Question:** how much comes for free from an ImageNet-pretrained backbone?

| Model | Method | Note |
|-------|--------|------|
| `T2-mahal` | Mahalanobis distance on a globally pooled embedding | Image-level only, ~20 lines, often shockingly strong. Regularize the covariance (Ledoit–Wolf shrinkage) — with 137 training images and a 1024-d feature, the empirical covariance is singular. |
| `T2-padim` | Per-position multivariate Gaussian over a feature map; random dimension selection | Fixed spatial grid means it is sensitive to part misalignment; on a fixed industrial rig that is an acceptable assumption, and saying so explicitly is part of the analysis. |
| `T2-spade` | kNN over a multi-scale feature pyramid, image retrieval then pixel alignment | The direct conceptual ancestor of PatchCore. Implementing it makes PatchCore's coreset contribution obvious. |

**Ablations:** backbone {ResNet18, ResNet50, WideResNet50-2} · layers {layer2, layer3, layer2+3} ·
covariance shrinkage {none, Ledoit–Wolf} · feature dimension {100, 256, 550}.

---

## Tier 3 — PatchCore  `[OWN]` — *the core model*

**Question:** what does a locally-aware memory of normal patches achieve, and what does it cost?

### Reference configuration

| Component | Setting |
|-----------|---------|
| Backbone | WideResNet50-2, ImageNet, frozen |
| Feature layers | `layer2` + `layer3`, layer3 bilinearly upsampled to layer2's grid |
| Patch aggregation | 3×3 average pooling over the feature map (local neighbourhood awareness) |
| Projection | Adaptive average pool to a fixed dimension (1024) |
| Memory bank | Greedy k-center coreset subsampling, ratio 1% (also ablate 10%, 25%, 100%) |
| Scoring | kNN distance (k = 1 for the map; the paper's re-weighting for the image score) |
| Map post-processing | Bilinear upsample to native resolution, Gaussian blur sigma = 4 |

### Why PatchCore is the right upgrade target

It needs no training — only a forward pass over the normal set — so the whole model is a frozen
backbone plus a stored matrix. That makes it (a) trivially reproducible, (b) inherently
explainable (every score points at a specific nearest normal patch — see Tier 6), and (c) a clean
platform for ablation, because there is no training noise to confound results.

### The pre-identified research question

The external matrix shows PatchCore at 28.8% AU-PRO@0.05 but **3.7% SegF1** on AD 2. Its scores
rank correctly but do not calibrate. Phase P8 investigates directly:

- Is the score distribution on validation normals so heavy-tailed that `mean + 3·std` is far off?
- Does per-category score standardization, rank normalization, or a quantile threshold fix it?
- Is this specific to the high-resolution / small-defect regime?

A clean answer to this is a more valuable outcome than a marginal AU-PRO gain, and it is not yet
well documented in the literature.

### Ablation matrix (the main study)

| Axis | Values | Question it answers |
|------|--------|---------------------|
| Backbone | WRN50-2, ResNet50, ResNet18, DINOv2 ViT-S/14, DINOv3 ViT-S/16 | Does a self-supervised foundation backbone beat ImageNet supervision for industrial texture? (2024–2026's clearest trend, and ViT-S/16 fits in 4 GB) |
| Layer set | L2, L3, **L2+L3**, L2+L3+L4 | Feature granularity vs semantic level. A companion study reports that layer3-only produces strong image AUROC but ~9% AU-PRO, and that adding layer2 changes the picture completely. **Treat that as a hypothesis to test, not a fact** — it comes from an unrefereed repository and may be measured on a different split. |
| Resolution | 256², 320², 448², 512², tiled-native | The dominant axis for `sheet_metal`. Expect the largest single effect here. |
| Tiling | none, 2×2, overlapping sliding window (50%), Swin-cropping | Does input tiling help once feature resolution is already addressed? |
| Coreset ratio | 0.1%, 1%, 10%, 25%, 100% | The accuracy/memory/latency curve. Produces a Pareto plot — a headline figure. |
| k in kNN | 1, 3, 5, 9 | Robustness of the score to memory-bank noise |
| Smoothing sigma | 0, 2, 4, 8 | How much of AU-PRO is post-processing rather than modelling |
| Projection dim | 128, 384, 768, 1024 | Memory reduction at fixed accuracy |
| Precision | fp32, fp16 | Required for the 4 GB card; quantify the accuracy cost |

Run as a **staged sweep, not a grid**: fix the reference config, vary one axis at a time, take the
winner, then re-vary the two axes with the largest effects jointly. A full grid is roughly 10⁴ runs
and is not affordable; a staged sweep is ~60 runs and captures the main effects plus the one
interaction that matters.

---

## Tier 4 — Modern comparators  `[LIB]`

**Question:** where does our own work sit against current published methods?

| Model | Year | Family | Why included |
|-------|------|--------|--------------|
| `T4-efficientad` | WACV 2024 | Student–teacher + autoencoder | Best published AU-PRO@0.05 on AD 2 (30.8%) *and* millisecond latency. The efficiency/accuracy reference point, and the most credible deployment target. |
| `T4-rd++` | 2023 | Reverse distillation | Best SegF1 among classical baselines (19.2%) — the calibration counterexample to PatchCore. |
| `T4-dinomaly` | CVPR 2025 | ViT encoder + minimalist decoder | Current SOTA on classic AD (99.6% multi-class). Tests whether a saturating-on-classic method transfers to AD 2. |
| `T4-inp-former` | CVPR 2025 | Intrinsic normal prototypes within a single image | Strong few-shot and multi-class behaviour; a genuinely different inductive bias. |

**Rules for this tier.** Library implementations, default published hyperparameters, *our* data
pipeline and *our* metric code. Never quote a number produced by someone else's evaluation
harness next to a number produced by ours — that comparison is uncontrolled. If a model will not
fit our budget, record that as its result (a method that cannot be run on 4 GB is a finding about
the method) rather than dropping it silently.

---

## Tier 5 — Competition-grade engineering  `[OWN]`

**Question:** on this benchmark, does engineering beat architecture?

The external evidence says yes — the VAND 3.0 challenge system roughly tripled SegF1 (21.8 → 51.0)
with preprocessing, augmentation, and thresholding, not a new architecture. These are therefore
first-class experiments, not polish.

| Technique | What it does | Expected effect |
|-----------|--------------|-----------------|
| **Overlapping tiling + stitching** | Slide a window over the native-resolution image, score each tile, stitch with cosine-weighted blending to remove seams | The defensible way to keep native resolution within 4 GB. Seam artifacts are a real failure mode; blending and seam inspection are part of the work. |
| **Lighting-robust memory augmentation** | Extract features from photometrically perturbed copies of the *training normals* (exposure, gamma, colour temperature) and add them to the memory bank | Directly targets AD 2's unseen-lighting split. Tests the hypothesis that lighting robustness is a *coverage* problem, not an invariance problem. |
| **Adaptive binarization** | Compare fixed `mean+3sigma`, per-image quantile, Otsu, and an entropy/stability-based adaptive threshold | The single highest-leverage lever on SegF1, per the external evidence. |
| **Score calibration** | Rank/quantile normalization of scores against the validation-normal distribution | The direct attack on PatchCore's SegF1 collapse. |
| **Test-time augmentation** | Score under horizontal flip and small photometric jitter, then aggregate | Cheap variance reduction; costs latency, so it is evaluated under DR-1's latency clause. |
| **Feature whitening / PCA** | Decorrelate patch features before the kNN | Often improves kNN geometry and shrinks the memory bank at once |
| **Multi-scale fusion** | Combine anomaly maps from two input scales | Separates "small defect" from "large diffuse defect" failure modes |
| *(stretch)* **Mask refinement** | Refine the binarized mask with a promptable segmentation model | Improves IoU/SegF1 without changing detection. Only if P8 has budget left. |

---

## Tier 6 — Explainability  `[OWN]` — *cross-cutting*

| Technique | Applies to | Mechanism |
|-----------|------------|-----------|
| **Anomaly heatmap + overlay** | All | The primary artifact. Fixed colormap and fixed scale bar across the entire case book, so cases are visually comparable. |
| **Grad-CAM** | `T1-cae`, `T1-dsvdd`, and a supervised control | Needs a differentiable scalar. Use reconstruction error (AE) or distance-to-centre (Deep SVDD) as the target scalar. |
| **Grad-CAM on PatchCore** | `T3` | Not standard, and worth doing properly: the kNN min is non-differentiable, so use a **soft-min (log-sum-exp over the k nearest memory entries)** as a differentiable surrogate and backpropagate to the input. This distinguishes *which pixels drive the feature* from *which patch is far from memory* — a genuinely informative distinction. |
| **Nearest-normal retrieval** | `T2-spade`, `T3` | For an anomalous patch, display the nearest normal patch in memory and its source image. This is the most convincing explanation available for a memory-based model: "this region does not look like any of these 300 good parts, the closest is *this* one." Put it in the demo UI. |
| **Failure taxonomy** | All | Every case in the book is tagged: missed-small-defect, false-positive-on-normal-variation, boundary-imprecision, lighting-induced, transparency/occlusion, tiling-seam artifact. Counts per tag become a figure. |

**Case book spec (Objective O6):** at least **15 cases** — minimum 4 per study category, spanning
at least 5 taxonomy tags, and including **at least 4 failures of the best model**. A case book of
successes is marketing. Each case: input, GT mask, heatmap, overlay, binarized mask at
`OP-3SIGMA`, image score vs threshold, nearest-normal retrieval, and two to four sentences of
written diagnosis.

---

## Experiment matrix — the master plan

Run IDs follow `{tier}-{method}-{category}-{res}-{seed}`, e.g. `T3-patchcore-sheet_metal-448-s0`.

| Tier | Methods | Categories | Resolutions | Seeds | Runs | Where |
|------|---------|-----------|-------------|-------|------|-------|
| T0 | 4 | 3 | 1 | 1 | 12 | Laptop |
| T1 | 5 (+ ablations) | 3 | 2 | 3 | ~120 | Laptop |
| T2 | 3 (+ ablations) | 3 | 2 | 1–3 | ~60 | Laptop |
| T3 reference | 1 | 3 | 4 | 3 | 36 | Laptop ≤448², Kaggle above |
| T3 ablations | staged sweep | 3 | varies | 1 (then 3 for winners) | ~60 | Kaggle |
| T4 | 4 | 3 | 1–2 | 1 | ~18 | Kaggle |
| T5 | 7 techniques | 3 | best | 1 (then 3 for winners) | ~45 | Kaggle |
| Robustness | best 3 models | 3 | best | 1 | 6 corruptions × 5 severities × 3 × 3 = 270 evaluations (inference only) | Laptop (eval is cheap) |

**Total: roughly 350 fitted runs and ~300 inference-only evaluations.** This is affordable only
because Tiers 0–2 are minutes each and the robustness sweep re-uses fitted models. Budget and
sequencing are in [04-roadmap.md](04-roadmap.md).

**Sequencing rule.** Never start a tier before the gate below it has passed. The temptation to
jump straight to PatchCore is exactly how projects end up with an impressive number and no way to
tell whether it is real.
