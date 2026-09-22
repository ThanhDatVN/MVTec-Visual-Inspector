# 09 — References

**Verification status** is tracked per entry, because a plan built on unchecked numbers produces a
report full of unchecked numbers.

- ✅ **Verified** — primary source read directly
- ◐ **Partial** — abstract, official dataset page, or documentation read; details need the full text
- ⚠ **Unverified** — obtained from a secondary source or search summary; **must be checked before
  it appears in the final report**

---

## Datasets

| Ref | Source | Status | Notes |
|-----|--------|--------|-------|
| **MVTec AD 2** | Heckler-Kram, Neudeck, Scheler, König, Steger. *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection.* IJCV 134(4), 2026. DOI 10.1007/s11263-026-02743-0 · [arXiv:2503.21622](https://arxiv.org/abs/2503.21622) · [dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) | ◐ | Abstract and official page verified. Per-category counts, resolutions, and the baseline table were read from the arXiv HTML — **re-verify against the published version (task at G1/G2)**. Author list needs confirmation. |
| **MVTec AD (classic)** | Bergmann, Fauser, Sattlegger, Steger. CVPR 2019, DOI 10.1109/CVPR.2019.00982 · Bergmann et al., IJCV 129(4):1038-1059, 2021, DOI 10.1007/s11263-020-01400-4 | ✅ | License and citation confirmed from the official dataset page |
| **MVTec AD-C** (corrupted AD, robustness) | [github.com/YurongChen1998/MVTec-AD-C](https://github.com/YurongChen1998/MVTec-AD-C) | ⚠ | Six algorithmic corruptions (defocus/Gaussian blur, dropout, Gaussian/Poisson noise, salt-and-pepper). Useful prior art for our C1–C6 design; check whether its severity definitions are reusable. |
| **Evaluation server** | [benchmark.mvtec.com](https://benchmark.mvtec.com/) | ◐ | Required for any `test_private` number. Confirm submission format (per-pixel maps, optional thresholds) and submission limits before P11. |

---

## Core methods

| Ref | Source | Status | Role |
|-----|--------|--------|------|
| **PatchCore** | Roth et al. *Towards Total Recall in Industrial Anomaly Detection.* [arXiv:2106.08265](https://arxiv.org/abs/2106.08265), CVPR 2022 | ◐ | Tier 3 core model. Read in full at P5; per-category numbers are the G5 reproduction target. |
| **PaDiM** | Defard et al., 2020 | ⚠ | Tier 2 |
| **SPADE** | Cohen & Hoshen, 2020 | ⚠ | Tier 2 |
| **Deep SVDD** | Ruff et al., ICML 2018 | ⚠ | Tier 1. Read the hypersphere-collapse constraints before implementing. |
| **SSIM autoencoder** | Bergmann et al. *Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders*, 2019 | ⚠ | Tier 1; the SSIM-loss variant |
| **EfficientAD** | Batzner, Heckler, König. WACV 2024. [arXiv:2303.14535](https://arxiv.org/abs/2303.14535) | ◐ | Tier 4; best published AU-PRO@0.05 on AD 2 (30.8%) and the millisecond-latency reference |
| **Reverse Distillation / RD++** | Deng & Li (CVPR 2022); Tien et al. (CVPR 2023) | ⚠ | Tier 4; best SegF1 among the classical AD 2 baselines |

---

## Recent work (2024–2026) — the "advanced" tier

| Ref | Source | Status | Role |
|-----|--------|--------|------|
| **Dinomaly** | Guo et al. *The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection.* CVPR 2025. [arXiv:2405.14325](https://arxiv.org/abs/2405.14325) · [code](https://github.com/guojiajeremy/Dinomaly) | ◐ | Tier 4. Reported 99.6% image AUROC on classic MVTec AD (multi-class), up to 99.8% scaled — the clearest evidence that classic AD is finished. |
| **Dinomaly2** | *One Dinomaly2 Detect Them All.* [arXiv:2510.17611](https://arxiv.org/abs/2510.17611) | ⚠ | Reported 99.9% on MVTec AD. Context, probably not a run. |
| **INP-Former** | Luo et al. *Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection.* CVPR 2025. [arXiv:2503.02424](https://arxiv.org/abs/2503.02424) · [code](https://github.com/luow23/INP-Former) | ◐ | Tier 4; a genuinely different inductive bias |
| **RoBiS** | Li et al. *Robust Binary Segmentation for High-Resolution Industrial Images.* CVPR 2025 VAND 3.0 challenge. [arXiv:2505.21152](https://arxiv.org/abs/2505.21152) · [code](https://github.com/xrli-U/RoBiS) | ◐ | **The most directly useful reference for Tier 5.** Swin-cropping, lighting/noise augmentation, adaptive binarization (`mean+3σ` combined with MEBin), mask refinement. Reported SegF1 21.8 → 51.0 on `TEST_priv` and 16.7 → 46.5 on `TEST_priv,mix`. Read in full before P8. |
| **AnomalyDINO** | Damm et al. *Boosting Patch-based Few-shot Anomaly Detection with DINOv2.* [arXiv:2405.14529](https://arxiv.org/abs/2405.14529) | ⚠ | Evidence for the DINO-backbone ablation in Tier 3 |
| **AD-DINOv3** | 2026 | ⚠ | DINOv3 ViT-L/16 with multi-level feature aggregation; reported ~94.2% AUROC / 44.6% F1 on industrial data. Background for OD-5. |
| **"Recovering Total Recall"** (PatchCore on AD 2) | [github.com/yyqmeow/patchcore-mvtec-ad2](https://github.com/yyqmeow/patchcore-mvtec-ad2) | ⚠ | Claims layer3-only PatchCore gives ~9% AU-PRO and that layer2+layer3 multi-scale fusion raises mean AU-PRO@5% to ~76%, at the cost of image AUROC (~90% → ~66%). **An unrefereed companion repo, and the 76% figure is far above the published leaderboard — most likely a different split or protocol.** Treat strictly as a *hypothesis* for the Tier 3 layer ablation. Do not cite the number without reproducing it. |

---

## Metrics

| Ref | Source | Status | Role |
|-----|--------|--------|------|
| **PRO / AU-PRO** | Bergmann et al., IJCV 2021 (see MVTec AD) | ◐ | Primary localization metric. Pin the FPR-normalization convention (OD-3). |
| **AUPIMO** | Bertoldo et al. *AUPIMO: Redefining Visual Anomaly Detection Benchmarks with High Speed and Low Tolerance.* BMVC 2024. [arXiv:2401.01984](https://arxiv.org/abs/2401.01984) | ◐ | Per-image metric enabling paired statistical tests; our model-selection metric of record alongside AU-PRO@0.05 |
| **ImageNet-C** | Hendrycks & Dietterich, ICLR 2019. [arXiv:1903.12261](https://arxiv.org/abs/1903.12261) | ✅ | Methodological template for the corruption suite. **Severity constants are not transferable** — our images are far larger; recalibrate (see [05](05-robustness-protocol.md) §2). |

---

## Tooling

| Tool | Source | Status | Role |
|------|--------|--------|------|
| **anomalib** (v2.2+) | [github.com/open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib) | ◐ | Tier 4 comparators; includes an `MVTecAD2` datamodule with a `test_type` parameter for the three test splits, and reference metric implementations for the G2 validation |
| **Grad-CAM** | Selvaraju et al., ICCV 2017 | ✅ | Tier 6 explainability |
| **Model cards** | Mitchell et al., FAT* 2019 | ✅ | Structure of `MODEL_CARD.md` |
| **Segment Anything** | Kirillov et al., 2023 | ⚠ | Optional Tier 5 mask refinement (stretch only) |

---

## Reading order

Read these before the phase that needs them, not all at once.

| Before | Read |
|--------|------|
| P1 | MVTec AD 2 paper (full), MVTec AD 2021 IJCV §on PRO |
| P2 | AUPIMO; anomalib's AU-PRO implementation source |
| P3 | SSIM-AE; Deep SVDD |
| P4 | PaDiM; SPADE |
| P5 | **PatchCore (full, carefully)**; the AD 2 baseline section |
| P7 | ImageNet-C; MVTec AD-C |
| P8 | **RoBiS (full)**; EfficientAD; Dinomaly |
| P10 | Model Cards for Model Reporting |
