# P3 — First Results on Real Data (VisA)

Stages A–C of [the run register](../docs/11-experiment-plan.md), executed on the laptop
(RTX 3050, 4 GB) against VisA's official one-class split. These are the project's **first numbers
on real data**; everything before this was on synthetic fixtures and proved only that the code was
correct.

Categories: `pcb1`, `macaroni2`, `capsules`. Dataset: VisA © Amazon, CC BY 4.0 — see
[ATTRIBUTION.md](../ATTRIBUTION.md).

---

## 0. The licence question is settled

The AWS registry and the paper say CC BY 4.0; some third-party documentation says CC BY-NC-SA 4.0.
[docs/10](../docs/10-datasets.md) recorded the discrepancy and adopted the stricter reading.

**The archive resolves it.** `VisA_20220922.tar` ships a `LICENSE-DATASET` file whose first line is
`Attribution 4.0 International` — the CC BY 4.0 text. The dataset's own licence file is the most
authoritative source available, so **VisA is CC BY 4.0**: attribution is the only obligation, with
no non-commercial and no share-alike clause. The third-party claim is wrong.

The project still redistributes no images, because the `.gitignore` and the CI guard do not need an
exception carved into them for one dataset.

## 1. Stage A — split integrity

| category | train | validation | test (N/A) | checks |
|----------|-------|------------|------------|--------|
| `pcb1` | 768 | 136 | 100 / 100 | 5/5 pass |
| `macaroni2` | 765 | 135 | 100 / 100 | 5/5 pass |
| `capsules` | 461 | 81 | 60 / 100 | 5/5 pass |

15/15 checks pass: no cross-split content duplication (SHA-256 over file bytes, not names), no
within-split duplicates, every anomalous test image has a mask. Validation carved from train with
seed 0; assignment digests recorded. **Gate A passes.**

## 2. Stage A — the resolution decision, and a correction to the plan

### 2.1 VisA defects are far smaller than the standard protocol assumes

| category | native | regions | median defect area | p5 area | median thin dim | median local contrast |
|----------|--------|---------|--------------------|---------|-----------------|----------------------|
| `pcb1` | 1404×1070 | 304 | 0.0232% | 0.0005% | 18.0 px | 58.2 grey levels |
| `macaroni2` | 1500×1000 | 378 | **0.0017%** | 0.0001% | 5.5 px | **10.6 grey levels** |
| `capsules` | 1500×1000 | 156 | 0.0791% | 0.0006% | 38.0 px | 22.9 grey levels |

What a resize does to the defect population:

| long side | `pcb1` regions <1px | `macaroni2` | `capsules` |
|-----------|---------------------|-------------|------------|
| 256 | 19.4% | **50.0%** | 11.5% |
| 320 | 13.2% | 43.9% | 9.6% |
| 448 | 8.6% | 35.7% | 7.1% |
| 512 | 4.9% | 27.5% | 4.5% |
| native | 0.0% | 0.0% | 0.0% |

**The literature's standard VisA protocol resizes to 256×256.** On `macaroni2` that puts half of
all annotated defect regions below one pixel, and the median defect's local contrast is 10.6 grey
levels before any resizing at all. Pixel-level metrics computed on `macaroni2` at 256² are
therefore scoring a ground truth that is, for half its regions, not present in the model's input.

This is not a claim that published numbers are wrong — they are computed correctly against masks
at whatever resolution the authors chose. It is a claim about **what those numbers can mean**, and
it is the measured justification for treating resolution as a study axis rather than a default.

### 2.2 The VRAM assumption was wrong, and the real constraint is elsewhere

[docs/06 §3](../docs/06-engineering-mlops-and-testing.md) framed the 4 GB card as "the binding
constraint" and the resolution decision used an assumed 1.0 MP single-pass budget. Both are now
measured, and both were wrong.

**Measured — peak VRAM, WideResNet50-2, `layer2+layer3`:**

| input | MP | batch 1 | batch 2 | batch 4 |
|-------|-----|---------|---------|---------|
| 256×195 | 0.05 | 307 MB | 322 MB | 354 MB |
| 512×390 | 0.20 | 364 MB | 431 MB | 592 MB |
| 1024×780 | 0.80 | 642 MB | 904 MB | 1535 MB |
| **1404×1070** | **1.50** | **959 MB** | 1459 MB | 2645 MB |

Native resolution costs **under 1 GB** at batch 1 on a 4096 MB card. The 1.0 MP assumption was too
conservative by at least a factor of three, and it had caused the analysis to demand tiling for all
three categories — a recommendation that was an artifact of the placeholder, not of the hardware.
The constant is now the measured 3.0 MP and the recommendation is native resolution, no tiling.

**The real constraint is the patch matrix in host RAM.** A `layer2` grid at 1404×1070 yields 23,584
patches per image:

| input | patches/img | × 768 train imgs | fp16 host RAM |
|-------|-------------|------------------|---------------|
| 256×195 | 800 | 614 k | 1.3 GB |
| 512×390 | 3,136 | 2.4 M | 4.9 GB |
| 1024×780 | 12,544 | 9.6 M | 19.7 GB |
| **1404×1070** | **23,584** | **18.1 M** | **37.1 GB** |

This laptop has 16.3 GB total and 7.8 GB free. A Kaggle session has ~30 GB. **Native-resolution
PatchCore fits in neither**, and the forward pass being cheap is irrelevant to that.

**Consequence for the plan.** The current implementation accumulates all patches before
subsampling, so the extraction peak, not the coreset, is what binds. Native-resolution PatchCore
requires **per-image patch subsampling during extraction**. That is a concrete, measured engineering
requirement that the plan did not previously contain, and it replaces "tiling for VRAM" as the
Tier 5 priority.

Until it exists, the reported PatchCore configuration runs at a reduced resolution and is
**handicapped by a known amount** — the region-loss table in §2.1 says exactly how much.

## 3. Stage B — Tier 0 floors

VisA test split, 3 seeds, 256 px, validation-derived thresholds.

| category | method | image AUROC | pixel AUROC | AU-PRO@0.05 | AU-PRO@0.30 |
|----------|--------|-------------|-------------|-------------|-------------|
| `pcb1` | random | 0.501 ± 0.030 | 0.499 | 0.032 | 0.161 |
| `pcb1` | mean_intensity | **0.801** | 0.755 | 0.038 | 0.296 |
| `pcb1` | histogram | **0.831** | 0.735 | 0.038 | 0.281 |
| `pcb1` | pixel_pca | 0.813 | 0.969 | 0.365 | 0.733 |
| `macaroni2` | random | 0.481 ± 0.073 | 0.499 | 0.023 | 0.153 |
| `macaroni2` | mean_intensity | 0.391 | 0.462 | 0.002 | 0.129 |
| `macaroni2` | histogram | 0.451 | 0.511 | 0.013 | 0.103 |
| `macaroni2` | pixel_pca | 0.734 | 0.890 | 0.177 | 0.603 |
| `capsules` | random | 0.505 ± 0.010 | 0.508 | 0.027 | 0.158 |
| `capsules` | mean_intensity | 0.505 | 0.480 | 0.054 | 0.204 |
| `capsules` | histogram | 0.554 | 0.494 | 0.052 | 0.200 |
| `capsules` | pixel_pca | 0.789 | 0.739 | 0.119 | 0.371 |

### 3.1 Gate G2 passes on real data

| control | measured | expected |
|---------|----------|----------|
| random, image AUROC | **0.4955** | 0.500 |
| random, pixel AUROC | **0.5017** | 0.500 |
| random, AU-PRO@0.05 | **0.0274** | 0.025 (= L/2) |
| random, AU-PRO@0.30 | **0.1573** | 0.150 (= L/2) |

All four land on their analytic values. The metric implementation is now validated on real data at
real scale, not only on fixtures. **Gate B passes.**

### 3.2 `pcb1` is substantially separable by a global statistic — report this prominently

A colour histogram reaches **0.831** image AUROC on `pcb1`, and the global mean intensity reaches
**0.801**. Both exceed the 0.80 threshold the plan set for flagging, and neither model has any
notion of *where* anything is — both emit a constant anomaly map.

This bounds what an image-level result on `pcb1` can claim. A method reporting 0.95 image AUROC
there has improved on a trivial baseline by 0.12, not by 0.45, and any claim that it "understands"
PCB defects has to survive that comparison. The honest reporting requirement is that **every
image-level number on `pcb1` is printed against its 0.83 trivial floor**.

`macaroni2` shows the opposite and is reassuring: trivial methods sit at or below chance
(0.391, 0.451), so its image-level scores are not explainable by global appearance. It is, as the
VisA paper suggests, the genuinely hard category.

### 3.3 The constant-map effect, now visible at scale

`mean_intensity` emits a constant map yet reaches 0.755 pixel AUROC and 0.296 AU-PRO@0.30 on
`pcb1`. This is the mechanism documented in [P2](P2-tier0-findings.md): with a constant map,
lowering the threshold switches whole images on at once, so a model that merely *ranks images*
collects whole regions at low FPR.

At fixture scale this was dominated by discretization. At 200 test images it is not — it is a real
property of AU-PRO. **A localization metric should not be read without a detection baseline
beside it**, which is precisely why the Tier 0 row belongs in every table.

### 3.4 `pixel_pca` again localizes better than it detects

0.969 pixel AUROC and 0.365 AU-PRO@0.05 on `pcb1`, from linear PCA on 64×64 downscaled pixels.
That is a high bar for the Tier 1 autoencoder: a conv AE that does not beat it has learned nothing
a linear projection did not already capture.

Its image AUROC (0.813) is below its localization quality, repeating the P2 finding that map
quality and map-to-scalar aggregation are separate abilities. Stage F4 ablates aggregation
separately for exactly this reason.

## 4. Stage C — PatchCore

*(Running at the time of writing; results appended when the three categories × three seeds
complete. Configuration: WideResNet50-2, `layer2+layer3`, 3×3 aggregation, 1% greedy coreset,
k=1, 320 px — reduced from native by the host-RAM constraint in §2.2.)*

## 5. What these results do not establish

- **No comparison against a published number yet.** Gate G5 — reproducing a PatchCore result on a
  known benchmark — is still open, so "our PatchCore" cannot yet be called "PatchCore".
- **AUPIMO is still unimplemented**, so paired tests pair over three categories rather than ~200
  images and have very little power. This remains the highest-priority gap in
  [docs/11 §6](../docs/11-experiment-plan.md).
- **The PatchCore numbers are at reduced resolution** and are handicapped by the amount §2.1
  quantifies. They are a development measurement, not the study's headline.
- **VisA has no lighting-shifted split**, so the robustness validity question stays blocked on
  MVTec AD 2.

---

Data: VisA © Amazon, CC BY 4.0 (confirmed from the archive's own `LICENSE-DATASET`).
No dataset images are redistributed in this repository.
