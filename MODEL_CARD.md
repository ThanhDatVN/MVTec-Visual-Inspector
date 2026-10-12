# Model Card — MVTec Visual Inspector, PatchCore recipe v1

Structure after Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019), from
[the template](docs/templates/model-card.md). It describes the model the service and demo load: the
recipe frozen in [ADR-11](docs/07-risks-and-decisions.md), fitted per VisA category. Every number
links to the report it comes from.

## Model details

| Field | Value |
|-------|-------|
| Name / version | `inspector-visa-<category>-patchcore`, recipe `confirmation-v1-patchcore-640` |
| Date | artifacts exported 2026-09-30; card written 2026-10-12 |
| Type | Unsupervised one-class anomaly detection and localization |
| Architecture | PatchCore: WideResNet50-2 (torchvision `IMAGENET1K_V1`), `layer2`+`layer3`, 3×3 neighbourhood pooling, 1024-d channel pooling; memory bank of **10,000** patches chosen by greedy k-center (128-d random projection) from a 10% per-image candidate pool; nearest-neighbour patch score; image score = anomalib's re-weighted maximum |
| Training regime | Normal images only. One pass of frozen-backbone feature extraction plus the coreset; no gradient training |
| Input | RGB, long side **640 px**, aspect preserved (area interpolation), ImageNet normalization |
| Output | Image score; anomaly map at native resolution (Gaussian σ = 4 input px); decision `normal` / `anomalous` / `refused` (input guards); optional overlay; the nearest normal training patch to the peak |
| Artifact size | 20 MB per category (bank 10,000 × 1024 in fp16, `meta.json`, `guards.json`, training ids) |
| Frameworks | PyTorch, torchvision, OpenCV; exact versions in each artifact's `meta.json` provenance and `requirements-lock.txt` |
| Code | <https://github.com/ThanhDatVN/MVTec-Visual-Inspector>; recipe frozen at `4ff5a6f`; artifacts exported at `5474702` (`pcb1`) and `eee82dd` |
| License (code) | MIT |
| License (memory bank) | Derived from VisA, CC BY 4.0 — attribution required (below) |

## Intended use

**Intended:** research and education in cold-start visual inspection: one model per VisA category,
under that dataset's fixed capture setup, as a screening aid whose every decision a person reviews.

**Out of scope:**
- Safety-critical inspection, or any use where a missed defect causes harm without review.
- Parts, cameras, fixtures or lighting other than those fitted on. The model is a memory of *these*
  normal images; a different rig is a different distribution (see Limitations).
- Unattended pass/fail gating at the operating point below: it misses many defects (recall
  0.39–0.84 on the development categories).

## Operating point

Frozen in each artifact on that category's validation normals; the service never re-derives it.

| category | threshold τ (raw score) | calibration normals | requested FPR | guaranteed bound | realized FPR on test normals (3 seeds) | recall |
|---|---|---|---|---|---|---|
| `pcb1` | 1.539 | 136 | 1% | 0.73% | **4.3%** | 0.84 |
| `macaroni2` | 1.601 | 135 | 1% | 0.74% | **1.7%** | 0.39 |
| `capsules` | 1.747 | 81 | 1% | 1.22% (1% unreachable, flagged) | **2.2%** | 0.54 |

Rule: conservative rank `k = floor(α(n+1))`, strict `score > τ` ([ADR-9](docs/07-risks-and-decisions.md)).
The bound is marginal and assumes calibration and test normals are exchangeable. On these three
categories they are not quite: capture-session debris sits in the test normals' upper tail
([development report](reports/E00-E04-development-findings.md) §4). The pixel threshold is the
validation mean + 3σ of the map (`OP-3SIGMA`).

## Performance

### Development categories (shaped the recipe — optimistic)

3 seeds, mean ± SD ([K1](reports/K1-kaggle-development.md), [E8](reports/E8-gate-e-followup.md)):

| category | image AUROC | AU-PRO@0.05 | AU-PRO@0.30 | pixel AUROC | SegF1 (3σ) | FPR@OP1 | recall@OP1 |
|---|---|---|---|---|---|---|---|
| `pcb1` | 0.979 ± 0.002 | 0.854 ± 0.004 | 0.948 | 0.995 | 0.29 | 4.3% | 0.84 |
| `macaroni2` | 0.887 ± 0.010 | 0.864 ± 0.011 | 0.950 | 0.991 | 0.08 | 1.7% | 0.39 |
| `capsules` | 0.914 ± 0.003 | 0.870 ± 0.004 | 0.965 | 0.993 | 0.54 | 2.2% | 0.54 |

### Held-out evidence (the recipe, refitted per category, nothing retuned)

| data | image AUROC | AU-PRO@0.05 | false alarms | report |
|---|---|---|---|---|
| 9 VisA confirmation categories, pre-registered (ADR-11) | **0.981** macro (0.971–0.996) | 0.766 | 10 observed / 19.7 expected over 27 runs; bound fails on `pipe_fryum` | [E12](reports/E12-confirmation-v1.md) |
| MPDD, 6 categories, other camera and factory, pre-registered (ADR-12) | **0.943** macro (≥ 0.88 each) | 0.861 | 31 / 28.4 at relaxed bounds (2.3–11.1%) | [H1](reports/H1-mpdd-transfer.md) |

### Performance across conditions

Synthetic corruptions of the test split, threshold frozen on clean data, seed 0
([E08](reports/E08-robustness.md)); "refused" is the share of all test images the input guards turn
away:

| condition | `pcb1` AUROC / FPR / refused | `macaroni2` | `capsules` |
|---|---|---|---|
| clean | 0.979 / 5% / 5% | 0.887 / 2% / 0% | 0.916 / 2% / 7% |
| Gaussian blur, σ = 0.55 × defect diameter | 0.652 / 100% / 100% | 0.832 / 28% / 100% | 0.628 / 100% / 100% |
| exposure +1.1 stop | 0.852 / 100% / 100% | 0.560 / 100% / 100% | 0.557 / 100% / 100% |
| exposure −1.1 stop | 0.968 / 100% / 100% | 0.921 / 7% / 100% | 0.915 / 7% / 100% |
| resize round-trip 0.49× | 0.976 / 5% / 100% | 0.878 / 2% / 44% | 0.889 / 3% / 96% |
| shift of 1.1% of the frame | 0.833 / 100% / 5% | 0.825 / 9% / 1% | 0.841 / 10% / 6% |

Unseen *real* lighting (MVTec AD 2's `test_private_mixed`) is not measured: that dataset is not
available to this project.

**By defect type:** VisA's one-class split labels every defect "anomaly", so no per-type breakdown
exists. By defect *size*, misses concentrate on small defects whose peaks fall below a threshold
set by background variation ([development report](reports/E00-E04-development-findings.md) §6,
measured at 320 px).

### Systems

Batch 1, 20 warm-up and 200 timed requests through the service path ([E8](reports/E8-gate-e-followup.md) §3):

| device | precision | long side | p50 | p95 | p99 | peak VRAM (allocated / reserved) | peak RSS |
|---|---|---|---|---|---|---|---|
| RTX 3050 Laptop GPU (4 GB) | fp16 features | 640 | 99 ms | 102 ms | 103 ms | 574 / 692 MB | 1.5 GB |
| Kaggle T4 | fp16 features | 640 | 90–102 ms per image in batch evaluation; not measured with this protocol | | | | |
| AMD Ryzen 5 6600H, 6 threads | fp16 features | 640 | 890 ms | 957 ms | 1,034 ms | – | 1.5 GB |

Fitting one category takes 92 s on the laptop GPU at 0.8 GB of VRAM and 3.8 GB of host memory.

## Training data

VisA (Amazon, 2022), official one-class split, development categories `pcb1`, `macaroni2`,
`capsules`; the remaining nine categories were held out for confirmation. Fit on the training
normals after carving 15% off as validation (seed 0): 768 / 765 / 461 images. No defect label is
used at fit time; validation normals (136 / 135 / 81) set the threshold, the input guards and
nothing else.

## Evaluation data

VisA's official test split: 200 / 200 / 160 images (100 / 100 / 100 defective), pixel masks for
every defect. These test sets were used throughout development, so their numbers are optimistic;
the held-out rows above are the unbiased estimate.

## Ethical considerations

Industrial parts only: no people, no personal data. The harm is economic and physical — a missed
defect ships a faulty part, a false alarm stops a line. The 1% false-alarm request favours
precision, and at it the model misses 16–61% of defective development images, so it is specified
as a screening aid with a human in the loop. A deployment that values escapes more should lower τ
on its own validation normals and say so.

## Limitations

- **One model per category and rig.** No cross-category use; a moved fixture, a new camera mode or
  a lighting change breaks the operating point long before it breaks the ranking (E08): a quarter
  to two thirds of a stop of over-exposure or a blur of 2–4 native pixels sends false alarms to
  45–100%. The guards refuse those inputs, but they cannot see a frame shift or new content at the
  frame edge.
- **The false-alarm bound is not guaranteed in practice.** It assumes exchangeable normals;
  capture-session debris broke it on `pcb1` (4.3% vs 0.73%) and on held-out `pipe_fryum`, and a
  larger calibration pool did not fix it ([negative results](reports/negative-results.md) #2).
- **Recall at the operating point is low** on `macaroni2` (0.39), where the image maximum is set by
  normal variation of the tray and pasta, not by the defects.
- **The development test sets were used for selection**, so the development numbers carry an
  optimism bias; only E12 and H1 are free of it, and both used up their data.
- **Compute limits.** Native resolution (≈ 1,400–1,500 px) was not evaluated; 960 and 1280 px were
  (E8 adopts 960 px with `layer2` as an unconfirmed candidate v2,
  [configs/recipes/candidate-v2-patchcore-layer2-960.yaml](configs/recipes/candidate-v2-patchcore-layer2-960.yaml)).
  The anomalib comparison (C2) ran at 320 px only.
- **Bounded by the negative results:** the from-scratch autoencoder never approached this model
  (0.17–0.67 image AUROC behind); object-region scoring, larger banks, k-NN averaging and timm
  weights were tried and not adopted ([negative results](reports/negative-results.md)).

## Attribution

This project uses the VisA dataset, provided by Amazon under the Creative Commons Attribution 4.0
International License (CC BY 4.0): <https://creativecommons.org/licenses/by/4.0/>

> Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, Onkar Dabeer. "SPot-the-Difference
> Self-Supervised Pre-training for Anomaly Detection and Segmentation." ECCV 2022.
> arXiv:2207.14315

The memory banks are derived from VisA images; no images are redistributed. MPDD (CC BY-NC-SA 4.0)
was used for evaluation only. Full attribution: [ATTRIBUTION.md](ATTRIBUTION.md).
