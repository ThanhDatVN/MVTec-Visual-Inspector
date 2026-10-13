# MVTec Visual Inspector

Industrial surface-defect **detection and localization** trained on normal images only
(cold-start / one-class anomaly detection), benchmarked under a pre-registered protocol,
with an explainability case book, a robustness study, and a containerized demo service.

> **Status:** protocol v2. Development (E00–E04) on three VisA categories selected a recipe —
> PatchCore at 640 px with a fixed 10k memory bank — which was frozen and pre-registered
> ([ADR-11](docs/07-risks-and-decisions.md)) and then run once on nine held-out categories (E12).
> Results: [confirmation](reports/E12-confirmation-v1.md) ·
> [development](reports/E00-E04-development-findings.md) · [robustness](reports/E08-robustness.md) ·
> [data efficiency](reports/E09-data-efficiency.md) · [MPDD transfer](reports/H1-mpdd-transfer.md) ·
> [Kaggle studies](reports/K1-kaggle-development.md) · [Gate E follow-up](reports/E8-gate-e-followup.md) ·
> [MVTec AD 2 confirmation](reports/E13-confirmation-v2-mvtec-ad2.md) ·
> [AD 2 development: lighting, adaptation, scoring](reports/E14-E15-ad2-development.md) · [case book](reports/case-book/README.md).

---

## The one-paragraph version

Given only defect-free training images, the detector learns what normal looks like and returns
an image-level anomaly score and a pixel-level map at inference time. The repository compares
simple baselines, a custom autoencoder, and PatchCore under one protocol; it then tests the frozen
recipe on held-out VisA categories and optical/geometric corruptions. The fitted detector can be
exported to the API or demo.

## Data

Two tracks, for a reason recorded in [ADR-8](docs/07-risks-and-decisions.md):

| | Dataset | Why | Status |
|---|---------|-----|--------|
| **Working** | **VisA** — 12 categories, 10,821 images, 1.93 GB | Downloadable today, no account, CC BY 4.0, official one-class split | In use |
| **Headline** | **MVTec AD 2** — 8 categories, 30.4 GB | Not saturated (best published AU-PRO@5% ≈ 31%), and the only dataset with a *real* lighting-shifted test split | In use since 2026-10-12 (confirmation v2, E13); now downloadable without registration |

The survey behind that choice — five datasets verified downloadable by HTTP probe, nine behind
accounts or application forms — is [docs/10-datasets.md](docs/10-datasets.md).

**Study categories (VisA):** `pcb1` (complex structure → resolution), `macaroni2` (multiple
instances, high normal variance → false positives), `capsules` (transparent, reflective → optics).
One per structural group; three PCBs would test one difficulty three times.

## Getting started

```bash
python -m venv .venv                                       # everything installs into the venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -e ".[notebook,dev]"

inspector fetch visa --out data/raw                       # 1.93 GB, no account
inspector audit -c configs/data/visa_pcb1.yaml --data-root data/raw/VisA_20220922
inspector eda   -c configs/data/visa_pcb1.yaml --data-root data/raw/VisA_20220922 \
                --categories pcb1 macaroni2 capsules
inspector run   -c configs/data/visa_pcb1.yaml -c configs/models/patchcore.yaml \
                --data-root data/raw/VisA_20220922 --categories pcb1 macaroni2 capsules --seeds 0 1 2 \
                --set preprocess.long_side=640 --set model.bank_ratio=null --set model.bank_size=10000
inspector results                                          # regenerate the table from reports/runs
inspector study --data-root data/raw/VisA_20220922 --roi-methods patchcore   # E02/E03, no refits
```

No dataset needed to run the tests — fixtures are generated procedurally:

```bash
pytest            # no GPU, no data needed
```

## Serving

A fitted detector is exported as an **artifact** — the memory bank plus a `meta.json` with the
hyperparameters, the input transform, the operating threshold frozen on that category's
validation normals, and provenance (no pickle). The service and the demo load an artifact and
never re-derive the threshold.

```bash
inspector export -c configs/data/visa_pcb1.yaml -c configs/recipes/confirmation-v1-patchcore-640.yaml \
                 --data-root data/raw/VisA_20220922 --category pcb1 --out artifacts/pcb1
inspector serve  --artifact artifacts/pcb1            # FastAPI: GET /health, POST /predict[?heatmap=true]
inspector demo   --artifact artifacts/pcb1            # Gradio (needs the [serve] extra)

docker build -t mvi-inspector .
docker run --rm -p 8000:8000 -v "$PWD/artifacts/pcb1:/artifact:ro" mvi-inspector
```

`/predict?explain=true` also names the nearest normal training patch to the peak — for a memory-bank
model, the most direct explanation of why a region scored high. `/predict` returns the score, the decision (`normal`, `anomalous`, or `refused` when the input
guards find the exposure or focus outside the training range), the rule `score > threshold`, the effective false-alarm bound
the threshold was calibrated to, the location of the map's peak, latency, and optionally a PNG
overlay on a colour scale fixed by the threshold — so a normal image looks calm instead of being
stretched to its own maximum.

Measured on a laptop (RTX 3050 4 GB, Ryzen 5 6600H; batch 1, 200 timed requests through the
service path): the frozen recipe answers in **99 / 102 ms** (p50 / p95) on the GPU with 0.6 GB of
GPU memory, and in 890 / 957 ms on the CPU; the 960 px candidate takes 196 / 198 ms and 0.75 GB.
`scripts/bench_inference.py`, [E8 §3](reports/E8-gate-e-followup.md). Intended use, operating points and
limitations: [MODEL_CARD.md](MODEL_CARD.md).

## Notebooks

| Notebook | Venue | Covers |
|----------|-------|--------|
| [`01_laptop_data_and_eda.ipynb`](notebooks/01_laptop_data_and_eda.ipynb) | Laptop, 4 GB | Acquisition, integrity audit, EDA and the resolution decision, Tier 0 floors, development PatchCore |
| [`02_kaggle_experiments.ipynb`](notebooks/02_kaggle_experiments.ipynb) | Kaggle T4 | Reference config, ablation sweep, Gate E follow-up, autoencoder, C2, MPDD — resumable across sessions |

Both are **generated** by `scripts/build_notebooks.py`, never hand-edited. Operational detail:
[docs/12-running-the-notebooks.md](docs/12-running-the-notebooks.md).

## Documents

| # | Document | What it fixes |
|---|----------|---------------|
| 01 | [Charter & frozen protocol](docs/01-charter-and-protocol.md) | Objectives, splits, leakage rules L1–L7, what "done" means |
| 02 | [Metrics & baselines](docs/02-metrics-and-baselines.md) | Metric definitions, external baseline matrix, decision rules |
| 03 | [Method ladder](docs/03-method-ladder-and-matrix.md) | Implemented and proposed method tiers |
| 04 | [Roadmap & gates](docs/04-roadmap.md) | Original phase plan and gates |
| 05 | [Robustness protocol](docs/05-robustness-protocol.md) | Corruption suite, severities, reporting |
| 06 | [Engineering & testing](docs/06-engineering-mlops-and-testing.md) | Current layout, run registry, CI, memory, serving |
| 07 | [Risks & decision log](docs/07-risks-and-decisions.md) | Risk register, ADR-1 … ADR-13 |
| 08 | [Licensing & attribution](docs/08-licensing-and-attribution.md) | **Read before publishing anything** |
| 09 | [References](docs/09-references.md) | Bibliography, each tagged ✅/◐/⚠ by verification status |
| 10 | [Dataset survey](docs/10-datasets.md) | What is downloadable today, and why VisA |
| 11 | [**Experiment plan & run register**](docs/11-experiment-plan.md) | Every planned run: purpose, venue, cost, decision rule |
| 12 | [Running the notebooks](docs/12-running-the-notebooks.md) | Laptop and Kaggle operation |
| 13 | [External review & research roadmap](docs/13-project-review-and-research-roadmap.md) | Historical review findings and research programme |
| 14 | [**Open problems & remedies**](docs/14-open-problems-and-remedies.md) | Every known gap, measured error and defect after E13, with the technique proposed for each |

## Compute

| Where | Hardware | Used for |
|-------|----------|----------|
| Laptop | RTX 3050, 4 GB VRAM, 16 GB RAM | Data work, EDA, Tier 0, the CAE, PatchCore up to 640 px, every study on stored predictions |
| Kaggle | T4 16 GB (free tier), **~30 GPU-h/week** | Reference configs, ablation sweep, autoencoder training |

Measured, not assumed: a WideResNet50-2 forward pass at VisA's native 1.5 MP costs under 1 GB of
VRAM; the binding constraint is host RAM for the patch matrix, which per-image candidate sampling
bounds and every run now records per stage. The binding
remote constraint is the weekly quota, not the session limit; the plan budgets ~41 GPU-hours of
the ~360 available.

## Headline results

**Confirmation on nine held-out VisA categories** (recipe and analysis pre-registered and pushed
before the runs; each category fitted and calibrated on its own normals; nothing retuned):

| | image AUROC | AU-PRO@0.05 | realized false alarms at a 1% request |
|---|---|---|---|
| PatchCore recipe, 640 px, fixed 10k bank (3 seeds) | **0.981** macro mean, 0.971–0.996 per category | 0.766 | 10 observed vs 19.7 expected over 27 runs |
| same recipe at 320 px | 0.949 | – | – |
| pixel PCA (linear floor) | 0.815 | – | – |

- **H2 confirmed (9/9):** PatchCore at 640 px beats the linear floor on every held-out category,
  macro +0.167 [+0.145, +0.189] image AUROC.
- **H1 not confirmed (4/9, rule required 6):** 640 px never hurt and gained +0.034 [+0.025, +0.043]
  on average, but most held-out categories were already at 0.96–0.99 at 320 px. Development had
  overestimated the effect (+0.19/+0.25) because its categories were hard at low resolution.
  *Exploratory:* the gain tracks the headroom left at 320 px (Spearman 0.93).
- **The conservative threshold holds on 8/9 categories.** Where it fails (`pcb1` in development,
  `pipe_fryum` in confirmation) the mechanism is the same: background debris from a capture
  session sits in the test normals' upper tail and not in validation — the same fibre appears in
  two `pipe_fryum` images, the same stray object in three `pcb1` images. The rank rule itself is
  sound: calibrated on held-out test normals, it hits its bound.

**Transfer to a second dataset** (MPDD: six painted metal parts, another camera, factory and
annotation team; pre-registered in ADR-12, recipe unchanged, 3 seeds): macro image AUROC **0.943**
and AU-PRO@0.05 0.861, at least 0.88 on every category, against 0.729 for pixel PCA; the recipe
beats PCA on 4 of 6 categories. MPDD's 8–43 calibration normals cannot support a 1% operating
point: bounds relax to 2.3–11.1%, and pooled false alarms (31 vs 28.4 expected) stay consistent
with them. [Report](reports/H1-mpdd-transfer.md).

**MVTec AD 2: the candidate recipe is not confirmed, and the recipe does not transfer** (E13,
pre-registered in ADR-13 before the download; 64 runs, `test_public`, not comparable with
published private-split numbers). Candidate v2 (`layer2`, 960 px) against v1: image AUROC +0.004
macro (one category up, one down), AU-PRO@0.05 0.357 → 0.409 but higher on only 6 of 8 categories
against a rule of 7 — v1 stays. Both are far below their VisA level (macro image AUROC 0.685, at
chance on `can` and `wallplugs`), and both operating points fail (160 and 195 false alarms against
29 expected): the guards show that 20–83% of the test normals lie outside the training normals'
exposure and focus range in six categories. [Report](reports/E13-confirmation-v2-mvtec-ad2.md).

**Why AD 2 breaks the operating point, and what does not fix it** (E14–E15, development).
`test_public` shows each object under the training lighting and under changed ones; the false
alarms are the changed light *direction and colour* (22.8% of those normals for v1), not exposure
(5–6%, at the bound). Five photometric remedies — per-image gain and standardization, an
exposure-augmented bank, flat-field normalization, a relit bank — were screened against
pre-registered rules and none carried; a moved specular highlight is not a gain or a smooth field.
Real normals halve the changed-lighting false alarms at the old threshold (E16, 41 → 19 of 187),
but at a threshold recalibrated on new-lighting normals they buy no recall (E17): on `can` the
changed-lighting normals and the defects are simply not separable with these features.
Resolution is the one lever that works on AD 2 (E19): v1 at 1280 px lifts `sheet_metal` from
0.63 to 0.86 image AUROC (3 seeds) and `fabric`'s AU-PRO from 0.15 to 0.52 — but applied to every
category it costs `vial` 0.14, so the pre-registered gate kept it off the server (ADR-14, E20).

**Held-out, MVTec's server (S1):** v1 scores AU-PRO@0.05 **37.6** on `test_private` and **35.1** on
`test_private_mixed`, SegF1 26.4 / 23.0 — above every baseline in the AD 2 paper (best: 30.8 AU-PRO,
21.8 SegF1, all at 256×256) and with the smallest lighting drop, but about half of the VAND 2025
challenge system RoBiS (67.3 / 51.0). ClassF1 (73.9 / 76.2) sits near the trivial reject-all floor,
as RoBiS's does. [Report](reports/E14-E15-ad2-development.md).

**Candidate v5 for the server (E21, ADR-16):** a tiled DINOv3 ViT-H+ nearest-neighbour model
(after the training-free VAND 4.0 winner SuperADD, reimplemented) fused with WRN50 at 1280 px,
masks tuned per server metric with one global rule (per-category thresholds lost out of sample).
On `test_public` (development, server-style scoring, mixed-lighting proxy): AU-PRO 0.351 → 0.555,
SegF1 0.205 → 0.609, ClassF1 0.776 → 0.802. Frozen and pre-registered before its export; the server
result is pending. [Report](reports/E21-ad2-localization.md).
The image-score failure is separate: it persists under the training lighting. The captures of one
object are one sample, so AD 2 intervals are computed over objects from here on.
[Report](reports/E14-E15-ad2-development.md).

**Robustness (E08, frozen threshold, development categories):** the ranking survives far longer
than the operating point. Sensor noise with JPEG down to quality 40, a 0.35× resize round-trip
and a 50% illumination falloff leave the false-alarm rate at its clean level; a quarter to two
thirds of a stop of over-exposure, or a blur of ~2–4 native pixels, sends it to 45–100% while
image AUROC drops by only 0.04. Two input guards (exposure and focus ranges from the training
normals) refuse every breaking corruption at or before its breaking severity, at a cost of 1 in
260 clean normals; the service returns `decision: refused` for them. A shift of 1–1.5% of the frame
also breaks it, but through the replicated border strip the shift leaves rather than the shift
itself, and the guards do not see it: content at the frame edge that no training image showed is
the open failure mode. [Report](reports/E08-robustness.md).

**Development findings that shaped the recipe** (three categories, details in the
[development report](reports/E00-E04-development-findings.md)): input resolution is the largest
effect and a fixed 10k bank costs nothing measurable; at 320 px PatchCore was not measurably
better than linear PCA on two of three categories; scoring only the object region helps where
background nuisance sets the threshold but is not a safe default; the from-scratch autoencoder
was budget-limited and its image maximum is dominated by high-frequency normal texture.

**Data efficiency (E09, development):** with the calibration normals held fixed, ten fitting
images at 640 px match or beat the full training set at 320 px on `macaroni2` and `capsules`;
image AUROC keeps improving up to the full training set, while localization is within 0.02
AU-PRO by 200 images. [Report](reports/E09-data-efficiency.md).

**Kaggle development studies** (126 runs, bit-identical across three sessions;
[report](reports/K1-kaggle-development.md)): resolution keeps paying past 640 px (+0.06 to +0.08
image AUROC on `macaroni2`/`capsules` at 960 px, for twice the cost) and `layer2` alone beats
`layer2+layer3` at no cost. Bank size,
k-NN averaging and re-weighting are flat. The from-scratch autoencoder at 640 px and 150 epochs
stays 0.17–0.67 behind PatchCore, closing that rung. Against anomalib v2.3.0's own pipeline (C2),
our PatchCore is not distinguishable after correction, at a fifth of the memory.

**Gate E follow-up** (E8, pre-registered, 3 seeds; [report](reports/E8-gate-e-followup.md)): the
screen replicated. `layer2` features at 960 px take `macaroni2` from 0.887 to **0.979** image
AUROC and `capsules` from 0.914 to 0.963, with AU-PRO@0.05 up 0.04–0.08 on every category; it is
the candidate recipe v2 (`configs/recipes/candidate-v2-patchcore-layer2-960.yaml`), unconfirmed
until it meets data that shaped nothing (MVTec AD 2). timm's backbone weights trade categories and
were not adopted.

## Findings from building the apparatus

Three came out before any model result existed:

1. **No downscale is safe for small-defect categories.** At `sheet_metal`'s real scale
   (4224×1056, 1.5–3.5 px scratches), resizing to 256 puts **100%** of defect regions below one
   pixel; 512 → 79%; 2048 → 27%. Only native resolution preserves the population, which makes
   tiling a prerequisite rather than a refinement.
2. **`OP-FPR1`'s 1% target is not always achievable** ([ADR-7](docs/07-risks-and-decisions.md)).
   A threshold from `n` validation scores floors the false-alarm rate at `1/(n+1)` — 5.0% for AD 2's
   `sheet_metal` (n=19). `numpy.percentile` does not complain; it returns a threshold delivering
   five times the advertised rate. Thresholds now use a conservative rank, `k = floor(α(n+1))`
   with a strict `>` ([ADR-9](docs/07-risks-and-decisions.md)), and an unachievable request is
   relaxed to `1/(n+1)` *and flagged*, never silently. VisA sharpens this: its floors land on
   **both sides** of 1% within one dataset.
3. **Localization and detection are separate abilities.** `pixel_pca` reached 0.985 pixel AUROC at
   0.53 image AUROC — good maps, bad map-to-scalar aggregation. Aggregation is now ablated
   separately from the model.

## Not a goal

We set **no accuracy target**. Targets chosen before the achievable range is known are theatre.
[docs/02 §3](docs/02-metrics-and-baselines.md#3-decision-rules-pre-registered) instead
pre-registers *decision rules*: a testable statement of when a change is adopted.

## License and attribution

Code: MIT. Data: **VisA** (CC BY 4.0) and **MVTec AD / AD 2** (CC BY-NC-SA 4.0). No dataset image
is committed to this repository, and a CI job checks the whole git history rather than only the
tip. Full obligations and the attribution block: [ATTRIBUTION.md](ATTRIBUTION.md) and
[docs/08](docs/08-licensing-and-attribution.md).

## Stack

PyTorch · torchvision · OpenCV · MLflow · FastAPI/Gradio · Docker · pytest
Third-party comparators via [anomalib](https://github.com/open-edge-platform/anomalib), labelled
`owner=lib` in every table.
