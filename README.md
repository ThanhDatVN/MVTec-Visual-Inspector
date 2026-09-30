# MVTec Visual Inspector

Industrial surface-defect **detection and localization** trained on normal images only
(cold-start / one-class anomaly detection), benchmarked under a pre-registered protocol,
with an explainability case book, a robustness study, and a containerized demo service.

> **Status:** protocol v2. Development (E00–E04) on three VisA categories selected a recipe —
> PatchCore at 640 px with a fixed 10k memory bank — which was frozen and pre-registered
> ([ADR-11](docs/07-risks-and-decisions.md)) and then run once on nine held-out categories (E12).
> Results: [confirmation](reports/E12-confirmation-v1.md) ·
> [development](reports/E00-E04-development-findings.md) · [robustness](reports/E08-robustness.md) ·
> [case book](reports/case-book/README.md).

---

## The one-paragraph version

Given only defect-free images of three industrial parts, learn a model of "normal", then at
inference time produce (a) an image-level anomaly score and (b) a pixel-level anomaly map.
We build the ladder from a hand-written autoencoder up to PatchCore and current (2025–2026)
feature-based methods, measure everything under one frozen protocol, stress the winner with
optical and geometric corruptions, and ship it behind an API in a container.

## Data

Two tracks, for a reason recorded in [ADR-8](docs/07-risks-and-decisions.md):

| | Dataset | Why | Status |
|---|---------|-----|--------|
| **Working** | **VisA** — 12 categories, 10,821 images, 1.93 GB | Downloadable today, no account, CC BY 4.0, official one-class split | In use |
| **Headline** | **MVTec AD 2** — 8 categories, ~30 GB | Not saturated (best published AU-PRO@5% ≈ 31%), and the only dataset with a *real* lighting-shifted test split | Blocked on registration |

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

`/predict` returns the score, the decision (`score > threshold`), the effective false-alarm bound
the threshold was calibrated to, the location of the map's peak, latency, and optionally a PNG
overlay on a colour scale fixed by the threshold — so a normal image looks calm instead of being
stretched to its own maximum.

## Notebooks

| Notebook | Venue | Covers |
|----------|-------|--------|
| [`01_laptop_data_and_eda.ipynb`](notebooks/01_laptop_data_and_eda.ipynb) | Laptop, 4 GB | Acquisition, integrity audit, EDA and the resolution decision, Tier 0 floors, development PatchCore |
| [`02_kaggle_experiments.ipynb`](notebooks/02_kaggle_experiments.ipynb) | Kaggle P100 | Reference config, ablation sweep, autoencoder, robustness grid — resumable across sessions |

Both are **generated** by `scripts/build_notebooks.py`, never hand-edited. Operational detail:
[docs/12-running-the-notebooks.md](docs/12-running-the-notebooks.md).

## Documents

| # | Document | What it fixes |
|---|----------|---------------|
| 01 | [Charter & frozen protocol](docs/01-charter-and-protocol.md) | Objectives, splits, leakage rules L1–L7, what "done" means |
| 02 | [Metrics & baselines](docs/02-metrics-and-baselines.md) | Metric definitions, external baseline matrix, decision rules |
| 03 | [Method ladder](docs/03-method-ladder-and-matrix.md) | Every model, tier 0 → tier 5 |
| 04 | [Roadmap & gates](docs/04-roadmap.md) | 12 phases, gates, compute allocation |
| 05 | [Robustness protocol](docs/05-robustness-protocol.md) | Corruption suite, severities, reporting |
| 06 | [Engineering & test strategy](docs/06-engineering-mlops-and-testing.md) | Repo layout, MLflow, pytest tiers, VRAM budget |
| 07 | [Risks & decision log](docs/07-risks-and-decisions.md) | Risk register, ADR-1 … ADR-11 |
| 08 | [Licensing & attribution](docs/08-licensing-and-attribution.md) | **Read before publishing anything** |
| 09 | [References](docs/09-references.md) | Bibliography, each tagged ✅/◐/⚠ by verification status |
| 10 | [Dataset survey](docs/10-datasets.md) | What is downloadable today, and why VisA |
| 11 | [**Experiment plan & run register**](docs/11-experiment-plan.md) | Every planned run: purpose, venue, cost, decision rule |
| 12 | [Running the notebooks](docs/12-running-the-notebooks.md) | Laptop and Kaggle operation |
| 13 | [External review & research roadmap](docs/13-project-review-and-research-roadmap.md) | Review findings F01–F12 and the E00–E12 programme |

## Compute

| Where | Hardware | Used for |
|-------|----------|----------|
| Laptop | RTX 3050, 4 GB VRAM, 16 GB RAM | Data work, EDA, Tier 0, the CAE, PatchCore up to 640 px, every study on stored predictions |
| Kaggle | P100 16 GB, **~30 GPU-h/week** | Reference configs, ablation sweep, autoencoder training |

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

**Robustness (E08, frozen threshold, development categories):** the ranking survives far longer
than the operating point. Sensor noise with JPEG down to quality 40, a 0.35× resize round-trip
and a 50% illumination falloff leave the false-alarm rate at its clean level; a quarter to two
thirds of a stop of over-exposure, or a blur of ~2–4 native pixels, sends it to 45–100% while
image AUROC drops by only 0.04. Deployment needs exposure and focus guards on the input.
[Report](reports/E08-robustness.md).

**Development findings that shaped the recipe** (three categories, details in the
[development report](reports/E00-E04-development-findings.md)): input resolution is the largest
effect and a fixed 10k bank costs nothing measurable; at 320 px PatchCore was not measurably
better than linear PCA on two of three categories; scoring only the object region helps where
background nuisance sets the threshold but is not a safe default; the from-scratch autoencoder
was budget-limited and its image maximum is dominated by high-frequency normal texture.

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
