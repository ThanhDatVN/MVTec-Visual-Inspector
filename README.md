# MVTec Visual Inspector

Industrial surface-defect **detection and localization** trained on normal images only
(cold-start / one-class anomaly detection), benchmarked under a pre-registered protocol,
with an explainability case book, a robustness study, and a containerized demo service.

> **Status:** protocol v2 (after an external review, [docs/13](docs/13-project-review-and-research-roadmap.md)).
> A PatchCore reference-agreement check (E00), the baseline ladder (E01), aggregation and
> calibration studies (E02/E03) and a resolution/memory study (E04) have run on three VisA
> development categories; nine further categories are frozen for confirmation
> ([ADR-10](docs/07-risks-and-decisions.md)).
> Results: [reports/E00-E04-development-findings.md](reports/E00-E04-development-findings.md).

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
| 07 | [Risks & decision log](docs/07-risks-and-decisions.md) | Risk register, ADR-1 … ADR-10 |
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

## Headline results (VisA development categories, protocol v2)

PatchCore reference configuration; operating point from a conservative rank threshold at a 1%
false-alarm request. Development evidence only — these categories shaped the hypotheses. Details,
controls and caveats: [reports/E00-E04-development-findings.md](reports/E00-E04-development-findings.md).

| category | image AUROC, 320 px (3 seeds) | image AUROC, 640 px (fixed 10 k bank) | AU-PRO@0.05, 320 → 640 px | realized FPR at 640 px (bound) |
|----------|------|------|------|------|
| `pcb1` | 0.944 ± 0.003 | **0.979** | 0.734 → 0.862 | **5.0%** (0.73%) |
| `macaroni2` | 0.708 ± 0.011 | **0.887** | 0.645 → 0.882 | 2.0% (0.74%) |
| `capsules` | 0.710 ± 0.044 | **0.916** | 0.430 → 0.853 | 1.7% (1.22%) |

What the controls behind this table say:

- **Input resolution is the largest effect, and it needs no larger memory bank.** From 320 to
  640 px image AUROC rises by +0.03 / +0.19 / +0.25 (paired bootstrap intervals exclude zero),
  and a bank fixed at 10,000 entries matches one that grows 4× with the pixel count while halving
  fit and prediction time. Peak host memory at 640 px is ~4 GB, set by the candidate pool.
- **At a common 320 px, PatchCore was not measurably better than linear PCA at detection** on
  `macaroni2` and `capsules` — a statement about a resolution-matched comparison that E04 shows is
  PatchCore's worst case. On `pcb1` a colour histogram alone reaches 0.83.
- **The `pcb1` false-alarm bound fails at every resolution, and the reason is on the table.** The
  highest-scoring normals in validation and test are debris on the background felt; three flagged
  test normals show the same stray object. The upper tail of test normals is not exchangeable
  with validation (exact permutation p 0.005–0.03) while the bulk is. The rank rule itself is
  sound: calibrated on held-out test normals, it hits its bound.
- **Scoring only the object region** — estimated from training normals — lifts `pcb1` recall at
  the operating point from 0.46 to 0.81 at 320 px, but costs `capsules` 2 points of AUROC: a
  confirmation candidate, not a default.

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
