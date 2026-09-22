# MVTec Visual Inspector

Industrial surface-defect **detection and localization** trained on normal images only
(cold-start / one-class anomaly detection), benchmarked under a pre-registered protocol,
with an explainability case book, a robustness study, and a containerized demo service.

> **Status:** Phases P0–P2 complete and committed; the models and experiment plan for P3–P7 are in
> place. 243 tests pass, `ruff` and `mypy` clean. No result on real data has been produced yet —
> everything measured so far is on synthetic fixtures and proves only that the code is correct.

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
python -m pip install -e ".[torch,track,dev]"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

inspector fetch visa --out data/raw                       # 1.93 GB, no account
inspector audit -c configs/data/visa_pcb1.yaml --data-root data/raw/VisA_20220922
inspector eda   -c configs/data/visa_pcb1.yaml --data-root data/raw/VisA_20220922 \
                --categories pcb1 macaroni2 capsules
inspector run   -c configs/data/visa_pcb1.yaml --data-root data/raw/VisA_20220922 --seeds 0 1 2
```

No dataset needed to run the tests — fixtures are generated procedurally:

```bash
pytest            # 243 tests, no GPU, no data
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
| 07 | [Risks & decision log](docs/07-risks-and-decisions.md) | Risk register, ADR-1 … ADR-8 |
| 08 | [Licensing & attribution](docs/08-licensing-and-attribution.md) | **Read before publishing anything** |
| 09 | [References](docs/09-references.md) | Bibliography, each tagged ✅/◐/⚠ by verification status |
| 10 | [Dataset survey](docs/10-datasets.md) | What is downloadable today, and why VisA |
| 11 | [**Experiment plan & run register**](docs/11-experiment-plan.md) | Every planned run: purpose, venue, cost, decision rule |
| 12 | [Running the notebooks](docs/12-running-the-notebooks.md) | Laptop and Kaggle operation |

## Compute

| Where | Hardware | Used for |
|-------|----------|----------|
| Laptop | RTX 3050, **4 GB VRAM** | Data work, EDA, Tier 0, PatchCore ≤448², all evaluation, robustness inference, demo |
| Kaggle | P100 16 GB, **~30 GPU-h/week** | Reference configs, ablation sweep, autoencoder training |

The 4 GB ceiling is a first-class design constraint — see the memory-bank arithmetic in
[docs/06 §3](docs/06-engineering-mlops-and-testing.md#3-vram-and-memory-budget). The binding
remote constraint is the weekly quota, not the session limit; the plan budgets ~41 GPU-hours of
the ~360 available.

## Findings so far

Three came out of building the measurement apparatus, before any model result existed:

1. **No downscale is safe for small-defect categories.** At `sheet_metal`'s real scale
   (4224×1056, 1.5–3.5 px scratches), resizing to 256 puts **100%** of defect regions below one
   pixel; 512 → 79%; 2048 → 27%. Only native resolution preserves the population, which makes
   tiling a prerequisite rather than a refinement.
2. **`OP-FPR1`'s 1% target is not always achievable** ([ADR-7](docs/07-risks-and-decisions.md)).
   A threshold from `n` validation scores floors the false-alarm rate at `1/(n+1)` — 5.0% for AD 2's
   `sheet_metal` (n=19). `numpy.percentile` does not complain; it returns a threshold delivering
   five times the advertised rate. Thresholds now use the distribution-free order statistic and
   refuse an unachievable target. VisA sharpens this: its floors land on **both sides** of 1%
   within one dataset.
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
