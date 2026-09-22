# MVTec Visual Inspector

Industrial surface-defect **detection and localization** trained on normal images only
(cold-start / one-class anomaly detection), benchmarked under a pre-registered protocol,
with an explainability case book, a robustness study, and a containerized demo service.

> **Status:** planning complete, implementation not started.
> This repository currently contains the project plan only. Read [docs/04-roadmap.md](docs/04-roadmap.md) first.

---

## The one-paragraph version

Given only defect-free images of three industrial parts, learn a model of "normal", then at
inference time produce (a) an image-level anomaly score and (b) a pixel-level anomaly map.
We build the ladder from a hand-written autoencoder up to PatchCore and current (2025–2026)
feature-based methods, measure everything under one frozen protocol, stress the winner with
optical and geometric corruptions, and ship it behind an API in a container.

## Why this is not a solved problem

The classic MVTec AD benchmark is saturated — current methods sit at ~99.6% image AUROC, so
it no longer separates good work from bad. We therefore benchmark on **MVTec AD 2**, where the
best published methods reach only ~31% AU-PRO@5% and where lighting shifts between train and
test are part of the dataset rather than something we have to simulate. See
[docs/02-metrics-and-baselines.md](docs/02-metrics-and-baselines.md) for the external baseline matrix.

## Plan documents

| # | Document | What it fixes |
|---|----------|---------------|
| 01 | [Charter & frozen protocol](docs/01-charter-and-protocol.md) | Objectives, the three categories, splits, leakage rules, what "done" means |
| 02 | [Metrics & baselines](docs/02-metrics-and-baselines.md) | Exact metric definitions, external baseline matrix, decision rules |
| 03 | [Method ladder & experiment matrix](docs/03-method-ladder-and-matrix.md) | Every model and ablation we will run, tier 0 → tier 5 |
| 04 | [Roadmap & schedule](docs/04-roadmap.md) | 12 phases, gates, deliverables, compute allocation |
| 05 | [Robustness protocol](docs/05-robustness-protocol.md) | Corruption suite, severities, reporting |
| 06 | [Engineering, MLOps & test strategy](docs/06-engineering-mlops-and-testing.md) | Repo layout, MLflow, pytest tiers, CI, Docker, API |
| 07 | [Risks & decision log](docs/07-risks-and-decisions.md) | Risk register, ADRs, open decisions |
| 08 | [Licensing & attribution](docs/08-licensing-and-attribution.md) | CC BY-NC-SA 4.0 compliance — read before publishing anything |
| 09 | [References](docs/09-references.md) | Bibliography with verification status |
| — | [Model card template](docs/templates/model-card.md) | Final deliverable skeleton |
| — | [Results table template](docs/templates/results-table.md) | Canonical reporting format |

## Stack

PyTorch · torchvision · OpenCV · MLflow · FastAPI + Gradio · Docker · pytest
Third-party comparators via [anomalib](https://github.com/open-edge-platform/anomalib) 2.2, clearly labelled as such.

## Compute

| Where | Hardware | Used for |
|-------|----------|----------|
| Laptop | RTX 3050 Laptop, **4 GB VRAM** | Development, unit/fixture tests, AE training ≤320², PaDiM, PatchCore ≤448² (fp16), all evaluation, API/Docker, demo |
| Google Colab | T4 16 GB (A100 when available) | Full-resolution and tiled PatchCore, EfficientAD/Dinomaly, the ablation sweep, robustness sweeps |

The 4 GB ceiling is a first-class design constraint, not an afterthought — see the VRAM budget in
[docs/06-engineering-mlops-and-testing.md](docs/06-engineering-mlops-and-testing.md#vram-and-memory-budget).

## Data & license — read this first

This project uses MVTec AD 2 and MVTec AD, both released under **CC BY-NC-SA 4.0**
(non-commercial, share-alike, attribution required). **No dataset image, mask, or derived
crop is ever committed to this repository.** CI runs on synthetic fixtures.
Full obligations and the attribution block: [docs/08-licensing-and-attribution.md](docs/08-licensing-and-attribution.md).

## Deliberately not a goal

We set **no accuracy target**. Targets picked before you know the achievable range are
theatre, and on AD 2 the achievable range is genuinely unknown for our compute budget.
Instead, [docs/02-metrics-and-baselines.md](docs/02-metrics-and-baselines.md#decision-rules) defines
*decision rules*: a pre-registered, testable statement of when a change is adopted, and against
which external anchors our numbers are read.
