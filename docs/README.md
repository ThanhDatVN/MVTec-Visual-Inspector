# Research documents

These documents were written before or alongside the experiments: what the project set out to
test, how it would measure it, and what it decided at each gate. The results are in
[reports/](../reports/README.md), starting with the
[final report](../reports/00-final-report.md).

A pre-registration is the commit that added a decision rule here *before* the run it governs. The
git history of these files is therefore part of the evidence.

| # | document | what it fixes |
|---|---|---|
| 01 | [Charter and frozen protocol](01-charter-and-protocol.md) | Problem statement, splits and their roles, leakage rules L1–L7, the operating point |
| 02 | [Metrics and baselines](02-metrics-and-baselines.md) | Metric definitions, the external baseline matrix, decision rules DR-1 … DR-5 |
| 03 | [Method ladder](03-method-ladder-and-matrix.md) | Model tiers from trivial floors to retrieval and foundation features |
| 04 | [Roadmap and gates](04-roadmap.md) | The original phase plan and its gates G1–G5 |
| 05 | [Robustness protocol](05-robustness-protocol.md) | Corruption suite, physically scaled severities, reporting |
| 06 | [Engineering, MLOps and testing](06-engineering-mlops-and-testing.md) | Package layout, run registry and identity, CI, memory, serving |
| 07 | [Risks and decision log](07-risks-and-decisions.md) | Risk register and **ADR-1 … ADR-17**, including every candidate recipe and the two server submissions |
| 08 | [Licensing and attribution](08-licensing-and-attribution.md) | Dataset licences and what may be published. Read before publishing anything |
| 09 | [References](09-references.md) | Bibliography, each entry tagged by verification status |
| 10 | [Dataset survey](10-datasets.md) | Which datasets were downloadable, and why VisA, MPDD and MVTec AD 2 |
| 11 | [Experiment plan and run register](11-experiment-plan.md) | Every planned stage: purpose, venue, cost, and the decision rule written before it ran |
| 13 | [External review and research roadmap](13-project-review-and-research-roadmap.md) | Review findings F01–F12 that led to protocol v2, and the research programme |
| 14 | [Open problems and remedies](14-open-problems-and-remedies.md) | Every known gap (A1–M9), measured error, and the remedy tested for it (R1–R11) |

[templates/model-card.md](templates/model-card.md) is the template behind
[MODEL_CARD.md](../MODEL_CARD.md).
