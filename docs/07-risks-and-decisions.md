# 07 — Risk Register and Decision Log

---

## 1. Risk register

Scored as Likelihood (L) × Impact (I), each 1–5. Anything at 12 or above has a mitigation that is
scheduled work, not an intention.

| ID | Risk | L | I | Score | Mitigation (scheduled) | Trigger / early warning |
|----|------|---|---|-------|------------------------|-------------------------|
| R1 | **Dataset access delayed** — registration or ~30 GB download fails or is slow | 3 | 5 | 15 | Start day 1 (task 0.1). Fall back to classic MVTec AD for the three categories and re-plan the robustness section around the synthetic suite only. | No data by end of week 0 |
| R2 | **4 GB VRAM OOM blocks the target resolution** | 4 | 4 | 16 | fp16 memmaps, chunked extraction, streaming coreset, tiled inference, Colab for ≥512² — all specified in [06](06-engineering-mlops-and-testing.md) §3 and scheduled in P5 | First OOM during P4 feature caching |
| R3 | **Colab quota exhausted mid-sweep** | 4 | 3 | 12 | Compute ledger; 2-hour run cap; checkpoint/resume on every run; prioritized run queue so the cheapest informative runs go first | Ledger exceeds 60% before P8 → trigger the `walnuts`→`can` swap |
| R4 | **Metric implementation subtly wrong** | 3 | 5 | 15 | Gate G2 validation against a reference implementation; analytic unit tests; golden regression files | Reference disagreement above 1e-3 |
| R5 | **Silent test-set leakage** (esp. anomaly-map normalization, L3) | 3 | 5 | 15 | Seven leakage tests; threshold provenance fields; an unusually good result is treated as a bug report until explained | A result that beats published SOTA by a wide margin. Treat it as a defect, not a breakthrough — it almost always is. |
| R6 | **PatchCore reproduction fails at G5** | 3 | 4 | 12 | Debug checklist in P5 (layers, 3×3 aggregation, coreset ratio, image-score re-weighting, native-resolution scoring); budget 3 extra days inside P5's two weeks | Classic-AD AUROC off by more than 1 point |
| R7 | **Scope creep** — a fourth category, a fifth method, one more ablation | 4 | 3 | 12 | Three categories frozen at G1; new ideas go to `reports/backlog.md`, never into the current phase | Any phase overruns by more than 3 days |
| R8 | **Results are uninteresting** (everything scores similarly) | 2 | 3 | 6 | The category choice already guards against this — three distinct failure physics. Null results are reported as results; the calibration question (P5.8) and the robustness-validity question (P7.6) produce findings regardless of the ranking. | — |
| R9 | **License violation** — dataset images pushed to a public repo | 2 | 5 | 10 | `.gitignore` + a CI license-guard job that fails the build; pre-commit hook; synthetic CI fixtures | The guard job failing is the system working |
| R10 | **Private-split overfitting** via repeated submissions | 2 | 3 | 6 | Hard cap of 2 submissions; `test_public` evaluation budget tracked in `reports/test_set_budget.md` | Budget tally approaching 40 per category |
| R11 | **Notebook drift** — real logic ends up living only in Colab | 3 | 4 | 12 | Notebooks are thin drivers by rule; CI runs only packaged code; a `notebooks/` line-count check in review | A notebook cell longer than ~20 lines |
| R12 | **Drive sync corrupts cached features or the dataset** | 2 | 4 | 8 | Re-verify manifest hashes after every sync; content-addressed feature cache so a corrupt entry is detected, not consumed | Hash mismatch |
| R13 | **Time runs out before serving and documentation** | 3 | 4 | 12 | The 8-week compressed lane in [04](04-roadmap.md) §4; P9/P10 are never the phases that get cut — an unmeasured, undocumented model is not a deliverable | End of P7 with P5 incomplete |
| R14 | **Published baseline numbers quoted wrongly** | 3 | 3 | 9 | The verification task in [02](02-metrics-and-baselines.md) §2.1, including the unresolved 60%-vs-31% AU-PRO discrepancy | — |

---

## 2. Decision log (ADRs)

Format: **ADR-n — Title** · Status · Context · Decision · Consequences.
Every reversal of a frozen protocol item gets a new ADR that names the results it invalidates.

---

### ADR-1 — MVTec AD 2 as the primary benchmark
**Status:** Accepted (planning)

**Context.** The scope permits classic MVTec AD or AD 2. Classic AD is saturated at ~99.6% image
AUROC; differences between methods there are within noise, and robustness must be entirely
simulated. AD 2 has published results in the 20–31% AU-PRO@0.05 range, an official normal-only
validation split, and real lighting-shifted test splits.

**Decision.** AD 2 is primary. Classic AD is used only as an implementation-correctness fixture.

**Consequences.** (+) Room for measurable progress; real distribution shift; honest thresholding;
an auditable held-out split. (−) ~30 GB download and registration; multi-megapixel images that
collide with 4 GB VRAM; no local GT for private splits, so our headline numbers are on
`test_public` and are **not** directly comparable to published `TEST_priv` numbers — a caveat that
must appear in every mixed table.

---

### ADR-2 — The three categories
**Status:** Accepted (planning)

**Context.** Exactly three categories, chosen from eight, to leave time for analysis.

**Decision.** `sheet_metal`, `fruit_jelly`, `walnuts` — selected to span three *different* failure
physics: resolution/small-defect, transparency/optics, and normal-variance/false-positives.

**Consequences.** Each category interrogates a different part of the system, so the ablations
generalize rather than repeating one lesson. Cost: all three are large images, and `walnuts` has
the largest training set, so this is the most compute-hungry admissible triple. The documented
`walnuts`→`can` swap rule in [01](01-charter-and-protocol.md) §3.3 is the release valve.

---

### ADR-3 — `test_public` is a development test set, not a held-out set
**Status:** Accepted (planning)

**Context.** AD 2's validation split contains only normal images, so no anomaly-aware
hyperparameter selection is possible without touching a labelled test split.

**Decision.** State plainly that `test_public` functions as a development set; bound its use
(40 evaluations per category); treat the private split as the only clean held-out number.

**Consequences.** Our reported numbers carry a stated optimism bias. Naming it is strictly better
than the common alternative of tuning on the test set while calling it held-out.

---

### ADR-4 — Own implementations for Tiers 0–3 and 5; library implementations for Tier 4
**Status:** Accepted (planning)

**Context.** The scope requires a custom baseline and PatchCore. Re-implementing EfficientAD,
Dinomaly, and INP-Former correctly would consume the entire schedule.

**Decision.** Implement Tiers 0–3 and 5 ourselves. Use anomalib for Tier 4 comparators, run through
our data pipeline and our metric code, tagged `owner=lib` in every table.

**Consequences.** Comparisons stay controlled (same data, same metrics). Any Tier 4 result is a
statement about a reference implementation at default hyperparameters, not about the method's
ceiling — and the report must say so.

---

### ADR-5 — No accuracy target; pre-registered decision rules instead
**Status:** Accepted (planning)

**Context.** The scope explicitly forbids an arbitrary accuracy target, and rightly so: on AD 2 the
achievable range at our compute budget is genuinely unknown.

**Decision.** Pre-register decision rules DR-1 … DR-5 in [02](02-metrics-and-baselines.md) §3.

**Consequences.** Adoption decisions are made by a stated rule with a significance test rather than
retrospectively. Cost: the rules are occasionally inconvenient — a change that "obviously helps" but
fails DR-1 is recorded as rejected. That inconvenience is the entire point.

---

### ADR-6 — No default train-time augmentation
**Status:** Accepted (planning)

**Context.** Augmentation is reflexive in supervised vision. In one-class anomaly detection it is
dangerous: widening the model of "normal" can absorb the very deviations the model must flag.

**Decision.** No augmentation by default. Augmentation appears only as an explicit Tier 5
experiment (lighting-robust memory augmentation), evaluated under DR-1.

**Consequences.** Baselines are clean and interpretable, and the augmentation question gets a
measured answer instead of an assumption.

---

### Open decisions (resolve before the stated gate)

| # | Question | Resolve by |
|---|----------|------------|
| OD-1 | Does `test_public` contain lighting variation, or is that confined to `test_private_mixed`? Determines whether the native-lighting control needs a leaderboard submission. | G1 |
| OD-2 | Which classic-AD categories and which published PatchCore numbers form the G5 reproduction target, and at what tolerance? | G2 |
| OD-3 | Which AU-PRO FPR-normalization convention does the reference implementation use (normal images only, vs all normal pixels)? Pin it in a docstring. | G2 |
| OD-4 | Resolve the AD 2 "below 60% AU-PRO" vs ~31% AU-PRO@0.05 discrepancy — different integration limit, or different split? | G2 |
| OD-5 | DINOv2/DINOv3 ViT backbones in the PatchCore ablation: worth the patch-token plumbing, or cut for time? | G5 |
| OD-6 | Do we make a leaderboard submission at all? | G8 |
