# 07 — Risk Register and Decision Log

---

## 1. Risk register

Scored as Likelihood (L) × Impact (I), each 1–5. Anything at 12 or above has a mitigation that is
scheduled work, not an intention.

| ID | Risk | L | I | Score | Mitigation (scheduled) | Trigger / early warning |
|----|------|---|---|-------|------------------------|-------------------------|
| R1 | **Dataset access delayed** — registration or ~30 GB download fails or is slow | 3 | 5 | 15 | Start day 1 (task 0.1). Fall back to classic MVTec AD for the three categories and re-plan the robustness section around the synthetic suite only. | No data by end of week 0 |
| R2 | **4 GB VRAM OOM blocks the target resolution** | 4 | 4 | 16 | fp16 memmaps, chunked extraction, streaming coreset, tiled inference, Kaggle for ≥512² — all specified in [06](06-engineering-mlops-and-testing.md) §3 and scheduled in P5 | First OOM during P4 feature caching |
| R3 | **Kaggle quota exhausted mid-sweep** (~30 GPU-h/week is the binding limit, not the 12-hour session) | 4 | 3 | 12 | Compute ledger; stages sized under 4 h; JSONL checkpoint keyed by run ID so a killed session resumes; cheapest informative runs first; budget uses ~41 of ~360 available hours, leaving room to re-run | Ledger exceeds 60% before P8 → trigger the category swap |
| R4 | **Metric implementation subtly wrong** | 3 | 5 | 15 | Gate G2 validation against a reference implementation; analytic unit tests; golden regression files | Reference disagreement above 1e-3 |
| R5 | **Silent test-set leakage** (esp. anomaly-map normalization, L3) | 3 | 5 | 15 | Seven leakage tests; threshold provenance fields; an unusually good result is treated as a bug report until explained | A result that beats published SOTA by a wide margin. Treat it as a defect, not a breakthrough — it almost always is. |
| R6 | **PatchCore reproduction fails at G5** | 3 | 4 | 12 | Debug checklist in P5 (layers, 3×3 aggregation, coreset ratio, image-score re-weighting, native-resolution scoring); budget 3 extra days inside P5's two weeks | Classic-AD AUROC off by more than 1 point |
| R7 | **Scope creep** — a fourth category, a fifth method, one more ablation | 4 | 3 | 12 | Three categories frozen at G1; new ideas go to `reports/backlog.md`, never into the current phase | Any phase overruns by more than 3 days |
| R8 | **Results are uninteresting** (everything scores similarly) | 2 | 3 | 6 | The category choice already guards against this — three distinct failure physics. Null results are reported as results; the calibration question (P5.8) and the robustness-validity question (P7.6) produce findings regardless of the ranking. | — |
| R9 | **License violation** — dataset images pushed to a public repo | 2 | 5 | 10 | `.gitignore` + a CI license-guard job that fails the build; pre-commit hook; synthetic CI fixtures | The guard job failing is the system working |
| R10 | **Private-split overfitting** via repeated submissions | 2 | 3 | 6 | Hard cap of 2 submissions; `test_public` evaluation budget tracked in `reports/test_set_budget.md` | Budget tally approaching 40 per category |
| R11 | **Notebook drift** — real logic ends up living only in a notebook | 3 | 4 | 12 | Notebooks are generated from `scripts/build_notebooks.py` and never hand-edited; a test fails the build on a class definition, an `nn.Module` or a `torch.optim` call in any cell | The regeneration test failing, which means someone edited a notebook directly |
| R12 | **Sync corrupts cached features or the dataset** | 2 | 4 | 8 | `inspector fetch --verify` re-hashes against the recorded manifest after every sync; content-addressed feature cache so a corrupt entry is detected, not consumed | Hash mismatch |
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

### ADR-7 — `OP-FPR1` targets the achievable rate, not 1%
**Status:** Accepted (P2). **Supersedes** the 1% figure in the original protocol §4.4.

**Context.** The protocol defined the primary operating point as "the 99th percentile of scores
over `validation/good`", targeting 1% false alarms. Implementing it at P2 exposed that the target
is not reachable. For a threshold set as the k-th largest of `n` exchangeable validation scores, a
fresh normal sample exceeds it with probability `k/(n+1)` — a distribution-free result. The most
extreme available choice, `k = 1` (the sample maximum), therefore floors the achievable
false-alarm rate at `1/(n+1)`.

MVTec AD 2's validation splits hold 19–48 normal images per category, so the floors are:

| category | validation `n` | minimum achievable FPR |
|----------|----------------|------------------------|
| `sheet_metal` | 19 | 5.0% |
| `fruit_jelly` | 37 | 2.6% |
| `walnuts` | 48 | 2.0% |

A simulation over 19 validation draws confirms the realized rate of the nominal "p99" threshold is
about 5%, not 1%. Nothing errors: `numpy.percentile` happily interpolates between the top two
order statistics and returns a number, and the report would have claimed 1% while the system
delivered five times that.

**Decision.**
1. The operating point is set **per category at its achievable floor**, not at a fixed 1%.
2. Thresholds are computed from the **order statistic** `k = ceil(target * (n+1))`, which is
   distribution-free, rather than from an interpolated percentile that assumes a tail shape 19
   points cannot support.
3. `from_validation_fpr(..., strict=True)` **raises** on an unachievable target. Clamping silently
   would reproduce the original failure.
4. The expected rate, the chosen `k`, `n`, and the floor are all recorded in the threshold's
   provenance and carried into the results table.

**Consequences.** (+) The reported target and the delivered rate agree, and the gap between
expected and realized FPR now measures genuine distribution shift rather than an estimator
artifact. (+) The constraint is visible: buying a lower false-alarm rate requires more normal
validation images, which is an actionable statement to make in the model card. (−) Operating
points differ across categories, so the per-category table cannot be collapsed into one FPR
column without stating each category's target. (−) `sheet_metal` is calibratable only to 5%,
which will look poor next to published work that quotes 1% without checking whether it was
achievable.

---

### ADR-8 — VisA as the immediately available working dataset
**Status:** Accepted (P2/P3). **Amends** ADR-1 without replacing it.

**Context.** ADR-1 chose MVTec AD 2 as the primary benchmark on its merits, which stand. What it
did not weigh was that AD 2 is behind an MVLogin registration and a ~30 GB download, so it blocks
*every* experiment until that completes. A survey of alternatives (docs/10), with availability
probed by HTTP rather than read off a paper, found five reputable datasets fetchable today: VisA
(1.93 GB), BTAD (1.23 GB), KolektorSDD2 (0.85 GB), KolektorSDD (0.10 GB) and MPDD. The newest and
largest benchmarks — Real-IAD Variety, MANTA, Kaputt — are all behind application forms.

**Decision.** VisA becomes the dataset the project runs on now. AD 2 remains the headline
benchmark and the target for the final report once its download completes. Study categories:
`pcb1`, `macaroni2`, `capsules` — one from each of VisA's three structural groups, mirroring the
one-failure-mode-per-category logic of ADR-2. MPDD is added as a **transfer probe**: the final
configuration applied once, unchanged, with no tuning.

**Consequences.**

(+) Experiments start immediately instead of waiting on a download and a registration.
(+) CC BY 4.0 rather than CC BY-NC-SA, which removes the sharpest constraint on outputs — though
the project keeps behaving as if the stricter reading applied while a licence discrepancy between
the AWS registry and third-party docs is unresolved.
(+) VisA's official `1cls.csv` split keeps results comparable with published work.
(+) It sharpens ADR-7 instead of dodging it: VisA's per-category validation sizes put the
achievable FPR floor on *both sides* of 1% within a single dataset, which demonstrates the finding
better than AD 2 alone.

(−) **VisA has no lighting-shifted test split.** Robustness on VisA is therefore entirely
synthetic, and the Phase P7 validity question — does synthetic corruption predict real
distribution shift? — cannot be answered on it at all. That question is the single most valuable
output of P7 and it stays blocked on AD 2. This is the real cost of the amendment and it must be
stated in the report rather than glossed.
(−) Lower resolution than `sheet_metal`, so the resolution axis is less punishing and the
tiling-is-mandatory finding cannot be reproduced on VisA.
(−) Two datasets in flight means two sets of numbers, and every table must say which.

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
