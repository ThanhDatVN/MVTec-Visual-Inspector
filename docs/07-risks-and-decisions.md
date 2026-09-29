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
**Decisions 2 and 3 are superseded by ADR-9:** the `ceil` rank rule overshoots the target
(review F01), and "raise on an unachievable target" became an explicit, recorded policy.

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

### ADR-9 — The operating threshold is a conservative split-conformal rank
**Status:** Accepted (protocol v2). **Supersedes** ADR-7 decisions 2 and 3.

**Context.** ADR-7 set `k = ceil(target * (n+1))` and took the k-th largest validation score as
the threshold. The external review (docs/13, F01) showed this is the wrong direction: `ceil` picks a
*larger* `k`, a *lower* threshold, and an expected false-alarm rate `k/(n+1)` that can exceed the
target — at `n = 136` and a 1% target it gives `k = 2` and 1.46%, not ≤ 1%. Decisions also used
`>=`, so a test score tied with the threshold raised an alarm, which the rank argument does not
cover.

**Decision.**
1. `k = floor(α(n+1))`; the threshold is the k-th largest calibration score; a test image is flagged
   iff `score > threshold` (strict). The expected rate `k/(n+1) ≤ α` for a fresh normal that is
   exchangeable with the calibration normals.
2. When `α(n+1) < 1` no rank meets the target. The behaviour is a declared **policy**, never an
   accident: `reject` raises; `relax` uses `k = 1` (the sample maximum) and records an effective
   rate `1/(n+1) > α` with `target_met = false`; `always_accept` returns an infinite threshold
   that never alarms and says so. The evaluation default is `relax`, because development tables must
   still show recall at the best achievable point — and the table carries the flag.
3. Every result records the requested rate, `n`, `k`, the effective rate, `target_met`, the
   comparator, and the realized test FPR, as separate columns. They answer different questions and
   are never merged into one "FPR".
4. The guarantee is stated with its limits wherever it is quoted: it is **marginal** over
   calibration draws, not conditional on one fitted model and one calibration set; it assumes the
   calibration and test normals are exchangeable; and its resolution is `1/(n+1)`.

**Consequences.** (+) A reported target is now an upper bound in expectation rather than a
number the rule can overshoot. (+) VisA's large categories (`n ≈ 135`) calibrate to 0.73–0.74% at a
1% request; its small ones (`n = 68`, and `capsules` at 81) cannot reach 1% and are visibly
relaxed. (−) The realized test FPR of one fitted model can still exceed the bound — the first
corrected PatchCore pilot showed 4% on `pcb1` against a 0.73% bound. Whether that is sampling
noise or a validation-vs-test shift is tested explicitly (E03, `analysis.normal_shift`), not
assumed. (−) Conservative thresholds cost recall; the study reports how much.

---

### ADR-10 — Protocol v2: development and confirmation roles, frozen before inspection
**Status:** Accepted (protocol v2).

**Context.** The three study categories have shaped the EDA, the resolution choice, the
hypotheses and the review. Their test results are development evidence and can no longer support
a held-out claim, however the configuration is frozen afterwards (docs/13 §5.1). Separately, the
review found that results were keyed by human-readable strings that omitted settings, so a
resumed run could silently reuse a result produced under different code or evaluation settings
(F04), and that every pre-fix PatchCore number used the uncorrected scoring (F02, F03).

**Decision.**
1. **Development categories:** VisA `pcb1`, `macaroni2`, `capsules`. All comparisons, ablations,
   failure analysis and recipe selection happen here, and their numbers are labelled development.
2. **Confirmation categories, frozen now:** VisA `candle`, `cashew`, `chewinggum`, `fryum`,
   `macaroni1`, `pcb2`, `pcb3`, `pcb4`, `pipe_fryum`. Up to this decision only their split
   *counts* (from the official `1cls.csv`) have been read — no image, mask, defect statistic or
   model output. Until confirmation: no EDA on their test splits, no per-image viewing, no runs.
   The docs/10 idea of `cashew` as an easy sanity row is withdrawn; it is a confirmation category.
3. **A confirmation run** uses a recipe frozen beforehand (config plus `implementation_id`),
   fits each category on its own training normals, calibrates on its own validation normals, runs
   once with `role = "confirmation"`, and is reported whatever it shows. Retuning after seeing a
   confirmation result starts a new development cycle, and those categories lose the confirmation
   label.
4. **Run identity.** A run's id hashes the dataset split listing, the model's effective
   hyperparameters (defaults included), the transform, the evaluation settings, the seed, the
   role, the protocol version, and an `implementation_id` over the result-producing source
   modules. A completed run is reused only under an identical id.
5. **Protocol version 2** starts with the F01–F11 corrections. Earlier VisA numbers are archived
   as exploratory (`reports/archive/`) and are not compared with v2 numbers in any table.

**Consequences.** (+) A confirmation table is possible at all, and cheap: nine categories, one
frozen recipe. (+) Reuse is safe by construction rather than by naming discipline. (−) Any
change to data, model, metric or evaluation code invalidates reuse; analysis and table-formatting
modules are deliberately outside the hash so that studying results does not force refits.
(−) Nine confirmation categories are a single dataset from a single source: they test recipe
transfer across VisA's object types, not industrial generality.

---

### ADR-11 — Confirmation v1: frozen recipe and pre-registered analysis
**Status:** Accepted. Committed and pushed **before** any confirmation run, so the git history
timestamps the plan ahead of the data.

**Context.** Development (E01–E04, [report](../reports/E00-E04-development-findings.md)) selected
PatchCore at 640 px with a memory bank fixed at 10,000 entries, and produced three claims worth
confirming: resolution is the dominant factor; at that resolution PatchCore beats a linear floor;
and the conservative rank threshold holds its false-alarm bound except where the normal upper
tail is not exchangeable (`pcb1`). Everything about them was chosen on `pcb1`, `macaroni2` and
`capsules`, so none of it is evidence yet.

**Decision.**

1. **Recipe.** [`configs/recipes/confirmation-v1-patchcore-640.yaml`](../configs/recipes/confirmation-v1-patchcore-640.yaml),
   composed after `configs/data/visa_pcb1.yaml` (VisA layout, official split, 15% validation
   carve with seed 0), evaluation protocol v2 from `configs/base.yaml`, implementation id
   `a2d2dcd0c230`, seeds 0, 1, 2, `role = confirmation`. The result-producing code is the E04
   code: `git diff 469e9b3 -- src/` touches only the presentation modules and the
   implementation-id function itself, which now normalizes line endings (a CRLF worktree and an
   LF working tree had produced different ids for identical sources). The analysis is frozen too:
   [`scripts/confirmation_analysis.py`](../scripts/confirmation_analysis.py), smoke-tested on the
   development categories only.
2. **Comparators, run once each with `role = confirmation`:** the same recipe at 320 px
   ([`confirmation-v1-patchcore-320.yaml`](../configs/recipes/confirmation-v1-patchcore-320.yaml),
   seed 0); pixel PCA and the colour histogram at their defaults and a 320 px long side, as in
   E01; the random scorer, seed 0, as a sanity control.
3. **Categories:** `candle`, `cashew`, `chewinggum`, `fryum`, `macaroni1`, `pcb2`, `pcb3`, `pcb4`,
   `pipe_fryum`. Each is fitted on its own training normals, calibrated on its own validation
   normals, and evaluated once on its official test split.
4. **Confirmatory hypotheses.** One family of 18 tests, Holm-adjusted together at α = 0.05; each
   test is a paired, class-stratified image bootstrap (10,000 resamples) of an image-AUROC
   difference with a two-sided (count + 1)/(n + 1) p-value:
   - **H1, resolution** (9 tests): AUROC(640 px, seed 0) − AUROC(320 px, seed 0) > 0.
   - **H2, against a linear floor** (9 tests): mean AUROC(640 px, seeds 0–2) − AUROC(pixel PCA) > 0.
   - **Dataset-level reading**, decided now: H1 (or H2) is *confirmed* if at least 6 of the 9
     categories are significant in the positive direction after Holm **and** none is
     significant in the negative direction; *not confirmed* otherwise. The macro-mean difference
     and its interval are reported either way.
5. **Calibration (a separate, descriptive family).** Per category, the exact tail-permutation
   p-value of the seed-0 false alarms above the validation threshold (Holm over 9), the realized
   FPR against the effective bound for all seeds, and the pooled count of false alarms against
   the pooled expected count. Development predicts that most categories hold their bound; any
   category that does not gets the `pcb1` diagnostics (calibration-source control, locations of
   the flagged peaks).
6. **Descriptive only, not tested:** AU-PRO@0.05 and @0.30, recall at the operating point, seed
   spread, resources, the histogram-shortcut flag (> 0.80 image AUROC), and — explicitly
   exploratory — the object-region variant computed on the stored maps.
7. **No retuning.** Nothing is rerun or changed after a result is seen. A run that fails (for
   example out of memory) is reported as a failure. A code fix is admissible only if it cannot
   change any completed run's numbers; otherwise the affected category loses its confirmation
   status, and the report says so.

**Consequences.** (+) The first evidence in this project that was not used to choose what it
evaluates. (+) A negative result is as reportable as a positive one, and the thresholds for
calling either were fixed in advance. (−) Nine categories from one dataset and one capture setup:
this confirms recipe transfer across VisA's object types, not industrial generality. (−) H1 uses
one seed per arm; the 3-seed spread at 640 px on development (≤ 0.006 AUROC) is small against the
effects being tested, but the test does not include fitting variance.

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
