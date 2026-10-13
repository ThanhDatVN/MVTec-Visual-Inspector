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

### ADR-12 — MPDD transfer probe: the frozen recipe on a second dataset
**Status:** Accepted; committed before any MPDD image is opened.

**Context.** Confirmation v1 (ADR-11) tested recipe transfer across VisA's object types — one
dataset, one capture rig. MPDD (six painted metal parts, MVTec-AD format) is a different camera,
factory and annotation team, which is the question a deployment poses. It cannot be fetched by
script (the authors share a SharePoint folder) and its licence is unspecified, so it runs on
Kaggle from an attached copy, for evaluation only.

**Decision.**
1. **Recipe:** unchanged — `configs/recipes/confirmation-v1-patchcore-640.yaml` composed after
   `configs/data/mpdd.yaml` (validation carved 15%, seed 0), seeds 0–2, `role = confirmation`.
   No parameter is changed for MPDD, including the 640 px input on its 1024 px images.
2. **Categories:** all six — `bracket_black`, `bracket_brown`, `bracket_white`, `connector`,
   `metal_plate`, `tubes`.
3. **Comparators** (seed 0): pixel PCA and the colour histogram at 320 px, as global-shortcut
   and linear floors.
4. **Reported, not tested:** per-category and macro image AUROC, AU-PRO@0.05, realized FPR
   against the effective bound, recall at the operating point; the share of categories where
   the recipe beats pixel PCA. MPDD is thinly reported in the literature and has no agreed
   protocol here, so no pass/fail rule is attached; the numbers are the result.
5. **No retuning,** as in ADR-11 §7.

**Consequences.** (+) The first evidence outside VisA's capture setup. (−) Six small categories
(tens of test images each), so per-category intervals are wide. (−) The dataset has to be
supplied by hand and its licence is not stated; no MPDD image or derived figure is published.

**Erratum (2026-10-11).** Two statements above were wrong, and neither touches the protocol. The
licence *is* stated: the authors' repository ships a CC BY-NC-SA 4.0 `LICENSE`, the same terms
as MVTec AD (attribution in [ATTRIBUTION.md](../ATTRIBUTION.md)). And the shared folder can be
fetched by script through SharePoint's guest API (`scripts/fetch_mpdd.py`, with a SHA-256
manifest). The no-publication rule for MPDD images and derived figures stays as decided.

### ADR-13 — Confirmation v2: candidate recipe v2 against v1 on MVTec AD 2
**Status:** Accepted, 2026-10-12. Committed and pushed **before any MVTec AD 2 archive was
downloaded**, so the git history timestamps the plan ahead of the data.

**Context.** Recipe v1 (ADR-11) is confirmed on nine held-out VisA categories (E12) and transfers
to MPDD (H1, ADR-12). The Gate E follow-up (E8) adopted, as a development result, `layer2`
features at 960 px — candidate v2 — with the largest gains on the categories where v1 was weakest
(`macaroni2` 0.887 → 0.979 image AUROC). Both earlier held-out sets are spent. MVTec AD 2, the
headline dataset of ADR-1 that ADR-8 deferred because it sat behind a registration, is now offered
as direct per-category downloads (checked 2026-10-12; CC BY-NC-SA 4.0). This project has never
loaded an AD 2 image, so AD 2 shaped nothing: it can confirm or refute v2.

**Decision.**

1. **Data.** All eight categories — `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`,
   `vial`, `wallplugs`, `walnuts` — fetched by `scripts/fetch_mvtec_ad2.py` (SHA-256 manifest).
   Each recipe is fitted on the category's `train/good`, calibrated on the provided
   `validation/good` (nothing carved), and evaluated once on `test_public`, the split with pixel
   ground truth. The private splits are read only for §6. Data config
   [`configs/data/mvtec_ad2.yaml`](../configs/data/mvtec_ad2.yaml).
2. **Recipes, unchanged files, seeds 0–2, `role = confirmation`:** v1,
   [`confirmation-v1-patchcore-640.yaml`](../configs/recipes/confirmation-v1-patchcore-640.yaml);
   v2, [`candidate-v2-patchcore-layer2-960.yaml`](../configs/recipes/candidate-v2-patchcore-layer2-960.yaml).
   Implementation `af89b5297f4d` (the E8 code with the GPU-memory fix, which changes no
   computation). Floors, seed 0, 320 px long side: pixel PCA and the colour histogram, as in
   ADR-12. Runs execute on the laptop; a run that fails for host or GPU memory is rerun unchanged
   on a Kaggle T4 and the venue recorded. A category whose arms are incomplete is reported as
   missing and counts as not significant.
3. **Confirmatory hypotheses**, analysed by
   [`scripts/ad2_confirmation_analysis.py`](../scripts/ad2_confirmation_analysis.py) (committed
   with this ADR, dry-run on the E8 development runs only):
   - **H3, detection.** Per category, mean image AUROC over seeds 0–2 of v2 minus that of v1,
     paired class-stratified image bootstrap (10,000 resamples), two-sided (count + 1)/(n + 1)
     p-value, Holm over the 8 categories at α = 0.05. *Holds* if at least 5 of 8 are
     significantly positive and none significantly negative.
   - **H3b, localization** (AU-PRO@0.05, the AD 2 headline metric). *Holds* if v2's mean over
     seeds exceeds v1's on at least 7 of 8 categories (one-sided exact sign test, p = 0.035).
   - **Verdict:** v2 is *confirmed* if H3 and H3b both hold — it then replaces v1 as the
     recommended recipe; *partially confirmed* if one holds; *not confirmed* otherwise, and v1
     stays. Whatever the verdict, both recipes' numbers are reported for every category.
4. **Descriptive only:** AU-PRO@0.30, pixel AUROC, SegF1 at the validation 3σ threshold, recall
   and realized FPR at the operating point against the effective bound (AD 2's 19–48 validation
   normals cannot support a 1% request; the rank rule relaxes to 1/(n+1) and flags it), the exact
   tail test per category and recipe (Holm over 16), pooled false alarms against the expected
   count, the floors, fit time and time per image.
5. **No retuning.** No parameter of either recipe changes for AD 2, including the input
   resolution on images far larger than VisA's. A surprising result starts a new development
   cycle, not a revision of this one.
6. **Official evaluation (optional, after §3).** For the recipe the verdict selects (v2 if
   confirmed, else v1), seed 0: float16 anomaly maps at native resolution for `test_private` and
   `test_private_mixed`, and binary masks at the validation mean + 3σ of the map (our
   `OP-3SIGMA`, also the baseline MVTec's utilities suggest), checked with MVTec's
   `check_and_prepare_data_for_upload.py`. The account and the upload are the repository owner's;
   the server's numbers are reported as received. Comparing `test_private` with
   `test_private_mixed` is the real-lighting evidence the synthetic suite (E08) could not give
   (G5), and is exploratory.

**Consequences.** (+) A confirmation on the dataset the project was designed for, with real
lighting variation behind the server. (+) H3 and H3b test the claim E8 makes, on data that shaped
nothing. (−) ~30 GB, fetched one category at a time. (−) Small validation sets make the operating
point loose (2.0–5.0% bounds). (−) CC BY-NC-SA 4.0: non-commercial, and no AD 2 image or derived
figure is published here.

**Outcome (2026-10-12, [E13](../reports/E13-confirmation-v2-mvtec-ad2.md)):** 64 runs, none
failed. H3 does not hold (1 category significantly positive, 1 negative; macro +0.004) and H3b
does not hold (v2 higher AU-PRO on 6 of 8). **v2 is not confirmed; v1 stays.** Both recipes'
operating points fail on AD 2 (160 and 195 false alarms against 29.2 expected), and `test_public`
is from now on development data.

---

### ADR-14 — Candidate v3 (v1 at 1280 px): development gate on `test_public`, held-out test on MVTec's server

**Status:** Accepted, 2026-10-13. Committed and pushed before E20 runs and before any server
result exists for either recipe.

**Context.** On AD 2 `test_public` (development since E13), E19 found that recipe v1 at a 1280 px
long side instead of 640 lifts `sheet_metal` image AUROC 0.580 → 0.871 (object bootstrap, Holm
p 0.003; seeds 1–2 in progress) and AU-PRO@0.05 0.18 → 0.35, and `fabric` AU-PRO 0.15 → 0.52.
AD 2's headline metric is pixel-level (AU-PRO@0.05, SegF1), which MVTec's server computes on the
private splits that nothing here has seen. v1 has been exported for that server (S1) and awaits
the owner's upload.

**Decision.**

1. **Recipe v3,** frozen as
   [`candidate-v3-patchcore-1280.yaml`](../configs/recipes/candidate-v3-patchcore-1280.yaml): v1
   with `long_side: 1280` (and `batch_size: 1`, which changes no computed value).
2. **E20, development gate.** v3, seed 0, on the five categories E19 did not run (`can`,
   `fruit_jelly`, `vial`, `wallplugs`, `walnuts`); with E19's three, all eight against E18-ref
   (v1, 640 px) at the same implementation (`5ab1a8956c23`), analysed by `scripts/e19_analysis.py`
   extended to the eight categories. v3 goes to the server if **no category is significantly
   worse in image AUROC** (object bootstrap, Holm over 8) **and macro AU-PRO@0.05 is higher** than
   v1's. Development evidence only; its purpose is to keep a regression off the server.
3. **S2, export.** If the gate passes: v3, seed 0, exported exactly as S1 (ADR-13 §6: float16
   maps at native resolution, masks at the validation mean + 3σ), checked with MVTec's
   `check_and_prepare_data_for_upload.py`. The owner uploads both S1 (v1) and S2 (v3), one
   submission each.
4. **H4, held-out.** From the server's per-category numbers, transcribed into
   `reports/studies/ad2-server/server_results.csv` and analysed by
   [`scripts/ad2_server_analysis.py`](../scripts/ad2_server_analysis.py) (committed with this ADR):
   v3's AU-PRO@0.05 on `test_private` exceeds v1's on **at least 7 of 8** categories (one-sided
   exact sign test, p = 0.035). *Confirmed* if it holds: v3 then replaces v1 for AD 2-like parts
   (large images, thin defects); *not confirmed* otherwise, and v1 stays. SegF1, the
   `test_private_mixed` split and each recipe's private → mixed drop are descriptive; if the
   server reports only means, H4 is not evaluable and the means are reported as received.
5. **No retuning** between the gate and the server, and no second submission of either recipe.

**Consequences.** (+) The first held-out test of a localization claim on the dataset the project
was designed for. (+) The `test_private` vs `test_private_mixed` comparison measures real lighting
change on data that shaped nothing. (−) Two of the owner's evaluation budget. (−) 1280 px doubles
fit time and host memory; still within the laptop.

**Outcome (2026-10-13, [E14–E20](../reports/E14-E15-ad2-development.md) §2):** the gate **fails**:
`vial` is significantly worse (−0.142, Holm p 0.028) although macro image AUROC rises 0.676 →
0.731 and macro AU-PRO@0.05 0.351 → 0.473 (higher on all eight categories). v3 is not exported,
H4 is not tested, and v1 remains the recipe; S1 is the only submission.

**S1 result (2026-10-13, owner's upload):** v1 AU-PRO@0.05 37.64 / 35.11, SegF1 26.42 / 23.01,
ClassF1 73.91 / 76.21 on `test_private` / `test_private_mixed` (server means; per-category numbers
not displayed). ([report](../reports/E14-E15-ad2-development.md), S1.)

---

### ADR-15 — Candidate v4 for MVTec's server: tiled DINOv3 fused with WRN50, masks per server metric

**Status:** Accepted, 2026-10-13. Committed and pushed before the v4 export and before any server
result for it. The last submission the protocol allows (two; S1 was the first).

**Context.** S1 put recipe v1 at AU-PRO@0.05 37.64 / 35.11 and SegF1 26.42 / 23.01 on
`test_private` / `test_private_mixed` — above every baseline of the AD 2 paper and about half of
the challenge systems (VAND 3.0 winner ISVL 53.81 / 51.43 SegF1; SuperADD, VAND 4.0, 57.42 /
54.35). E21 (development, `test_public`) rebuilt the localization pipeline from the published
training-free methods (SuperAD, SuperADD) and measured each server metric separately, with a proxy
that reproduces the server's numbers for v1 within −1.2 to +3.2 points
(`scripts/mask_rule_study.py`; `regular` captures for the private split, all captures for the
mixed one).

**Decision.**

1. **Maps (AU-PRO).** Two components, each fitted on `train/good` (seed 0):
   [`candidate-v4-component-dinov3l.yaml`](../configs/recipes/candidate-v4-component-dinov3l.yaml)
   — DINOv3 ViT-L/16 blocks 5, 11, 17, 23 over 640 px tiles of the image scaled by 0.625, a
   density-subsampled 100k bank per block, brightness augmentation 0.8–1.2 — and
   [`candidate-v3-patchcore-1280.yaml`](../configs/recipes/candidate-v3-patchcore-1280.yaml)
   (frozen by ADR-14). Each map is divided by the 95th percentile of its own validation pixels at
   a quarter of native resolution; the submitted map is their maximum.
2. **Masks (SegF1).** Threshold: the fused validation pixels' 95th percentile × 1.25. Shape:
   `MaskRule` — line closing at 16 orientations, radius 26 px at quarter resolution, kept where
   the map exceeds 0.8 × the threshold, holes filled, 1 px erosion.
3. **Image decision (ClassF1).** `mark = 0.9`: an image whose mask is empty but whose map
   exceeds 0.9 × the threshold receives one quarter-resolution pixel at its maximum. The server
   derives the image decision from mask emptiness; this encodes an image-level decision at a
   lower threshold than the pixel one, at a cost of 16 pixels per marked image.
4. **Frozen as one global setting:**
   [`candidate-v4-fusion.yaml`](../configs/recipes/candidate-v4-fusion.yaml). No category has its
   own setting. Chosen on `test_public`: ViT-L over ViT-H+ (four-category screen: AU-PRO 0.595
   vs 0.581, half the cost), max fusion over DINOv3 alone (AU-PRO 0.500 → 0.564) and over mean
   fusion, the gain and the mark from grids.
5. **Export** with `scripts/export_ad2_submission.py` from a clean worktree at this commit,
   checked with MVTec's `check_and_prepare_data_for_upload.py`; the owner uploads it once.
6. **H5, pre-registered predictions** on the server's means (development estimates in brackets,
   private / mixed: AU-PRO 58.1 / 56.4, SegF1 59.5 / 57.1, ClassF1 — / 80.6):
   - H5a: AU-PRO@0.05 at least **10 points** above v1 on both splits (≥ 47.64 and ≥ 45.11);
   - H5b: SegF1 at least **20 points** above v1 on both splits (≥ 46.42 and ≥ 43.01);
   - H5c: ClassF1 not below v1's (73.91 / 76.21).
   v4 is *confirmed* as the project's localization recipe for AD 2 if H5a and H5b hold. The
   development estimates are selection-biased (D1: twice before, development effects shrank on
   held-out data), which is why the bars sit well below them.

**Consequences.** (+) The pipeline closest to the published state of the art that the 4 GB card
can run. (+) One configuration for every category, as VAND 4.0 requires. (−) About 2 s per image
on the RTX 3050 and two models per category; v1 stays the recipe where an image-level decision at a
calibrated false-alarm rate is what matters (VisA, MPDD, E12). (−) DINOv3 weights are Meta's
(DINOv3 License); SuperAD/SuperADD are reimplemented from their papers, no code copied (theirs is
CC BY-NC 4.0).

---

### Open decisions (resolve before the stated gate)

| # | Question | Resolve by |
|---|----------|------------|
| OD-1 | Does `test_public` contain lighting variation, or is that confined to `test_private_mixed`? Determines whether the native-lighting control needs a leaderboard submission. | **Resolved (E13):** the guards fitted on training normals refuse 20–83% of `test_public` normals in six of eight categories (0–8% of validation normals); `test_public` contains capture shifts. |
| OD-2 | Which classic-AD categories and which published PatchCore numbers form the G5 reproduction target, and at what tolerance? | G2 |
| OD-3 | Which AU-PRO FPR-normalization convention does the reference implementation use (normal images only, vs all normal pixels)? Pin it in a docstring. | G2 |
| OD-4 | Resolve the AD 2 "below 60% AU-PRO" vs ~31% AU-PRO@0.05 discrepancy — different integration limit, or different split? | **Resolved (2026-10-13):** different integration limit — the abstract's figure is AU-PRO@0.30 (EfficientAD 58.7 on TEST_priv, AD 2 paper Table IX). |
| OD-5 | DINOv2/DINOv3 ViT backbones in the PatchCore ablation: worth the patch-token plumbing, or cut for time? | **Resolved (E18):** built and screened; not carried (`rice` +0.33, `vial` −0.17, more lighting-sensitive). |
| OD-6 | Do we make a leaderboard submission at all? | **Resolved (ADR-13 §6, ADR-14):** v1 exported and checked (S1); v3 after its gate (S2); uploads are the owner's. |
