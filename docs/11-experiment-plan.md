# 11 — Experiment Plan and Run Register

The executable counterpart to [03-method-ladder-and-matrix.md](03-method-ladder-and-matrix.md).
Where that document says *what kinds of method* the study covers, this one says *which runs
happen*, in what order, on which machine, at what cost, and what decides whether each one was
worth doing.

**Scope of this document:** VisA, three study categories, laptop + Kaggle. The MVTec AD 2 plan is
unchanged and resumes when that download completes; see [ADR-8](07-risks-and-decisions.md).

A run that is not in this register does not get reported. That is not bureaucracy — it is the
only defence against the pattern where thirty exploratory runs happen, three look good, and those
three become the paper.

---

## 1. Compute budget

| Venue | Hardware | Constraint | Reserved for |
|-------|----------|------------|--------------|
| Laptop | RTX 3050, **4 GB VRAM** | Hard OOM ceiling around 1 MP single-pass | Data work, EDA, Tier 0, development-scale PatchCore, all evaluation, the demo |
| Kaggle | P100 16 GB or T4×2 | **~30 GPU-h/week**, 12 h interactive session, ~9 h on commit, 20 GB working dir | Reference configurations, the ablation sweep, autoencoder training, robustness grid |

**The binding constraint is the weekly Kaggle quota, not the session limit.** 30 hours a week
across a 12-week plan is ~360 GPU-hours total, and §3 budgets ~78 of them. The headroom is
deliberate: sweeps overrun, and a plan that consumes its entire quota on the first pass has no
capacity to re-run anything after a bug is found.

**Session-limit discipline.** Every Kaggle stage writes to a JSONL checkpoint keyed by
configuration, so a killed session resumes instead of restarting. Any single stage is sized to
finish inside ~4 hours, well under the 9-hour commit cap, so a stage never straddles a session
boundary.

---

## 2. Run-ID scheme

```
{stage}|{variant}|{category}|{resolution}|s{seed}
```

e.g. `ref|patchcore|pcb1|320|s0`, `abl-backbone|resnet18|macaroni2|320|s0`.

The ID is the checkpoint key, the MLflow run name, and the `notes` column of `results.csv`. One
string ties a row in a table to the exact configuration that produced it, which is what makes the
table auditable rather than merely printed.

---

## 3. The register

Costs are estimates to be replaced by measurements in `reports/compute_ledger.md`. An estimate
that is never reconciled against the actual is how a budget silently doubles.

### Stage A — Data and protocol (laptop, ~2 h, no GPU)

| # | Run | Purpose | Decides |
|---|-----|---------|---------|
| A1 | Fetch + hash VisA | Reproducible acquisition | Gate G0 |
| A2 | Split audit, 3 categories | Rules L1–L7 | **Blocks everything downstream** |
| A3 | EDA: defect-size distribution, resolution impact, intensity shift | The resolution decision | `RESOLUTION` for every later stage |
| A4 | pHash near-duplicate calibration | Rule L6 threshold, justified rather than guessed | Reported, not gating |

**Gate A.** No fit runs until A2 passes on all three categories. A leaked split produces a better
score, not an error, so it has to be excluded mechanically before any number exists.

### Stage B — Tier 0 floors (laptop, ~1 h)

| # | Run | n | Purpose |
|---|-----|---|---------|
| B1 | `random` × 3 categories × 3 seeds | 9 | **Gate G2 control.** Image AUROC ≈ 0.5, AU-PRO@L ≈ L/2. A deviation means the metric is broken and every later number is wrong. |
| B2 | `mean_intensity`, `histogram` × 3 × 3 | 18 | Is any category separable by a global statistic? If so, a deep model's success there measures the lighting. |
| B3 | `pixel_pca` × 3 × 3 | 9 | The honest floor for the autoencoder: a conv AE that cannot beat linear PCA has learned nothing a linear projection did not. |

**Gate B.** B1 within tolerance. Any category where B2 exceeds 0.80 image AUROC is flagged in
bold in the final report.

### Stage C — Development PatchCore (laptop, ~2 h)

| # | Run | n | Purpose |
|---|-----|---|---------|
| C1 | `resnet18`, 256², 1% coreset × 3 categories | 3 | Confirm the pipeline runs end to end on real data and measure laptop peak VRAM |
| C2 | Same, 3 seeds, best category | 3 | Coreset seed variance — how much of a difference is noise? |

**Gate C.** Peak VRAM recorded. C2's seed spread establishes the noise floor below which no
ablation difference in Stage E may be called real.

### Stage D — Reference configuration (Kaggle, ~4 GPU-h)

| # | Run | n | Purpose |
|---|-----|---|---------|
| D1 | PatchCore WRN50-2, `layer2+3`, 320², 1% coreset × 3 categories × 3 seeds | 9 | **The row every ablation is measured against.** Frozen; changes need an ADR. |

**Gate D.** Mean ± std reported per category. If the seed std exceeds C2's noise floor by a wide
margin, the coreset pre-subsample is too aggressive — investigate before sweeping.

### Stage E — Ablation sweep (Kaggle, ~25 GPU-h)

Staged, one axis at a time from D1, single seed. A full grid is ~10⁴ runs; this is ~60 and
captures the main effects.

| Axis | Values | n | Question |
|------|--------|---|----------|
| E1 backbone | resnet18, resnet50, wide_resnet50_2 | 9 | Does backbone capacity pay for itself at this resolution? |
| E2 layers | L2, L3, L2+L3, L2+L3+L4 | 12 | Feature granularity vs semantic level. **Tests the layer3-only hypothesis** from docs/09 — treat that claim as unverified. |
| E3 resolution | 224, 320, 448 | 9 | Expected to be the largest single effect |
| E4 coreset ratio | 0.001, 0.01, 0.1, 0.25 | 12 | The accuracy/memory/latency Pareto figure |
| E5 k | 1, 3, 9 | 9 | Score robustness to memory-bank noise |
| E6 projection dim | 128, 384, 1024 | 9 | Memory reduction at fixed accuracy |

**E7 — joint re-vary.** Take the two axes with the largest measured effects and vary them
together (~12 runs). One interaction, chosen by evidence rather than guessed in advance.

**Gate E.** Every axis has a written one-paragraph conclusion. Winners re-run at 3 seeds before
any adoption decision.

### Stage F — Tier 1 autoencoder (Kaggle, ~12 GPU-h)

| # | Run | n | Purpose |
|---|-----|---|---------|
| F1 | loss ∈ {L2, SSIM, L2+SSIM} × 3 categories × 3 seeds | 27 | The most instructive Tier 1 ablation. L2 blurs; a blurry reconstruction errs everywhere rather than at the defect. |
| F2 | latent ∈ {32, 128} at the best loss | 18 | Capacity vs the identity-function failure |
| F3 | residual ∈ {raw, multiscale} | 18 | Raw residuals are dominated by edge misalignment |
| F4 | aggregation ∈ {mean, max, top-k mean} | 27 | **From the P2 finding:** `pixel_pca` reached 0.985 pixel AUROC at 0.53 image AUROC. Map quality and map-to-scalar aggregation are separate abilities and must be ablated separately. |

**Gate F.** The headline comparison — what the pretrained prior bought, in points, per category —
is computed and written down, including any category where it bought nothing.

### Stage G — Robustness (laptop for inference, ~6 h wall-clock)

Models are fitted once on clean data and **never re-thresholded**. Re-thresholding under
corruption answers a different and much easier question.

| # | Run | n | Purpose |
|---|-----|---|---------|
| G1 | 7 corruptions × 5 severities × 3 categories × best PatchCore | 105 | Degradation curves |
| G2 | Same for best AE | 105 | A reconstruction model and a memory model can fail in opposite directions; two strong models would be less informative |
| G3 | Clean baselines | 6 | The reference the curves are read against |
| G4 | Translation, reported separately | 30 | Registration sensitivity. Excluded from the headline mean: on a fixed rig a geometric shift is out of distribution by design |

**Gate G.** Three quantities reported per cell: relative degradation, **realized FPR at the frozen
threshold**, and recall. The second is the deployment-critical one — a model can hold its AUROC
while its alarm rate triples.

**G5 — the validity question, blocked on AD 2.** Does synthetic illumination corruption predict
behaviour under *real* lighting shift? Answerable only against AD 2's `test_private_mixed` split.
Recorded here as blocked rather than quietly dropped, because it is the highest-value hour in P7
and its absence is a real limitation of a VisA-only study.

### Stage H — Transfer probe (laptop, ~1 h)

| # | Run | n | Purpose |
|---|-----|---|---------|
| H1 | Final configuration on MPDD, unchanged, no tuning | 6 | Does the method survive a different camera, factory and annotator? |

**Gate H.** One paragraph. Tuning on MPDD would turn three study categories into nine and is
exactly the scope creep R7 exists to prevent.

### Stage I — Explainability and reporting (laptop, ~8 h)

| # | Deliverable | Spec |
|---|-------------|------|
| I1 | Case book, ≥15 cases | ≥4 per category, ≥5 taxonomy tags, **≥4 failures of the best model** |
| I2 | Nearest-normal retrieval panels | The most convincing explanation a memory-based model offers |
| I3 | Pareto figure | Accuracy vs p95 latency vs memory-bank size, with the 4 GB feasibility boundary marked |
| I4 | Failure taxonomy counts | Turns a scalar into an actionable profile |
| I5 | Negative results | `reports/negative-results.md`, one entry per abandoned line |

---

## 4. Budget summary

| Stage | Venue | Runs | Est. cost |
|-------|-------|------|-----------|
| A | laptop | — | 2 h |
| B | laptop | 36 | 1 h |
| C | laptop | 6 | 2 h |
| D | **Kaggle** | 9 | 4 GPU-h |
| E | **Kaggle** | ~72 | 25 GPU-h |
| F | **Kaggle** | ~90 | 12 GPU-h |
| G | laptop | 246 (inference only) | 6 h |
| H | laptop | 6 | 1 h |
| I | laptop | — | 8 h |
| **Total** | | **~465 runs** | **~41 GPU-h on Kaggle**, ~20 h laptop |

41 GPU-hours is under a week and a half of Kaggle quota, leaving room for the re-runs that a real
study needs. If Stage E overruns, cut E5 and E6 before cutting seeds: a three-seed result on four
axes is worth more than a one-seed result on six.

---

## 5. Pre-registered decision rules

Restated here so a decision can be checked against them without leaving this document.

**DR-1, adoption.** A change enters the main line only if: AU-PRO@0.05 improves on ≥2 of 3
categories and regresses the third by ≤1 point; the paired test gives p < 0.05 after
Holm–Bonferroni within its family; p95 latency stays within budget; peak VRAM stays under 3.5 GB.
Passing (1)–(2) but failing (3)–(4) makes it an *accuracy-only variant*, reported but not
deployed.

**DR-2, "better".** Never on a mean. The paired test, the effect size, and the per-category table
are all shown. Where a method wins on one category and loses on another, that disagreement is the
finding.

**DR-3, stopping.** After its budgeted hours, a line that has met DR-1 on no category is
abandoned and written into `reports/negative-results.md` with configs tried.

**DR-4, the headline table.** One row per method at its best *validation-selected* configuration.
Anything test-selected is marked `(test-selected)` and read as an upper bound.

**DR-5, comparability.** VisA numbers and MVTec AD 2 numbers never share a table without a caption
saying so. Our VisA numbers are on its official test split; published AD 2 numbers are on
`TEST_priv`. These are not comparable quantities.

**DR-6 (new, from ADR-7), operating points.** Each category's operating point targets its own
achievable FPR floor `1/(n+1)`, which on VisA ranges 0.73%–1.45%. No single headline FPR is
quoted across categories, and every reported FPR carries the target it was calibrated to.

---

## 6. Known gaps in this plan

Stated because a plan that lists no gaps has not been read carefully.

| Gap | Effect | Status |
|-----|--------|--------|
| **AUPIMO not implemented** | The paired tests in DR-1/DR-2 currently pair over *categories* (n=3), which has very little power. Per-image AUPIMO would pair over ~200 images. | Blocks the statistical strength of every adoption decision. Highest-priority gap. |
| **No reference-implementation metric check on real data** | Gate G2's analytic half passes; the external-agreement half does not exist | Gate G5 |
| **No PatchCore reproduction of a published number** | We cannot yet distinguish "our PatchCore" from "PatchCore" | Gate G5 |
| **Single-pass VRAM budget is an assumption** | The resolution decision uses 1.0 MP from docs/06 §3, not a measurement | Gate G5, Stage C |
| **Robustness validity unanswerable on VisA** | The strongest evidence P7 could produce is unavailable | Blocked on AD 2 |
| **Tier 4 comparators not planned here** | EfficientAD, RD++, Dinomaly are in docs/03 but have no stage yet | Add after Stage F, budget ~15 GPU-h |
| **No AD 2 leaderboard submission planned** | Our only truly held-out number would be missing | Decide at Gate E (OD-6) |

---

## 7. Execution order

```
A ──► B ──► C ──► D ──► E ──► F ──► G ──► H ──► I
│     │     │     │     │
│     │     │     │     └─ winners re-run at 3 seeds before DR-1 is applied
│     │     │     └─ frozen; an ADR is required to change it
│     │     └─ establishes the noise floor that Stage E differences are judged against
│     └─ Gate G2: if the random control is off, stop and fix the metric
└─ Gate A: if a split leaks, nothing downstream means anything
```

Never start a stage before the gate below it passes. The temptation to jump to Stage D is exactly
how a project ends up with an impressive number and no way to tell whether it is real.
