# 11 — Experiment Plan and Run Register

The executable counterpart to [03-method-ladder-and-matrix.md](03-method-ladder-and-matrix.md).
Where that document says *what kinds of method* the study covers, this one says *which runs
happen*, in what order, on which machine, at what cost, and what decides whether each one was
worth doing.

**Scope of this document:** VisA, three development categories plus nine frozen confirmation
categories, laptop + Kaggle. The MVTec AD 2 plan is unchanged and resumes when that download
completes; see [ADR-8](07-risks-and-decisions.md).

**Protocol v2 (ADR-9, ADR-10).** This register was revised after the external review
([docs/13](13-project-review-and-research-roadmap.md)). Every VisA number produced before the
F01–F11 corrections is archived as exploratory (`reports/archive/`) and is not compared with v2
numbers. Stages below carry the review's experiment ids (E00–E12) where they map onto them.

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

**Session-limit discipline.** Every run on every venue goes through `inspector.runner` and lands
in a run registry (`reports/runs/<run_id>/`), so a killed session resumes instead of restarting,
and a failed run leaves a `failed` record with its reason. Any single stage is sized to finish
inside ~4 hours, well under the 9-hour commit cap.

**Measured, not assumed (P3).** The laptop's VRAM is *not* the binding constraint: a
WideResNet50-2 forward pass at VisA's native ~1.5 MP costs under 1 GB at batch 1. What binds is
host RAM for the patch matrix (37 GB in fp16 at native resolution), which per-image candidate
sampling bounds. The corrected PatchCore reference at 320 px therefore runs on the laptop, and
Stage D moved there.

---

## 2. Run identity

The v1 scheme — a readable key `{stage}|{variant}|{category}|{resolution}|s{seed}` — omitted
settings, so a resumed run could silently reuse a result produced under different code or
evaluation settings (review F04). A run id is now a hash of a `RunSpec`:

| part | what it pins |
|---|---|
| dataset | layout, category, test split, validation carve fraction and seed, a hash of each split's image-id listing |
| model | name, class, and **effective** hyperparameters — constructor defaults included |
| transform | resize mode, long side, normalisation |
| evaluation | smoothing sigma and units, target FPR, threshold policy, AU-PRO limits and grid, negative-sampling budget and seed, metrics version |
| seed, role, protocol version | `development` or `confirmation`; protocol `2` |
| `implementation_id` | SHA-256 over the result-producing source modules (data, features, models, metrics, post-processing, evaluation, runner); analysis and presentation modules are excluded |

The registry directory holds `spec.json`, `result.json` (status, metrics, provenance: git sha,
dirty flag, dataset manifests, library versions, device), per-split predictions (`*.predictions.json`),
and raw anomaly maps (`*.maps.npz`, git-ignored). `inspector results` regenerates
`reports/results_registry.csv` from the registry; the table is never edited by hand.

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

### Stage C — Reference agreement (E00; laptop)

| # | Run | n | Purpose |
|---|-----|---|---------|
| C1 | Own PatchCore vs a pinned reference (anomalib) on identical small tensors | — | Scoring, re-weighting and distance units agree to a stated tolerance; the difference ledger (`PatchCore.reference_differences()`) lists what cannot match |
| C2 | The same on one development category, identical inputs and bank | 1 | Prediction-level agreement on real data, not only a metric match |

**Gate C.** A written tolerance and a met or failed verdict. Every later PatchCore conclusion is
conditional on it.

*Status.* **C1 met.** anomalib v2.3.0's scoring and k-center loop, excerpted verbatim as a test
oracle (`tests/reference/`), agree with ours on identical tensors: patch scores within
`rtol 1e-4, atol 1e-3` (the float32 error of anomalib's distance expansion), image scores within
`1e-4`, for support sizes 1, 3 and 9 (`tests/models/test_reference_agreement.py`). The coreset
matches its greedy order except that anomalib does not return its random start point — recorded
in the difference ledger. **C2 open:** it needs the full anomalib pipeline on real images, and its
descriptor (1536-d, no channel reduction) and coreset (all patches, sparse projection) differ from
ours by design, so C2 will measure the effect of those choices rather than agreement.

### Stage D — Baseline ladder (E01; laptop)

All at a common 320 px long side, so the comparison is not also a resolution comparison.

| # | Run | n | Purpose |
|---|-----|---|---------|
| D1 | Corrected PatchCore reference (`configs/models/patchcore.yaml`) × 3 categories × 3 seeds | 9 | **The row every ablation is measured against.** Frozen; changes need an ADR. Seeds vary candidate sampling and the coreset start. |
| D2 | CAE, L2 and SSIM loss × 3 × 3 | 18 | The from-scratch baseline; what the pretrained prior buys |
| D3 | Tier 0 floors at 320 px (Stage B rerun under protocol v2) | 18 | Same evaluator, same smoothing — the v1 Tier 0 table used a different sigma |

**Gate D.** Mean ± std over seeds per category, and the realized test FPR next to the effective
bound. The seed spread is the noise floor below which no Stage E difference is called real.

*Status: done (45 runs, commit `c53e7f6`).* Results and paired comparisons:
[reports/E00-E04-development-findings.md](../reports/E00-E04-development-findings.md) §2. The CAE
never triggered early stopping in 60 epochs; its rung is budget-limited, not measured at its best.

### Stage D′ — Aggregation and calibration on stored predictions (E02, E03; laptop, no refits)

| # | Study | Question |
|---|-------|----------|
| D′1 | E02: image score = stored (re-weighted) / max / mean / mean of top 0.1%, 1%, 5% of the map | Is the detection gap a map-quality problem or a map-to-scalar problem? Maps are fixed, so localization metrics must not move — an informative control. Paired class-stratified bootstrap per category. |
| D′2 | E03: rank rule vs mean+3σ vs median+3·MAD, on calibration subsets of 20/40/80 normals and the full pool, at 1/2/5% targets | Distribution of threshold, realized FPR and recall per rule and size; which targets a given `n` can support |
| D′3 | Exchangeability diagnostic | Are validation and test normals exchangeable? AUROC of validation-vs-test-normal scores (0.5 under exchangeability) and a Mann-Whitney p; a realized FPR far above the bound with a shift here is a split problem, not a threshold bug |

Both reuse the registry's predictions (`inspector.analysis`), so they cost minutes, not GPU-hours.

*Status: done.* Added along the way, because the `pcb1` operating point overshot its bound: an
exact tail-exchangeability test, a calibration-source control, a model-free train-proximity
check, the object-region aggregation domain (E02b) and localization by defect size. Findings §3–§6
and §8 of the report above; generated tables in `reports/studies/`.

### Stage D″ — Resolution under a fixed memory budget (E04; laptop)

| # | Run | n | Purpose |
|---|-----|---|---------|
| D″1 | PatchCore at 320 / 448 / 640 px × {1% bank, fixed 10 k bank}, seed 0 | 18 | Separate the resolution effect from the bank-size effect, with per-stage host RSS |
| D″2 | 640 px, fixed 10 k bank, seeds 1–2 | 6 | Seed spread of the candidate recipe |

*Status: D″1 done* — the largest effect measured (+0.03 / +0.19 / +0.25 image AUROC from 320 to
640 px, intervals excluding zero), with no measurable cost from fixing the bank at 10 k. *D″2
done:* 0.979 ± 0.001 / 0.884 ± 0.005 / 0.919 ± 0.006 over 3 seeds at ~4 GB peak host RSS. Report §7.
Native resolution is open; it needs a smaller candidate fraction or tiling to stay within 16 GB RAM.

### Stage E — Ablation sweep (Kaggle, ~25 GPU-h)

Staged, one axis at a time from D1, single seed. A full grid is ~10⁴ runs; this is ~50 and
captures the main effects. It is a **screen**: it nominates variants for 3-seed reruns and a
paired test, and adopts nothing by itself.

| Axis | Values | n | Question |
|------|--------|---|----------|
| E1 backbone | resnet18, resnet50 (vs WRN50-2 in D1) | 6 | Does backbone capacity pay for itself at this resolution? |
| E2 layers | L2, L3, L2+L3+L4 (vs L2+L3) | 9 | Feature granularity vs semantic level. **Tests the layer3-only hypothesis** from docs/09 — treat that claim as unverified. |
| E3 resolution | 224, 448 (vs 320) | 6 | Expected to be the largest single effect; the 448 row is bounded by host RAM, not VRAM |
| E4 bank ratio | 0.001, 0.01, 0.05, 0.1, all with `candidate_fraction = 0.25` | 12 | The accuracy/memory/latency Pareto figure. The bank is selected from the per-image candidates, so a ratio above `candidate_fraction` is capped — the v1 axis {0.1, 0.25} at 10% candidates would have measured the same bank twice (F08). `bank_effective` and `bank_capped` are logged per run. |
| E5 neighbours | k ∈ {3, 9} with `mean` reduction, k = 3 with `kth` | 9 | Score robustness to bank noise. With `nearest` reduction k does not enter the patch score at all, so the v1 axis k ∈ {1, 3, 9} would have been three copies of one run (F02). |
| E6 projection dim | 256, 512 (vs 1024) | 6 | Memory reduction at fixed accuracy |
| E6b re-weighting | off (vs on) | 3 | What the reference image-score re-weighting (F03) contributes on these categories |

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
| F4 | aggregation ∈ {mean, max, top-k mean} | 0 | **From the P2 finding:** `pixel_pca` reached high pixel AUROC at chance image AUROC. Map quality and map-to-scalar aggregation are separate abilities. Now done by E02 (Stage D′) on stored maps for every model, without refitting. |

**Gate F.** The headline comparison — what the pretrained prior bought, in points, per category —
is computed and written down, including any category where it bought nothing.

### Stage G — Robustness (laptop for inference, ~6 h wall-clock)

Models are fitted once on clean data and **never re-thresholded**. Re-thresholding under
corruption answers a different and much easier question. Both normal and anomalous test images
are corrupted — only normals can show a false alarm — each with its own corruption seed (a stable
hash of its path) shared across severities and methods, and all metrics come from the same
evaluator as the clean runs, so the clean and corrupted negative populations match (F09).

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

### Stage H′ — Confirmation (E12; laptop + Kaggle)

| # | Run | n | Purpose |
|---|-----|---|---------|
| H′1 | The frozen recipe on the nine confirmation categories (ADR-10), `role = confirmation`, 3 seeds where stochastic | 27 per method | Does the development conclusion hold on categories that shaped nothing? |

*Status: done ([report](../reports/E12-confirmation-v1.md)).* Pre-registered in ADR-11 (`4ff5a6f`);
63 runs. H2 confirmed 9/9; H1 not confirmed (4/9 against a rule of 6/9); calibration holds on 8/9.

**Gate H′.** The recipe and `implementation_id` are recorded *before* the first confirmation
run. Every category is reported, including failures and resource-limit outcomes, with the macro
mean, the worst category, and development vs confirmation shown separately. Retuning after this
point starts a new development cycle.

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
| C | laptop | — | 3 h |
| D | laptop | 45 | ~6 h wall-clock (measured in `reports/runs`) |
| D′ | laptop | — (stored predictions) | < 1 h |
| E | **Kaggle** | ~51 | ~20 GPU-h |
| F | **Kaggle** | 45 | ~12 GPU-h |
| G | **Kaggle** | ~108 cells (one fit per category) | ~4 GPU-h |
| H | laptop | 6 | 1 h |
| H′ | laptop + Kaggle | ~27 per method | ~6 GPU-h |
| I | laptop | — | 8 h |
| **Total** | | **~320 runs** | **~42 GPU-h on Kaggle**, ~22 h laptop |

Under a week and a half of Kaggle quota, leaving room for the re-runs a real study needs. If
Stage E overruns, cut E6 and E6b before cutting seeds: a three-seed result on four axes is worth
more than a one-seed result on seven. Confirmation (H′) is never the stage that gets cut.

---

## 5. Pre-registered decision rules

Restated here so a decision can be checked against them without leaving this document.

**DR-1, adoption.** A change enters the main line only if: AU-PRO@0.05 improves on ≥2 of 3
categories and regresses the third by ≤1 point; the paired image-level bootstrap interval of the
difference (class-stratified, `inspector.stats.paired_bootstrap_difference`) excludes zero after
Holm correction over the family of variants screened; p95 latency stays within budget; peak VRAM
stays under 3.5 GB. Passing (1)–(2) but failing (3)–(4) makes it an *accuracy-only variant*,
reported but not deployed. Adoption on development categories is a development result; it
becomes a finding only if it holds on the confirmation categories.

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

**DR-6 (ADR-7, corrected by ADR-9), operating points.** The requested rate is 1%; the threshold
is the conservative rank `k = floor(α(n+1))` with strict `score > threshold`. Where `α(n+1) < 1`
the threshold relaxes to the sample maximum and the row is flagged `target_met = false` — on VisA
the effective rate spans 0.73%–1.45%. Every reported FPR states the requested rate, the effective
bound and the realized test rate separately, and no single headline FPR is quoted across
categories.

---

## 6. Known gaps in this plan

Stated because a plan that lists no gaps has not been read carefully.

| Gap | Effect | Status |
|-----|--------|--------|
| **AUPIMO not implemented** | Image-level AUROC differences are now paired over images by a class-stratified bootstrap; localization differences still have no per-image paired measure | Needed before any localization adoption claim |
| **No reference-implementation agreement (E00)** | We cannot yet distinguish "our PatchCore" from "PatchCore"; the corrected scoring is only checked against hand-computed cases | Stage C, before any PatchCore conclusion is written as a finding |
| **No PatchCore reproduction of a published number** | Three categories cannot reproduce a 12-category mean | After E00; needs all VisA categories under the published protocol |
| ~~Single-pass VRAM budget is an assumption~~ | Measured in P3: under 1 GB at native resolution, batch 1. Host RAM binds instead. | Closed |
| **Group structure unknown** | VisA has no lot or session ids, so image independence in bootstrap intervals cannot be checked | Stated as a limitation |
| **Robustness validity unanswerable on VisA** | The strongest evidence P7 could produce is unavailable | Blocked on AD 2 |
| **Tier 4 comparators not planned here** | EfficientAD, RD++, Dinomaly are in docs/03 but have no stage yet | Add after Stage F, budget ~15 GPU-h |
| **No AD 2 leaderboard submission planned** | Our only truly held-out number would be missing | Decide at Gate E (OD-6) |

---

## 7. Execution order

```
A ──► B ──► C ──► D ──► D′ ──► E ──► F ──► G ──► H ──► H′ ──► I
│     │     │     │     │      │
│     │     │     │     │      └─ winners re-run at 3 seeds before DR-1 is applied
│     │     │     │     └─ no refits: aggregation and calibration on stored predictions
│     │     │     └─ frozen reference; its seed spread is the noise floor for Stage E
│     │     └─ E00: PatchCore conclusions are conditional on reference agreement
│     └─ Gate G2: if the random control is off, stop and fix the metric
└─ Gate A: if a split leaks, nothing downstream means anything
```

Never start a stage before the gate below it passes. The temptation to jump to Stage D is exactly
how a project ends up with an impressive number and no way to tell whether it is real.
