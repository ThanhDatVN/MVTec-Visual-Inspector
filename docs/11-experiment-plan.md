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
| Kaggle | T4×2 (one used; P100 if offered), 16 GB | **~30 GPU-h/week**, 12 h interactive session, ~9 h on commit, 20 GB working dir | Reference configurations, the ablation sweep, autoencoder training, robustness grid |

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
in the difference ledger. **C2 done ([report](../reports/K1-kaggle-development.md) §4):** anomalib
v2.3.0's own pipeline, vendored, at 320 px, 3 seeds. As shipped it is not distinguishable from ours
on any category after Holm correction (+0.001 / +0.020 / +0.065 image AUROC); its pipeline and its
backbone weights each move AUROC by up to ±0.06 in category-dependent directions, at about five
times our host memory and fit time.

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

### Stage D‴ — Data efficiency (E09; laptop)

| # | Run | n | Purpose |
|---|-----|---|---------|
| D‴1 | The frozen recipe fitted on nested subsets of 10 / 25 / 50 / 100 / 200 training normals × 3 subset seeds; calibration and test unchanged | 45 | How much normal data the recipe needs, with fit shots and calibration counts reported separately (docs/13 E09) |

*Status: done — [report](../reports/E09-data-efficiency.md).* No plateau before the full training
set on image AUROC (gap at 200 images 0.02 / 0.055 / 0.02); localization saturates earlier; ten
images at 640 px match or exceed the full training set at 320 px on `macaroni2` and `capsules`.
The patches-per-image contrast at a fixed bank was not run.

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

*Status: E1–E6b and the resolution axis (960, 1280 px) done — [report](../reports/K1-kaggle-development.md)
§2, 51 runs, Holm over 51.* Resolution keeps paying past 640 px on `macaroni2`/`capsules` (+0.06 to
+0.08) and `layer2` alone beats `layer2+layer3` at no cost; bank size, k-NN averaging, re-weighting
and projection are flat; `layer4` and `layer3`-only are catastrophic. The 3-seed reruns and E7 are
E8 below; adoption on held-out data needs fresh data (ADR-11, ADR-12).

**E8 — Gate E follow-up, pre-registered 2026-10-12 before any of its runs.** Four variants of
the frozen recipe, each at seeds 0–2 on the three development categories, in one Kaggle session
with its own 3-seed reference at the same implementation (notebook 02 §4b, `RUN_FOLLOWUP`):

| variant | why |
|---|---|
| `layer2` only | screening: +0.064 / +0.035 on `macaroni2` / `capsules` at no cost |
| 960 px | screening: +0.058 / +0.062, twice the cost |
| `layer2` at 960 px | E7, the joint re-vary of the two largest screening effects |
| timm `racm_in1k` weights | C2: weights moved image AUROC by up to ±0.06 at 320 px |

*Family:* 4 × 3 = 12 comparisons of seed-averaged image AUROC against the reference, paired,
class-stratified bootstrap with 10,000 resamples, Holm over the 12. *DR-1, made operational:*
(1) mean AU-PRO@0.05 improves on ≥ 2 categories and drops by ≤ 0.01 on the third; (2) the
image-AUROC difference is significantly positive after Holm on ≥ 2 categories and significantly
negative on none; (3) no latency budget is set for this phase (docs/04), so latency is reported,
not gated; (4) peak inference VRAM < 3.5 GB. (1), (2) and (4) → adopted as a development result;
(1) and (2) without (4) → accuracy-only variant; otherwise not adopted. Computed by
`scripts/kaggle_studies.py` (`followup.csv`, `followup_decisions.csv`). Expected from the screen:
960 px peaks at 4.1–4.4 GB of inference VRAM on a T4 and can at best be an accuracy-only variant.
A development adoption does not change the recipe frozen in ADR-11: the confirmation categories
and MPDD are spent, so a claim on held-out data needs data that shaped nothing (MVTec AD 2).
*Disclosure:* the notebook's smoke test (laptop, `capsules`, seed 0, a separate registry) ran the
`layer2` and timm-weights variants after this rule was written and before it was committed. It
changed nothing above, and its runs are not part of the analysis.

*Measurement amendment (2026-10-12, after the runs; a first reading of the decision used the
contaminated records below and is void):* criterion (4)
cannot be taken from the runner's VRAM records. Until the fix in `FeatureExtractor` (its forward
hooks closed over the extractor — a reference cycle), every finished run kept its backbone on the
GPU until Python's cycle collector ran; within a session the recorded peaks climb by ≈ 280 MB a
run and fall back when the collector runs. The 4.1–4.4 GB expected above for 960 px came from such
records. (4) is evaluated instead with `scripts/bench_inference.py`: each configuration fitted and
exported in its own process and measured in a fresh one on the laptop's 4 GB RTX 3050 (the
deployment card), `pcb1`, peak allocated VRAM over 20 warm-up and 200 timed service-path
predictions. The rule is unchanged; only its measurement is.

*Status: done — [report](../reports/E8-gate-e-followup.md).* `layer2` only, 960 px and `layer2` at
960 px are **adopted (development)** — e.g. `macaroni2` 0.887 → 0.979 image AUROC with both — at
0.57–0.75 GB inference VRAM; timm's `racm_in1k` weights are not (−0.055 on `macaroni2`).
Candidate recipe v2: `configs/recipes/candidate-v2-patchcore-layer2-960.yaml`, unconfirmed. Gate E
closed.

### Stage F — Tier 1 autoencoder (Kaggle, ~12 GPU-h)

| # | Run | n | Purpose |
|---|-----|---|---------|
| F1 | loss ∈ {L2, SSIM, L2+SSIM} × 3 categories × 3 seeds | 27 | The most instructive Tier 1 ablation. L2 blurs; a blurry reconstruction errs everywhere rather than at the defect. |
| F2 | latent ∈ {32, 128} at the best loss | 18 | Capacity vs the identity-function failure |
| F3 | residual ∈ {raw, multiscale} | 18 | Raw residuals are dominated by edge misalignment |
| F4 | aggregation ∈ {mean, max, top-k mean} | 0 | **From the P2 finding:** `pixel_pca` reached high pixel AUROC at chance image AUROC. Map quality and map-to-scalar aggregation are separate abilities. Now done by E02 (Stage D′) on stored maps for every model, without refitting. |

**Gate F.** The headline comparison — what the pretrained prior bought, in points, per category —
is computed and written down, including any category where it bought nothing.

*Status: F1 done at one seed per loss — [report](../reports/K1-kaggle-development.md) §3. At 640 px
and 150 epochs (no run triggered early stopping) the CAE reaches image AUROC 0.31–0.75; PatchCore is
ahead by +0.17 to +0.67 on every run. Rung closed as a negative result; F2/F3 not run.*

### Stage G — Robustness (laptop for inference, ~6 h wall-clock)

*Status: G1 and G4 done — [report](../reports/E08-robustness.md) (G4 in its own section). G2 dropped:
the autoencoder rung closed without a model worth stress-testing (Stage F). Run on the laptop with the ADR-11 recipe (640 px, fixed 10 k bank) on the three
development categories — `scripts/robustness_study.py`, output `reports/studies/e08/`. The
confirmation categories are spent, so this stage is development evidence.*

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

*Status: done as pre-registered in ADR-12 — [report](../reports/H1-mpdd-transfer.md).* Unchanged
recipe, 6 categories × 3 seeds: macro image AUROC 0.943, AU-PRO@0.05 0.861 (pixel PCA 0.729); the
recipe beats pixel PCA on 4/6. The 1% operating point is unreachable with 8–43 calibration normals;
pooled false alarms 31 against 28.4 expected at the relaxed bounds, no exceedance significant.

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

### Stage H″ — Confirmation v2 on MVTec AD 2 (E13; laptop)

| # | Run | n | Purpose |
|---|-----|---|---------|
| H″1 | Recipes v1 and candidate v2, unchanged, on all eight AD 2 categories, seeds 0–2, `role = confirmation` | 48 | Does the E8 adoption hold on data that shaped nothing? (ADR-13: H3 detection, H3b localization) |
| H″2 | Pixel PCA and the colour histogram, 320 px, seed 0 | 16 | Floors |
| H″3 | Optional: the selected recipe's `test_private` / `test_private_mixed` maps for benchmark.mvtec.com | — | Official numbers, and real lighting shift (G5) |

*Status: done — [report](../reports/E13-confirmation-v2-mvtec-ad2.md).* 64 runs. **Not confirmed:** H3 1/8 positive, 1/8 negative (macro +0.004); H3b 6/8 (rule 7). v1 stays. H″3 (server submission) not yet made.

### Stage H‴ — Development on MVTec AD 2 after E13 (E14; laptop)

E13 exposed two separate failures on AD 2 `test_public`, which is development data from now on:

- **(A) the image score.** The maps localize (pixel AUROC 0.76–0.96 for v1) while the image score
  — the re-weighted map maximum — is near chance on `can`, `wallplugs`, `rice` and `sheet_metal`.
- **(B) the operating point.** The test normals are a shifted population: the guards refuse 20–83%
  of them in six categories (0–8% of validation normals), and on `can` test normals out-score
  validation normals with AUROC 0.79–0.83. Both recipes' false alarms are 5–7× the bound.

Written down before any E14 run (2026-10-12):

| # | Run | n | Addresses |
|---|-----|---|-----------|
| E14a | Aggregation rules and object-region scoring on the stored E13 maps (`inspector.analysis`, no refit) | 0 | A |
| E14-ref | Recipe v1, seed 0, rerun at the E14 implementation | 8 | must reproduce E13's seed-0 predictions |
| E14b | v1 + per-image gain normalization (`preprocess.normalize: gain`) | 8 | B: global exposure |
| E14c | v1 + per-image standardization (`preprocess.normalize: standardize`) | 8 | B: exposure and contrast |
| E14d | v1 + exposure-augmented bank (`model.bank_exposures: [-0.5, 0.5]`, the E08 exposure model) | 8 | B: lighting coverage |

*Screen → 3 seeds.* Against E14-ref, all eight categories, seed 0: a variant goes to seeds 1–2 if
pooled false alarms fall by at least half with no category significantly worse in image AUROC
(paired bootstrap, Holm over 8), **or** image AUROC is significantly better on ≥ 3 categories and
worse on none. A carried variant must also cost no more than 0.01 image AUROC on the VisA
development categories (seed 0). Everything here is development evidence: the held-out arbiter is
MVTec's server on the private splits, with v1 and the best variant submitted once each.

*Outcome (2026-10-13).* E14-ref reproduced E13's seed-0 prediction digests on all eight
categories. No variant carried: gain 47 → 73 pooled false alarms, 1 gain / 1 loss; standardize
47 → 64, 1 / 2; exposure-augmented bank 47 → 49, 2 / 0. E14d first failed at its turn (a `--set`
of a key no config has) and was rerun from an overlay at the same implementation. E14a: top-k
aggregation +0.01–0.02 macro, no rule repairs `can`, `rice`, `wallplugs`. Report:
[E14–E15](../reports/E14-E15-ad2-development.md).

### Stage H⁗ — Lighting remedies on MVTec AD 2 (E15; laptop)

*Diagnosis (2026-10-13, `scripts/ad2_condition_study.py` on the stored E13 predictions).*
`test_public` shows every object under several captures, `<object>_<condition>.png`: `regular`,
`overexposed`, `underexposed` and `shift_1`..`shift_k`. Train and validation hold `regular` only.
Against the `regular` capture of the same object, over- and under-exposure are a pure global gain
(grey-level correlation ≥ 0.992, equal channel gains 0.76–1.18), while `shift_k` changes the
spatial structure of the lighting (correlation 0.82–0.99: highlights and shading move) and its
colour (blue/red ratio down to 0.78). E13's excess false alarms sit almost entirely in `shift`:

| v1 (v2) | regular | overexposed | underexposed | shift |
|---|---|---|---|---|
| false alarms / normals, pooled | 5.7% (2.6%) | 5.7% (3.6%) | 5.2% (4.2%) | 22.8% (31.2%) |
| macro image AUROC | 0.686 (0.742) | 0.694 (0.734) | 0.698 (0.740) | 0.684 (0.659) |

(Correction, same day: the `shift` correlation range is 0.82–1.00; `vial`'s shifts are
near-identical to `regular` at this scale.)

So the operating point holds under the training lighting, and E14b–d, which target global
exposure, address a condition that is not failing; they run to completion as registered. The
six captures of one object are also not independent: E13's image bootstrap understated the
uncertainty (object bootstrap: `sheet_metal` +0.156 is no longer significant, `vial` −0.111 still
is; the verdict, not confirmed, stands). From E15 on, the object is the resampling unit on AD 2.

Written down before any E15 run, after the diagnosis above and after a dry run of the analysis
script on the partial E14 `gain` runs (2026-10-13):

| # | Run | n | Addresses |
|---|-----|---|-----------|
| E15-ref | Recipe v1, seed 0, at the E15 implementation | 8 | must reproduce E13's seed-0 predictions |
| E15a | v1 + flat-field normalization (`preprocess.normalize: flatfield`: each channel divided by its blur at σ = 0.1 × long side) | 8 | smooth shading and colour casts, removed |
| E15b | v1 + relit bank (`model.bank_relight: 2`: two copies of each training image under a random smooth light field, gain 0.6–1.2 and colour cast, `transforms.relight`) | 8 | the same, covered in the bank |

*Screen → 3 seeds* (`scripts/e15_analysis.py`). Against E15-ref, all eight categories, seed 0: a
variant goes to seeds 1–2 if the pooled false alarms among `shift` normals fall by at least half
with no category significantly worse in image AUROC, **or** image AUROC is significantly better on
≥ 3 categories and worse on none — significance by the paired, class-stratified **object**
bootstrap (10,000 resamples, Holm over 8). A carried variant must also cost no more than 0.01 image
AUROC on the VisA development categories (seed 0). If both carry, their combination is screened
next, and a carried remedy is also screened on v2 (`layer2`, 960 px), which is the better recipe
under `regular` lighting. The arbiter stays MVTec's server: `test_private_mixed` is the mixed-lighting
split, so it is the direct held-out test of any lighting remedy.

*Outcome (2026-10-13).* E15-ref reproduced E13's seed-0 digests. Neither variant carried:
flat-field raised the `shift` false alarms 41 → 60 of 187; the relit bank left them at 39, with no
category significantly changed. The conditional follow-ups (combination, v2 screen) do not run.

**E16 — few-shot adaptation to changed lighting (R11).** Written down before any E16 fit
(2026-10-13), after E15's outcome. What a line does when its lighting changes is photograph a
few good parts under it and add them to the bank. Protocol (`scripts/e16_adaptation.py`): per
category, the `test_public` objects are split into two folds by a seeded permutation (seed 0),
separately for normal and anomalous objects. For fold f, recipe v1 (seed 0) is fitted on
`train/good` plus every capture of n normal objects of the other fold — n = 1, and n = all of
that fold (2–6 objects) — and evaluated on fold f's images, so no evaluated object was seen in
fitting and the two folds partition the test set. The operating point stays calibrated on
`validation/good` alone. Reference: E15-ref on the same images (the script with n = 0 reproduces
its scores exactly). Reading: adaptation with n objects *works* if the pooled false alarms among
`shift` normals fall by at least half and no category is significantly worse in image AUROC
(object bootstrap, Holm over 8). A measurement for deployment guidance (docs/14 R11), not a
recipe change; 32 fits.

*Outcome (2026-10-13).* 32 fits, clean worktree (`7685063`). One object: `shift` false alarms
41 → 28 of 187, does not work. One fold's objects (2–6): 41 → 19, no category significantly worse —
**works**, narrowly, and almost entirely on `can` (35 → 13). Macro AUROC 0.676 → 0.688; macro
recall at the unchanged operating point 0.239 → 0.195. Report:
[E14–E16](../reports/E14-E15-ad2-development.md).

### Stage J — After E16 (laptop; owner for the server)

Written down on 2026-10-13, before any run below. Order: S1 (export, running), E17, E18, then
E19 if E18 leaves the image-score problem open.

| # | Run | n | Addresses |
|---|-----|---|-----------|
| S1 | Recipe v1 exported for MVTec's server (`test_private`, `test_private_mixed`; ADR-13 §6, 3σ masks) | 8 fits | the held-out lighting shift, measured once — **upload is the owner's** |
| E17 | Recalibration under the new lighting, with and without an adapted bank (`scripts/e17_recalibration.py`) | 16 fits | B1: E16's threshold no longer bounds false alarms under the new lighting |
| E18-ref | Recipe v1, seed 0, at the E18 implementation | 8 | must reproduce E13's seed-0 predictions |
| E18a | v1 with DINOv2 ViT-S/14 features (`backbone: vit_small_patch14_dinov2`, `weights: timm:lvd142m`, blocks 8 + 11 after the final norm, no channel projection, 672 px) | 8 | R8: A1, A5, and lighting robustness |
| E18b | the same with DINOv2 ViT-B/14 | 8 | R8 |
| E19 | R1 tiling for `sheet_metal`, `fabric`, `rice` — designed after E18 | – | A2 |

*E17 reading.* Same folds as E16; the other fold's normal objects are split, the first half into
the adapted bank, the rest (every capture) into the calibration set beside `validation/good`.
Recalibration restores the bound if the reference recalibrated (R+cal) raises no more false alarms
than its expected count plus two Poisson SDs; adaptation pays at a valid operating point if the
adapted, recalibrated arm's macro recall exceeds R+cal's (object bootstrap 95% CI above zero) with
pooled false alarms no higher.

*E17 outcome (2026-10-13).* 16 fits, pinned worktree (`e781cd2`, implementation `67985a22598d`).
R+cal: 17 false alarms against 7.6 expected (limit 13.1) — the bound is **not** restored. A+cal −
R+cal macro recall −0.014 [−0.034, +0.003] — adaptation does **not** pay at a valid operating
point. Report: [E14–E17](../reports/E14-E15-ad2-development.md).

*E18 screen → 3 seeds* (`scripts/e15_analysis.py`, its variant label extended by the backbone).
Against E18-ref, all eight AD 2 categories, seed 0: a variant goes to seeds 1–2 if image AUROC
is significantly better on ≥ 3 categories and worse on none, **or** the false alarms among
`shift` normals fall by at least half with none worse — object bootstrap, Holm over 8. A carried
variant must also cost no more than 0.01 image AUROC on the VisA development categories
(seed 0). The 14 px token grid is coarser than WRN50's 8 px `layer2` grid, so AU-PRO is reported
and may fall; it is not part of the rule.

*E18 outcome (2026-10-13).* E18-ref reproduced E13's seed-0 digests on all eight categories.
Neither variant carried: ViT-S 1 gain (`rice` +0.334) / 1 loss (`vial` −0.172), ViT-B 1 / 1
(+0.265 / −0.170); `shift` false alarms 41 → 95 of 187 for both.

**E19 — resolution for the thin-defect categories (R1, A2).** Written down before any E19 run
(2026-10-13), after E18. R1 as planned split images into tiles; on these images the 4 GB card
embeds the whole image at the target resolution with batch size 1, so the experiment is the
resolution itself, with recipe v1 otherwise unchanged (`layer2+3`, 10k bank, 10% candidates).
Host memory for the candidate pool bounds the levels: `sheet_metal` (4224×1056, 137 training
images) at 1280, 1920 and 2816 px; `fabric` and `rice` (2448×2048, 387 and 313 images) at 1280 px
(1920 px would need ~11 GB for the pool). Seed 0, against E18-ref (640 px) at the same
implementation; image AUROC by the object bootstrap, Holm over the five comparisons. Reading: a
resolution goes to seeds 1–2 for its category if image AUROC is significantly better and AU-PRO
not lower. `sheet_metal` has four normal objects, so only a large effect can reach significance;
effect sizes are reported either way. 5 runs.

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
