# 01 — Project Charter and Frozen Protocol

**Status:** DRAFT until Gate P1, then **FROZEN**.
Changes after freezing require an entry in [07-risks-and-decisions.md](07-risks-and-decisions.md#decision-log-adrs)
stating what changed, why, and which already-recorded results are invalidated.

---

## 1. Problem definition

### 1.1 Formal statement

Let `X_norm` be a set of images of a defect-free industrial part, captured on a fixed rig.
We observe **only** `X_norm` at training time. At inference we receive an image `x` and must output:

| Output | Symbol | Type | Used for |
|--------|--------|------|----------|
| Image anomaly score | `s(x)` | scalar, higher = more anomalous | Pass/fail decision |
| Anomaly map | `M(x)`, shape H×W | per-pixel score | Localization, operator explanation |
| Binary decision | `s(x) > tau` | bool | The actual factory output |
| Binary mask | `M(x) > tau_pix` | H×W bool | Defect region for rework/report |

`tau` and `tau_pix` are chosen **using normal-only validation data**, never using test data.
This is the single most commonly violated rule in published anomaly-detection code and we
test for it (see §5).

### 1.2 Why this framing

The one-class framing is not an academic preference — it is the economics of manufacturing.
Defects are rare, diverse, and their catalogue is open-ended: you cannot collect a balanced,
labelled defect set for a part that has a 0.3% defect rate and whose failure modes change
when a supplier changes. Normal images, by contrast, are free and abundant. Any method that
needs defect labels to train is not deployable at line start-up ("cold start").

### 1.3 Non-goals

- Multi-class / unified models across categories (noted as future work; we train per-category).
- 3D, multi-view, or logical anomalies (MVTec 3D-AD / LOCO are out of scope).
- Beating the MVTec AD 2 leaderboard. We are building a rigorous, reproducible, *deployable*
  pipeline and an honest measurement of where it sits. A leaderboard submission is an optional
  stretch deliverable, not a success criterion.
- Any commercial use. Precluded by the dataset license — see [08](08-licensing-and-attribution.md).

---

## 2. Objectives

Objectives are stated so that each one is **falsifiable** and maps to a gate in
[04-roadmap.md](04-roadmap.md).

| ID | Objective | Verified by |
|----|-----------|-------------|
| **O1** | A leakage-free, hash-verified data pipeline for 3 AD 2 categories | Test suite `tests/data/` green; split manifests SHA-256 recorded |
| **O2** | A metric implementation provably equal to a reference implementation | `tests/metrics/` agreement within 1e-6 on fixtures, plus reproduction of a published PatchCore number on classic AD within tolerance |
| **O3** | A from-scratch AE and one-class baseline, honestly reported | 3-seed results table, all 3 categories, in MLflow |
| **O4** | A from-scratch PatchCore that reproduces its published behaviour | Classic-AD reproduction gate passes; AD 2 numbers read against external anchors |
| **O5** | A quantified comparison: AE vs pretrained-features vs PatchCore vs modern comparators | Results table with paired significance tests, not eyeballed means |
| **O6** | Explainability a process engineer could act on | 15 or more documented defect cases plus a failure taxonomy |
| **O7** | A robustness profile under optical and geometric perturbation | Degradation curves over 5 severities × 6 corruptions, plus AD 2's native lighting-shift split |
| **O8** | A deployable service with measured latency and memory | Dockerized FastAPI + Gradio; p50/p95 latency and peak RSS/VRAM reported |
| **O9** | Reproducibility by a stranger | Fresh-clone run reproduces the headline table from documented commands |
| **O10** | Full license compliance | Attribution block present; no dataset bytes in git history |

---

## 3. Dataset selection

### 3.1 Primary: MVTec AD 2

**Chosen as the primary benchmark.** Rationale:

1. **It is not saturated.** Published methods reach roughly 20–31% AU-PRO@5% on the private
   test split. Classic MVTec AD sits at ~99.6% image AUROC — work done there measures noise.
2. **Robustness is built in, not simulated.** AD 2 ships test splits captured under unseen
   lighting (over/under-exposure, extra light sources). That is a *real* distribution shift to
   report alongside our synthetic corruption suite, which is strictly stronger evidence.
3. **It has an official validation split** of normal images — exactly what honest threshold
   selection requires, and what classic AD lacks.
4. **It has a held-out, server-evaluated test split**, so our protocol can be audited.

Costs we accept: ~30 GB download, multi-megapixel images, registration required, and no local
ground truth for the private splits.

### 3.2 Secondary: MVTec AD (classic) — a test fixture, not a study subject

Classic AD is used **only** as a correctness harness (Gates P2 and P5): we reproduce a published
PatchCore result on one or two classic categories to prove our metric and model implementations
are right before trusting any AD 2 number. These categories are **not** study categories and
their results appear only in an appendix. This keeps the analysis honestly at "exactly three
categories" while buying implementation confidence cheaply.

Suggested fixture categories: `screw` (small defects, stresses localization) and `carpet`
(texture, stresses false positives). Confirm the published PatchCore reference values from the
PatchCore paper before use.

### 3.3 The three study categories

Chosen so that each fails for a *different physical reason*. A three-category study where all
three are texture-like teaches you one thing three times.

| Category | Resolution (W×H) | Train / Val / Test_pub | Dominant difficulty | What it tests in our system |
|----------|------------------|------------------------|---------------------|------------------------------|
| **`sheet_metal`** | 4224 × 1056 | 137 / 19 / 114 | Extremely small defects on a huge, high-aspect-ratio, dark-field image | The **resolution** axis. A naive resize to 256² destroys the defect. Forces tiling and feature-granularity work. Smallest train set, so fastest iteration. |
| **`fruit_jelly`** | 2100 × 1520 | 263 / 37 / 80 | Transparent and overlapping objects, back-lighting | The **optics/ambiguity** axis. This is where published PatchCore segmentation F1 collapses (~3.7% mean) — the most instructive failure in the benchmark. |
| **`walnuts`** | 2448 × 2048 | 432 / 48 / 150 | Very high variance in *normal* appearance (natural product) | The **false-positive** axis. Natural variation looks like a defect to a poorly regularized normality model. Largest train set, so largest memory bank — stresses the VRAM budget. |

Counts are taken from the AD 2 paper and **must be re-verified against the downloaded data** at
Gate P1 (`tests/data/test_dataset_shape.py`). The private and private-mixed splits add roughly
142–276 images per category with withheld ground truth.

**Documented swap rule.** If at Gate P5 the Colab GPU-hours spent exceed 60% of budget, swap
`walnuts` → `can` (2232×1024, 412 train images, back-lit, ~45% fewer pixels). Record the swap as
an ADR. Do not swap for any other reason, and never mid-phase.

### 3.4 Split semantics — what each split may be used for

This table is the protocol. Violating it is a bug, and §5 lists the tests that catch it.

| Split | Contents | GT masks | May be used for |
|-------|----------|----------|-----------------|
| `train/good` | normal only | — | Fitting models, memory banks, feature statistics |
| `validation/good` | normal only | — | **Threshold selection**, early stopping, hyperparameter selection, anomaly-map normalization statistics |
| `test_public` | normal + anomalous | yes | **Reporting.** Read once per frozen configuration. |
| `test_private` | normal + anomalous, seen lighting | withheld | Optional leaderboard submission at P11 |
| `test_private_mixed` | normal + anomalous, **seen and unseen lighting** | withheld | Optional leaderboard submission at P11 |

> **The hyperparameter trap.** With no anomalous validation images, you cannot tune for AUROC on
> validation. You must either tune on train/validation normals only (e.g. held-out normal
> reconstruction error, or another normal-only proxy), *or* accept that tuning on `test_public`
> turns `test_public` into a validation set — in which case you must report it as one, leaving
> the private split as your only clean test.
> **We choose the latter, explicitly:** `test_public` is our development test set, we bound how
> many times we look at it (§4.3), and we treat the private split as the one honest held-out
> number. Pretending otherwise is the most common silent dishonesty in this field.

---

## 4. Frozen experimental protocol

### 4.1 Preprocessing

| Parameter | Value | Note |
|-----------|-------|------|
| Colour | RGB, ImageNet mean/std normalization | Backbones are ImageNet/DINO pretrained |
| Resize policy | **Aspect-preserving**, defined per category, recorded in config | Never a blind square resize — `sheet_metal` is 4:1 |
| Interpolation | `INTER_AREA` for downscale, `INTER_LINEAR` for upscale | Fixed. Area-averaging preserves small dark defects better than bilinear |
| Resolution ladder | 256², 320², 448², 512², tiled-native | Resolution is an explicit ablation axis, not a hidden default |
| Train-time augmentation | **None by default** | Augmenting normal-only data widens the normality model and can *hide* defects. Any augmentation is an explicit, ablated experiment (see [03](03-method-ladder-and-matrix.md), Tier 5). |
| Mask handling | Nearest-neighbour resize; masks binarized at > 0 | Never interpolate a label mask bilinearly |
| Anomaly map output | Upsampled bilinearly to **native image resolution** before scoring | Scoring at reduced resolution inflates PRO; all metrics are computed at native size |

### 4.2 Determinism and seeds

- Global seed set for `random`, `numpy`, `torch`, `torch.cuda`; `cudnn.deterministic=True` and
  `cudnn.benchmark=False` for reported runs.
- **Seeds 0, 1, 2** for every stochastic method. Report **mean ± std**, never a single run.
- Stochastic components: AE / Deep SVDD initialization and batch order; PatchCore greedy coreset
  (random start point); any random pre-subsampling of patch features.
- Deterministic components (PaDiM, Mahalanobis) are run once; the results table records
  `n_seeds=1` explicitly rather than leaving it implied.
- Every run records: git commit SHA (plus dirty flag), config hash, dataset manifest SHA-256,
  library versions, device name, CUDA/cuDNN version.

### 4.3 Test-set discipline

- `test_public` may be evaluated at most **once per frozen configuration**. A configuration is
  frozen when its config file is committed.
- A running tally of `test_public` evaluations per category lives in
  `reports/test_set_budget.md`. Budget: **40 evaluations per category** for the whole project.
  Exceeding it is not forbidden, but it must then appear in the model card's limitations —
  because it changes what the number means.
- `test_private` / `test_private_mixed`: at most **2 submissions total** for the project.

### 4.4 Operating point definition

The factory question is not "what is your AUROC", it is "at the alarm rate I can staff, how many
defects escape?" So every reported model carries defined operating points:

1. **`OP-FPR1` (primary)** — `tau` is the 99th percentile of `s(x)` over `validation/good`,
   targeting 1% false alarms on normal parts. Report the *realized* FPR on `test_public` normals
   and the recall at that point.
2. **`OP-3SIGMA` (official AD 2)** — `tau_pix = mean + 3·std` of anomaly-map values over the
   validation set, as the AD 2 benchmark specifies. Used for thresholded `SegF1` so our numbers
   stay comparable to the leaderboard.
3. **`OP-F1MAX` (oracle)** — the threshold maximizing F1 on `test_public`. This uses test labels;
   it is reported only as an upper bound and is **always labelled `(oracle)`**.

Reporting all three, with the oracle honestly labelled, is the difference between a benchmark and
a sales pitch.

---

## 5. Leakage rules and the tests that enforce them

| # | Rule | Enforcing test |
|---|------|----------------|
| L1 | No file appears in more than one split | `test_splits_disjoint` — set intersection on SHA-256 of file bytes, not filenames |
| L2 | No test image influences any fitted parameter | `test_no_test_in_fit` — the fit function's loader dataset is asserted to be split in {train, val} |
| L3 | **Anomaly maps are never min-max normalized using test-set statistics** | `test_normalization_stats_from_val_only` — normalizer is fitted once on validation and frozen; asserts identical fitted min/max for two different test subsets |
| L4 | Thresholds derive from validation normals only (except the labelled oracle) | `test_threshold_provenance` — threshold objects carry `source_split`, asserted `== "validation"` unless `oracle=True` |
| L5 | No augmentation statistic, normalization statistic, or PCA basis is fitted on test data | `test_transform_fit_provenance` |
| L6 | Near-duplicate check between train and test | `test_no_near_duplicates` — perceptual hash Hamming distance above a calibrated threshold; collisions are reported as a dataset finding, not silently dropped |
| L7 | Private splits carry no ground truth and no code path reads labels from them | `test_private_split_has_no_gt` |

> **On L3.** Min-max normalizing an anomaly map over the test set is endemic in public
> repositories. It leaks the test distribution into every score and can move pixel AUROC by
> several points. We fit the normalizer once on validation and freeze it.

> **On L6.** MVTec categories are captured on a fixed rig, so images are genuinely similar by
> design and pHash will flag many pairs. The test exists to catch *exact or near-exact duplicate
> captures across splits*, so calibrate its threshold on within-train pairs first and record the
> calibration alongside the test.

---

## 6. Definition of Done (project level)

The project is done when all of the following hold at once:

- [ ] All ten objectives (§2) verified by their stated evidence.
- [ ] `pytest` green on a fresh clone with **no dataset present** (synthetic fixtures).
- [ ] `docker run` serves the demo and a documented `curl` returns a scored heatmap.
- [ ] The headline results table is regenerable by one documented command per row.
- [ ] Model card complete, with a limitations section a skeptic would accept.
- [ ] README benchmark table cites external anchors and marks every oracle number.
- [ ] Attribution and license obligations satisfied; git history contains no dataset bytes.
- [ ] Every negative result is written down. A report containing only successes is a report that
      has been filtered, and the filter is the finding.
