# 02 — Metrics, External Baselines, and Decision Rules

Everything in this document is fixed at Gate P2 and implemented in `src/inspector/metrics/`.
Any metric reported anywhere in this project must be defined here first.

---

## 1. Metric definitions

### 1.1 Localization (pixel-level)

| Metric | Definition | Why we report it | Trap it avoids / has |
|--------|------------|------------------|----------------------|
| **AU-PRO@0.05** | Area under the Per-Region-Overlap curve, integrated to FPR = 0.05, normalized to [0,1]. PRO averages overlap **per connected defect region**, so a 40-pixel scratch counts as much as a 40 000-pixel stain. | **Primary localization metric.** It is the official AD 2 metric, and the tight 0.05 limit is what makes it meaningful when defects are tiny relative to a multi-megapixel frame. | Pixel AUROC is dominated by the enormous normal-pixel majority and by whichever defect happens to be largest. |
| **AU-PRO@0.30** | Same, integrated to FPR = 0.30. | Comparability with the classic MVTec AD literature, which uses 0.30. | **Never mix the two.** Always print the integration limit in the column header. A 0.30 number next to a 0.05 number with no label is a reporting bug. |
| **Pixel AUROC** | ROC-AUC over all pixels pooled. | Required by the project scope and needed for comparison with older papers. | Reported with an explicit caveat: it is inflated and near-saturating; it is not used for model selection. |
| **AUPIMO** | Area Under the Per-IMage Overlap curve. Computes a curve **per image**, with the false-positive axis defined using normal images only. Yields a *distribution* of per-image scores. | The modern fix for the above. Because it gives one score per image, it enables **paired statistical tests** between models and exposes variance that a single pooled number hides. This is our model-selection metric of record alongside AU-PRO@0.05. | Pooled metrics report a mean that no individual image achieves. |
| **SegF1** | F1 of the binarized mask at `tau_pix` from `OP-3SIGMA` (validation mean + 3 std). | The AD 2 benchmark's threshold-dependent metric. Measures whether the model is *usable*, not merely rankable. | A model can have good AU-PRO and unusable SegF1 if its scores are not calibratable — exactly PatchCore's reported failure on AD 2 (~3.7% mean SegF1 against ~28.8% AU-PRO@5%). |
| **IoU @ OP-3SIGMA** | Standard intersection-over-union of the binarized mask. | Intuitive for the case book and the demo UI. | — |

**Implementation notes for AU-PRO.** Compute connected components on the ground-truth mask with
8-connectivity. For each threshold, PRO is the mean over regions of (intersection / region area).
FPR is computed over the pixels of *normal* images plus normal pixels of anomalous images —
follow the reference implementation exactly and pin the choice in a docstring, because the two
conventions differ by a few points. Sweep thresholds over the union of unique anomaly-map values
subsampled to a fixed grid (e.g. 512 points) for tractability at multi-megapixel resolution, and
record the grid size as part of the metric config.

### 1.2 Detection (image-level)

| Metric | Definition | Why |
|--------|------------|-----|
| **Image AUROC** | ROC-AUC of `s(x)` over the test split. | The field's standard; required by scope. |
| **Image AUPR** | Average precision. | Honest under class imbalance, which is the real factory condition. |
| **F1-max (oracle)** | Best achievable F1 over all thresholds. | Upper bound, always labelled `(oracle)`. |
| **FPR @ OP-FPR1** | Realized false-positive rate on test normals at the validation-derived threshold. | **Required by scope.** The gap between the targeted 1% and the realized rate is the calibration error, and it is often large under lighting shift. This gap is a headline finding, not a footnote. |
| **Recall @ OP-FPR1** | Defect catch rate at that same threshold. | The number a plant manager actually asks for. |
| **Escape rate** | 1 − Recall @ OP-FPR1, reported per defect type. | Turns a scalar into an actionable failure profile. |

### 1.3 Systems metrics

| Metric | How measured | Reported as |
|--------|--------------|-------------|
| **Latency p50 / p95 / p99** | 200 timed single-image inferences after 20 warm-up iterations, `torch.cuda.synchronize()` around each, batch size 1 (the factory condition). | ms, with device, resolution, precision, and batch size in the same row. A latency number without those four is meaningless. |
| **Throughput** | Images/s at the largest batch size that fits in 4 GB. | img/s |
| **Peak VRAM** | `torch.cuda.max_memory_allocated()` and `max_memory_reserved()`. | MB, both |
| **Peak host RAM** | `resource` / `psutil` RSS high-water mark across fit and predict. | MB |
| **Model / memory-bank size on disk** | Serialized artifact size. | MB |
| **Fit wall-clock** | End-to-end time to build the model from raw images. | s |
| **Cold-start time** | Process start to first served prediction. | s (matters for the container) |

Latency is measured on **both** devices (laptop RTX 3050 and Colab T4) and on **CPU**, because
"can this run without a GPU" is a real deployment question and the answer changes the
architecture choice.

### 1.4 Statistics

- 3 seeds for stochastic methods; report **mean ± std**.
- **Model comparison**: paired **Wilcoxon signed-rank test** over the per-image AUPIMO scores
  (paired by image). Report the p-value and the median of paired differences. With ~80–150 test
  images per category this has adequate power and makes no normality assumption.
- **Confidence intervals** on pooled metrics: bootstrap over images, 1000 resamples, BCa, 95%.
- Apply **Holm–Bonferroni** correction across the comparisons within a single results table, and
  state the family size. Running 30 ablations and reporting the one with p < 0.05 is not a result.

---

## 2. External baseline matrix

These are **published anchors**, not our results. They tell us what "normal" looks like on this
benchmark so we can recognize a broken implementation.

### 2.1 MVTec AD 2, mean over all 8 categories, `TEST_priv`

| Method | AU-PRO@0.05 (%) | SegF1 (%) | Family |
|--------|-----------------|-----------|--------|
| EfficientAD | 30.8 | 15.4 | Student–teacher distillation |
| **PatchCore** | **28.8** | **3.7** | Memory bank / kNN |
| RD++ | 27.1 | 19.2 | Reverse distillation |
| RD | 26.4 | 18.1 | Reverse distillation |
| MSFlow | 24.3 | 21.8 | Normalizing flow |
| SimpleNet | 21.1 | 17.7 | Feature adaptation + synthetic anomalies |
| DSR | 20.3 | 10.9 | Quantized latent reconstruction |
| RoBiS (VAND 3.0 challenge) | — | **51.0** | Challenge system: Swin-cropping + augmentation + adaptive binarization |

> **Verification task (P2).** These figures were extracted from the arXiv HTML of the AD 2 paper
> and the RoBiS paper. Before any of them appears in the final report, re-read the published IJCV
> table and confirm each cell. Note also that the AD 2 abstract describes state-of-the-art
> performance as "below 60% average AU-PRO", which does not match the ~20–31% in the per-method
> table — most likely a different integration limit or split. **Resolve this discrepancy and
> record the resolution in an ADR.** Quoting a number whose provenance you have not checked is
> how a benchmark section becomes wrong.

**What this table tells us before we write a line of code:**

1. **PatchCore's AU-PRO is competitive but its SegF1 is catastrophic (3.7%).** Its scores rank
   well but do not calibrate — `mean + 3·std` on validation lands nowhere near a useful pixel
   threshold. This is a concrete, pre-identified research question for Phase P8, and it is more
   interesting than a leaderboard place.
2. **The spread across families is narrow (20–31%).** Nobody has this benchmark solved. A modest,
   well-measured contribution is a real contribution here.
3. **The challenge system nearly triples SegF1 through engineering**, not a new loss: cropping
   strategy, lighting augmentation, adaptive binarization. That is an explicit signal about where
   our effort pays off — see Tier 5 in [03](03-method-ladder-and-matrix.md).

### 2.2 MVTec AD classic — the reproduction anchor

| Method | Image AUROC (%) | Note |
|--------|-----------------|------|
| PatchCore (published) | ~99.1 | Our Gate P5 reproduction target on fixture categories |
| Dinomaly (CVPR 2025) | 99.6 (multi-class) | Current SOTA; illustrates saturation |
| Dinomaly2 (2025) | 99.9 (multi-class) | The benchmark is finished |

Pull the exact per-category PatchCore numbers from the PatchCore paper at P5 and set the
reproduction tolerance from them (see [04](04-roadmap.md), Gate P5).

---

## 3. Decision rules (pre-registered)

We set **no accuracy target**. We instead pre-register how decisions get made, so that results
cannot be rationalized after the fact.

### DR-1 — Adopting a change

A configuration change is adopted into the main line if **all** hold:

1. Mean AU-PRO@0.05 improves on **at least 2 of the 3** categories and does not regress by more
   than 1 point on the third.
2. The paired Wilcoxon test over per-image AUPIMO gives p < 0.05 after Holm–Bonferroni correction
   within that experiment family.
3. p95 latency stays within the phase's stated budget.
4. Peak VRAM stays under 3.5 GB at the deployment resolution (leaving headroom under the 4 GB card).

If (1) and (2) pass but (3) or (4) fail, the change is recorded as an **accuracy-only variant**
and kept out of the deployed model. Both are reported.

### DR-2 — Declaring a method "better"

Never on a mean alone. A method is called better only with the paired test, the effect size
(median paired difference), and the per-category table all shown. Where a method wins on one
category and loses on another, that is the finding — report it as such rather than averaging it away.

### DR-3 — Stopping an experiment line

Abandon a line if, after its budgeted GPU-hours, it has not met DR-1 on any category. Write the
negative result into `reports/negative-results.md` with the configurations tried. A documented
dead end is a deliverable.

### DR-4 — What goes in the headline table

One row per method at its **best validation-selected configuration**, never its best test-selected
configuration. If a configuration was chosen by looking at `test_public`, the row is marked
`(test-selected)` and read as an upper bound.

### DR-5 — Interpreting our numbers against the external matrix

Our AD 2 numbers are on `test_public`; the published anchors in §2.1 are on `TEST_priv`. **These
are not directly comparable** and every table that shows both must say so in its caption. Only the
optional leaderboard submission produces a directly comparable number.

---

## 4. Canonical results table schema

Every experiment writes one row to `reports/results.csv` with exactly these columns. MLflow is the
source of truth; the CSV is generated from it, never hand-edited.

```
run_id, git_sha, dirty, config_hash, dataset, category, split, method, tier,
backbone, resolution, tiling, precision, coreset_ratio, seed, n_seeds,
image_auroc, image_aupr, f1max_oracle, fpr_at_op1, recall_at_op1,
pixel_auroc, aupro_005, aupro_030, aupimo_mean, aupimo_median, aupimo_p10,
segf1_3sigma, iou_3sigma,
latency_p50_ms, latency_p95_ms, latency_p99_ms, device, batch_size,
peak_vram_mb, peak_rss_mb, artifact_mb, fit_seconds,
threshold_source, oracle_flag, notes
```

`threshold_source` and `oracle_flag` are mandatory, not optional. They are what make the table
auditable.

See [templates/results-table.md](templates/results-table.md) for the presentation format.
