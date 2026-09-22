# 04 — Roadmap, Schedule, and Gates

**Planning assumption:** ~12–15 focused hours per week over 12 weeks (~160 hours total).
A compressed 8-week lane is given in §4. Adjust the calendar, keep the gate order.

**The gate rule.** A phase is not finished when the code runs. It is finished when its gate
passes. A gate is a binary, externally checkable condition — not "looks good". If a gate fails,
you fix it before moving on; you do not carry a known-broken foundation into the next phase,
because every number built on it will have to be thrown away.

---

## Phase map

| Phase | Week | Theme | Gate | Scope item |
|-------|------|-------|------|------------|
| P0 | 0 | Foundations, data acquisition, CI skeleton | G0 | — |
| P1 | 1 | Problem definition, loaders, splits, EDA | G1 | 1, 2 |
| P2 | 2 | Metrics module, Tier 0 floors | G2 | 5 |
| P3 | 3 | Autoencoder / one-class baseline | G3 | 3 |
| P4 | 4 | Pretrained-feature methods | G4 | 4 |
| P5 | 5–6 | PatchCore + ablation sweep | G5 | 4 |
| P6 | 7 | Explainability and the case book | G6 | 6 |
| P7 | 8 | Robustness study | G7 | 7 |
| P8 | 9 | Advanced tier and comparators | G8 | 4 |
| P9 | 10 | Serving: API, Docker, latency | G9 | 8 |
| P10 | 11 | Model card, benchmark report, repro audit | G10 | 9 |
| P11 | 12 | Buffer, stretch goals, optional submission | — | — |

---

## P0 — Foundations (Week 0)

**Do first, on day one:** register for the MVTec AD 2 download. Access is gated behind an account
and the download is ~30 GB. If this slips, everything slips — it is the only true external
dependency in the project.

| # | Task | Output |
|---|------|--------|
| 0.1 | Register and download MVTec AD 2; download MVTec AD classic for fixtures | Data on disk, outside the repo, path in `.env` |
| 0.2 | Verify integrity; record SHA-256 of each archive and a per-file manifest | `data/manifests/*.sha256` (committed — hashes are not dataset content) |
| 0.3 | `git init`; repo skeleton per [06](06-engineering-mlops-and-testing.md); `.gitignore` that excludes every image extension and the data dir | Repo |
| 0.4 | Environment: `pyproject.toml`, pinned `requirements.txt` + `requirements-lock.txt`, CUDA-matched torch wheel for the RTX 3050 | Reproducible env |
| 0.5 | **Synthetic fixture generator** — procedurally generated "parts" with injected defects and masks, ~20 images | `tests/fixtures/` (safe to commit; contains no MVTec data) |
| 0.6 | CI: GitHub Actions running ruff, mypy, pytest on CPU | Green badge |
| 0.7 | MLflow tracking store; run-metadata logging helper | `mlruns/` + `src/inspector/tracking.py` |
| 0.8 | `LICENSE`, `ATTRIBUTION.md`, `NOTICE` per [08](08-licensing-and-attribution.md) | Compliance |

**Gate G0** — all true:
- [ ] `pytest` green on a machine with **no dataset present**.
- [ ] CI green on a clean checkout.
- [ ] `git log -p | grep -c` for image magic bytes returns 0; `.gitignore` verified by a test.
- [ ] Dataset manifest hashes recorded and a re-verification command documented.

> **Why the synthetic fixture generator comes before any model.** The dataset cannot be committed
> (license) and CI has no GPU. Without synthetic fixtures the test suite can only run on your
> laptop with data mounted, which means it will stop being run. Twenty procedurally generated
> images with known ground truth make the entire pipeline testable by anyone, forever.

---

## P1 — Problem definition, loaders, splits, EDA (Week 1)

| # | Task | Output |
|---|------|--------|
| 1.1 | Freeze [01-charter-and-protocol.md](01-charter-and-protocol.md); confirm the three categories | Signed protocol |
| 1.2 | `MVTecAD2Dataset` + `MVTecADDataset` (torch `Dataset`), split-aware, manifest-driven | `src/inspector/data/` |
| 1.3 | Deterministic transform pipeline with recorded provenance | `src/inspector/data/transforms.py` |
| 1.4 | The seven leakage tests L1–L7 | `tests/data/` |
| 1.5 | Verify the paper's per-category counts and resolutions against the data on disk | `reports/eda/dataset_audit.md` |
| 1.6 | EDA: intensity distributions per split, defect-size histogram, defect-type inventory, defect-area-to-image-area ratio, train/val/test lighting comparison | `reports/eda/` notebooks + figures |
| 1.7 | Per-category resize policy decided from the defect-size histogram, not by habit | Config files |

**The EDA that matters:** plot the distribution of **defect area as a fraction of image area** per
category, then compute what a 256×256 resize does to the median defect — for `sheet_metal` it will
likely reduce some defects to sub-pixel size. That single plot justifies the entire resolution axis
of the study and belongs in the final report.

**Gate G1**
- [ ] Counts and resolutions match the paper, or the discrepancy is documented.
- [ ] All seven leakage tests pass.
- [ ] Split manifests hashed and committed.
- [ ] Defect-size analysis complete; per-category resolution policy chosen **and justified in writing**.

---

## P2 — Metrics and Tier 0 (Week 2)

| # | Task | Output |
|---|------|--------|
| 2.1 | Implement AU-PRO (both integration limits), pixel AUROC, image AUROC/AUPR, SegF1, IoU | `src/inspector/metrics/` |
| 2.2 | Implement or integrate AUPIMO | same |
| 2.3 | **Validate against a reference implementation** on fixtures and on a real category | `tests/metrics/` |
| 2.4 | Analytic unit tests: perfect predictor → 1.0; random → chance; inverted → symmetric | `tests/metrics/` |
| 2.5 | Bootstrap CIs and paired Wilcoxon helper | `src/inspector/stats.py` |
| 2.6 | Tier 0 floors on all three categories | First MLflow rows |
| 2.7 | Results-table generator (MLflow → CSV → markdown) | `scripts/make_results.py` |

**Gate G2**
- [ ] Our AU-PRO agrees with the reference implementation within 1e-6 on fixtures and within 1e-3
      on a real category (differences above that indicate a convention mismatch — resolve it, do
      not tolerate it).
- [ ] `T0-random` gives image AUROC in [0.45, 0.55]; a perfect oracle gives 1.0.
- [ ] Tier 0 floors recorded; any category where a trivial method exceeds 0.8 AUROC is flagged
      loudly in the report.
- [ ] Results table regenerates from MLflow with one command.

---

## P3 — Autoencoder / one-class baseline (Week 3)

| # | Task | Output |
|---|------|--------|
| 3.1 | Conv AE with L2 loss, trainer, early stopping on validation reconstruction | `src/inspector/models/cae.py` |
| 3.2 | SSIM loss variant | same |
| 3.3 | Deep SVDD with collapse-prevention constraints | `src/inspector/models/dsvdd.py` |
| 3.4 | Multi-scale residual scoring | `src/inspector/models/scoring.py` |
| 3.5 | Latent-dimension and loss ablations, 3 seeds | MLflow |
| 3.6 | First qualitative heatmaps | `reports/figures/` |

**Latency budget for this phase:** none — baselines are not deployment candidates.
**VRAM:** trains comfortably at 256²–320² on 4 GB at batch size 8–16.

**Gate G3**
- [ ] Full 3-category × 3-seed table for at least `T1-cae-l2`, `T1-cae-ssim`, `T1-dsvdd`.
- [ ] Deep SVDD hypersphere collapse checked for and reported (variance of embeddings > 0).
- [ ] Training curves logged; loss decreased and validation reconstruction is not diverging.
- [ ] Written paragraph on *where* the AE fails, supported by at least three heatmaps.

---

## P4 — Pretrained features (Week 4)

| # | Task | Output |
|---|------|--------|
| 4.1 | Backbone feature extractor with hooks, cached to disk | `src/inspector/features/` |
| 4.2 | Mahalanobis with Ledoit–Wolf shrinkage | `src/inspector/models/mahalanobis.py` |
| 4.3 | PaDiM | `src/inspector/models/padim.py` |
| 4.4 | SPADE | `src/inspector/models/spade.py` |
| 4.5 | Backbone and layer ablations | MLflow |
| 4.6 | **Paired significance test vs Tier 1** | `reports/t1_vs_t2.md` |

**Feature caching is the key engineering decision here.** Extract once per (category, backbone,
layer, resolution), store as fp16 `.npy` memmaps, and let every Tier 2–3 model read from cache.
This converts the rest of the project from GPU-bound to IO-bound and is what makes ~350 runs
affordable on a laptop plus Colab.

**Gate G4**
- [ ] Tier 2 beats Tier 1 with p < 0.05 (paired, corrected) — or, if not, the anomaly is
      investigated and explained before proceeding.
- [ ] Feature cache is content-addressed and its invalidation is tested.
- [ ] Backbone comparison table complete.

---

## P5 — PatchCore (Weeks 5–6)

| # | Task | Output |
|---|------|--------|
| 5.1 | Patch aggregation, projection, greedy k-center coreset | `src/inspector/models/patchcore.py` |
| 5.2 | **Streaming/chunked coreset** for high resolution (see §3) | same |
| 5.3 | kNN scoring with the paper's image-score re-weighting | same |
| 5.4 | **Reproduction gate on classic AD fixture categories** | `reports/reproduction.md` |
| 5.5 | Reference config on all three AD 2 categories, 3 seeds | MLflow |
| 5.6 | Staged ablation sweep (backbone → layers → resolution → coreset → k → sigma) | MLflow, Colab |
| 5.7 | Coreset ratio vs accuracy vs latency vs memory **Pareto figure** | `reports/figures/pareto.png` |
| 5.8 | Investigate the AU-PRO / SegF1 divergence | `reports/patchcore_calibration.md` |

**Gate G5** — the most important gate in the project:
- [ ] Our PatchCore reproduces the published image AUROC on the classic-AD fixture categories to
      within **1.0 point**. If not, the implementation is wrong — find it before running anything
      on AD 2. (Check first: layer choice, the 3×3 aggregation, coreset ratio, the image-score
      re-weighting, and whether the map is scored at native resolution.)
- [ ] AD 2 reference-config numbers recorded for all three categories, 3 seeds.
- [ ] Ablation sweep complete; one-page written conclusion on which axis dominated.
- [ ] Pareto figure produced.
- [ ] Peak VRAM recorded for each resolution, and the maximum resolution that fits in 4 GB is
      established empirically.

---

## P6 — Explainability and case book (Week 7)

| # | Task | Output |
|---|------|--------|
| 6.1 | Heatmap/overlay renderer with a fixed colormap and scale bar | `src/inspector/viz/` |
| 6.2 | Grad-CAM for AE and Deep SVDD | `src/inspector/viz/gradcam.py` |
| 6.3 | Soft-min Grad-CAM surrogate for PatchCore | same |
| 6.4 | Nearest-normal-patch retrieval | `src/inspector/viz/retrieval.py` |
| 6.5 | **Case book: 15+ cases**, ≥4 per category, ≥5 taxonomy tags, **≥4 failures** | `reports/case-book/` |
| 6.6 | Failure taxonomy counts figure | `reports/figures/failure_taxonomy.png` |

**Gate G6**
- [ ] 15+ cases, each with all seven required elements (see [03](03-method-ladder-and-matrix.md), Tier 6).
- [ ] At least four are failures of the best model, diagnosed rather than merely displayed.
- [ ] Taxonomy counts produce at least one actionable conclusion for P8.

---

## P7 — Robustness (Week 8)

Full specification in [05-robustness-protocol.md](05-robustness-protocol.md).

| # | Task | Output |
|---|------|--------|
| 7.1 | Corruption suite: blur, brightness/exposure, resize/rescale, noise, JPEG, contrast — 5 severities each | `src/inspector/robustness/` |
| 7.2 | Severity calibration against a documented physical reference | `reports/robustness/calibration.md` |
| 7.3 | Evaluate the top 3 models across the full grid | MLflow |
| 7.4 | Degradation curves and a relative-robustness table | `reports/figures/robustness_*.png` |
| 7.5 | **AD 2 native lighting-shift split** as the real-world control | `reports/robustness/lighting.md` |
| 7.6 | Correlate synthetic-corruption ranking with the native-lighting ranking | `reports/robustness/validity.md` |

**7.6 is the phase's most valuable output.** Does synthetic lighting corruption predict behaviour
under *real* unseen lighting? If yes, synthetic robustness testing is validated as a cheap proxy —
useful far beyond this project. If no, that is a finding the literature needs, and it is publishable-
grade evidence about a practice the field relies on.

**Gate G7**
- [ ] Full grid evaluated for the top 3 models.
- [ ] Degradation curves plotted with the clean baseline marked.
- [ ] Threshold stability quantified: how far does realized FPR at `OP-FPR1` drift under corruption?
      (This is the deployment-critical number and is usually worse than the AUROC drop suggests.)
- [ ] Synthetic-vs-native correlation reported with its rank correlation coefficient.

---

## P8 — Advanced tier (Week 9)

| # | Task | Output |
|---|------|--------|
| 8.1 | Tier 4 comparators via anomalib, our pipeline and metrics | MLflow |
| 8.2 | Tier 5: tiling + stitching | `src/inspector/models/tiling.py` |
| 8.3 | Tier 5: lighting-robust memory augmentation | same |
| 8.4 | Tier 5: adaptive binarization and score calibration | `src/inspector/postproc/` |
| 8.5 | Apply **DR-1** to each technique; adopt or reject, in writing | `reports/adoption_log.md` |
| 8.6 | Final model selection under DR-4 | `configs/final.yaml` |

**Gate G8**
- [ ] Every Tier 5 technique has an explicit adopt/reject decision with its test statistic.
- [ ] Rejected techniques are in `reports/negative-results.md` with configs tried.
- [ ] The final configuration is chosen by validation, and any test-selected element is labelled.
- [ ] Comparator numbers come from *our* harness, not from quoted papers.

---

## P9 — Serving (Week 10)

| # | Task | Output |
|---|------|--------|
| 9.1 | FastAPI: `POST /predict` (image → score, map, mask, decision), `GET /health`, `GET /metadata` | `src/inspector/api/` |
| 9.2 | Gradio UI: upload, heatmap overlay, threshold slider, nearest-normal panel | `src/inspector/app/` |
| 9.3 | Multi-stage Dockerfile, CPU and CUDA variants; non-root user; pinned base digest | `Dockerfile` |
| 9.4 | Latency benchmark harness: p50/p95/p99, three devices, batch 1 | `scripts/bench_latency.py` |
| 9.5 | Load test (concurrency sweep) and memory high-water mark | `reports/serving/` |
| 9.6 | API contract tests + Docker smoke test in CI | `tests/api/`, CI job |
| 9.7 | Model artifact versioning: memory bank + config + threshold as one signed bundle | `artifacts/` |

**Gate G9**
- [ ] `docker run` → documented `curl` → correct JSON with a base64 heatmap.
- [ ] p50/p95/p99 reported for laptop GPU, Colab GPU, and CPU.
- [ ] Container image size recorded; cold-start time measured.
- [ ] The container runs with **no network access at inference time** (no weight download at
      startup — bake weights into the image, or fail loudly).
- [ ] API rejects malformed input with a 4xx and a useful message; tested.

---

## P10 — Documentation and audit (Week 11)

| # | Task | Output |
|---|------|--------|
| 10.1 | Model card from [templates/model-card.md](templates/model-card.md) | `MODEL_CARD.md` |
| 10.2 | Benchmark report: headline table, Pareto, robustness curves, case book links | `reports/BENCHMARK.md` |
| 10.3 | README rewrite with results and quickstart | `README.md` |
| 10.4 | **Reproducibility audit**: fresh clone, fresh env, follow your own docs verbatim | `reports/repro_audit.md` |
| 10.5 | License compliance check; attribution verified | — |
| 10.6 | Negative results and limitations consolidated | `reports/negative-results.md` |

**Gate G10**
- [ ] A fresh clone reproduces the headline table (documented tolerance for stochastic rows).
- [ ] Model card complete, including intended use, out-of-scope use, and the NC license restriction.
- [ ] Every table caption states split, threshold source, and oracle status.
- [ ] The repro audit lists every step where your own documentation was wrong, and the docs are fixed.

---

## P11 — Buffer and stretch (Week 12)

Buffer first. If there is time left: leaderboard submission (max 2, per §4.3 of the protocol),
SAM mask refinement, ONNX/TensorRT export with a re-measured latency table, a fourth category as
a generalization check, or a short write-up of the PatchCore calibration finding.

---

## 2. Compute allocation

| Workload | Device | Est. GPU-hours |
|----------|--------|----------------|
| Tier 0–2 + feature caching | Laptop (4 GB) | 8 |
| Tier 1 training, 3 seeds, ablations | Laptop | 20 |
| PatchCore ≤ 448² | Laptop | 12 |
| PatchCore ≥ 512², tiled, sweep | **Colab T4** | 35 |
| Tier 4 comparators | **Colab T4/A100** | 25 |
| Tier 5 techniques | **Colab** | 20 |
| Robustness (inference only, cached features) | Laptop | 10 |
| Serving benchmarks | Both | 4 |
| **Total** | | **~134 GPU-hours** (~100 on Colab) |

**Colab discipline — non-negotiable rules:**

1. Notebooks are **thin drivers only**: mount Drive, `pip install -e .` from the repo, call the
   same CLI the laptop calls. No model code in a notebook, ever. Code that exists only in a
   notebook cannot be tested, reviewed, or reproduced.
2. Every run is **checkpointed and resumable**; assume the session dies at any moment.
3. Cap any single run at ~2 hours. Split sweeps into queued chunks.
4. Each machine logs to its own `mlruns/` directory; merge with `mlflow-export-import` rather than
   writing concurrently to one SQLite backend over Drive (which will corrupt).
5. Track Colab hours in `reports/compute_ledger.md`. The `walnuts` → `can` swap rule in the
   protocol triggers off this ledger.
6. Re-verify dataset hashes after any Drive sync. Drive sync corruption is silent and will cost
   you a week if it goes unnoticed.

---

## 3. Known-hard engineering problems, called out in advance

These are the tasks most likely to blow their estimate. Each is scheduled with slack.

| Problem | Where | Why it is hard | Planned approach |
|---------|-------|----------------|------------------|
| **Coreset on a huge patch set** | P5 | `walnuts` at 512²: 432 images × 4096 patches ≈ 1.77 M patches × 1024 dims. In fp32 that is ~7 GB — it does not fit in 4 GB VRAM, and greedy k-center is O(N·k) distance computations. | Extract in chunks to an fp16 memmap on disk; random pre-subsample to ~10%; run greedy k-center on the subsample with distances computed in GPU-sized tiles; keep the running min-distance vector on GPU (N floats only). Verify the coreset is a genuine subset and that its size matches the ratio. |
| **Native-resolution inference in 4 GB** | P5, P8 | `sheet_metal` is 4224×1056. A full forward pass at native resolution will OOM. | Overlapping tiled inference with cosine-weighted stitching; check seams explicitly in the case book. |
| **AU-PRO at multi-megapixel scale** | P2 | Connected components and threshold sweeps over 8 MP × 150 images is slow in pure Python. | Vectorize with `scipy.ndimage`; subsample the threshold grid to a fixed size and record it; cache per-image region maps. Profile it — a slow metric silently discourages re-evaluation, which corrupts the protocol. |
| **fp16 numerical behaviour** | P5 | Distance computations in fp16 can lose precision at large magnitudes. | Store features in fp16, accumulate distances in fp32. Ablate fp32 vs fp16 and report the accuracy delta rather than assuming it is zero. |
| **Threshold transfer under lighting shift** | P7 | A validation-derived threshold may be badly wrong on the shifted split. | This is a *result*, not a bug. Measure and report the FPR drift; it is one of the project's more useful findings. |

---

## 4. Compressed 8-week lane

If the schedule must shrink, cut **scope**, never gates.

- Merge P0 and P1 into one week (skip the pHash duplicate analysis, keep L1–L5).
- P3: drop VAE and OC-SVM; keep CAE-L2, CAE-SSIM, Deep SVDD.
- P4: drop SPADE; keep Mahalanobis and PaDiM.
- P5: keep the full reproduction gate (never cut this), reduce the ablation sweep to backbone,
  resolution, and coreset only.
- P7: 4 corruptions × 3 severities, still including the native-lighting control.
- P8: Tier 4 reduced to EfficientAD only; Tier 5 reduced to tiling and adaptive binarization.
- P11 buffer absorbed.

**Never cut:** G0 (no dataset in git), G2 (metric validation), G5 (reproduction), G10 (repro audit).
These four are what separate a project whose numbers mean something from one whose numbers do not.
