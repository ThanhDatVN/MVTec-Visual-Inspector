# 06 — Engineering, MLOps, and Test Strategy

---

## 1. Repository layout

```
MVTec Visual Inspector/
├── README.md, LICENSE, ATTRIBUTION.md, NOTICE, MODEL_CARD.md
├── pyproject.toml, requirements.txt, requirements-lock.txt
├── Dockerfile, docker-compose.yml, .dockerignore
├── .github/workflows/ci.yml
├── configs/                      # one YAML per experiment; the unit of reproducibility
│   ├── base.yaml
│   ├── data/{sheet_metal,fruit_jelly,walnuts}.yaml
│   ├── models/{cae,dsvdd,padim,patchcore,...}.yaml
│   └── final.yaml
├── src/inspector/
│   ├── data/          # datasets, splits, manifests, transforms
│   ├── features/      # backbone extraction + content-addressed cache
│   ├── models/        # cae, vae, dsvdd, mahalanobis, padim, spade, patchcore, tiling
│   ├── postproc/      # normalization, thresholding, binarization, calibration
│   ├── metrics/       # aupro, aupimo, image metrics, segf1
│   ├── robustness/    # corruptions + severity calibration
│   ├── viz/           # heatmaps, gradcam, retrieval, case-book rendering
│   ├── api/           # FastAPI service
│   ├── app/           # Gradio UI
│   ├── tracking.py    # MLflow wrapper; records git sha, config hash, env
│   ├── stats.py       # bootstrap CIs, paired Wilcoxon, Holm-Bonferroni
│   └── cli.py         # `inspector fit|predict|evaluate|bench|report`
├── scripts/           # sweep drivers, results generation, benchmarks
├── notebooks/         # THIN Kaggle drivers only — no model code
├── tests/
│   ├── fixtures/      # synthetic images + masks (committed, license-safe)
│   ├── data/ metrics/ models/ robustness/ api/ integration/
│   └── golden/        # stored expected metric values
├── reports/           # eda, case-book, figures, benchmark, negative results, ledgers
└── data/              # GITIGNORED except manifests/
    └── manifests/*.sha256
```

**One principle above all: everything runs through `inspector` CLI + a YAML config.** The laptop
and Kaggle execute the *same* code path with the same configs. A result that can only be produced
by a notebook cell is not a result; it cannot be re-run, reviewed, or trusted six weeks later.

---

## 2. Configuration and reproducibility

- **Config = the experiment.** Every run is `inspector fit --config configs/....yaml --seed 0`.
- Configs compose (`base.yaml` ← data ← model ← overrides) and every run stores its **fully
  resolved** config, not the composition — resolved configs are what you can actually re-run.
- **`config_hash`** = SHA-256 of the resolved config, and it is the primary experiment key.
- Every run records: git SHA + dirty flag, config hash, dataset manifest hash, `pip freeze`,
  device name, driver/CUDA version, and the RNG seeds actually used.
- **A dirty git tree marks the run `dirty=True` in MLflow, and dirty runs may not enter the
  headline table.** This one rule prevents more irreproducibility than any other.

---

## 3. VRAM and memory budget

> **Corrected by measurement (2026-09-22).** This section previously called the 4 GB card "the
> binding constraint". It is not. Measured peak VRAM for a WideResNet50-2 `layer2+layer3` forward
> pass on the RTX 3050:
>
> | input | MP | batch 1 | batch 2 | batch 4 |
> |-------|-----|---------|---------|---------|
> | 256×195 | 0.05 | 307 MB | 322 MB | 354 MB |
> | 512×390 | 0.20 | 364 MB | 431 MB | 592 MB |
> | 1024×780 | 0.80 | 642 MB | 904 MB | 1535 MB |
> | 1404×1070 | 1.50 | 959 MB | 1459 MB | 2645 MB |
>
> Native VisA resolution costs under 1 GB at batch 1. **The binding constraint is the patch matrix
> in host RAM**: 23,584 patches per image at 1404×1070 × 768 training images = 18.1 M patches =
> **37 GB in fp16**, beyond this laptop (16.3 GB) and a Kaggle session (~30 GB) alike.
>
> The consequence replaces a planned mitigation. Tiling was scheduled to solve a VRAM problem that
> does not exist; what is actually needed is **per-image patch subsampling during extraction**, so
> the matrix never fully materializes. The current implementation subsamples only *after*
> accumulating everything, so the extraction peak is what binds.
> Full measurements: [reports/P3-visa-findings.md](../reports/P3-visa-findings.md) §2.2.

The arithmetic below remains the right way to plan; only the conclusion about which resource binds
has changed.

### PatchCore memory-bank arithmetic

For input H×W, `layer2` features have stride 8, so patches = (H/8)·(W/8). After aggregation and
projection, each patch is a 1024-d vector.

| Category | Input | Patches/img | Train imgs | Total patches | fp32 size | fp16 size | 1% coreset |
|----------|-------|-------------|-----------|---------------|-----------|-----------|------------|
| `sheet_metal` | 1024×256 | 4 096 | 137 | 561 k | 2.3 GB | 1.1 GB | 23 MB |
| `fruit_jelly` | 448×320 | 2 240 | 263 | 589 k | 2.4 GB | 1.2 GB | 24 MB |
| `walnuts` | 512×512 | 4 096 | 432 | 1.77 M | **7.2 GB** | 3.6 GB | 72 MB |
| `walnuts` | 1024×1024 | 16 384 | 432 | 7.08 M | **29 GB** | 14.5 GB | 290 MB |

**Consequences, decided in advance rather than discovered at 2 a.m.:**

1. The full patch matrix **never lives in VRAM**. Extract in chunks, write to an **fp16 disk
   memmap**, and stream.
2. Greedy k-center coreset runs on a **random pre-subsample** (10%) with distances computed in
   GPU-sized tiles. Only the running min-distance vector (N floats) and the current tile stay
   resident.
3. Accumulate distances in **fp32** even when features are fp16.
4. Above ~512², use **tiled inference**. `sheet_metal` at native resolution is tiling-only.
5. Budget: fitting must stay under **3.5 GB** peak VRAM; inference under **2.5 GB** (the container
   must also run alongside a desktop session).
6. Every run logs `torch.cuda.max_memory_allocated()`. An OOM that happens twice is a planning
   failure, not bad luck.

---

## 4. MLflow

- **Backend:** local `mlruns/` per machine; merged with `mlflow-export-import`.
  Do **not** point two machines at one SQLite file on Google Drive — it will corrupt.
- **Logged per run:** params (full resolved config, flattened), metrics (every column of the
  canonical schema), artifacts (anomaly maps for a fixed 10-image subset, heatmap grid, training
  curves, resolved config, environment snapshot), and tags (`tier`, `category`, `owner=own|lib`,
  `dirty`, `oracle`).
- **Fixed-subset artifacts matter more than they look.** Logging the same 10 images' anomaly maps
  for every run gives you a visual diff across the entire project history for free, and it is how
  you notice that a "2-point improvement" is actually a normalization change.
- `reports/results.csv` is **generated** from MLflow by `scripts/make_results.py`. Never edited by
  hand. Hand-edited results tables drift from reality, always.

---

## 5. Test strategy

Five tiers. CI runs tiers 1–4 on CPU with synthetic fixtures; tier 5 is local, gated by a marker.

### Tier 1 — Unit (fast, hermetic, no data)

| Area | Tests |
|------|-------|
| Metrics | Analytic cases (perfect / random / inverted predictors); AU-PRO at both integration limits; AU-PRO invariance to a monotone score transform; SegF1 vs a hand-computed confusion matrix; AUPIMO shape and range |
| Transforms | Determinism; aspect-preservation; mask stays binary after resize; INTER_AREA used for downscale |
| Coreset | Output is a strict subset of the input; size matches the ratio; determinism given a seed; the k-center objective is non-increasing as the coreset grows |
| Scoring | Anomaly map shape equals native image shape; score is invariant to batch size; a memory bank containing an image's own features gives it near-zero score |
| Postproc | Thresholds carry `source_split`; normalizer is fit-once-then-frozen |
| Config | Resolved config round-trips; `config_hash` is stable across dict ordering |

### Tier 2 — Property / invariant

- Identity transform does not change any score (any model).
- Monotone rescaling of an anomaly map leaves AUROC and AU-PRO unchanged but *does* change SegF1 —
  asserting this pins down which metrics are threshold-dependent, which is the exact distinction
  behind PatchCore's SegF1 collapse.
- A constant image produces a finite, non-NaN score.
- Severity-0 corruption is the identity (see [05](05-robustness-protocol.md) §6).

### Tier 3 — Data contract (needs the dataset; skipped in CI with a clear skip reason)

L1–L7 leakage tests from [01](01-charter-and-protocol.md) §5, plus per-category count and
resolution assertions and a mask/image pairing check.

### Tier 4 — Golden regression

Run the full pipeline on the 20 synthetic fixtures; assert each metric is within tolerance of a
stored value in `tests/golden/`. **This is the test that catches a silent change in a
normalization constant three weeks after it happened** — the class of bug that otherwise costs a
re-run of every experiment.

Golden values are regenerated only by an explicit `--update-golden` flag, and the regeneration must
appear as its own commit with a written justification in the message. A golden file updated in the
same commit as the code change it was supposed to catch is not a test.

### Tier 5 — Integration and service

- End-to-end `fit → predict → evaluate` on fixtures, asserting the results CSV schema.
- FastAPI contract tests via `TestClient`: happy path, oversized image, wrong content type,
  corrupt bytes, missing field — each asserting both the status code and the error body.
- Docker smoke test in CI: build, run, `GET /health`, `POST /predict` with a fixture, assert 200
  and a well-formed response.
- Latency regression guard: assert p95 on the fixture set is under a generous ceiling, so an
  accidental `O(n²)` in postprocessing cannot reach main.

### CI pipeline (`.github/workflows/ci.yml`)

```
lint (ruff) → typecheck (mypy) → unit+property+golden (pytest, CPU)
  → integration (fixtures) → docker build + smoke → docs link check
```

Plus a **license guard job**: fails if any file with an image extension, or any path under `data/`
other than `manifests/`, is present in the commit. Cheap, and it enforces the one obligation whose
violation cannot be undone once pushed.

---

## 6. Serving

### API

| Endpoint | Behaviour |
|----------|-----------|
| `POST /predict` | multipart image → `{score, decision, threshold, threshold_source, anomaly_map_png_b64, mask_png_b64, latency_ms, model_version}` |
| `GET /health` | liveness + model-loaded flag |
| `GET /metadata` | model version, category, backbone, resolution, threshold and its provenance, training-set size, **license notice** |

`/metadata` returning the threshold *and its provenance* is a deliberate design choice: the single
most common production failure in anomaly detection is a threshold whose origin nobody remembers.

### Container

- Multi-stage build; CPU and CUDA variants; base image pinned **by digest**.
- Non-root user; read-only root filesystem where possible.
- **Model weights and memory bank baked into the image.** No network access at inference time — a
  container that downloads weights on start is a container that fails in an air-gapped factory.
- Healthcheck; graceful shutdown; image size recorded in the benchmark report.

### Gradio demo

Upload → score with the decision at the frozen threshold → heatmap overlay → adjustable threshold
slider (showing how the mask changes) → **nearest-normal-patch panel** ("closest matching good
region from training") → the license/attribution footer.

The nearest-normal panel is what makes the demo convincing to a non-ML audience. "This is a 0.83
anomaly score" persuades nobody; "this region matches none of the 432 good walnuts, the closest is
this one" persuades everybody.
