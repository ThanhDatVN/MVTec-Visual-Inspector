# 12 — Running the Notebooks

Two notebooks split the work along the only line that matters here: what fits in 4 GB of laptop
VRAM, and what does not.

| Notebook | Venue | Wall-clock | Covers |
|----------|-------|-----------|--------|
| [`01_laptop_data_and_eda.ipynb`](../notebooks/01_laptop_data_and_eda.ipynb) | Laptop, RTX 3050 | ~4 h | Stages A, B, D of [the register](11-experiment-plan.md): acquisition, audit, EDA, Tier 0, the PatchCore reference |
| [`02_kaggle_experiments.ipynb`](../notebooks/02_kaggle_experiments.ipynb) | Kaggle P100/T4 | ~38 GPU-h across sessions | Stages D–G: reference config, ablation screen, autoencoder, robustness |

Both are **generated** by `scripts/build_notebooks.py` and must not be hand-edited. A notebook
edited in place accumulates hidden state, stale outputs, and cells that ran once in an order
nobody recorded. Edit the builder and regenerate; a test enforces that the committed notebooks
match what the builder emits.

```bash
python scripts/build_notebooks.py
```

---

## 1. Notebook 01 — laptop

### Prerequisites

Everything installs into the project's virtual environment, never the system Python:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -e ".[notebook,dev]"
.venv/Scripts/python -m ipykernel install --user --name mvi --display-name "MVI (.venv)"
```

`requirements-lock.txt` pins the versions the committed results were produced with.

> **Windows note.** If `import torch` raises `OSError: [WinError 4551] An Application Control
> policy has blocked this file`, Smart App Control is inspecting the unsigned CUDA DLLs. In
> practice this cleared itself once the reputation check completed; if it persists, the options
> are to turn Smart App Control off (Windows Security → App & browser control — **irreversible
> without a Windows reset**), use WSL2, or run all torch work on Kaggle.

### What each section does

| § | Section | Notes |
|---|---------|-------|
| 0 | Environment | Prints git SHA and the dirty flag. **A dirty tree tags every run dirty, and dirty runs may not enter the headline table.** Commit before a reportable run. |
| 1 | Acquire VisA | 1.93 GB over HTTPS, no account. Records a SHA-256 manifest; re-run `verify` after any sync. |
| 2 | The official split | Prints the per-category table including each category's **achievable FPR floor** — see [ADR-7 and ADR-9](07-risks-and-decisions.md). |
| 3 | Study categories | `pcb1`, `macaroni2`, `capsules`, one per structural group. The other nine are frozen confirmation categories (ADR-10): the notebook never opens their images. |
| 4 | **Split integrity** | Rules L1–L7. The notebook `assert`s here; nothing downstream runs if it fails, by design |
| 5 | **EDA and the resolution decision** | The section that decides the study's most consequential parameter |
| 6 | Visual check | Always look at the data before trusting a statistic about it |
| 7 | Tier 0 floors | Gate G2's control, at the common 320 px, through the shared runner and registry |
| 8 | PatchCore reference | The corrected reference (`configs/models/patchcore.yaml`, WRN50-2 at 320 px) × 3 seeds. VRAM is not the limit on this laptop; host RAM is, and candidate sampling bounds it |
| 9 | Qualitative check | From stored predictions, with a fixed colour scale — the number and the picture must agree |
| 10 | Hand-off | Appends to the compute ledger |

### Expected runtime

| Section | Time |
|---------|------|
| Download + extract | 15–30 min (network-bound) |
| Audit, 3 categories | ~3 min (SHA-256 over ~3,500 images) |
| EDA | ~5 min |
| Tier 0, 18 runs | ~20 min |
| PatchCore, 9 runs | measured in the registry (`fit_seconds`, `eval_seconds`) |

### If it fails

| Symptom | Cause | Fix |
|---------|-------|-----|
| `split CSV not found` | `1cls.csv` missing | Re-run the fetch cell; it copies the CSV into `<root>/split_csv/` |
| `AssertionError` in §4 | Split integrity | **Do not proceed.** Read the report — it names the offending files |
| CUDA OOM in §8 | 4 GB ceiling | Lower `batch_size` in the model config; the failed run is recorded in the registry |
| `MemoryError` in §8 | Host RAM for the patch matrix | Lower `candidate_fraction`; the run id changes with it, so nothing is silently reused |
| `image listed in the split CSV is missing on disk` | Partial extraction | Delete the extracted dir and re-run the fetch |

---

## 2. Notebook 02 — Kaggle

### Settings

| Setting | Value | Why |
|---------|-------|-----|
| Accelerator | **GPU P100** (16 GB) | T4×2 also works; the code uses one device |
| Internet | **On** | Required to fetch VisA and the package |
| Persistence | Variables **and** files | Otherwise the checkpoint is lost between sessions |

Limits to plan around: ~30 GPU-h/week, 12 h interactive session, ~9 h on commit, 20 GB working
directory.

### Getting the package there

The notebook needs a checkout, not only the package, because run configurations live in
`configs/`. Three routes, in the notebook's order of preference:

1. **Local checkout** — when the same notebook runs on the laptop.
2. **Kaggle dataset** — zip the repo, upload via *Add Data → Upload* as
   `mvtec-visual-inspector`; the notebook adds its `src/` to `sys.path`.
3. **`REPO_URL`** — a shallow clone of the GitHub repository (Internet on). Set `REPO_REF` to a
   commit or tag for a run you intend to report; the notebook `chdir`s into the clone so every
   run records that commit.

Only packages missing from Kaggle's image are installed (`opencv-python-headless`, `pyyaml`).
The notebook never upgrades numpy or torch inside a running kernel.

### Getting the data there

Either attach VisA as a Kaggle dataset at `/kaggle/input/visa-anomaly/VisA_20220922`, or let the
notebook download it (1.93 GB, a few minutes on Kaggle's network). Attaching is better across
multiple sessions: it costs nothing per session and survives restarts.

### The run registry — the part that matters

Every run goes through `inspector.runner`, the same code path as the CLI and the laptop notebook.
A run's id is a hash of everything that can change its numbers (docs/11 §2), and each run writes
`/kaggle/working/runs/<run_id>/`. A restarted session **reuses** completed runs and reloads their
results, so the summary tables cover the whole registry and not only this session. A failed run
(for example an out-of-memory at 448 px) leaves a `failed` record with its reason, and the sweep
continues.

**Without this, the 12-hour session limit turns a 20-hour sweep into a gamble.** With it, the
sweep is simply run across as many sessions as it takes. To force a rerun, delete that run's
directory.

The robustness grid checkpoints per cell to `robustness.csv` and skips cells already present.

**Download `runs_export.zip` (and `robustness.csv`) before the session ends.** Kaggle discards
`/kaggle/working` unless the notebook is committed or the outputs are saved.

### Stage sizing

Each stage is sized to finish inside ~4 hours, comfortably under the 9-hour commit cap:

| Cells | Stage | Est. |
|-------|-------|------|
| §3 | Reference = the ADR-11 recipe (640 px, fixed 10 k bank), 9 runs; also runnable on the laptop | ~1 GPU-h |
| §4 | Ablation screen around it, ~51 runs, including 960 and 1280 px | ~12 GPU-h — **split across several sessions** |
| §5 | Autoencoder at 640 px, 3 losses × 3 seeds × 3 categories, up to 150 epochs | ~6 GPU-h |
| §6 | Robustness grid, one fit per category | ~4 GPU-h |

All of it runs on the **development** categories. The nine confirmation categories are touched
only by the pre-registered ADR-11 runs.

Run §4 in chunks by commenting out axes in the `AXES` dict. The registry makes the chunking
invisible in the results.

### If it fails

| Symptom | Cause | Fix |
|---------|-------|-----|
| `git clone` fails | Internet off | Turn Internet on, or attach the repo as a dataset |
| Session killed mid-sweep | 12 h limit | Re-run the notebook; completed runs are reused |
| Quota exhausted | 30 GPU-h/week | Cut E6/E6b before cutting seeds |
| `FAILED ... out of memory` at 448 px | Host RAM for the patch matrix, not VRAM | Lower `candidate_fraction` for that row and say so; the failure record stays in the registry |

---

## 3. Merging results back

```bash
# after a Kaggle session: unzip into the repository registry, then regenerate the table
unzip ~/Downloads/runs_export.zip -d reports/runs/
inspector results --registry reports/runs --out reports/results_registry.csv --markdown
```

Run directories are content-addressed, so merging registries from several sessions or machines
cannot overwrite a different run. `reports/results_registry.csv` is **generated, never
hand-edited** (docs/06 §4). A hand-edited table drifts from the runs that produced it, and by the
time anyone notices, nobody remembers which version was right. The `*.maps.npz` files are
git-ignored derived data; keep them locally for the E02/E03 studies.

---

## 4. What the notebooks deliberately do not do

| Not in a notebook | Where it lives | Why |
|-------------------|----------------|-----|
| Model definitions | `src/inspector/models/` | Code in a cell cannot be tested, reviewed, or re-run |
| Metric implementations | `src/inspector/metrics/` | Gate G2 validates them under `pytest`, not by eye |
| Threshold logic | `src/inspector/postproc/` | Rules L3/L4 are enforced structurally; a notebook could bypass them |
| Corruption functions | `src/inspector/robustness/` | Wrong corruption code is invisible — a corrupted image still looks corrupted |

A test fails the build if any cell defines a class, an `nn.Module`, or calls `torch.optim`. The
notebooks are drivers; the project is the package.

---

## 5. Reproducing a single number

Every row in `results.csv` carries `run_name`, `config_hash`, `git_sha` and `notes` (the run ID).
To reproduce one:

```bash
git checkout <git_sha>
inspector run -c configs/data/visa_pcb1.yaml --methods patchcore --seeds 0 \
              --data-root data/raw/VisA_20220922
```

If `dirty` was `True` for that row, the commit does not fully describe it and the number is not
reproducible — which is why dirty runs are barred from the headline table rather than merely
flagged.
