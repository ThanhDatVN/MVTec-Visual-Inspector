# 12 — Running the Notebooks

Two notebooks split the work along the only line that matters here: what fits in 4 GB of laptop
VRAM, and what does not.

| Notebook | Venue | Wall-clock | Covers |
|----------|-------|-----------|--------|
| [`01_laptop_data_and_eda.ipynb`](../notebooks/01_laptop_data_and_eda.ipynb) | Laptop, RTX 3050 | ~4 h | Stages A–C of [the register](11-experiment-plan.md): acquisition, audit, EDA, Tier 0, development PatchCore |
| [`02_kaggle_experiments.ipynb`](../notebooks/02_kaggle_experiments.ipynb) | Kaggle P100/T4 | ~41 GPU-h across sessions | Stages D–G: reference config, ablation sweep, autoencoder, robustness |

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

```bash
python -m pip install -e ".[torch,track,dev]"
python -m pip install matplotlib          # notebook-only, not a package dependency
```

Torch with CUDA for the RTX 3050:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

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
| 2 | The official split | Prints the per-category table including each category's **achievable FPR floor** — see [ADR-7](07-risks-and-decisions.md). |
| 3 | Study categories | `pcb1`, `macaroni2`, `capsules`, one per structural group |
| 4 | **Split integrity** | Rules L1–L7. The notebook `assert`s here; nothing downstream runs if it fails, by design |
| 5 | **EDA and the resolution decision** | The section that decides the study's most consequential parameter |
| 6 | Visual check | Always look at the data before trusting a statistic about it |
| 7 | Tier 0 floors | Gate G2's control |
| 8 | Development PatchCore | `resnet18` at 256², fits 4 GB |
| 9 | Qualitative check | The number and the picture must agree |
| 10 | Hand-off | Appends to the compute ledger |

### Expected runtime

| Section | Time |
|---------|------|
| Download + extract | 15–30 min (network-bound) |
| Audit, 3 categories | ~3 min (SHA-256 over ~3,500 images) |
| EDA | ~5 min |
| Tier 0, 36 runs | ~25 min |
| PatchCore ×3 | ~20 min |

### If it fails

| Symptom | Cause | Fix |
|---------|-------|-----|
| `split CSV not found` | `1cls.csv` missing | Re-run the fetch cell; it copies the CSV into `<root>/split_csv/` |
| `AssertionError` in §4 | Split integrity | **Do not proceed.** Read the report — it names the offending files |
| CUDA OOM in §8 | 4 GB ceiling | Lower `RESOLUTION`, or `batch_size=2`, or `backbone='resnet18'` |
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

Three routes, in the notebook's order of preference:

1. **`REPO_URL`** — set it to a pushed repository and the notebook `pip install`s from git. The
   cleanest route, and the only one where the notebook records which commit it ran.
2. **Kaggle dataset** — zip the repo, upload via *Add Data → Upload*, and the notebook adds
   `src/` to `sys.path`. Use this until the repo is pushed.
3. **Local fallback** — for running the same notebook off Kaggle.

### Getting the data there

Either attach VisA as a Kaggle dataset at `/kaggle/input/visa-anomaly/VisA_20220922`, or let the
notebook download it (1.93 GB, a few minutes on Kaggle's network). Attaching is better across
multiple sessions: it costs nothing per session and survives restarts.

### The checkpoint — the part that matters

Every completed run appends to `runs.jsonl`, keyed by its run ID. A restarted session skips what
is already done.

**Without this, the 12-hour session limit turns a 25-hour sweep into a gamble.** With it, the
sweep is simply run across as many sessions as it takes.

```python
# resume is automatic; to force a re-run, delete its key
CHECKPOINT = WORK / 'runs.jsonl'
```

**Download `runs.jsonl` and `results_visa.csv` before the session ends.** Kaggle discards
`/kaggle/working` unless the notebook is committed or the outputs are saved. Merge them into the
repository's `reports/` afterwards.

### Stage sizing

Each stage is sized to finish inside ~4 hours, comfortably under the 9-hour commit cap, so no
stage straddles a session boundary:

| Cells | Stage | Est. |
|-------|-------|------|
| §3 | Reference config, 9 runs | ~4 GPU-h |
| §4 | Ablation sweep, ~72 runs | ~25 GPU-h — **split across 3+ sessions** |
| §5 | Autoencoder, ~90 runs | ~12 GPU-h |
| §6 | Robustness grid | ~3 GPU-h |

Run §4 in chunks by commenting out axes in the `AXES` dict. The checkpoint makes the chunking
invisible in the results.

### If it fails

| Symptom | Cause | Fix |
|---------|-------|-----|
| `inspector package not found` | No `REPO_URL`, no attached dataset | Set one of the three routes in §0 |
| Session killed mid-sweep | 12 h limit | Re-run the notebook; the checkpoint resumes |
| Quota exhausted | 30 GPU-h/week | Check the ledger cell; cut E5/E6 before cutting seeds |
| OOM on P100 | Resolution × batch size | Lower `batch_size` in `REFERENCE`; 16 GB should hold 448² at batch 8 |

---

## 3. Merging results back

```bash
# after a Kaggle session
cp ~/Downloads/results_visa.csv reports/
cp ~/Downloads/runs.jsonl       reports/
```

`reports/results.csv` is **generated from MLflow, never hand-edited** (docs/06 §4). A hand-edited
table drifts from the runs that produced it, and by the time anyone notices, nobody remembers
which version was right.

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
