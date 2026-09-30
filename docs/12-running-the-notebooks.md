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

Runs what the laptop cannot: an ablation screen around the frozen recipe (including 960 and
1280 px, which need more than 16 GB of host RAM) and the autoencoder at 640 px with a converging
schedule. Development categories only.

### Step by step

1. **Account.** Kaggle needs a phone-verified account for GPU and Internet.
2. **Import.** *Create → New Notebook → File → Import Notebook* →
   `notebooks/02_kaggle_experiments.ipynb` (upload the file, or give the GitHub URL).
3. **Settings** (right-hand panel):

   | Setting | Value | Why |
   |---------|-------|-----|
   | Accelerator | **GPU P100** (16 GB) or T4 ×2 | the code uses one device |
   | Internet | **On** | clones the repository and downloads VisA |
   | Persistence | not needed | the registry is carried by the output, see step 6 |

4. **Data.** Nothing to prepare: the notebook downloads VisA (1.93 GB, SHA-256 verified) into
   `/tmp` in a couple of minutes. If you prefer to attach a copy (*Add Input → Datasets*), any
   dataset whose files contain `split_csv/1cls.csv` within three folder levels is found
   automatically; otherwise it falls back to the download.
5. **Choose what the session runs** in the first code cell:

   | session | `RUN_SWEEP` | `RUN_AE` | `RUN_ROBUSTNESS` | est. P100 time |
   |---|---|---|---|---|
   | 1 | `True` | `False` | `False` | ~4–6 h |
   | 2 | `False` | `True` | `False` | ~3–6 h |

   The PatchCore reference (3 runs × 3 categories) always runs first, or is reused. Robustness
   was already run on the laptop (`reports/E08-robustness.md`); enable it only to reproduce it.
6. **Run unattended:** *Save Version → Save & Run All (Commit)*. The commit keeps running after
   you close the browser, and saves `/kaggle/working` as the version's output.
7. **Next session:** open the notebook again, *Add Input → Your Work →* this notebook (its
   latest version), flip the flags, and commit again. The first cells unzip every attached
   `runs_export.zip` into the registry; completed runs are recognised by their content hash and
   skipped, so a killed session loses at most the run that was in progress.
8. **Bring the results home:** from the version's *Output* tab download `runs_export.zip`
   (and `robustness.csv` if you ran it), then merge them as in §3.

### What is where

| Path | Content | Saved as output? |
|---|---|---|
| `/tmp/mvi/repo` | shallow clone of the repository (`REPO_REF` pins a commit) | no |
| `/tmp/mvi/data` | VisA, tar deleted after extraction | no |
| `/kaggle/working/runs/` | the run registry: spec, result, predictions, raw maps | yes |
| `/kaggle/working/runs_export.zip` | registry without map files over 20 MB (the CAE's full-resolution maps) | yes — **this is the file to download and to attach next time** |
| `/kaggle/working/results_registry.csv`, `session_summary.md` | tables for a quick look | yes |

### Checking the notebook without a GPU session

`MVI_SMOKE=1` runs every section on one category with one variant per axis and a one-epoch
autoencoder, into a separate registry (`smoke_runs`). It exists to test the notebook itself;
its numbers mean nothing. Locally: `MVI_SMOKE=1 MVI_DATA_ROOT=<VisA root> jupyter nbconvert
--to notebook --execute notebooks/02_kaggle_experiments.ipynb` (~15 min on the laptop).

### If it fails

| Symptom | Cause | Fix |
|---------|-------|-----|
| `git clone` fails | Internet off, or account not phone-verified | turn Internet on; or attach the repository as a dataset named `mvtec-visual-inspector` |
| Download stalls | network | re-run the data cell; a completed extraction is reused |
| Session killed mid-run | 12 h limit or a crash | commit again with the previous output attached (step 7) |
| `FAILED ... out of memory` at 1280 px | host RAM for the candidate pool | recorded as a failed run; lower `candidate_fraction` for that row and say so |
| Quota exhausted | ~30 GPU-h/week | the registry resumes next week; the AE section is the one to postpone |

---

## 3. Merging results back

```bash
# after a Kaggle session: unzip into the repository registry, then regenerate tables
unzip -n ~/Downloads/runs_export.zip -d reports/runs/
inspector results --registry reports/runs --out reports/results_registry.csv
inspector study --registry reports/runs --data-root data/raw/VisA_20220922                 --implementation <id prefix of the Kaggle runs> --out reports/studies/kaggle
```

Run directories are content-addressed, so merging registries from several sessions or machines
cannot overwrite a different run (`-n` never overwrites). `reports/results_registry.csv` is
**generated, never hand-edited** (docs/06 §4). The Kaggle runs carry their own implementation id
(a Linux checkout of the same commit hashes to the same id as any other checkout, since line
endings are normalized), and `inspector study --implementation` keeps one study to one code
version.

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
