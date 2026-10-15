# Changelog

## 1.0.1 — 2026-10-15

- The Kaggle notebook (`02_kaggle_experiments.ipynb`) and the Kaggle upload paths are removed.
  This covers the MPDD `--zip` archive, the repository-as-dataset option, and the private-dataset
  instructions. The experiment definitions the notebook held are now
  `scripts/visa_k1_experiments.py`, which runs the same groups through the shared runner on any
  CUDA machine. Its analysis script is renamed `scripts/visa_k1_analysis.py`.
- Four notebooks, generated, in project order:
  - `01_laptop_data_and_eda`;
  - `02_visa_gpu_experiments`, which replaces the Kaggle notebook and has no platform-specific code;
  - `03_mvtec_ad2_pipeline`, from E13 to the v6 export;
  - `04_results_and_figures`, which needs no data.

  `python -m inspector` now runs the command line.
- Final report §6.6–6.7: each component against the basic technique and the alternatives; v6
  against the published systems.

## 1.0.0 — 2026-10-15

Final release after the second and last MVTec AD 2 server submission (S2).

**Results**

- v6 on MVTec's server: AU-PRO@0.05 62.97 / 60.73, SegF1 53.76 / 51.67, ClassF1 79.28 / 79.45
  (`test_private` / `test_private_mixed`). It meets the pre-registered H5 bar on both splits
  (ADR-17).
- `reports/00-final-report.md`: one technical report covering the whole project, with eight
  figures generated from committed result files (`scripts/report_figures.py`).

**Layout**

- Reports are numbered in reading order (`reports/00-final-report.md` …
  `reports/14-negative-results.md`). `reports/README.md` is the experiment register.
- Scripts are named `<stage>_<experiment>_<topic>.py`, with stages `data`, `visa`, `ad2`,
  `server`, `deploy` and `report`. `scripts/README.md` gives the index and each script's former
  name.
- Package modules were renamed to say what they do: `analysis` → `studies`,
  `report` → `eda_report`, `models.features` → `models.backbone`,
  `models.anomalib_ref` → `models.anomalib_patchcore`. The empty `features` package was removed.
- `docs/` keeps the research documents. The notebook how-to and the Vietnamese guides stay local.

**Fixes**

- The CI lint job failed from 2026-10-12 on a mypy error in `evaluate.py`; fixed.

Renaming package files changes the implementation id (a hash of the package source) of any *new*
run. Stored runs keep the ids they were made with, and the frozen analyses select runs by those
stored ids.

## 0.x — 2026-09-22 … 2026-10-14

Protocol v1 and v2, VisA development and confirmation, MPDD, MVTec AD 2 development and the first
server submission (S1). See the experiment register in `reports/README.md`.
