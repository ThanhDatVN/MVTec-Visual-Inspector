# Scripts

Study, analysis and export scripts. The reusable code lives in `src/inspector/`: models, metrics,
data, post-processing and the shared runner. These scripts call it and write their outputs under
`reports/studies/`.

**Naming:** `<stage>_<experiment>_<topic>.py`, so a directory listing follows the project.

| prefix | stage | report |
|---|---|---|
| `data_` | acquisition and audits | 03 |
| `visa_` | VisA development and confirmation (E03b–E12, K1) | 04–09 |
| `ad2_` | MVTec AD 2 development (E13–E23, open problems B3, R2, R4, R10) | 11–13 |
| `server_` | MVTec's server: proxy, expectation, export of the submissions (S1, S2) | 12–13 |
| `deploy_` | serving benchmark | 00 §5.6 |
| `report_` | figures and generated notebooks | 00 |

Run the scripts from the repository root, inside the project's virtual environment. Each script's
docstring gives its command line. Scripts that read a dataset expect it under `data/raw/`. Those
marked *stored* work from saved predictions in `reports/runs/` and refit nothing. The AD 2 scripts
import each other (for example `ad2_e23_fusion_head` imports `ad2_e21_mask_rules`), which is why
the folder stays flat.

## Data acquisition and audits

| script | purpose | former name |
|---|---|---|
| `data_fetch_mvtec_ad2.py` | Download MVTec AD 2 one category at a time, verify it, write a SHA-256 manifest | `fetch_mvtec_ad2.py` |
| `data_fetch_mpdd.py` | Download and verify MPDD (optionally zip it for Kaggle) | `fetch_mpdd.py` |
| `data_near_duplicate_audit.py` | A4: near-duplicate captures across splits (leakage rule L6) | `near_duplicate_audit.py` |

## VisA: development, robustness, confirmation

| script | experiment | former name |
|---|---|---|
| `visa_e03b_crossfit_calibration.py` | E03b: does a larger calibration pool repair the conformal threshold? *(stored)* | `crossfit_calibration.py` |
| `visa_e05_smoothing.py` | E05: how much the map smoothing decides the localization numbers *(stored)* | `smoothing_study.py` |
| `visa_e08_robustness.py` | E08: seven corruptions (and translation) × five severities, threshold frozen on clean data | `robustness_study.py` |
| `visa_e08b_input_guards.py` | E08b: do the exposure / focus guards catch what breaks the operating point? | `guard_study.py` |
| `visa_g4_translate_border.py` | G4: is the translation failure the shift or the replicated border strip? | `translate_border_study.py` |
| `visa_k1_kaggle_studies.py` | K1 and E8: Kaggle sweep, Gate E follow-up, autoencoder, anomalib comparison (C2), MPDD | `kaggle_studies.py` |
| `visa_e12_confirmation_analysis.py` | **E12**: the pre-registered confirmation of the frozen v1 recipe (ADR-11) | `confirmation_analysis.py` |
| `visa_pareto_figure.py` | Accuracy against latency and memory (`reports/figures/pareto.*`) | `pareto_figure.py` |
| `visa_case_book.py` | Case-book panels, rendered locally (they contain dataset pixels and are never committed) | `case_book.py` |

## MVTec AD 2: why the recipe broke (E13–E20)

| script | experiment | former name |
|---|---|---|
| `ad2_e13_confirmation_analysis.py` | **E13**: candidate v2 against v1 (ADR-13) | `ad2_confirmation_analysis.py` |
| `ad2_e14_capture_conditions.py` | `test_public` by capture condition; the object bootstrap used by later studies | `ad2_condition_study.py` |
| `ad2_e14_photometric_screen.py` | E14: per-image gain, standardization, exposure-augmented bank | `e14_analysis.py` |
| `ad2_e15_lighting_screen.py` | E15: flat-field normalization, relit bank | `e15_analysis.py` |
| `ad2_e16_fewshot_adaptation.py` | E16: a few new-lighting normals added to the bank | `e16_adaptation.py` |
| `ad2_e17_recalibration.py` | E17: recalibration under a new lighting | `e17_recalibration.py` |
| `ad2_e19_resolution.py` | E19: whole-image resolution for the thin-defect categories | `e19_analysis.py` |
| `ad2_b3_pixel_threshold.py` | Open problem B3: the 3σ pixel threshold against a validation pixel quantile *(stored)* | `pixel_threshold_study.py` |
| `ad2_r2_location_normalization.py` | Remedy R2: location-conditional map normalization *(stored)* | `location_normalization_study.py` |
| `ad2_r4_global_branch.py` | Remedy R4: an image-level branch beside the patch score | `global_branch_study.py` |
| `ad2_r10_shading_guard.py` | Remedy R10: a lighting-structure input guard | `shading_guard_study.py` |

## MVTec AD 2: localization rebuilt (E21–E23)

| script | role | former name |
|---|---|---|
| `ad2_e21_mask_rules.py` | Threshold and mask rules (MaskRule, image mark) for SegF1 / ClassF1 *(stored)* | `mask_rule_study.py` |
| `ad2_e21_fusion.py` | P95-normalized map fusion of two or three models *(stored)* | `fusion_study.py` |
| `ad2_e21_threshold_transfer.py` | Split-half test: per-category against one global mask rule *(stored)* | `threshold_transfer_study.py` |
| `ad2_e23_head_features.py` | E23 step 1 (GPU): per-block maps of both components on held-out and synthetic images | `head_features.py` |
| `ad2_e23_fusion_head.py` | E23 step 2 (CPU): the head trained on synthetic defects, against the fixed rule | `fusion_head_study.py` |
| `ad2_e23_hybrid_check.py` | v6's maps built exactly as the export builds them, scored on `test_public` | `hybrid_maps_check.py` |
| `ad2_e23_hybrid_bootstrap.py` | Object bootstrap of v6's AU-PRO gain over v5 | `hybrid_bootstrap.py` |

## MVTec's server (S1, S2)

| script | role | former name |
|---|---|---|
| `server_proxy.py` | The server's metrics recomputed on `test_public` *(stored)* | `ad2_server_proxy.py` |
| `server_s1_expectation.py` | What the server should report for v1, written before the result | `ad2_server_expectation.py` |
| `server_h4_analysis.py` | ADR-14 H4: v3 against v1 on the private splits (prepared; v3 failed its gate and was never submitted) | `ad2_server_analysis.py` |
| `server_export_submission.py` | Export a submission from a recipe: native float16 maps, PNG masks, optionally the component maps | `export_ad2_submission.py` |
| `server_export_v6_hybrid.py` | Build v6 from v5's kept component maps on the CPU (head maps, v5 masks) | `export_hybrid_from_components.py` |

The five-input variant of the head export failed its pre-registered per-category guard and was
never run. It is in the git history as `scripts/export_head_maps.py` (commit `1a45ebd`).

## Deployment and reporting

| script | output | former name |
|---|---|---|
| `deploy_bench_inference.py` | Latency (p50 / p95 / p99) and peak memory, one process per measurement | `bench_inference.py` |
| `report_figures.py` | `reports/figures/fig1`–`fig8`, the figures of the final report (metrics only, no dataset pixels) | — |
| `report_build_notebooks.py` | `notebooks/*.ipynb`, generated and never hand-edited | `build_notebooks.py` |

Records written before the renaming, such as the run registry and the submission JSON files, name
the scripts as they were called then. Each record also stores its git SHA.
