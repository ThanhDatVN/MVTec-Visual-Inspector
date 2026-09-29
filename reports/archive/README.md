# Archived results

`results_visa_exploratory_v1.csv` holds the first VisA runs (2026-09-22/23). They are kept as
**exploratory historical evidence** and must not be promoted to a headline table:

- every row was made from a dirty working tree (`dirty=True`);
- the nine PatchCore rows have blank `config_hash` and `git_sha` — their exact configuration
  cannot be proven, and the hashes are deliberately **not** filled in after the fact;
- Tier 0 ran at smoothing sigma 4 (native pixels) through the CLI while PatchCore ran at sigma 0
  through an inline script, so the two were compared under different post-processing;
- thresholds used the pre-ADR-9 rank rule, whose effective FPR could exceed the target
  (1.46% for a 1% request on `pcb1`), with `>=` decisions;
- PatchCore image scores used a re-weighting rule that differs from the reference (docs/13, F03),
  and its `k` setting had no effect (F02).

The corrected, fully identified runs live in `reports/runs/` (protocol v2) and the table
regenerated from them is `reports/results_registry.csv`. See docs/13 and ADR-9/ADR-10.
