# E12 — Confirmation v1 on nine held-out VisA categories

The recipe, the comparators, the hypotheses, the dataset-level decision rule and the analysis
script were committed and pushed in [ADR-11](../docs/07-risks-and-decisions.md) (commit
`4ff5a6f`) **before** any run on these categories. Every run below comes from a clean worktree at
that commit (implementation id `a2d2dcd0c230`), `role = confirmation`, and was run once. Nothing
was rerun or retuned. The generated output of the frozen analysis is
[studies/confirmation-v1/confirmation.md](studies/confirmation-v1/confirmation.md).

Dataset: VisA © Amazon, CC BY 4.0 — see [ATTRIBUTION.md](../ATTRIBUTION.md).

| | |
|---|---|
| Categories | `candle`, `cashew`, `chewinggum`, `fryum`, `macaroni1`, `pcb2`, `pcb3`, `pcb4`, `pipe_fryum` — none had been opened before this run |
| Recipe | PatchCore, WideResNet50-2 `layer2+3`, 640 px long side, memory bank fixed at 10,000 entries, seeds 0–2 ([config](../configs/recipes/confirmation-v1-patchcore-640.yaml)) |
| Comparators | the same recipe at 320 px (seed 0); pixel PCA and colour histogram at 320 px; random scorer |
| Runs | 63, all completed; no failures |
| Hardware | RTX 3050 laptop, 4 GB VRAM, 16 GB RAM; peak host RSS 2.4–4.6 GB |

---

## Verdicts (pre-registered rule: ≥ 6 of 9 categories significant and positive after Holm, none negative)

| hypothesis | categories significant (+ / −) | macro-mean difference [95% CI] | verdict |
|---|---|---|---|
| **H1** — 640 px beats 320 px | 4 / 0 | +0.034 [+0.025, +0.043] | **not confirmed** |
| **H2** — PatchCore 640 px beats pixel PCA | 9 / 0 | +0.167 [+0.145, +0.189] | **confirmed** |

Family: 18 paired, class-stratified image-bootstrap tests (10,000 resamples), Holm-adjusted
together; the smallest attainable adjusted p-value is 0.0036.

| category | H1: 640 − 320 px [95% CI] | p (Holm) | H2: 640 px − pixel PCA [95% CI] | p (Holm) |
|---|---|---|---|---|
| `candle` | +0.017 [−0.002, +0.038] | 0.32 | +0.175 [+0.111, +0.241] | 0.0036 |
| `cashew` | +0.011 [−0.011, +0.036] | 0.99 | +0.116 [+0.052, +0.187] | 0.0036 |
| `chewinggum` | −0.000 [−0.022, +0.020] | 1 | +0.262 [+0.181, +0.348] | 0.0036 |
| `fryum` | **+0.073** [+0.039, +0.113] | 0.0036 | +0.136 [+0.067, +0.212] | 0.0036 |
| `macaroni1` | **+0.079** [+0.045, +0.119] | 0.0036 | +0.270 [+0.197, +0.344] | 0.0036 |
| `pcb2` | **+0.035** [+0.012, +0.062] | 0.0084 | +0.097 [+0.046, +0.152] | 0.0036 |
| `pcb3` | **+0.074** [+0.037, +0.117] | 0.0036 | +0.273 [+0.201, +0.347] | 0.0036 |
| `pcb4` | +0.003 [−0.006, +0.014] | 1 | +0.057 [+0.026, +0.095] | 0.0036 |
| `pipe_fryum` | +0.009 [+0.000, +0.021] | 0.29 | +0.114 [+0.062, +0.174] | 0.0036 |

## Results per category

| category | I-AUROC 640 px | AU-PRO@0.05 | AU-PRO@0.30 | recall@OP | FPR@OP (bound) | I-AUROC 320 px | pixel PCA | histogram |
|---|---|---|---|---|---|---|---|---|
| `candle` | 0.971 ± 0.003 | 0.907 | 0.967 | 0.31 | 0.0% (0.74%) | 0.957 | 0.795 | 0.513 |
| `cashew` | 0.975 ± 0.005 | 0.825 | 0.911 | 0.68 | 0.0% (1.45%) | 0.968 | 0.860 | **0.962** |
| `chewinggum` | 0.986 ± 0.004 | 0.653 | 0.817 | 0.90 | 0.0% (1.45%) | 0.982 | 0.724 | 0.643 |
| `fryum` | 0.976 ± 0.000 | 0.642 | 0.890 | 0.66 | 0.0% (1.45%) | 0.903 | 0.839 | 0.747 |
| `macaroni1` | 0.977 ± 0.003 | 0.909 | 0.973 | 0.45 | 0.0% (0.74%) | 0.901 | 0.708 | 0.796 |
| `pcb2` | 0.977 ± 0.002 | 0.788 | 0.939 | 0.64 | 0.7% (0.74%) | 0.940 | 0.879 | **0.881** |
| `pcb3` | 0.978 ± 0.002 | 0.729 | 0.932 | 0.67 | 0.0% (0.73%) | 0.905 | 0.705 | 0.637 |
| `pcb4` | 0.996 ± 0.000 | 0.598 | 0.889 | 0.71 | 0.3% (0.73%) | 0.992 | 0.939 | 0.423 |
| `pipe_fryum` | 0.996 ± 0.002 | 0.847 | 0.964 | 0.98 | **4.7%** (1.45%) | 0.990 | 0.882 | **0.851** |
| **macro mean** | **0.981** | **0.766** | **0.920** | 0.67 | | 0.949 | 0.815 | 0.717 |

Mean ± sd over 3 seeds for the recipe; AU-PRO and recall are seed means. The random scorer lands
at 0.42–0.55 image AUROC in every category (sanity control).

## Reading

**H1 failed its pre-registered rule, and the failure is informative.** No category got worse at
640 px, and the macro-mean gain is positive with an interval well away from zero. But only four
of nine gains are individually significant, where the rule asked for six. The development
estimate (+0.03 / +0.19 / +0.25) was inflated by selection: the development categories were
chosen *because* they were hard at low resolution. On categories picked without that knowledge,
the typical gain is +0.03.

*Exploratory, not pre-registered:* the gain follows the headroom left at 320 px almost
monotonically (Spearman 0.93 over the nine categories, 0.96 with the development three). The
five categories that were not significant were already at 0.957–0.992 image AUROC at 320 px; the
four that were significant sat at 0.90–0.94 and gained +0.035 to +0.079. So the defensible
statement is conditional: *where 320 px leaves room, 640 px closes most of it; where it does not,
there is nothing to close.* Headroom and gain share the 320 px estimate, so part of this
correlation is regression to the mean.

**H2 is confirmed without exception.** The development finding that PatchCore was no better than
linear PCA at 320 px does not survive at the recipe's resolution, on any held-out category.

**The recipe transfers.** Macro-mean image AUROC 0.981 and AU-PRO@0.05 0.766 on categories it
never saw, fitted and calibrated per category with nothing retuned, on a 4 GB laptop at ≤ 4.6 GB
host memory.

**Global shortcuts exist here too.** The colour histogram reaches 0.962 on `cashew`, 0.881 on
`pcb2` and 0.851 on `pipe_fryum` with a constant anomaly map. On those categories the image-level
result says little about whether PatchCore sees the defect; localization (AU-PRO) does.

**Calibration mostly holds.** Across 27 recipe runs the rank rule produced 10 false alarms against
19.7 expected at the effective bounds — conservative overall. Eight of nine categories are at or
below their bound in every seed; no seed-0 tail test is significant after Holm. The exception is
`pipe_fryum` (2, 1 and 4 false alarms among 50 test normals, bound 1.45%). ADR-11 required the
`pcb1` diagnostics for it:

| diagnostic | `pipe_fryum`, 640 px |
|---|---|
| Bulk shift (test vs validation normals) | AUROC 0.51–0.57, p 0.17–0.92: not shifted |
| Tail, exact permutation p per seed (uncorrected) | 0.18, 0.42, 0.03 |
| Calibration source (n = 25, k = 1) | from validation: 7.6% FPR; from a test half: 3.89% (bound 3.85%) |
| Flagged images (seed 2) | `245`, `337`: **the same curved fibre on the black background**; `467`: a speck at the frame corner; `352`: the part's own deformed rim, flagged in every seed and at 320 px |

The same mechanism as `pcb1`, found independently: the upper tail of test normals contains
background debris from a capture session that the validation split does not, and one piece of
debris appears in two images. It is weaker here — `pipe_fryum`'s test split has only 50 normals
and its bound is 1.45% — but it is the same failure, now seen on a category that played no part
in discovering it.

## What this does and does not establish

- **It establishes recipe transfer within VisA:** a configuration chosen on three categories
  works, without retuning, on nine others of different object types.
- **It does not establish industrial generality.** One dataset, one capture rig, one annotation
  team. VisA provides no lot or session identifiers, so the session effect in the calibration
  results can be observed but not controlled.
- **The confirmation categories are now spent.** Anything tuned from here on is development
  again, and needs new held-out data (MVTec AD 2, or MPDD as a transfer probe) to be confirmed.
