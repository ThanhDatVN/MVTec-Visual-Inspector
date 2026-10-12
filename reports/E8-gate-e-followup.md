# E8 — Gate E follow-up: three seeds for the screening winners, and the deployment cost

**Development evidence, pre-registered.** The family, the analysis and DR-1 made operational were
written into [docs/11](../docs/11-experiment-plan.md) (Stage E, E8) and pushed (`e7c71b0`) before
any of these runs. One Kaggle session (Tesla T4), 45 runs at implementation `fc9dbe93129d`, clean
checkout: the 3-seed reference and four variants × 3 categories × 3 seeds. Seed-averaged image
AUROC is compared with the reference on the same test images (paired, class-stratified bootstrap,
10,000 resamples), Holm over the 12 comparisons; **bold** survives it. Raw tables:
[studies/kaggle_e8/](studies/kaggle_e8/) and [studies/bench/](studies/bench/), from
`scripts/kaggle_studies.py --implementation fc9dbe93129d --vram-csv studies/bench/latency.csv`
and `scripts/bench_inference.py`.

Dataset: VisA © Amazon, CC BY 4.0 — see [ATTRIBUTION.md](../ATTRIBUTION.md).

## Summary

1. **The screen replicates.** `layer2` alone, 960 px, and both together pass DR-1 at three seeds;
   the single-seed estimates were within 0.007 of the 3-seed ones.
2. **`layer2` at 960 px is the strongest:** `macaroni2` 0.887 → **0.979** image AUROC, `capsules`
   0.914 → 0.963, AU-PRO@0.05 +0.04 to +0.08 on every category — at twice the latency
   (p95 ≈ 200 ms on a laptop RTX 3050). `layer2` at 640 px gains less (+0.069 / +0.028) at no cost.
3. **timm's `racm_in1k` weights are not adopted:** +0.050 on `capsules`, but −0.055 on
   `macaroni2` (significant) — and at 320 px (C2) the same weights had *helped* `macaroni2`. The
   weights effect changes sign with the resolution.
4. **The runner's VRAM records were wrong, and are fixed.** Every finished run left its backbone
   on the GPU until Python's cycle collector ran, so recorded peaks grew by ≈ 280 MB per run
   within a session. Measured cleanly, every variant needs **≤ 0.75 GB** of GPU memory at inference
   and ≤ 1.5 GB to fit: all of them run on the 4 GB laptop card.
5. **Deployment latency (O8), laptop:** the frozen recipe answers in 99 / 102 ms (p50 / p95) on
   the RTX 3050 and 890 / 957 ms on the CPU.

## 1. Accuracy at three seeds

| variant | category | I-AUROC (3 seeds) | Δ I-AUROC [95% CI] | AU-PRO@0.05 | Δ AU-PRO | Δ FPR@OP1 | Δ recall@OP1 |
|---|---|---|---|---|---|---|---|
| reference | `pcb1` | 0.979 ± 0.002 | – | 0.854 | – | (4.3%) | (0.84) |
| reference | `macaroni2` | 0.887 ± 0.010 | – | 0.864 | – | (1.7%) | (0.39) |
| reference | `capsules` | 0.914 ± 0.003 | – | 0.870 | – | (2.2%) | (0.54) |
| `layer2` only | `pcb1` | 0.987 ± 0.001 | +0.008 [-0.003, +0.021] | 0.867 | +0.012 | -2.0 pt | +0.05 |
| `layer2` only | `macaroni2` | 0.955 ± 0.003 | **+0.069** [+0.040, +0.101] | 0.930 | +0.066 | -0.3 pt | +0.05 |
| `layer2` only | `capsules` | 0.942 ± 0.005 | **+0.028** [+0.007, +0.052] | 0.891 | +0.020 | -2.2 pt | -0.15 |
| 960 px | `pcb1` | 0.988 ± 0.003 | +0.009 [-0.001, +0.022] | 0.885 | +0.031 | -2.0 pt | +0.06 |
| 960 px | `macaroni2` | 0.948 ± 0.002 | **+0.061** [+0.034, +0.091] | 0.894 | +0.030 | +3.3 pt | +0.25 |
| 960 px | `capsules` | 0.976 ± 0.003 | **+0.062** [+0.032, +0.098] | 0.927 | +0.056 | -1.1 pt | +0.22 |
| `layer2` at 960 px | `pcb1` | 0.983 ± 0.002 | +0.004 [-0.011, +0.020] | 0.897 | +0.043 | -1.7 pt | +0.07 |
| `layer2` at 960 px | `macaroni2` | 0.979 ± 0.001 | **+0.092** [+0.059, +0.129] | 0.949 | +0.084 | +1.3 pt | +0.40 |
| `layer2` at 960 px | `capsules` | 0.963 ± 0.004 | **+0.049** [+0.018, +0.085] | 0.942 | +0.071 | -0.6 pt | +0.12 |
| timm `racm_in1k` weights | `pcb1` | 0.968 ± 0.001 | -0.011 [-0.024, -0.000] | 0.850 | -0.004 | -2.0 pt | -0.22 |
| timm `racm_in1k` weights | `macaroni2` | 0.832 ± 0.021 | **-0.055** [-0.096, -0.015] | 0.830 | -0.035 | -0.3 pt | -0.21 |
| timm `racm_in1k` weights | `capsules` | 0.964 ± 0.002 | **+0.050** [+0.019, +0.085] | 0.888 | +0.018 | -1.7 pt | +0.08 |

The reference reproduced the earlier sessions exactly: its nine runs gave the same predictions,
bit for bit, as session A's at the previous implementation.

### DR-1

| variant | (1) AU-PRO ≥ 2 up, third ≥ −0.01 | (2) Holm-significant gain ≥ 2, loss none | (4) inference VRAM < 3.5 GB | decision |
|---|---|---|---|---|
| `layer2` only | yes (3/3) | yes (2 gains) | 0.57 GB | **adopted (development)** |
| 960 px | yes (3/3) | yes (2 gains) | 0.75 GB | **adopted (development)** |
| `layer2` at 960 px | yes (3/3) | yes (2 gains) | 0.75 GB | **adopted (development)** |
| timm `racm_in1k` weights | no (1/3) | no (a significant loss) | 0.57 GB | not adopted |

(3), latency, is reported but not gated: no budget was set for this phase.

### Reading

- **The screen held.** Single seed vs three seeds: `layer2` +0.064 / +0.035 → +0.069 / +0.028;
  960 px +0.058 / +0.062 → +0.061 / +0.062 (`macaroni2` / `capsules`). `pcb1` is saturated near
  0.98–0.99 in every variant, so no change there is significant.
- **The two levers combine.** `layer2` at 960 px is the best on `macaroni2` (0.979, AU-PRO 0.949)
  and on AU-PRO everywhere; on `capsules` image AUROC, 960 px with `layer2+layer3` is slightly
  higher (0.976 vs 0.963). Finer features and more pixels attack the same problem — defects that
  are small relative to the frame — so the gains overlap rather than add.
- **Operating point.** The false-alarm rate is not part of DR-1 and mostly moves little, but
  960 px with `layer2+layer3` raises it on `macaroni2` from 1.7% to 5.0% (bound 0.74%); `layer2`
  at 960 px keeps it at 3.0%. Recall at the operating point rises at 960 px, by up to 0.40 on
  `macaroni2`; on `capsules`, `layer2` at 640 px loses 0.15 of it.
- **Weights are not a free axis.** The `racm_in1k` weights trade categories (−0.055 on
  `macaroni2`, +0.050 on `capsules`), and the trade reverses between 320 px (C2: +0.063 on
  `macaroni2`) and 640 px. An effect whose sign depends on the resolution is not something to
  select on three categories.

## 2. The VRAM records, and the fix

Within session D the recorded peaks climbed run by run — 803, 1,083, 1,304, 1,582, 1,861 MB … —
by about the size of WideResNet50-2's weights, and fell back to ~550 MB at irregular points. The
cause: `FeatureExtractor`'s forward hooks closed over the extractor, so extractor → model → hook →
extractor formed a reference cycle that reference counting cannot free. A finished run's backbone
(and its last feature maps) stayed on the GPU until Python's cycle collector happened to run, and
each later run's peak included it. Reproduced on the laptop: before the fix four consecutive
extractors left 301, 596, 890 and 1,183 MB allocated; after it, 8 MB each time. The hooks now
hold only the feature dict, the dict is cleared after each batch, and a regression test drops an
extractor with the collector disabled.

The fix releases references and changes no computation. Its consequences for earlier reports:
**per-run peak VRAM recorded in a session with several runs is an upper bound** (E01's and E04's
VRAM columns, K1's 960/1280 px figures); host-memory (RSS) figures are essentially unaffected,
since the retained memory was on the GPU. DR-1's criterion (4) was therefore evaluated on the clean
benchmark below, as the amendment in docs/11 records.

## 3. Deployment cost (charter objective O8)

`scripts/bench_inference.py`: each configuration fitted and exported in its own process on
`pcb1`, then loaded in a fresh process and timed through the service path (`Inspector.predict`:
input guards, backbone and k-NN, upsampling and smoothing to native size), batch 1, 20 warm-up
and 200 timed predictions with `torch.cuda.synchronize()` around each. Laptop: NVIDIA GeForce RTX
3050 Laptop GPU (4 GB), AMD Ryzen 5 6600H (6 threads), PyTorch 2.14; features in fp16.

| configuration | device | long side | p50 | p95 | p99 | peak VRAM, allocated / reserved | fit (time, VRAM, host RSS) |
|---|---|---|---|---|---|---|---|
| recipe v1 | GPU | 640 | 99 ms | 102 ms | 103 ms | 574 / 692 MB | 92 s, 802 MB, 3.8 GB |
| recipe v1 | CPU | 640 | 890 ms | 957 ms | 1,034 ms | – | (fitted on the GPU) |
| `layer2` | GPU | 640 | 97 ms | 99 ms | 105 ms | 573 / 662 MB | 83 s, 643 MB, 3.8 GB |
| 960 px | GPU | 960 | 200 ms | 201 ms | 202 ms | 746 / 978 MB | 172 s, 1,470 MB, 6.8 GB |
| `layer2` at 960 px | GPU | 960 | 196 ms | 198 ms | 202 ms | 746 / 868 MB | 170 s, 1,105 MB, 6.8 GB |
| timm `racm_in1k` | GPU | 640 | 100 ms | 101 ms | 101 ms | 574 / 692 MB | 86 s, 802 MB, 3.8 GB |

- **Everything fits the 4 GB card with room to spare**, fitting included; DR-1's 3.5 GB limit is
  nowhere near binding. Neither is the CUDA context, which these figures exclude.
- **Latency is set by the resolution, not the feature level:** `layer2` costs nothing over
  `layer2+layer3`, 960 px doubles it. p95 sits within 3% of p50 on the GPU: the service has no
  long tail.
- **On the CPU** the recipe needs just under a second per image (p95 957 ms).
- The Kaggle T4 was not measured with this protocol; the registry's batch-mode time per image
  there is 90–102 ms at 640 px (report K1).

## Decision and what it does not establish

- **Candidate recipe v2: `layer2` at 960 px**
  ([configs/recipes/candidate-v2-patchcore-layer2-960.yaml](../configs/recipes/candidate-v2-patchcore-layer2-960.yaml)),
  the adopted variant with the largest AU-PRO gain; `layer2` at 640 px is the cost-neutral
  alternative. Selected on the development categories' test sets, which is what Stage E does —
  a development result, not a finding.
- **The frozen recipe of ADR-11 is unchanged**, and so are the confirmation (E12) and MPDD (H1)
  results built on it. The nine VisA confirmation categories and MPDD are spent; confirming v2
  needs data that shaped nothing — MVTec AD 2, with an ADR that freezes the candidate file first.
- Three categories, one dataset, one capture rig; latency on one laptop.
