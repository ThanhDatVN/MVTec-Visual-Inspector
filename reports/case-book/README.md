# Case book

Individual images where the model's behaviour teaches something a table cannot: each case names
the images, the run, the mechanism, the evidence, and what it changed in the project. Figures
contain dataset pixels, so they are not committed; every panel is reproducible locally:

```bash
python scripts/case_book.py --data-root <VisA root> --run <run id> --ids <image ids...> \
    --out artifacts/case-book/<name>.png
```

Dataset: VisA © Amazon, CC BY 4.0 — see [ATTRIBUTION.md](../../ATTRIBUTION.md). Image ids are
paths relative to the VisA root.

---

## 1. Background debris sets the operating threshold — `pcb1`

| | |
|---|---|
| Run | `00b777fff4506da9` (PatchCore, 320 px, seed 0); the same images at 640 px |
| Images | test normals `pcb1/Data/Images/Normal/0214.JPG`, `0694.JPG`, `0920.JPG`, `0306.JPG`; validation normals `0347.JPG`, `0332.JPG`, `0452.JPG` |
| Observed | 4–6 of 100 test normals exceed the threshold, against a 0.73% bound |
| Mechanism | The map peaks on the black felt, not on the board: a white fibre (`0214`) and **one dark hook-shaped object that appears in three images** (`0694`, `0920`, `0306`) — a single piece of debris in one capture session. The highest-scoring *validation* normals are debris too (fibres, tape residue), so the threshold itself is set by background, not by the part |
| Evidence | Bulk of test vs validation normal scores not shifted (AUROC 0.53, p ≥ 0.3); upper tail is (exact permutation p 0.005–0.03); a threshold from held-out *test* normals hits its bound, so the rank rule is sound; model-free pixel proximity to training images shows no difference |
| Consequence | The false-alarm guarantee's exchangeability assumption fails at the session level, which VisA does not annotate. Scoring only the object region recovers recall (0.46 → 0.81 at 320 px) but not calibration: the hook lies within the margin of the board. See [development report](../E00-E04-development-findings.md) §4–§5 |
| Nearest normal (`/predict?explain=true`, 640 px artifact) | For `0214` the closest stored normal patch is **also a white fibre on the felt** (training image `0134`), at distance 1.76 against a threshold of 1.54: debris exists in the training set, but a 10k coreset keeps too few examples of its variety for a new fibre to land close. For a defective image with bent pins (`Anomaly/005`) the nearest normal patch is a row of *straight* pins (`0817`) — the explanation a person reads at a glance |

## 2. The same mechanism on a held-out category — `pipe_fryum`

| | |
|---|---|
| Run | `3a85b4ec5650114f` (confirmation recipe, 640 px, seed 2) |
| Images | `pipe_fryum/Data/Images/Normal/245.JPG`, `337.JPG` (same curved fibre on the background), `467.JPG` (a speck at the frame corner), `352.JPG` (the part's own deformed rim, flagged in every seed and at 320 px) |
| Observed | 2 / 1 / 4 of 50 test normals flagged across seeds, bound 1.45% |
| Evidence | Calibration from validation gives 7.6% FPR on a held-out half of test normals; from the other test half, 3.89% against a 3.85% bound |
| Consequence | Found on a category that played no part in discovering case 1 — the debris/session mechanism is a property of the capture setup, not of `pcb1`. See [confirmation report](../E12-confirmation-v1.md) |

## 3. A reconstruction model scoring a normal texture — CAE-SSIM on `pcb1`

| | |
|---|---|
| Run | `2ff8bfca3f3fd9cb` (CAE, SSIM loss, 320 px, seed 0) |
| Observed | Image AUROC 0.408 ± 0.076 over three seeds — below chance |
| Mechanism | On every image the SSIM error peaks on the two ultrasonic transducers' fine metal mesh, a high-frequency texture the autoencoder cannot reconstruct. The image maximum ranks images by how hard their mesh is to reconstruct (pose, glare), not by defects, which are smaller errors elsewhere |
| Consequence | The map mean scores 0.612 instead: part of the CAE's weakness is the map-to-score rule. The classic failure mode of reconstruction models on high-frequency normal structure |

## 4. Small defects are in the map but not in the image score — `pcb1` at 320 px

| | |
|---|---|
| Runs | PatchCore 320 px seeds 0–2 (E01) |
| Observed | Missed defective images carry smaller defects (median 2,485 vs 3,739 px, Mann-Whitney p = 3.6e-5) |
| Mechanism | At a 5% pixel false-positive rate the map still finds 89% of regions under 64 px; their *peaks* sit below an image threshold set by background debris (case 1) |
| Consequence | Restricting the score to the object region recovers most of these misses at 320 px; at 640 px resolution recovers them on its own |

## 5. Normal variation outscores defects of every size — `macaroni2`

| | |
|---|---|
| Runs | PatchCore 320 px seeds 0–2 (E01) |
| Observed | 3 of 100 defective images detected at the operating point; image score uncorrelated with defect size (Spearman −0.06) |
| Mechanism | The image maximum comes from the textured tray and the pasta's normal variation; the defects are localized (AU-PRO@0.05 0.65) but never the highest peak |
| Consequence | Object region +0.06 image AUROC at 320 px; at 640 px image AUROC reaches 0.884 ± 0.005 |

## 6. A global shortcut — colour histograms

| | |
|---|---|
| Runs | Tier-0 `histogram`, 320 px |
| Observed | Image AUROC 0.829 on `pcb1`, 0.962 on `cashew`, 0.881 on `pcb2`, 0.851 on `pipe_fryum` — with a constant anomaly map |
| Consequence | On these categories an image-level score cannot show that a model sees the defect; localization metrics (AU-PRO) and the floor row must be read beside it |
