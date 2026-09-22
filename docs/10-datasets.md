# 10 — Dataset Survey and Selection

Written because MVTec AD 2 sits behind a registration wall and a ~30 GB download, which blocks
every experiment. This surveys the reputable alternatives that can be fetched **today**, and
records which one the project adopted and why.

All availability figures below were verified by an HTTP request on 2026-09-22, not taken from a
paper. Sizes are the actual `Content-Length` returned.

---

## 1. Candidates

Sizes are decimal GB from the `Content-Length` actually returned, probed 2026-09-22. The
**Access** column is the finding that matters: most of the field's headline datasets cannot be
fetched today without an account or an application, which is what makes the shortlist short.

### Verified downloadable without an account

| Dataset | Venue / origin | Images | Size | HTTP | License | Pixel masks |
|---------|----------------|--------|------|------|---------|-------------|
| **VisA** | ECCV 2022, Amazon Science | 10,821 (9,621 normal / 1,200 anomalous), 12 categories | **1.93 GB** | 200 | **CC BY 4.0** | yes |
| **BTAD** | beanTech, 2021 | 2,830, 3 products | **1.23 GB** | 200 | not stated on the download page — ask the authors before any redistribution | yes |
| **KolektorSDD2** | ViCoS Ljubljana, 2021 | ~3,335 | **0.85 GB** | 200 | CC BY-NC-SA 4.0 | yes |
| **KolektorSDD** | ViCoS Ljubljana, 2019 | 399 | **0.10 GB** | 200 | CC BY-NC-SA 4.0 | yes |
| **MPDD** | 2021 | 1,346, 6 painted metal parts | small (GitHub zip) | 200 | check the repository | yes |

### Gated — an account, a form, or a login

| Dataset | Images | Gate | License |
|---------|--------|------|---------|
| MVTec AD 2 | >8,000, 8 categories, ~30 GB | MVLogin registration | CC BY-NC-SA 4.0 |
| MVTec AD (classic) | >5,000, 15 categories, ~4.9 GB | Registration (Kaggle / HF mirrors exist) | CC BY-NC-SA 4.0 |
| MVTec LOCO AD | 3,644, 5 categories | Registration | CC BY-NC-SA 4.0 |
| Eyecandies | synthetic, 10 categories | HTTP 401 on the Hugging Face archive | check |
| DAGM 2007 | ~16,100 synthetic | Kaggle account | research use |
| Real-IAD | 151,050, 30 objects, multi-view | Application form | restricted |
| Real-IAD Variety (2026) | 198,950, 160 categories | Application form | restricted |
| MANTA (CVPR 2025) | multi-view + text | Project page / form | check |
| Kaputt (ICCV 2025) | large-scale retail defects | Project page / form | check |

**Note on the newest and largest benchmarks.** Real-IAD Variety (160 categories), MANTA and
Kaputt are the most interesting datasets to appear recently, and all three are behind a form.
They are worth applying for in parallel with this project rather than waiting on — an application
costs an email and the reply may arrive long before it is needed.

## 2. Decision — VisA becomes the working dataset

**Adopted as the dataset the project runs on now**, with MVTec AD 2 retained as the headline
benchmark once its download completes. See
[ADR-8](07-risks-and-decisions.md#adr-8--visa-as-the-immediately-available-working-dataset).

Why VisA and not the others:

1. **It is the field's second benchmark.** After MVTec AD, VisA is the most reported dataset in
   the anomaly-detection literature, so our numbers can be read against published work. BTAD,
   MPDD and KolektorSDD2 are real datasets but thinly reported, and a result on them is harder to
   situate.
2. **No registration, and a permissive licence.** CC BY 4.0 requires attribution and nothing else
   — no non-commercial clause, no share-alike. That is materially more permissive than MVTec's
   CC BY-NC-SA and removes the sharpest constraint on this project's outputs.

   > **Licence discrepancy, recorded rather than resolved.** The AWS Open Data registry and the
   > SPot-the-Difference paper both state CC BY 4.0. Some third-party documentation (including
   > anomalib's) states CC BY-NC-SA 4.0. Until the discrepancy is settled, this project behaves as
   > if the stricter reading applied: it stays non-commercial and redistributes no images. The
   > cost of that caution is zero, and the cost of guessing wrong is not.
3. **It has an official one-class split.** `split_csv/1cls.csv` defines the train/test partition
   used by published results. Inventing a split would produce numbers comparable with nothing.
4. **It is small enough to be practical.** 1.93 GB downloads in minutes and fits Kaggle's 20 GB
   working directory with room to spare, unlike AD 2's 30 GB.
5. **Its difficulty structure supports a principled three-category choice** (§4).

What VisA does **not** give us, and why AD 2 is still the headline:

- No lighting-shifted test split, so robustness must be entirely synthetic. AD 2's real
  distribution shift remains the stronger evidence, and the plan's Phase P7 validity check
  (does synthetic corruption predict real shift?) is only possible on AD 2.
- No held-out server-evaluated split.
- Lower resolution, so the resolution axis is less punishing than `sheet_metal`'s.

## 3. Structure and protocol

```
VisA_20220922/
  <category>/
    Data/Images/Normal/0000.JPG
    Data/Images/Anomaly/000.JPG
    Data/Masks/Anomaly/000.png
    image_anno.csv            # per-defect-type labels
  split_csv/1cls.csv          # THE official one-class split
```

Verified counts from `1cls.csv` (10,821 rows):

| category | group | train (normal) | test normal | test anomalous | masks | val @15% | min achievable FPR |
|----------|-------|----------------|-------------|----------------|-------|----------|--------------------|
| `candle` | multiple instances | 900 | 100 | 100 | 100 | 135 | 0.74% |
| `capsules` | multiple instances | 542 | 60 | 100 | 100 | 81 | 1.22% |
| `cashew` | single instance | 450 | 50 | 100 | 100 | 68 | 1.45% |
| `chewinggum` | single instance | 453 | 50 | 100 | 100 | 68 | 1.45% |
| `fryum` | single instance | 450 | 50 | 100 | 100 | 68 | 1.45% |
| `macaroni1` | multiple instances | 900 | 100 | 100 | 100 | 135 | 0.74% |
| `macaroni2` | multiple instances | 900 | 100 | 100 | 100 | 135 | 0.74% |
| `pcb1` | complex structure | 904 | 100 | 100 | 100 | 136 | 0.73% |
| `pcb2` | complex structure | 901 | 100 | 100 | 100 | 135 | 0.74% |
| `pcb3` | complex structure | 905 | 101 | 100 | 100 | 136 | 0.73% |
| `pcb4` | complex structure | 904 | 101 | 100 | 100 | 136 | 0.73% |
| `pipe_fryum` | single instance | 450 | 50 | 100 | 100 | 68 | 1.45% |

Two structural facts verified programmatically: **the train split is normal-only** in every
category, and **every anomalous test image has a mask**.

### VisA sharpens ADR-7 rather than escaping it

The achievability floor `1/(n+1)` from [ADR-7](07-risks-and-decisions.md) applies here too, and
VisA makes it a more interesting constraint than a blanket rule:

- Large categories (`candle`, `macaroni*`, `pcb*`, ~135 validation images) reach **0.73–0.74%**,
  so the 1% target **is** achievable.
- Small categories (`cashew`, `chewinggum`, `fryum`, `pipe_fryum`, 68 images) floor at **1.45%**,
  so it is **not**.

The same protocol therefore yields an achievable operating point on some categories and not
others *within one dataset*. That is a cleaner demonstration of the finding than AD 2 alone
provides, and it is a good reason to report the per-category target rather than a single headline
FPR.

Note the floors depend on the 15% carve fraction, which is our choice, not VisA's. Raising it buys
a lower floor at the cost of training images — a trade-off worth an explicit ablation rather than
a default.

## 4. Category selection

Three categories, one from each of VisA's structural groups, so each fails for a different
reason. Picking three PCBs would test one difficulty three times.

| category | group | why it is in the study | train |
|----------|-------|------------------------|-------|
| **`pcb1`** | complex structure | Dense small components; defects are tiny and structural against a busy background. The **resolution / small-defect** axis, and VisA's nearest analogue to `sheet_metal`. | 904 |
| **`macaroni2`** | multiple instances | High variance in the normal arrangement with subtle defects — reported as the hardest VisA category. The **false-positive** axis, analogous to `walnuts`. | 900 |
| **`capsules`** | multiple instances | Transparent, reflective shells; specularities look like defects. The **optics/ambiguity** axis, analogous to `fruit_jelly`. Smallest train split of the three, so the fastest to iterate on. | 542 |

`cashew` (single instance, aligned, the easiest group) is kept as an optional contrast row: a
method that cannot separate `cashew` is broken, which makes it a useful sanity control rather than
a study category.


## 4b. A second dataset, for the generalization question

Choosing three categories within one dataset answers "does this method work on these three
parts". It cannot answer "does this method survive a different camera, a different factory and a
different annotator", which is the question a deployment actually poses. One small second dataset
makes that a measurable rather than a rhetorical distinction.

**MPDD** is the right one: 1,346 images of six painted metal parts, small enough to run on the
laptop, and deliberately harder in exactly the way VisA is not — its images have non-homogeneous
backgrounds, varying spatial orientation and varying light intensity, where VisA is shot on a
fixed rig. A method tuned on VisA and applied unchanged to MPDD is a genuine transfer test.

Scope discipline: MPDD is a **transfer probe, not a study dataset**. It gets the final
configuration applied once, with no tuning, and one paragraph in the report. Tuning on it would
turn three study categories into nine and is exactly the scope creep the charter forbids (R7).

BTAD (1.23 GB, 3 products) is the fallback if MPDD proves unsuitable, with the caveat that its
download page states no licence — worth resolving with the authors before any figure derived from
it is published.

## 5. Attribution — required by CC BY 4.0

```
VisA (Visual Anomaly) dataset, (c) Amazon.com, Inc.
Licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

  Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, Onkar Dabeer.
  "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
  and Segmentation." ECCV 2022. arXiv:2207.14315

Accessed 2026-09-22 from https://registry.opendata.aws/visa

Changes made: images were resized and photometrically/geometrically corrupted for
robustness evaluation; model-generated anomaly heatmaps are overlaid on some images.
```

Attribution is added to `ATTRIBUTION.md`, the model card, the Gradio footer and the API
`/metadata` response, exactly as for MVTec. No VisA images are redistributed in this repository —
the same rule applies regardless of licence, because the `.gitignore` and CI guard do not need an
exception carved into them.

## 6. Reproducing the acquisition

```bash
inspector fetch visa --out data/raw        # 1.93 GB tar + official split CSV
inspector audit -c configs/data/visa_pcb1.yaml --data-root data/raw/VisA_20220922
inspector eda   -c configs/data/visa_pcb1.yaml --categories pcb1 macaroni2 capsules \
                --data-root data/raw/VisA_20220922
```

The tar's SHA-256 is recorded in `data/manifests/` on first fetch and re-verified afterwards, so a
corrupted or silently updated download is detected rather than trained on.
