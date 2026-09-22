# 08 — Licensing, Attribution, and Compliance

> Practical compliance guidance for this project, not legal advice. If the project's status ever
> changes from personal/research to anything commercial, stop and get the question answered
> properly — MVTec invites contact where non-commercial status is unclear.

---

## 1. What license applies

**MVTec AD 2** and **MVTec AD (classic)** are both released under
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

Three obligations:

| Clause | Obligation | What it means here |
|--------|-----------|--------------------|
| **BY** — Attribution | Credit the creators, link the license, indicate changes | The attribution block in §4 appears in the README, the model card, the Gradio footer, and the API `/metadata` response |
| **NC** — NonCommercial | No use primarily aimed at commercial advantage or monetary compensation | No commercial deployment, no paid demo, no use in a product, no consulting deliverable. A portfolio project or coursework is fine. |
| **SA** — ShareAlike | Adaptations must be distributed under the same license | See §3 — this is the clause that requires thought |

---

## 2. Hard rules for this repository

1. **No dataset image, mask, or crop is ever committed.** Not in `tests/`, not in `reports/`, not
   in a notebook output cell, not in the Docker image.
2. **CI fixtures are synthetic** — procedurally generated in task 0.5.
3. `.gitignore` excludes `data/` (except `data/manifests/`) and every common image extension.
4. A **CI license-guard job** fails the build if an image file or a `data/` path other than
   manifests appears in the tree.
5. A **pre-commit hook** runs the same check locally.
6. **The Docker image contains no dataset images** — only model artifacts (memory bank, weights,
   config, threshold).
7. If the repository is ever made public, verify the *history* too, not just the tip:
   `git log --all --diff-filter=A --name-only` and check for image extensions. A file deleted in a
   later commit is still published.

> Rule 7 exists because the usual failure is not "I committed the dataset", it is "I committed one
> example image in week 2, removed it in week 3, and pushed the history in week 12."

---

## 3. The ShareAlike question — handled honestly

CC BY-NC-SA's ShareAlike clause covers **adapted material**. Whether a *model trained on* a
CC BY-NC-SA dataset constitutes adapted material is **not settled**, and reasonable people
disagree. Rather than pick a convenient answer, this project takes the conservative route:

| Artifact | Treatment |
|----------|-----------|
| **Source code** (no dataset content) | MIT, in `LICENSE` |
| **Trained weights / PatchCore memory bank** | Distributed only if needed; marked "derived from MVTec AD 2 (CC BY-NC-SA 4.0); non-commercial use only", carrying the attribution block. The memory bank is a particularly clear case: it literally stores features computed from dataset images. |
| **Anomaly maps, heatmaps, overlays, case-book figures** rendered over dataset images | These reproduce dataset imagery. Treated as adapted material: **CC BY-NC-SA 4.0**, with attribution, and used only in non-commercial contexts. |
| **Metric tables, plots of scores, aggregate statistics** | Facts about the data, not reproductions. No restriction claimed, but attribution is given anyway. |
| **Manifests and SHA-256 hashes** | Not dataset content; freely committed (and essential to reproducibility). |

**Publishing the case book.** It contains dataset imagery, so wherever it is published it carries
the attribution block, a CC BY-NC-SA 4.0 notice, and a non-commercial statement. Do not post
case-book figures to a site whose terms claim a commercial or sublicensable right over uploads
without checking those terms first — that would place the obligation in conflict.

**"Indicate changes"** is a real BY obligation, not boilerplate: state that images were resized,
corrupted, or overlaid with model output. Figure captions should say so.

---

## 4. Attribution block

Reproduce verbatim in `ATTRIBUTION.md`, the README, `MODEL_CARD.md`, the Gradio footer, and the
API `/metadata` payload.

```
This project uses the MVTec Anomaly Detection datasets, provided by MVTec Software GmbH
under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
(CC BY-NC-SA 4.0): https://creativecommons.org/licenses/by-nc-sa/4.0/

MVTec AD 2:
  Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca König, Carsten Steger.
  "The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection."
  International Journal of Computer Vision, 134(4), 2026.
  DOI: 10.1007/s11263-026-02743-0   ·   arXiv:2503.21622

MVTec AD (classic):
  Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.
  "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection."
  IEEE/CVF CVPR, pp. 9592-9600, 2019.  DOI: 10.1109/CVPR.2019.00982

  Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger.
  "The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
  Unsupervised Anomaly Detection."
  International Journal of Computer Vision, 129(4):1038-1059, 2021.
  DOI: 10.1007/s11263-020-01400-4

Changes made: images were resized, tiled, and photometrically/geometrically corrupted for
robustness evaluation; model-generated anomaly heatmaps are overlaid on some images.

This work is non-commercial. No dataset images are redistributed in this repository.
```

**Verify the AD 2 author list and page numbers against the published paper at Gate P1.** Author
lists taken from secondary sources are wrong often enough that checking is worth the two minutes —
and an attribution block with the wrong authors fails the one obligation it exists to satisfy.

---

## 5. Third-party code and model licenses

Every dependency that ships weights or code into our artifacts gets a row. Complete this at P0 and
re-verify at P10.

| Component | License | Obligation |
|-----------|---------|------------|
| PyTorch, torchvision | BSD-3-Clause | Notice |
| OpenCV (`opencv-python-headless`) | Apache-2.0 | Notice |
| MLflow | Apache-2.0 | Notice |
| FastAPI / Starlette / Uvicorn | MIT / BSD | Notice |
| Gradio | Apache-2.0 | Notice |
| anomalib | Apache-2.0 | Notice; comparator results attributed to it |
| **torchvision ImageNet weights** (WRN50-2, ResNet) | BSD-3-Clause (weights: check the specific card) | Confirm redistribution terms before baking into a public image |
| **DINOv2 / DINOv3 weights** | Check the model card — these differ from the code license and from each other | **Resolve before ADR-5 (OD-5).** Do not bake weights into a distributed image before confirming. |
| Base Docker image | Varies | Pin by digest; record the license |

Collect notices in `NOTICE`. This is ordinary hygiene, and it is also the part reviewers of a
professional project actually check.

---

## 6. Compliance checklist (verify at G0 and again at G10)

- [ ] `LICENSE` (MIT, code) present
- [ ] `ATTRIBUTION.md` present and matching §4 verbatim
- [ ] `NOTICE` lists all third-party licenses
- [ ] `.gitignore` excludes `data/` and image extensions; covered by a test
- [ ] CI license-guard job present and proven to fail on a deliberately staged image file
- [ ] Pre-commit hook installed
- [ ] `git log --all --diff-filter=A --name-only` shows no dataset files anywhere in history
- [ ] Docker image contains no dataset images (verified by inspecting the built layers)
- [ ] Gradio footer and `/metadata` both carry attribution
- [ ] Model card states the non-commercial restriction under "out-of-scope use"
- [ ] AD 2 citation verified against the published paper
- [ ] Any published figure containing dataset imagery carries the CC BY-NC-SA notice and a
      statement of the changes made
