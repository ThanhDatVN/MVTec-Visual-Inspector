# Attribution

This project uses the MVTec Anomaly Detection datasets, provided by MVTec Software GmbH
under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
(CC BY-NC-SA 4.0): https://creativecommons.org/licenses/by-nc-sa/4.0/

**MVTec AD 2:**

> Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca König, Carsten Steger.
> "The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection."
> International Journal of Computer Vision, 134(4), 2026.
> DOI: 10.1007/s11263-026-02743-0 · arXiv:2503.21622

**MVTec AD (classic):**

> Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.
> "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection."
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
> DOI: 10.1109/CVPR.2019.00982

> Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger.
> "The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
> Unsupervised Anomaly Detection."
> International Journal of Computer Vision, 129(4):1038-1059, 2021.
> DOI: 10.1007/s11263-020-01400-4

**Changes made:** images were resized, tiled, and photometrically/geometrically corrupted
for robustness evaluation; model-generated anomaly heatmaps are overlaid on some images.

**This work is non-commercial. No dataset images are redistributed in this repository.**

---

## VisA (Visual Anomaly)

This project also uses the VisA dataset, provided by Amazon under the
**Creative Commons Attribution 4.0 International License (CC BY 4.0)**:
https://creativecommons.org/licenses/by/4.0/

> Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, Onkar Dabeer.
> "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
> and Segmentation." ECCV 2022. arXiv:2207.14315

Accessed 2026-09-22 from https://registry.opendata.aws/visa

**Changes made:** images were resized and photometrically/geometrically corrupted for
robustness evaluation; model-generated anomaly heatmaps are overlaid on some images.

> **Licence discrepancy, resolved.** Some third-party documentation states CC BY-NC-SA 4.0.
> The archive `VisA_20220922.tar` ships a `LICENSE-DATASET` file containing the CC BY 4.0
> text, which agrees with the AWS Open Data registry and the source paper; see
> [reports/P3-visa-findings.md](reports/P3-visa-findings.md) §0. The project still redistributes
> no images, so the repository guard needs no dataset-specific exception.

**No VisA images are redistributed in this repository.**

---

## Third-party code used as a test oracle

`tests/reference/anomalib_patchcore.py` excerpts PatchCore scoring and k-center selection from
[anomalib](https://github.com/open-edge-platform/anomalib) v2.3.0 (commit `091ca6a`),
© 2022-2025 Intel Corporation, under the Apache License 2.0 (full text in
`tests/reference/LICENSE-APACHE-2.0.txt`). The modifications are listed at the top of that
file. It is used only by the agreement tests and is not part of the `inspector` package.


---

> ⚠ **Verification pending (Gate P1).** The MVTec AD 2 author list and page numbers above were
> taken from the official dataset page and secondary sources. Confirm them against the published
> IJCV article before this file is published anywhere. An attribution block with the wrong authors
> fails the one obligation it exists to satisfy.
> Tracked as OD in [docs/07-risks-and-decisions.md](docs/07-risks-and-decisions.md).
