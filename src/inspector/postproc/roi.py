"""Object region of interest, estimated from training normals only.

Why it exists: on `pcb1` the image-level operating threshold turned out to be
set by debris on the background felt — fibres, tape residue, a stray object —
rather than by the board being inspected. An inspection specification names
what is inspected; a region of interest makes the image score say the same.

How: a colour model of the background is fitted on the border band of
*training* images (the object is assumed not to fill the frame edge — checked
by `border_foreground_share` on the same training images). Per image, pixels
far from that model are foreground; small specks are removed; components much
smaller than the largest are dropped (debris), holes are filled (the object's
interior may share the background colour), and the region is dilated by a
margin so a defect on the object's edge stays inside. No test image, label or
mask is used to choose any of this.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


def _cv2():
    import cv2

    return cv2


def _lab(image: np.ndarray, work_size: int) -> np.ndarray:
    cv2 = _cv2()
    h, w = image.shape[:2]
    scale = work_size / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)


def _border(shape: tuple[int, int], border: float) -> np.ndarray:
    h, w = shape
    b = max(1, round(border * min(h, w)))
    band = np.zeros((h, w), dtype=bool)
    band[:b] = band[-b:] = True
    band[:, :b] = band[:, -b:] = True
    return band


@dataclass(frozen=True)
class BackgroundModel:
    """A Gaussian colour model (CIELAB) of the background."""

    mean: np.ndarray
    inv_cov: np.ndarray
    threshold: float  # squared Mahalanobis distance separating foreground
    work_size: int
    border: float

    def distance2(self, lab: np.ndarray) -> np.ndarray:
        diff = lab.reshape(-1, 3) - self.mean
        return np.einsum("ij,jk,ik->i", diff, self.inv_cov, diff).reshape(lab.shape[:2])

    def border_foreground_share(self, image: np.ndarray) -> float:
        """Share of the border band classified as foreground — the assumption
        check: if the object often reaches the frame edge, the model is wrong."""
        lab = _lab(image, self.work_size)
        return float(np.mean(self.distance2(lab)[_border(lab.shape[:2], self.border)] > self.threshold))


def fit_background(
    images: Iterable[np.ndarray],
    *,
    work_size: int = 256,
    border: float = 0.05,
    quantile: float = 0.995,
    trim: float = 0.05,
) -> BackgroundModel:
    """Fit on the border band of training images.

    One trimming pass drops the `trim` share of border pixels farthest from the
    first estimate, so an object that occasionally touches the edge does not
    inflate the covariance. `threshold` is the `quantile` of *all* border
    pixels' distances under the trimmed model — taking it over the trimmed
    pixels instead would leave about `trim` of clean background above it.
    """
    pixels = []
    for image in images:
        lab = _lab(image, work_size)
        pixels.append(lab[_border(lab.shape[:2], border)])
    px = np.concatenate(pixels).astype(np.float64)

    def moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cov = np.cov(x.T) + 1e-3 * np.eye(3)
        return x.mean(axis=0), np.linalg.inv(cov)

    def d2(x: np.ndarray, mean: np.ndarray, inv: np.ndarray) -> np.ndarray:
        diff = x - mean
        return np.einsum("ij,jk,ik->i", diff, inv, diff)

    mean, inv = moments(px)
    dist = d2(px, mean, inv)
    mean, inv = moments(px[dist <= np.quantile(dist, 1.0 - trim)])
    threshold = float(np.quantile(d2(px, mean, inv), quantile))
    return BackgroundModel(mean=mean, inv_cov=inv, threshold=threshold, work_size=work_size, border=border)


def object_mask(
    image: np.ndarray,
    model: BackgroundModel,
    *,
    relative_area: float = 0.1,
    margin: float = 0.03,
) -> np.ndarray:
    """Boolean object region at the model's work size.

    Args:
        relative_area: components smaller than this share of the largest one
            are debris, not objects. Multi-instance categories keep every
            instance of similar size.
        margin: dilation radius as a share of the image diagonal.

    Returns an all-true mask when no foreground is found, so a segmentation
    failure means "no restriction", never "nothing to inspect".
    """
    cv2 = _cv2()
    lab = _lab(image, model.work_size)
    fg = (model.distance2(lab) > model.threshold).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.ones(lab.shape[:2], dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(areas >= relative_area * areas.max())
    roi = np.isin(labels, keep).astype(np.uint8)

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = cv2.drawContours(np.zeros_like(roi), contours, -1, 1, thickness=cv2.FILLED)

    h, w = roi.shape
    r = max(1, round(margin * float(np.hypot(h, w))))
    roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return roi.astype(bool)


def mask_on_grid(mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """A region mask resampled onto a model's raw-map grid (cells at least half
    inside count). Never empty: an empty result falls back to the full grid."""
    cv2 = _cv2()
    frac = cv2.resize(mask.astype(np.float32), (grid[1], grid[0]), interpolation=cv2.INTER_AREA)
    cells = frac >= 0.5
    return cells if cells.any() else np.ones(grid, dtype=bool)
