"""Synthetic defects on normal images, for learning how to combine anomaly maps (E23).

No test image and no test label is used anywhere: defects are painted on held-out *normal*
training images, and a model's response to them teaches a small head which of its maps see which
kind of defect. Two families, after DRAEM / CutPaste (texture displaced into a smooth random
region) and the thin scratches that dominate MVTec AD 2's hardest categories:

* **blob** — a region shaped by thresholded multi-scale value noise (Perlin-like), covering
  0.05-2% of the image, filled with the same image's texture from another place (shifted, possibly
  flipped), scaled in brightness and alpha-blended;
* **stroke** — a polyline 1-6 px wide (native pixels) of a darker or brighter shade of the local
  colour: a scratch, a hair, a crack.

Everything is drawn from the generator passed in, so a defect is reproducible from its seed.
"""

from __future__ import annotations

import numpy as np


def value_noise(h: int, w: int, rng: np.random.Generator, octaves: int = 3) -> np.ndarray:
    """Smooth noise in [0, 1]: random grids at a few coarse scales, upsampled bicubically and summed."""
    import cv2

    out = np.zeros((h, w), dtype=np.float32)
    base = rng.integers(4, 12)
    for o in range(octaves):
        cells = int(base * 2**o)
        grid = rng.random((max(2, h * cells // max(h, w)), max(2, w * cells // max(h, w)))).astype(np.float32)
        out += cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC) / 2**o
    out -= out.min()
    return out / max(float(out.max()), 1e-6)


def blob_defect(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    h, w = image.shape[:2]
    noise = value_noise(h // 4, w // 4, rng)
    area = rng.uniform(0.0005, 0.02)
    region = noise > np.quantile(noise, 1.0 - area * rng.uniform(1.0, 3.0))
    # keep one connected region, near the size asked for
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
    if n > 1:
        target = area * region.size
        k = 1 + int(np.argmin(np.abs(stats[1:, cv2.CC_STAT_AREA] - target)))
        region = labels == k
    mask = cv2.resize(region.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    source = np.roll(image, (int(rng.integers(h // 8, h - h // 8)), int(rng.integers(w // 8, w - w // 8))), (0, 1))
    if rng.random() < 0.5:
        source = source[:, ::-1]
    source = np.clip(source.astype(np.float32) * rng.uniform(0.6, 1.4), 0, 255)
    alpha = rng.uniform(0.6, 1.0)
    out = image.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * source[mask]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), mask


def stroke_defect(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    h, w = image.shape[:2]
    n_points = int(rng.integers(2, 6))
    start = np.array([rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h])
    length = rng.uniform(0.03, 0.25) * max(h, w)
    angle = rng.uniform(0, np.pi)
    points = [start]
    for _ in range(n_points - 1):
        angle += rng.normal(0, 0.4)
        points.append(points[-1] + length / (n_points - 1) * np.array([np.cos(angle), np.sin(angle)]))
    pts = np.round(np.array(points)).astype(np.int32).reshape(-1, 1, 2)
    width = int(rng.integers(1, 7))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(mask, [pts], False, 1, thickness=width)
    mask = mask.astype(bool)
    if not mask.any():
        return image.copy(), mask
    factor = rng.uniform(0.35, 0.75) if rng.random() < 0.5 else rng.uniform(1.3, 1.9)
    out = image.astype(np.float32)
    out[mask] = np.clip(out[mask] * factor, 0, 255)
    return np.rint(out).astype(np.uint8), mask


def synthetic_defect(image: np.ndarray, rng: np.random.Generator, p_stroke: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """One defect on an HxWx3 uint8 normal image: (defective image, boolean mask)."""
    for _ in range(5):  # a stroke can fall outside a tiny image; retry rather than return no defect
        out, mask = stroke_defect(image, rng) if rng.random() < p_stroke else blob_defect(image, rng)
        if mask.any():
            return out, mask
    return blob_defect(image, rng)
