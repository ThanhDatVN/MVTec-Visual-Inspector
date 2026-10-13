"""From a continuous anomaly map to a defect mask: thresholding plus shape post-processing.

The MVTec AD 2 server scores binary masks (SegF1, and ClassF1 from "any pixel set"), and the
VAND 3.0/4.0 entries show that most of the gap between a good map and a good mask is closed
after thresholding (VAND 3.0 report, arXiv:2509.17615; SuperADD, arXiv:2605.14808). This module
reimplements, from SuperADD's description, the post-processing that took a training-free
memory-bank method to the best published SegF1 on AD 2:

1. threshold the map at ``tau``;
2. close gaps along thin, elongated defects with line structuring elements at several
   orientations (a scratch broken into pieces becomes one region), the union over orientations;
3. keep only closed pixels whose score is above ``lower * tau`` (closing cannot spread far into
   clearly normal surface);
4. fill the holes of every region;
5. erode by ``erosion`` pixels.

The work happens at ``1/work_downscale`` of the native resolution (SuperADD: 1/4, closing radius
26 px there) and the mask is returned at native resolution. ``tau`` itself comes from normal
images only — validation maps, never test labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def line_kernel(radius: int, angle_deg: float, thickness: int = 1) -> np.ndarray:
    """A centred line segment of half-length `radius` at `angle_deg` (0 = horizontal)."""
    import cv2

    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    c = radius
    dx, dy = radius * np.cos(np.deg2rad(angle_deg)), radius * np.sin(np.deg2rad(angle_deg))
    cv2.line(kernel, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, thickness)
    return kernel


def oriented_closing(mask: np.ndarray, radius: int, n_angles: int) -> np.ndarray:
    """Union of the input and its line closings at `n_angles` orientations in [0, 180).

    Each closing dilates with a thick line (3 px) and erodes with a thin one (1 px): a defect a
    pixel wide, rasterized at an angle between two of the sampled orientations, would not contain
    a thin line element of the dilation's width, and a symmetric closing would then fail to bridge
    its gaps. The asymmetry tolerates that mismatch at the cost of a slightly wider result.
    """
    import cv2

    if radius <= 0 or n_angles <= 0:
        return mask.astype(bool)
    src = mask.astype(np.uint8)
    pad = radius + 1
    padded = np.pad(src, pad)  # zero padding: the border does not close regions against it
    out = padded.copy()
    for i in range(n_angles):
        angle = 180.0 * i / n_angles
        grown = cv2.dilate(padded, line_kernel(radius, angle, thickness=3))
        out = np.maximum(out, cv2.erode(grown, line_kernel(radius, angle, thickness=1)))
    return out[pad:-pad, pad:-pad].astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every region's interior (outer contours drawn filled)."""
    import cv2

    src = mask.astype(np.uint8)
    contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(src)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    import cv2

    if radius <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


@dataclass(frozen=True)
class MaskRule:
    """Everything that turns a map into a mask, given its threshold. Recorded with a submission."""

    lower: float = 0.8
    closing_radius: int = 26
    closing_angles: int = 16
    erosion: int = 1
    work_downscale: int = 4
    fill: bool = True

    def as_dict(self) -> dict:
        return asdict(self)

    def apply(self, anomaly_map: np.ndarray, tau: float) -> np.ndarray:
        """Native-resolution boolean mask for a native-resolution map."""
        import cv2

        h, w = anomaly_map.shape
        f = max(1, int(self.work_downscale))
        small = anomaly_map if f == 1 else cv2.resize(
            anomaly_map.astype(np.float32), (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        mask = small > tau
        if mask.any():
            mask = oriented_closing(mask, self.closing_radius, self.closing_angles)
            mask &= small > self.lower * tau
            if self.fill:
                mask = fill_holes(mask)
            mask = erode(mask, self.erosion)
        if f == 1:
            return mask
        return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


#: Only the threshold, no shape processing (the masks S1 submitted, at its 3-sigma threshold).
PLAIN = MaskRule(lower=1.0, closing_radius=0, closing_angles=0, erosion=0, work_downscale=1, fill=False)
