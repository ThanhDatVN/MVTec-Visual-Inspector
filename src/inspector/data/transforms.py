"""Deterministic image and mask transforms (protocol §4.1).

Three rules are enforced here because each one, violated, corrupts results in a
way that no later test would catch:

1. **Aspect-preserving resize.** `sheet_metal` is 4224x1056, a 4:1 frame. A
   blind square resize squashes it by 4x in one axis and turns a round pit into
   a line. The resize policy is therefore explicit and recorded, never implied
   by a default.
2. **INTER_AREA when downscaling.** Area-averaging integrates over the source
   pixels a target pixel covers. Bilinear samples a few of them, so a 3-pixel
   defect being downscaled 4x has a good chance of simply not being sampled.
3. **Nearest-neighbour for masks, then re-binarize.** Interpolating a label mask
   produces intermediate values, and every pixel metric computed against it is
   then quietly wrong.

The transform carries its own provenance so a run can record exactly what was
done to its pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

ResizeMode = Literal["aspect_preserving", "square", "none"]
Interpolation = Literal["area", "linear", "cubic", "nearest"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "opencv is required for resizing; install with pip install "
            "'mvtec-visual-inspector[torch]'"
        ) from exc
    return cv2


def _interp_flag(name: Interpolation) -> int:
    cv2 = _cv2()
    return {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
    }[name]


def target_size(
    width: int,
    height: int,
    *,
    mode: ResizeMode,
    long_side: int | None,
    short_side: int | None = None,
) -> tuple[int, int]:
    """Resolve the output (width, height) for one source size.

    Kept separate from the actual resize so the EDA can ask "what would this
    policy do to my defects?" without touching a pixel.
    """
    if mode == "none" or long_side is None:
        return width, height
    if mode == "square":
        return long_side, long_side
    if mode != "aspect_preserving":
        raise ValueError(f"unknown resize mode: {mode!r}")

    if short_side is not None:
        scale = short_side / min(width, height)
    else:
        scale = long_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


@dataclass(frozen=True)
class ImageTransform:
    """A frozen, recordable preprocessing policy."""

    mode: ResizeMode = "aspect_preserving"
    long_side: int | None = 448
    short_side: int | None = None
    interpolation_down: Interpolation = "area"
    interpolation_up: Interpolation = "linear"
    #: `imagenet` (the default) standardizes with the ImageNet statistics. Two photometric
    #: options remove capture-to-capture exposure differences per image before that:
    #: `gain` scales each image so its mean luminance is ImageNet's mean (contrast ratios kept);
    #: `standardize` gives each image's channels zero mean and unit variance (exposure and
    #: contrast removed); `flatfield` divides each channel by its own heavily blurred copy, which
    #: removes smooth shading and colour casts (a changed light direction or colour) and keeps
    #: edges and texture. All apply identically at fit and at test time.
    normalize: Literal["imagenet", "gain", "standardize", "flatfield", "unit", "none"] = "imagenet"

    # -- provenance --------------------------------------------------------
    def as_dict(self) -> dict:
        return asdict(self)

    def output_size(self, width: int, height: int) -> tuple[int, int]:
        return target_size(
            width, height, mode=self.mode, long_side=self.long_side, short_side=self.short_side
        )

    def scale_factor(self, width: int, height: int) -> float:
        """Linear scale applied to the image. <1 means downscaling."""
        out_w, out_h = self.output_size(width, height)
        return float(np.sqrt((out_w * out_h) / max(1, width * height)))

    # -- application -------------------------------------------------------
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize an HxWx3 uint8 image with the direction-appropriate filter."""
        cv2 = _cv2()
        h, w = image.shape[:2]
        out_w, out_h = self.output_size(w, h)
        if (out_w, out_h) == (w, h):
            return image
        downscaling = out_w * out_h < w * h
        flag = _interp_flag(self.interpolation_down if downscaling else self.interpolation_up)
        return cv2.resize(image, (out_w, out_h), interpolation=flag)

    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize a boolean mask with nearest-neighbour, then re-binarize.

        Re-binarizing after a nearest resize is redundant in principle and cheap
        in practice; it is here so that swapping the interpolation for
        experimentation cannot silently produce a non-binary mask.
        """
        cv2 = _cv2()
        h, w = mask.shape[:2]
        out_w, out_h = self.output_size(w, h)
        if (out_w, out_h) == (w, h):
            return mask.astype(bool)
        resized = cv2.resize(
            mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST
        )
        return resized.astype(bool)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """uint8 HxWx3 -> float32 CxHxW, normalized."""
        array = image.astype(np.float32) / 255.0
        if self.normalize == "imagenet":
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
        elif self.normalize == "gain":
            luminance = float(array.mean())
            array = array * (float(IMAGENET_MEAN.mean()) / max(luminance, 1e-3))  # no clipping
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
        elif self.normalize == "standardize":
            mean = array.mean(axis=(0, 1))
            std = array.std(axis=(0, 1))
            array = (array - mean) / np.maximum(std, 1e-3)
        elif self.normalize == "flatfield":
            array = array / np.maximum(shading(array), 0.05) * IMAGENET_MEAN
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
        elif self.normalize not in ("unit", "none"):
            raise ValueError(f"unknown normalization: {self.normalize!r}")
        return np.ascontiguousarray(array.transpose(2, 0, 1))

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None):
        out_image = self.normalize_image(self.resize_image(image))
        out_mask = self.resize_mask(mask) if mask is not None else None
        return out_image, out_mask


def shading(array: np.ndarray, *, sigma_fraction: float = 0.1) -> np.ndarray:
    """The smooth illumination component of an HxWxC float image: a Gaussian blur with
    sigma = `sigma_fraction` x the long side, computed on a 16x reduced copy (the blur is far
    wider than the reduction, so nothing it keeps is lost) and resized back."""
    cv2 = _cv2()
    h, w = array.shape[:2]
    small = cv2.resize(array, (max(1, w // 16), max(1, h // 16)), interpolation=cv2.INTER_AREA)
    sigma = sigma_fraction * max(small.shape[:2])
    blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR).reshape(array.shape)


def relight(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random change of light direction, intensity and colour on a uint8 image (E15).

    A smooth multiplicative field — a linear gradient in a random direction and one broad
    highlight or shadow — times a global gain and a warm/cool colour cast. The ranges cover the
    changed-lighting captures (`shift_k`) of MVTec AD 2 as measured against its `regular` ones
    (docs/14): channel gains 0.60-1.17 and a blue/red ratio down to 0.78. Sharp specular
    highlights that move with the light are outside what this model produces.
    """
    h, w = image.shape[:2]
    unit = float(max(h, w))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = xx / unit - 0.5 * w / unit
    yy = yy / unit - 0.5 * h / unit
    theta = rng.uniform(0.0, 2.0 * np.pi)
    field = 1.0 + rng.uniform(-0.4, 0.4) * (xx * np.cos(theta) + yy * np.sin(theta))
    cx, cy = rng.uniform(-0.5, 0.5) * w / unit, rng.uniform(-0.5, 0.5) * h / unit
    radius = rng.uniform(0.15, 0.4)
    field += rng.uniform(-0.3, 0.3) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * radius**2))
    tint = rng.uniform(-0.12, 0.12)
    gain = np.array([1.0 + tint, 1.0, 1.0 - tint], dtype=np.float32) * rng.uniform(0.6, 1.2)
    out = image.astype(np.float32) * field[..., None] * gain
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def exposure_shift(image: np.ndarray, stops: float) -> np.ndarray:
    """A global exposure change of `stops`, with the matching gamma, on a uint8 image.

    The same model of over- and under-exposure as the E08 corruption suite
    (`robustness.corruptions.exposure`), so a memory bank augmented with it covers
    exactly the shifts that suite applies.
    """
    gain = 2.0**stops
    gamma = 1.0 / (1.0 + 0.15 * stops)
    scaled = np.power(np.clip(image.astype(np.float32) / 255.0 * gain, 0.0, 1.0), gamma)
    return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)


def load_image(path, *, rgb: bool = True) -> np.ndarray:
    """Read an image as HxWx3 uint8 RGB.

    Reads via PIL rather than cv2.imread: cv2 returns BGR and silently returns
    None on a path it cannot read, which turns a typo into a confusing crash
    several frames later.
    """
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB" if rgb else "L"), dtype=np.uint8)


def load_mask(path) -> np.ndarray:
    """Read a ground-truth mask as a boolean HxW array."""
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8) > 0


def upsample_map(anomaly_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinearly upsample an anomaly map to (width, height).

    Protocol §4.1: every metric is computed at native resolution. Scoring a
    downscaled map against a downscaled mask inflates PRO, because a defect
    occupies proportionally more of a smaller image.
    """
    cv2 = _cv2()
    width, height = size
    if anomaly_map.shape[:2] == (height, width):
        return anomaly_map.astype(np.float64)
    return cv2.resize(
        anomaly_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    ).astype(np.float64)


def smooth_map(anomaly_map: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth an anomaly map (PatchCore's sigma=4 by default).

    Smoothing is an explicit, ablated post-processing step, not a hidden
    default: part of a reported AU-PRO is always attributable to it, and
    docs/03 makes the sigma sweep one of the ablation axes for that reason.
    """
    if sigma <= 0:
        return anomaly_map
    cv2 = _cv2()
    ksize = int(2 * round(3 * sigma) + 1)  # 3 sigma each side, forced odd
    return cv2.GaussianBlur(
        anomaly_map.astype(np.float32), (ksize, ksize), sigmaX=sigma, sigmaY=sigma
    ).astype(np.float64)
