"""Corruption suite for the robustness study (docs/05, Phase P7).

Robustness here means: the camera, the lamp or the lens changed slightly and
nobody retrained. That is the normal condition of a deployed inspection system —
bulbs age, a technician nudges a lens, a maintenance swap changes a sensor.

Four design rules, each of which prevents a specific way of producing a
meaningless robustness section:

1. **Only test images are corrupted.** The model is fitted once on clean data
   and never re-thresholded. Re-thresholding under corruption answers a
   different and much easier question.
2. **Severity is physically interpretable**, defined relative to the *measured
   median defect diameter* of the category rather than to ImageNet-C's
   constants. Our images are far larger than ImageNet's, so those constants do
   not transfer.
3. **Ground truth stays aligned.** Geometric corruptions transform the mask
   identically; photometric ones leave it untouched. Both are tested.
4. **Deterministic.** Same seed and severity give bit-identical output, so a
   robustness number can be reproduced.

Each family is a pure function `(image, mask, severity, d_med, rng) ->
(image', mask')` on uint8 RGB in [0, 255].
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

#: Severity levels. 0 is the identity and is used to assert exactly that.
SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)


def _cv2():
    import cv2

    return cv2


def _as_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Optical
# ---------------------------------------------------------------------------


def gaussian_blur(image, mask, severity, d_med, rng):
    """Defocus/vibration as a Gaussian kernel, scaled to the defect size.

    sigma runs from 0.1 to 1.0 times the median defect diameter, so severity 5
    blurs over roughly a whole defect — the point at which a human inspector
    would also struggle, which is where the scale should stop.
    """
    cv2 = _cv2()
    sigma = float(np.linspace(0.1, 1.0, len(SEVERITIES))[severity - 1] * d_med)
    if sigma <= 0:
        return image, mask
    ksize = 2 * round(3 * sigma) + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma), mask


def defocus_blur(image, mask, severity, d_med, rng):
    """True optical defocus is a *disc*, not a Gaussian.

    The difference matters here: a disc kernel has sharp support, so it
    preserves small-scale contrast further than a Gaussian of equal energy and
    then loses it abruptly. Reporting only Gaussian blur would overstate how
    gracefully a model degrades under real defocus.
    """
    cv2 = _cv2()
    radius = max(1, round(float(np.linspace(0.1, 0.8, len(SEVERITIES))[severity - 1] * d_med)))
    size = 2 * radius + 1
    yy, xx = np.mgrid[:size, :size] - radius
    kernel = ((xx**2 + yy**2) <= radius**2).astype(np.float32)
    kernel /= kernel.sum()
    return _as_uint8(cv2.filter2D(image.astype(np.float32), -1, kernel)), mask


# ---------------------------------------------------------------------------
# Illumination
# ---------------------------------------------------------------------------


def exposure(image, mask, severity, d_med, rng, *, direction: int = 1):
    """Global exposure shift in stops, plus the matching gamma.

    Over- and under-exposure fail differently — one clips highlights, the other
    buries the signal in sensor noise — so `direction` is a parameter and the
    two are reported separately rather than averaged.
    """
    stops = float(np.linspace(0.25, 2.0, len(SEVERITIES))[severity - 1]) * direction
    gain = 2.0**stops
    gamma = 1.0 / (1.0 + 0.15 * stops)
    scaled = np.power(np.clip(image.astype(np.float32) / 255.0 * gain, 0, 1), gamma)
    return _as_uint8(scaled * 255.0), mask


def exposure_up(image, mask, severity, d_med, rng):
    return exposure(image, mask, severity, d_med, rng, direction=1)


def exposure_down(image, mask, severity, d_med, rng):
    return exposure(image, mask, severity, d_med, rng, direction=-1)


def spatial_light(image, mask, severity, d_med, rng):
    """An off-axis lamp: a smooth multiplicative gradient across the frame.

    This is the closest synthetic analogue to MVTec AD 2's real lighting-shift
    split, so the correlation between behaviour here and behaviour there is the
    validity check that gives the whole synthetic suite its meaning.
    """
    falloff = float(np.linspace(0.05, 0.5, len(SEVERITIES))[severity - 1])
    h, w = image.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    axis = (xx / max(w - 1, 1)) * np.cos(angle) + (yy / max(h - 1, 1)) * np.sin(angle)
    axis = (axis - axis.min()) / (np.ptp(axis) + 1e-9)
    gain = (1.0 - falloff) + 2.0 * falloff * axis
    return _as_uint8(image.astype(np.float32) * gain[..., None]), mask


# ---------------------------------------------------------------------------
# Geometric / transport
# ---------------------------------------------------------------------------


def resize_roundtrip(image, mask, severity, d_med, rng):
    """Downscale then upscale back to native.

    Operationally the most common corruption — a wrong camera mode, a
    bandwidth-limited link, a thumbnail pipeline — and the one most likely to
    erase a small defect outright, since it destroys exactly the high-frequency
    content a defect consists of. The mask is untouched: the ground truth of
    where the defect *was* does not change because the image was degraded.
    """
    cv2 = _cv2()
    factor = float(np.linspace(0.9, 0.35, len(SEVERITIES))[severity - 1])
    h, w = image.shape[:2]
    small = cv2.resize(
        image, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), mask


def noise_jpeg(image, mask, severity, d_med, rng):
    """Sensor noise then JPEG, in that order.

    The order is the physical one: gain noise happens at the sensor, compression
    happens afterwards in the pipeline. Reversing it would let JPEG smooth away
    the noise and understate the damage.
    """
    cv2 = _cv2()
    sigma = float(np.linspace(1.0, 10.0, len(SEVERITIES))[severity - 1])
    quality = int(np.linspace(95, 40, len(SEVERITIES))[severity - 1])

    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)
    noisy = _as_uint8(noisy)
    ok, buffer = cv2.imencode(".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover - encoder failure is not expected
        return noisy, mask
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR), mask


def translate(image, mask, severity, d_med, rng):
    """Small rigid translation — a fixture that moved.

    Reported separately from the main robustness mean, never inside it: on a
    fixed industrial rig a large geometric shift is out of distribution by
    design, so folding it into an average would misrepresent the deployment
    condition. It is kept because position-dependent methods (PaDiM especially)
    fail here first, which predicts what a fixture change would cost.
    """
    cv2 = _cv2()
    frac = float(np.linspace(0.002, 0.02, len(SEVERITIES))[severity - 1])
    h, w = image.shape[:2]
    dx, dy = round(w * frac), round(h * frac)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    shifted = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    shifted_mask = cv2.warpAffine(
        mask.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    return shifted, shifted_mask


#: (image, mask, severity, median_defect_diameter, rng) -> (image, mask)
CorruptionFn = Callable[
    [np.ndarray, np.ndarray, int, float, np.random.Generator],
    "tuple[np.ndarray, np.ndarray]",
]


@dataclass(frozen=True)
class Corruption:
    name: str
    fn: CorruptionFn
    family: str
    geometric: bool = False
    #: Excluded from the headline robustness mean; see `translate`.
    in_main_suite: bool = True


CORRUPTIONS: dict[str, Corruption] = {
    c.name: c
    for c in (
        Corruption("gaussian_blur", gaussian_blur, "optical"),
        Corruption("defocus_blur", defocus_blur, "optical"),
        Corruption("exposure_up", exposure_up, "illumination"),
        Corruption("exposure_down", exposure_down, "illumination"),
        Corruption("spatial_light", spatial_light, "illumination"),
        Corruption("resize_roundtrip", resize_roundtrip, "transport"),
        Corruption("noise_jpeg", noise_jpeg, "transport"),
        Corruption("translate", translate, "geometric", geometric=True, in_main_suite=False),
    )
}

MAIN_SUITE: tuple[str, ...] = tuple(n for n, c in CORRUPTIONS.items() if c.in_main_suite)


def apply_corruption(
    image: np.ndarray,
    mask: np.ndarray | None,
    name: str,
    severity: int,
    *,
    median_defect_diameter: float = 8.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply one corruption at one severity.

    Args:
        image: HxWx3 uint8.
        mask: HxW bool ground truth, or None for a normal image.
        name: a key of `CORRUPTIONS`.
        severity: 0 (identity) through 5.
        median_defect_diameter: from the EDA, in native pixels. This is what
            makes severity comparable across categories with different defect
            scales; a fixed kernel size would mean something different on each.
        seed: makes the result reproducible.
    """
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}; known: {sorted(CORRUPTIONS)}")
    if severity == 0:
        return image, mask
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be 0 or one of {SEVERITIES}, got {severity}")
    if image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError(f"expected HxWx3 uint8 image, got {image.shape} {image.dtype}")

    corruption = CORRUPTIONS[name]
    rng = np.random.default_rng([seed, severity, abs(hash(name)) % (2**31)])
    working_mask = mask if mask is not None else np.zeros(image.shape[:2], dtype=bool)

    out_image, out_mask = corruption.fn(image, working_mask, severity, median_defect_diameter, rng)
    return _as_uint8(out_image), (out_mask if mask is not None else None)


def severity_grid(
    names: Sequence[str] | None = None, severities: Sequence[int] = SEVERITIES
) -> list[tuple[str, int]]:
    """The (corruption, severity) grid to evaluate."""
    return [(name, s) for name in (names or MAIN_SUITE) for s in severities]
