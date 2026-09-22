"""Procedural fixture generator.

Why this exists (docs/04, task 0.5): the MVTec datasets are CC BY-NC-SA and must
never enter the repository, and CI has no GPU and no data mount. Without
synthetic fixtures the test suite would only run on one laptop with the dataset
attached, which means it would quietly stop being run.

These fixtures deliberately mirror the *difficulty axes* of the three chosen
study categories rather than their appearance:

    synth_strip  <- sheet_metal   tiny defects, high aspect ratio, dark field
    synth_blob   <- fruit_jelly   translucent, overlapping, back-lit
    synth_grain  <- walnuts       high variance in normal appearance

They are not a substitute for the real data and no result is ever reported on
them. Their job is to make every code path — loaders, metrics, leakage checks,
the API — executable and assertable by anyone, forever.

The on-disk layout mirrors MVTec AD 2 so the same loader reads both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Defect classes per synthetic category. Each maps to a test_public subfolder.
DEFECT_TYPES: dict[str, tuple[str, ...]] = {
    "synth_strip": ("pit", "scratch"),
    "synth_blob": ("contamination", "bubble"),
    "synth_grain": ("crack", "chip"),
}

CATEGORY_SIZES: dict[str, tuple[int, int]] = {
    # (width, height) — synth_strip keeps the 4:1 aspect that makes a blind
    # square resize destructive, which is exactly the trap we want testable.
    "synth_strip": (256, 64),
    "synth_blob": (128, 128),
    "synth_grain": (128, 128),
}


@dataclass(frozen=True)
class FixtureSpec:
    """How many images of each kind to generate."""

    n_train: int = 12
    n_validation: int = 4
    n_test_good: int = 4
    n_test_per_defect: int = 3
    categories: tuple[str, ...] = field(default=tuple(DEFECT_TYPES))

    @property
    def n_images_per_category(self) -> int:
        n_defects = max(len(DEFECT_TYPES[c]) for c in self.categories)
        return (
            self.n_train
            + self.n_validation
            + self.n_test_good
            + self.n_test_per_defect * n_defects
        )


# ---------------------------------------------------------------------------
# Normal-image synthesis
# ---------------------------------------------------------------------------


def _smooth_noise(rng: np.random.Generator, h: int, w: int, scale: float) -> np.ndarray:
    """Band-limited noise: upsampling coarse noise gives spatially correlated
    texture, unlike per-pixel noise which no real sensor produces."""
    ch, cw = max(2, int(h / scale)), max(2, int(w / scale))
    coarse = rng.normal(size=(ch, cw))
    img = Image.fromarray(((coarse - coarse.min()) / (np.ptp(coarse) + 1e-9) * 255).astype(np.uint8))
    img = img.resize((w, h), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(1.0))
    return np.asarray(img, dtype=np.float32) / 255.0


def _normal_strip(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Dark-field brushed metal: horizontal grain plus a faint lighting gradient."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grain = 0.06 * np.sin(yy * rng.uniform(1.6, 2.4) + rng.uniform(0, 6.3))
    base = 0.22 + grain + 0.05 * _smooth_noise(rng, h, w, 6.0)
    base += 0.05 * (xx / w) * rng.uniform(-1.0, 1.0)  # mild illumination tilt
    return np.clip(base, 0.0, 1.0)


def _normal_blob(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Back-lit translucent discs that overlap: bright field, soft alpha
    compositing, so 'normal' contains a range of local intensities."""
    canvas = np.full((h, w), 0.88, dtype=np.float32)
    for _ in range(rng.integers(3, 6)):
        cx, cy = rng.uniform(0.15, 0.85, size=2) * (w, h)
        r = rng.uniform(0.14, 0.26) * min(w, h)
        yy, xx = np.mgrid[0:h, 0:w]
        disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r**2
        alpha = rng.uniform(0.12, 0.22)
        canvas[disc] *= 1.0 - alpha
    canvas += 0.02 * _smooth_noise(rng, h, w, 8.0)
    return np.clip(canvas, 0.0, 1.0)


def _normal_grain(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """A natural-product analogue: an ellipse whose size, tilt and internal
    texture vary a lot between samples. This is the false-positive stressor —
    a weak normality model will flag ordinary variation."""
    canvas = np.full((h, w), 0.12, dtype=np.float32)
    cx, cy = w / 2 + rng.uniform(-6, 6), h / 2 + rng.uniform(-6, 6)
    rx, ry = rng.uniform(0.30, 0.42) * w, rng.uniform(0.30, 0.42) * h
    theta = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xr = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    yr = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
    body = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0
    texture = 0.45 + 0.28 * _smooth_noise(rng, h, w, rng.uniform(3.0, 7.0))
    ridges = 0.06 * np.sin(xr * rng.uniform(0.35, 0.65) + rng.uniform(0, 6.3))
    canvas[body] = np.clip(texture + ridges, 0.0, 1.0)[body]
    return canvas


_NORMAL_FN = {
    "synth_strip": _normal_strip,
    "synth_blob": _normal_blob,
    "synth_grain": _normal_grain,
}


# ---------------------------------------------------------------------------
# Defect injection — returns (image, mask) with the mask exact by construction
# ---------------------------------------------------------------------------


def _inject(
    rng: np.random.Generator, img: np.ndarray, defect: str
) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if defect == "pit":
        # Sub-1% of image area: survives only if resolution is preserved.
        for _ in range(rng.integers(1, 3)):
            cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.2, 0.8) * h
            r = rng.uniform(1.5, 3.0)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif defect == "scratch":
        x0, y0 = rng.uniform(0.1, 0.6) * w, rng.uniform(0.2, 0.8) * h
        x1 = x0 + rng.uniform(0.15, 0.35) * w
        y1 = y0 + rng.uniform(-0.2, 0.2) * h
        draw.line([x0, y0, x1, y1], fill=255, width=int(rng.integers(1, 3)))
    elif defect == "contamination":
        cx, cy = rng.uniform(0.25, 0.75, size=2) * (w, h)
        r = rng.uniform(0.05, 0.10) * min(w, h)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif defect == "bubble":
        cx, cy = rng.uniform(0.25, 0.75, size=2) * (w, h)
        r = rng.uniform(0.04, 0.08) * min(w, h)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif defect == "crack":
        pts, x, y = [], rng.uniform(0.3, 0.5) * w, rng.uniform(0.3, 0.5) * h
        for _ in range(5):
            x += rng.uniform(3, 9)
            y += rng.uniform(-5, 5)
            pts.append((float(x), float(y)))
        draw.line(pts, fill=255, width=2, joint="curve")
    elif defect == "chip":
        cx, cy = rng.uniform(0.3, 0.7, size=2) * (w, h)
        r = rng.uniform(0.08, 0.14) * min(w, h)
        draw.pieslice(
            [cx - r, cy - r, cx + r, cy + r],
            start=float(rng.uniform(0, 360)),
            end=float(rng.uniform(120, 260)),
            fill=255,
        )
    else:  # pragma: no cover - guarded by DEFECT_TYPES
        raise ValueError(f"unknown defect type: {defect}")

    m = np.asarray(mask, dtype=np.uint8) > 0
    out = img.copy()
    if defect in ("pit", "scratch"):
        out[m] = np.clip(out[m] + rng.uniform(0.35, 0.55), 0, 1)  # bright on dark field
    elif defect == "bubble":
        # Deliberately the low-contrast case, to keep a near-threshold defect in
        # the suite. It *darkens*: brightening on an already bright back-lit
        # field clips at 1.0, which destroys the contrast it was meant to add.
        out[m] = np.clip(out[m] - rng.uniform(0.07, 0.11), 0, 1)
    else:
        out[m] = np.clip(out[m] - rng.uniform(0.30, 0.45), 0, 1)  # dark inclusion
    return out, m


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _to_rgb_png(arr: np.ndarray) -> Image.Image:
    u8 = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(np.repeat(u8[:, :, None], 3, axis=2), mode="RGB")


def generate(
    root: str | Path,
    spec: FixtureSpec | None = None,
    seed: int = 0,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the full synthetic dataset tree and return its root.

    Layout mirrors MVTec AD 2::

        root/<category>/train/good/*.png
        root/<category>/validation/good/*.png
        root/<category>/test_public/good/*.png
        root/<category>/test_public/<defect>/*.png
        root/<category>/test_public/ground_truth/<defect>/*_mask.png

    Every image is derived from `seed`, so the tree is byte-reproducible.
    """
    spec = spec or FixtureSpec()
    root = Path(root)
    if root.exists() and not overwrite and any(root.iterdir()):
        return root

    for cat_idx, category in enumerate(spec.categories):
        w, h = CATEGORY_SIZES[category]
        normal_fn = _NORMAL_FN[category]
        # Per-category stream offset keeps categories independent: regenerating
        # one category cannot change the pixels of another.
        rng = np.random.default_rng([seed, cat_idx])

        for split, count in (
            ("train", spec.n_train),
            ("validation", spec.n_validation),
            ("test_public", spec.n_test_good),
        ):
            out_dir = root / category / split / "good"
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                _to_rgb_png(normal_fn(rng, w, h)).save(out_dir / f"{i:03d}.png")

        for defect in DEFECT_TYPES[category]:
            img_dir = root / category / "test_public" / defect
            mask_dir = root / category / "test_public" / "ground_truth" / defect
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            for i in range(spec.n_test_per_defect):
                img, mask = _inject(rng, normal_fn(rng, w, h), defect)
                _to_rgb_png(img).save(img_dir / f"{i:03d}.png")
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
                    mask_dir / f"{i:03d}_mask.png"
                )

    return root
