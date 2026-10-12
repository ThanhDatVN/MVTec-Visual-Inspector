"""Photometric options: per-image exposure normalization and exposure shifts (E14)."""

from __future__ import annotations

import numpy as np
import pytest

from inspector.data.transforms import ImageTransform, exposure_shift, relight


def _image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(20, 120, size=(16, 12, 3), dtype=np.uint8) * 2  # even, so halving is exact


def test_gain_normalization_cancels_a_global_exposure_change():
    image = _image()
    darker = (image // 2).astype(np.uint8)
    t = ImageTransform(long_side=None, normalize="gain")
    assert np.allclose(t.normalize_image(image), t.normalize_image(darker), atol=1e-5)


def test_standardization_cancels_gain_and_offset():
    image = _image(1).astype(np.float32)
    shifted = np.clip(image * 0.6 + 30, 0, 255).astype(np.uint8)
    t = ImageTransform(long_side=None, normalize="standardize")
    out_a, out_b = t.normalize_image(image.astype(np.uint8)), t.normalize_image(shifted)
    assert np.allclose(out_a.mean(axis=(1, 2)), 0, atol=1e-5)
    assert np.allclose(out_a, out_b, atol=0.05)  # up to uint8 rounding


def test_imagenet_normalization_is_unchanged():
    image = _image(2)
    out = ImageTransform(long_side=None).normalize_image(image)
    expected = (image.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    assert np.allclose(out, expected.transpose(2, 0, 1), atol=1e-6)


def test_unknown_normalization_is_rejected():
    with pytest.raises(ValueError):
        ImageTransform(long_side=None, normalize="bogus").normalize_image(_image())  # type: ignore[arg-type]


def test_exposure_shift_is_identity_at_zero_and_monotone():
    image = _image(3)
    assert np.array_equal(exposure_shift(image, 0.0), image)
    assert exposure_shift(image, 0.5).mean() > image.mean() > exposure_shift(image, -0.5).mean()
    assert exposure_shift(image, 0.5).dtype == np.uint8


def test_exposure_shift_matches_the_corruption_suite():
    from inspector.robustness import apply_corruption

    image = _image(4)
    # severity 1 of `exposure_up` is +0.25 stop
    corrupted, _ = apply_corruption(image, None, "exposure_up", 1)
    assert np.abs(corrupted.astype(int) - exposure_shift(image, 0.25).astype(int)).max() <= 1


def _scene(h: int = 64, w: int = 96) -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8)


def test_flatfield_cancels_a_smooth_shading_gradient_and_a_colour_cast():
    scene = _scene().astype(np.float32)
    ramp = np.linspace(0.7, 1.2, scene.shape[1], dtype=np.float32)[None, :, None]
    cast = np.array([1.1, 1.0, 0.85], dtype=np.float32)
    lit = np.clip(np.rint(scene * ramp * cast), 0, 255).astype(np.uint8)
    t = ImageTransform(long_side=None, normalize="flatfield")
    a, b = t.normalize_image(scene.astype(np.uint8)), t.normalize_image(lit)
    plain = ImageTransform(long_side=None)
    raw_gap = np.abs(plain.normalize_image(scene.astype(np.uint8)) - plain.normalize_image(lit)).mean()
    assert np.abs(a - b).mean() < 0.25 * raw_gap


def test_relight_is_seeded_bounded_and_not_the_identity():
    image = _scene()
    one = relight(image, np.random.default_rng((0, 1, 0)))
    assert np.array_equal(one, relight(image, np.random.default_rng((0, 1, 0))))
    assert not np.array_equal(one, relight(image, np.random.default_rng((0, 1, 1))))
    assert one.dtype == np.uint8 and one.shape == image.shape
    ratio = one.astype(np.float64).mean() / image.astype(np.float64).mean()
    assert 0.3 < ratio < 1.8
