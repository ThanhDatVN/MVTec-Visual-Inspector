"""Synthetic defects for learned fusion (E23)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from inspector.data.synthetic_defects import (
    blob_defect,
    stroke_defect,
    synthetic_defect,
    value_noise,
)


def _image(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(60, 200, w, dtype=np.float32)[None, :, None] * np.ones((h, 1, 3), np.float32)
    return np.clip(base + rng.normal(0, 8, (h, w, 3)), 0, 255).astype(np.uint8)


def test_value_noise_is_smooth_and_normalized():
    n = value_noise(64, 96, np.random.default_rng(1))
    assert n.shape == (64, 96) and n.min() == 0.0 and n.max() == pytest.approx(1.0)
    assert np.abs(np.diff(n, axis=1)).mean() < 0.05


def test_blob_changes_only_its_mask_and_is_small():
    img = _image()
    out, mask = blob_defect(img, np.random.default_rng(2))
    assert mask.any() and mask.mean() < 0.06
    assert np.array_equal(out[~mask], img[~mask])
    assert not np.array_equal(out[mask], img[mask])


def test_stroke_is_thin_and_shifts_brightness():
    img = _image()
    out, mask = stroke_defect(img, np.random.default_rng(3))
    assert mask.any() and mask.mean() < 0.05
    assert np.array_equal(out[~mask], img[~mask])
    assert abs(out[mask].astype(float).mean() - img[mask].astype(float).mean()) > 5


def test_synthetic_defects_are_reproducible():
    img = _image()
    a = synthetic_defect(img, np.random.default_rng(7))
    b = synthetic_defect(img, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
