"""Mask post-processing for the MVTec AD 2 server (SuperADD-style, reimplemented)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from inspector.postproc.binarize import PLAIN, MaskRule, fill_holes, oriented_closing


def test_closing_reconnects_a_broken_scratch_in_any_direction():
    for angle in (0, 30, 90, 135):
        mask = np.zeros((120, 120), dtype=bool)
        t = np.linspace(-40, 40, 161)
        xs = np.rint(60 + t * np.cos(np.deg2rad(angle))).astype(int)
        ys = np.rint(60 + t * np.sin(np.deg2rad(angle))).astype(int)
        keep = (np.abs(t) > 6)  # a 12 px gap in the middle
        mask[ys[keep], xs[keep]] = True
        closed = oriented_closing(mask, radius=10, n_angles=16)
        assert closed[60, 60], angle
        assert closed[mask].all()  # closing never removes a pixel


def test_closing_does_not_bridge_far_blobs():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:14, 10:14] = True
    mask[80:84, 80:84] = True
    closed = oriented_closing(mask, radius=10, n_angles=16)
    assert not closed[47, 47]


def test_fill_holes_fills_a_ring():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True
    mask[15:35, 15:35] = False
    assert fill_holes(mask)[25, 25]


def test_lower_threshold_limits_what_closing_adds():
    m = np.zeros((40, 200), dtype=np.float32)
    m[20, 20:60] = 1.0
    m[20, 70:110] = 1.0  # a 10 px gap at score 0
    rule = MaskRule(lower=0.5, closing_radius=8, closing_angles=8, erosion=0, work_downscale=1)
    assert not rule.apply(m, tau=0.9)[20, 65]  # the gap is clearly normal: not bridged
    m[20, 60:70] = 0.6  # now the gap is moderately anomalous
    assert rule.apply(m, tau=0.9)[20, 65]


def test_rule_returns_native_shape_and_empty_masks_for_clean_maps():
    rng = np.random.default_rng(0)
    m = rng.random((203, 157)).astype(np.float32) * 0.5
    out = MaskRule().apply(m, tau=1.0)
    assert out.shape == m.shape and out.dtype == bool and not out.any()


def test_plain_rule_is_a_bare_threshold():
    m = np.random.default_rng(1).random((64, 48))
    assert np.array_equal(PLAIN.apply(m, 0.7), m > 0.7)
