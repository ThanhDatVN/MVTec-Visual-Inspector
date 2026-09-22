"""AU-PRO correctness (Gate G2).

These are the analytic tests. The reference-implementation comparison that Gate
G2 also requires lives in `test_aupro_reference.py` and is skipped until
anomalib is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from inspector.metrics.aupro import au_pro, pro_curve


def test_perfect_predictor_scores_one(toy_segmentation):
    value = au_pro(
        toy_segmentation["perfect"],
        toy_segmentation["masks"],
        integration_limit=0.05,
        num_thresholds=None,
    )
    assert value == pytest.approx(1.0, abs=1e-9)


def test_inverted_predictor_scores_zero(toy_segmentation):
    value = au_pro(
        toy_segmentation["inverted"],
        toy_segmentation["masks"],
        integration_limit=0.05,
        num_thresholds=None,
    )
    assert value == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("limit", [0.05, 0.10, 0.30])
def test_random_predictor_approaches_half_the_limit(limit):
    """For an uninformative scorer PRO(fpr) = fpr, so AU-PRO@L = L/2 after
    normalization. Deviating from this is the clearest sign of a bug in the FPR
    axis or the integration."""
    rng = np.random.default_rng(7)
    h = w = 128
    masks, scores = [], []
    for _ in range(30):
        m = np.zeros((h, w), dtype=bool)
        for _ in range(4):
            y, x = rng.integers(0, h - 30), rng.integers(0, w - 30)
            m[y : y + 25, x : x + 25] = True
        masks.append(m)
        scores.append(rng.random((h, w)))

    value = au_pro(scores, masks, integration_limit=limit, num_thresholds=2048)
    assert value == pytest.approx(limit / 2, rel=0.08)


def test_invariant_to_monotone_rescaling(toy_segmentation, rng):
    """AU-PRO is rank-based, so any strictly increasing transform of the scores
    must leave it unchanged. SegF1, by contrast, must change — that asymmetry is
    exactly what separates threshold-free from threshold-dependent metrics, and
    it is the mechanism behind PatchCore's SegF1 collapse on AD 2."""
    masks = toy_segmentation["masks"]
    scores = [rng.random(m.shape) for m in masks]

    plain = au_pro(scores, masks, integration_limit=0.3, num_thresholds=None)
    rescaled = au_pro(
        [3.5 * s + 10.0 for s in scores], masks, integration_limit=0.3, num_thresholds=None
    )
    squashed = au_pro(
        [np.expm1(s) for s in scores], masks, integration_limit=0.3, num_thresholds=None
    )

    assert rescaled == pytest.approx(plain, abs=1e-9)
    assert squashed == pytest.approx(plain, abs=1e-9)


def test_small_and_large_regions_weigh_equally():
    """The defining property of PRO: a tiny region counts as much as a huge one.

    One image has a 400-pixel region found perfectly and a 4-pixel region missed
    entirely, at zero false positives. A pixel-weighted overlap would read
    400/404 = 0.990; PRO must read exactly 0.5, because one of two *regions* was
    recovered.
    """
    h = w = 64
    mask = np.zeros((h, w), dtype=bool)
    mask[4:24, 4:24] = True  # 400 px
    mask[50:52, 50:52] = True  # 4 px

    score = np.zeros((h, w))
    score[4:24, 4:24] = 1.0  # large region found, small region missed

    curve = pro_curve(
        [score, np.zeros((h, w))],
        [mask, np.zeros((h, w), dtype=bool)],
        integration_limit=0.3,
        num_thresholds=None,
    )
    assert curve.n_regions == 2

    pixel_weighted = 400 / 404
    at_zero_fpr = curve.pro[curve.fpr == 0.0].max()
    assert at_zero_fpr == pytest.approx(0.5, abs=1e-12)
    assert pixel_weighted == pytest.approx(0.990, abs=1e-3)  # what PRO refuses to report


def test_connectivity_is_eight_way():
    """A diagonal chain is one scratch, not a string of separate defects."""
    mask = np.zeros((16, 16), dtype=bool)
    for i in range(6):
        mask[2 + i, 2 + i] = True

    curve = pro_curve(
        [np.zeros((16, 16))],
        [mask],
        integration_limit=0.3,
        num_thresholds=None,
    )
    assert curve.n_regions == 1


def test_normal_images_contribute_negatives():
    """Pixels of normal images must enter the FPR denominator. Excluding them
    flatters the metric exactly where deployment false alarms come from."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:8, 4:8] = True
    score = np.zeros((32, 32))

    without = pro_curve([score], [mask], integration_limit=0.3, num_thresholds=None)
    with_normals = pro_curve(
        [score, np.zeros((32, 32))],
        [mask, np.zeros((32, 32), dtype=bool)],
        integration_limit=0.3,
        num_thresholds=None,
    )
    assert with_normals.n_negative_pixels > without.n_negative_pixels
    assert with_normals.n_negative_pixels == without.n_negative_pixels + 32 * 32


def test_integration_limit_is_carried_not_implied(toy_segmentation):
    """A 0.05 result and a 0.30 result must be distinguishable after the fact."""
    curve = pro_curve(
        toy_segmentation["perfect"], toy_segmentation["masks"], integration_limit=0.05
    )
    assert curve.integration_limit == 0.05


def test_subsampled_grid_approximates_exact_grid(rng):
    masks, scores = [], []
    for _ in range(8):
        m = np.zeros((64, 64), dtype=bool)
        m[10:30, 10:30] = True
        masks.append(m)
        scores.append(rng.random((64, 64)) + m * 0.4)

    exact = au_pro(scores, masks, integration_limit=0.05, num_thresholds=None)
    sampled = au_pro(scores, masks, integration_limit=0.05, num_thresholds=512)
    assert sampled == pytest.approx(exact, abs=0.01)


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        au_pro([np.zeros((8, 8))], [np.ones((4, 4), dtype=bool)])


def test_rejects_empty_ground_truth():
    with pytest.raises(ValueError, match="no ground-truth regions"):
        au_pro([np.zeros((8, 8))], [np.zeros((8, 8), dtype=bool)])


def test_rejects_invalid_limit(toy_segmentation):
    with pytest.raises(ValueError, match="integration_limit"):
        au_pro(
            toy_segmentation["perfect"], toy_segmentation["masks"], integration_limit=0.0
        )
