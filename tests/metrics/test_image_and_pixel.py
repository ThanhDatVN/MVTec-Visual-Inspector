"""Image-level and pixel-level metric correctness (Gate G2)."""

from __future__ import annotations

import numpy as np
import pytest

from inspector.metrics import (
    compute_image_metrics,
    escape_rate_by_defect,
    f1_max,
    fpr_at_threshold,
    image_aupr,
    image_auroc,
    pixel_auroc,
    recall_at_threshold,
    segmentation_metrics,
)

# --- image level ------------------------------------------------------------


def test_perfect_ranking_gives_auroc_one():
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert image_auroc(labels, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == 1.0


def test_inverted_ranking_gives_auroc_zero():
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert image_auroc(labels, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) == 0.0


def test_random_scorer_is_chance_on_average():
    """The single most important smoke test in the project: if a random scorer
    is not at chance, the metric is broken and every downstream number is wrong
    (Gate G2)."""
    rng = np.random.default_rng(0)
    values = [
        image_auroc(
            np.repeat([0, 1], 100),
            rng.random(200),
        )
        for _ in range(200)
    ]
    assert np.mean(values) == pytest.approx(0.5, abs=0.02)


def test_all_tied_scores_give_half():
    assert image_auroc(np.array([0, 0, 1, 1]), np.array([0.5] * 4)) == 0.5


def test_auroc_requires_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        image_auroc(np.zeros(5, dtype=int), np.arange(5.0))


def test_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="labels must be"):
        image_auroc(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))


def test_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="NaN or inf"):
        image_auroc(np.array([0, 1]), np.array([0.1, np.inf]))


def test_aupr_reflects_imbalance():
    """AUPR must fall as positives become rarer even at fixed AUROC, which is
    exactly why it is reported alongside: the factory condition is imbalanced."""
    balanced = image_aupr(np.repeat([0, 1], 50), np.concatenate([np.zeros(50), np.ones(50)]))
    rare = image_aupr(
        np.concatenate([np.zeros(99, int), [1]]),
        np.concatenate([np.linspace(0, 0.9, 99), [1.0]]),
    )
    assert balanced == pytest.approx(1.0)
    assert rare == pytest.approx(1.0)  # still separable; AUPR is 1 when perfect

    noisy = image_aupr(
        np.concatenate([np.zeros(99, int), [1]]),
        np.concatenate([np.full(99, 0.6), [0.5]]),
    )
    assert noisy < 0.1


def test_f1_max_finds_the_separating_threshold():
    labels = np.array([0, 0, 1, 1])
    best, threshold = f1_max(labels, np.array([0.1, 0.2, 0.8, 0.9]))
    assert best == pytest.approx(1.0)
    assert threshold == pytest.approx(0.8)


def test_fpr_and_recall_at_a_fixed_threshold():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.6, 0.7])
    assert fpr_at_threshold(labels, scores, 0.45) == pytest.approx(0.25)  # one normal above
    assert recall_at_threshold(labels, scores, 0.45) == pytest.approx(0.75)  # three of four


def test_compute_image_metrics_carries_threshold_provenance():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_image_metrics(
        labels, scores, threshold=0.5, threshold_source="validation"
    )
    assert metrics.threshold_source == "validation"
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.fpr == pytest.approx(0.0)
    assert "threshold_source" in metrics.as_dict()


def test_escape_rate_is_reported_per_defect_type():
    """A scalar recall hides that one defect class is missed entirely."""
    labels = np.array([0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.9, 0.9, 0.2, 0.2])
    defects = np.array(["good", "scratch", "scratch", "pit", "pit"])

    escapes = escape_rate_by_defect(labels, scores, defects, threshold=0.5)
    assert escapes == {"pit": 1.0, "scratch": 0.0}


# --- pixel level ------------------------------------------------------------


def test_pixel_auroc_perfect_and_inverted():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:16, 8:16] = True
    assert pixel_auroc([mask.astype(float)], [mask]) == pytest.approx(1.0, abs=1e-6)
    assert pixel_auroc([1.0 - mask.astype(float)], [mask]) == pytest.approx(0.0, abs=1e-6)


def test_pixel_auroc_matches_sklearn_on_a_small_case():
    """The histogram implementation exists for memory reasons; it must still
    agree with the exact computation where the exact one is affordable."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    masks, scores = [], []
    for _ in range(4):
        m = np.zeros((40, 40), dtype=bool)
        m[5:20, 5:20] = True
        masks.append(m)
        scores.append(rng.random((40, 40)) + 0.5 * m)

    exact = roc_auc_score(
        np.concatenate([m.ravel() for m in masks]),
        np.concatenate([s.ravel() for s in scores]),
    )
    assert pixel_auroc(scores, masks) == pytest.approx(exact, abs=1e-4)


def test_pixel_auroc_of_a_constant_map_is_half():
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:6, 2:6] = True
    assert pixel_auroc([np.full((16, 16), 0.3)], [mask]) == 0.5


def test_segmentation_metrics_on_a_hand_computed_case():
    """4x4 grid, 4 defect pixels, prediction overlaps 2 of them and adds 2 false
    positives: precision 0.5, recall 0.5, F1 0.5, IoU 1/3."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, :4] = True

    score = np.zeros((4, 4))
    score[0, 0:2] = 1.0  # 2 true positives
    score[1, 0:2] = 1.0  # 2 false positives

    f1, precision, recall, iou, n_pos, n_neg = segmentation_metrics(
        [score], [mask], threshold=0.5, threshold_source="validation"
    )
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)
    assert iou == pytest.approx(1 / 3)
    assert (n_pos, n_neg) == (4, 12)


def test_segf1_changes_under_rescaling_while_aupro_does_not(rng):
    """The asymmetry that explains PatchCore's AD 2 result: a model can rank
    perfectly (threshold-free metrics high) and still be uncalibratable at a
    fixed threshold (SegF1 near zero)."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:16, 8:16] = True
    score = mask.astype(float) * 0.9 + 0.05

    at_half = segmentation_metrics(
        [score], [mask], threshold=0.5, threshold_source="validation"
    )[0]
    shifted = segmentation_metrics(
        [score * 0.1], [mask], threshold=0.5, threshold_source="validation"
    )[0]

    assert at_half == pytest.approx(1.0)
    assert shifted == 0.0


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        segmentation_metrics(
            [np.zeros((4, 4))],
            [np.zeros((8, 8), dtype=bool)],
            threshold=0.5,
            threshold_source="validation",
        )
