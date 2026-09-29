"""The shared evaluator (docs/13, F07/F09).

Torch-free: a linear-PCA baseline on synthetic fixtures exercises every code path.
"""

from __future__ import annotations

import numpy as np
import pytest

from inspector.data.transforms import ImageTransform
from inspector.evaluate import EvalConfig, SplitPredictions, evaluate
from inspector.metrics import au_pro
from inspector.models import PixelPCAScorer


@pytest.fixture(scope="module")
def predictions(tmp_path_factory):
    from inspector.data import SYNTHETIC, discover_category
    from inspector.fixtures import FixtureSpec, generate

    root = generate(
        tmp_path_factory.mktemp("ev"),
        spec=FixtureSpec(n_train=10, n_validation=6, n_test_good=5, n_test_per_defect=4),
        overwrite=True,
    )
    indices = discover_category(root, "synth_strip", layout=SYNTHETIC)
    model = PixelPCAScorer(ImageTransform(long_side=128), n_components=4, work_size=32)
    model.fit(indices["train"])
    val = SplitPredictions.from_model(model, indices["validation"])
    test = SplitPredictions.from_model(model, indices["test_public"])
    return model, indices, val, test


CFG = EvalConfig(target_fpr=0.2, threshold_policy="relax", aupro_num_thresholds=256)


def test_evaluation_produces_every_headline_metric(predictions):
    _, _, val, test = predictions
    out = evaluate(val, test, CFG)
    for key in ("image_auroc", "fpr_at_op1", "recall_at_op1", "requested_fpr", "effective_fpr",
                "pixel_auroc", "aupro_005", "aupro_030", "segf1_3sigma", "metrics_version"):
        assert key in out, key
    assert out["threshold_comparator"] == ">"
    assert out["tp"] + out["fn"] == int(test.labels.sum())
    assert out["fp"] + out["tn"] == int((test.labels == 0).sum())


def test_severity_zero_reproduces_the_clean_evaluation(predictions):
    """F09: the robustness path and the clean path are one evaluator, so an
    identity corruption must give bit-identical metrics."""
    from inspector.robustness import apply_corruption

    model, indices, val, test = predictions
    clean = evaluate(val, test, CFG)

    def identity(image, mask, sample):
        return apply_corruption(image, mask, "gaussian_blur", 0)

    corrupted = SplitPredictions.from_model(model, indices["test_public"], image_fn=identity)
    again = evaluate(val, corrupted, CFG)
    for key in ("image_auroc", "pixel_auroc", "aupro_005", "aupro_030", "segf1_3sigma", "fpr_at_op1"):
        assert again[key] == clean[key], key


def test_corrupted_evaluation_keeps_normal_images(predictions):
    """The old robustness code scored AU-PRO over anomalous images only, so its
    FPR denominator silently differed from the clean one."""
    from inspector.robustness import apply_corruption

    model, indices, val, _ = predictions

    def blur(image, mask, sample):
        return apply_corruption(image, mask, "gaussian_blur", 3, median_defect_diameter=4.0)

    corrupted = SplitPredictions.from_model(model, indices["test_public"], image_fn=blur)
    assert (corrupted.labels == 0).sum() == indices["test_public"].n_normal
    out = evaluate(val, corrupted, CFG)
    assert out["fp"] + out["tn"] == indices["test_public"].n_normal


def test_geometric_corruption_moves_the_masks(predictions):
    from inspector.robustness import apply_corruption

    model, indices, _, _ = predictions

    def shift(image, mask, sample):
        return apply_corruption(image, mask, "translate", 5)

    corrupted = SplitPredictions.from_model(model, indices["test_public"], image_fn=shift)
    moved = [m for m in corrupted.mask_overrides if m is not None]
    assert moved, "translation must produce transformed masks"


def test_changing_the_image_score_leaves_localization_unchanged(predictions):
    """The negative control for the aggregation study (docs/13, E02): a new
    map-to-score rule must move image metrics and leave every pixel metric
    exactly where it was, because the maps did not change."""
    _, _, val, test = predictions
    base = evaluate(val, test, CFG)
    alt_val = val.with_scores(np.array([m.max() for m in val.raw_maps]))
    alt_test = test.with_scores(np.array([m.max() for m in test.raw_maps]))
    other = evaluate(alt_val, alt_test, CFG)
    for key in ("pixel_auroc", "aupro_005", "aupro_030", "segf1_3sigma"):
        assert other[key] == base[key], key


def test_streaming_aupro_matches_the_in_memory_computation(predictions):
    """With no negative subsampling, the one-pass evaluator and the reference
    in-memory `au_pro` must agree."""
    _, _, val, test = predictions
    streamed = evaluate(val, test, EvalConfig(target_fpr=0.2, threshold_policy="relax",
                                              aupro_num_thresholds=512, max_negative_samples=10**9))
    maps = list(test.native_maps(4.0))
    masks = list(test.masks())
    direct = au_pro(maps, masks, integration_limit=0.05, num_thresholds=512, max_negative_samples=None)
    assert streamed["aupro_005"] == pytest.approx(direct, abs=5e-3)


def test_predictions_round_trip_through_disk(predictions, tmp_path):
    _, _, val, test = predictions
    val.save(tmp_path)
    test.save(tmp_path)
    val2 = SplitPredictions.load(tmp_path, val.split)
    test2 = SplitPredictions.load(tmp_path, test.split)
    assert val2.digest() == val.digest()
    a, b = evaluate(val, test, CFG), evaluate(val2, test2, CFG)
    assert b["image_auroc"] == a["image_auroc"]
    # maps are stored as float16: localization agrees to storage precision
    assert b["aupro_005"] == pytest.approx(a["aupro_005"], abs=2e-3)


def test_smoothing_is_applied_in_input_pixels():
    """sigma is in input-image pixels (anomalib's convention): the same raw map
    smoothed at sigma=4 must be broader than at sigma=0 by the same amount
    regardless of native size, because blurring happens before upsampling."""
    from inspector.models.base import postprocess_map

    raw = np.zeros((16, 16), dtype=np.float32)
    raw[8, 8] = 1.0
    sharp = postprocess_map(raw, input_size=(64, 64), native_size=(256, 256), sigma=0.0)
    smooth = postprocess_map(raw, input_size=(64, 64), native_size=(256, 256), sigma=4.0)
    assert smooth.max() < sharp.max()
    assert (smooth > 0.5 * smooth.max()).sum() > (sharp > 0.5 * sharp.max()).sum()
