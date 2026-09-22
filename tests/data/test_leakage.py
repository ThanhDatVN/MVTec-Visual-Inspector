"""Protocol §5 leakage rules L1-L7.

These are the tests that decide whether any later number means anything. A
leaked split does not produce an error, it produces a *better* result — which is
why it has to be caught mechanically rather than noticed.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from inspector.data import SYNTHETIC, discover_category, discover_split
from inspector.data.integrity import (
    calibrate_phash_threshold,
    check_duplicates_within,
    check_mask_binary,
    check_masks_present,
    check_near_duplicates,
    check_splits_disjoint,
    hamming,
    phash,
)
from inspector.postproc import ScoreNormalizer, Threshold, from_validation_percentile

# --- L1: split disjointness ------------------------------------------------


def test_L1_splits_are_disjoint(strip_indices):
    report = check_splits_disjoint(strip_indices)
    assert report.ok, str(report)


def test_L1_detects_a_planted_duplicate(synthetic_root, tmp_path):
    """The rule must catch the same *content* under a different name, which is
    what a filename-based check would miss."""
    src = tmp_path / "copy"
    shutil.copytree(synthetic_root / "synth_grain", src / "synth_grain")

    train_image = next((src / "synth_grain" / "train" / "good").iterdir())
    planted = src / "synth_grain" / "test_public" / "good" / "zz_renamed_copy.png"
    shutil.copyfile(train_image, planted)

    indices = discover_category(src, "synth_grain", layout=SYNTHETIC)
    report = check_splits_disjoint(indices)

    assert not report.ok
    assert any("zz_renamed_copy" in b for _, b in report.offenders)
    assert "zz_renamed_copy" in str(report)


def test_L1_within_split_duplicates_reported_separately(strip_indices):
    for name, index in strip_indices.items():
        assert check_duplicates_within(index).ok, name


# --- L2: fit never sees a test split ---------------------------------------


def test_L2_index_carries_its_split(strip_indices):
    """Fit code asserts on `split_name`, so it must be present and truthful on
    every index and on every sample within it."""
    for name, index in strip_indices.items():
        assert index.split_name == name
        assert {s.split for s in index} == {name}


def test_L2_fit_helper_rejects_a_test_split(strip_indices):
    def fit(index):
        if index.split_name not in (*index.layout.fit_splits, "validation"):
            raise ValueError(f"refusing to fit on {index.split_name!r} (protocol rule L2)")
        return len(index)

    assert fit(strip_indices["train"]) == len(strip_indices["train"])
    with pytest.raises(ValueError, match="rule L2"):
        fit(strip_indices["test_public"])


# --- L3: normalization statistics come from validation only ----------------


def test_L3_normalizer_refuses_to_fit_on_test(rng):
    normalizer = ScoreNormalizer()
    with pytest.raises(ValueError, match="rule L3"):
        normalizer.fit([rng.random((8, 8))], split_name="test_public")


def test_L3_normalizer_refuses_to_refit(rng):
    normalizer = ScoreNormalizer().fit([rng.random((8, 8))], split_name="validation")
    with pytest.raises(RuntimeError, match="already fitted"):
        normalizer.fit([rng.random((8, 8))], split_name="validation")


def test_L3_bounds_are_independent_of_the_test_subset(rng):
    """The decisive check: two different test subsets must be scaled by the same
    frozen constants. If the normalizer were refitted per subset, an image's
    score would depend on which other images it was evaluated alongside."""
    normalizer = ScoreNormalizer().fit(
        [rng.random((16, 16)) for _ in range(5)], split_name="validation"
    )
    bounds_before = normalizer.bounds

    subset_a = [rng.random((16, 16)) * 3.0 for _ in range(4)]
    subset_b = [rng.random((16, 16)) * 0.1 for _ in range(4)]
    normalizer.transform(subset_a)
    normalizer.transform(subset_b)

    assert normalizer.bounds == bounds_before
    assert normalizer.source_split == "validation"


def test_L3_out_of_range_scores_are_not_clipped(rng):
    """A test score above the validation maximum is the signal, not an error."""
    normalizer = ScoreNormalizer().fit([np.zeros((4, 4)), np.ones((4, 4))], split_name="validation")
    out = normalizer.transform(np.full((4, 4), 5.0))
    assert out.max() > 1.0


# --- L4: threshold provenance ----------------------------------------------


def test_L4_threshold_from_test_must_be_flagged_oracle():
    with pytest.raises(ValueError, match="rule L4"):
        Threshold(value=0.5, method="percentile", source_split="test_public")


def test_L4_validation_threshold_cannot_claim_oracle():
    with pytest.raises(ValueError, match="not an oracle"):
        Threshold(value=0.5, method="percentile", source_split="validation", oracle=True)


def test_L4_percentile_threshold_records_its_source(rng):
    threshold = from_validation_percentile(rng.random(50), percentile=99.0)
    assert threshold.source_split == "validation"
    assert threshold.oracle is False
    assert threshold.params["percentile"] == 99.0
    assert "(oracle)" not in threshold.label


def test_L4_refuses_too_few_validation_scores():
    with pytest.raises(ValueError, match="at least 4"):
        from_validation_percentile(np.array([0.1, 0.2]))


# --- L6: near duplicates ---------------------------------------------------


def test_L6_phash_identifies_an_identical_image(synthetic_root):
    image = next((synthetic_root / "synth_grain" / "train" / "good").iterdir())
    assert hamming(phash(image), phash(image)) == 0


def test_L6_calibration_reports_the_within_split_distribution(strip_indices):
    """The threshold has to be calibrated on within-train pairs before the
    cross-split check means anything: images shot on a fixed rig are similar by
    design, so an uncalibrated threshold flags the whole dataset."""
    stats = calibrate_phash_threshold(strip_indices["train"])
    assert stats["n_pairs"] > 0
    assert 0 <= stats["min"] <= stats["median"] <= stats["max"] <= 64


def test_L6_cross_split_check_runs(strip_indices):
    report = check_near_duplicates(strip_indices, max_distance=0)
    assert report.rule == "L6-near-duplicates"
    assert "comparisons" in report.detail


# --- L7: private splits carry no ground truth ------------------------------


def test_L7_layout_declares_private_splits_unlabeled():
    from inspector.data import MVTEC_AD2

    for split in ("test_private", "test_private_mixed"):
        assert split in MVTEC_AD2.unlabeled_splits
        assert not MVTEC_AD2.masks_available(split)


def test_L7_unlabeled_split_yields_no_masks(tmp_path):
    """AD 2's private splits hold images directly, with no class subfolders and
    no ground truth, so discovery must take a different branch."""
    from PIL import Image

    from inspector.data.integrity import check_private_split_unlabeled

    private = tmp_path / "cat" / "test_private"
    private.mkdir(parents=True)
    for name in ("a.png", "b.png"):
        Image.new("RGB", (8, 8), (12, 12, 12)).save(private / name)

    index = discover_split(tmp_path, "cat", "test_private", layout="mvtec_ad2")
    assert len(index) == 2
    assert all(s.mask_path is None for s in index)
    assert check_private_split_unlabeled(index).ok


# --- structural checks -----------------------------------------------------


def test_masks_present_for_every_anomalous_sample(strip_indices):
    assert check_masks_present(strip_indices["test_public"]).ok


def test_masks_are_binary(strip_indices):
    """A mask that was ever bilinearly resized would have intermediate values,
    and every pixel metric computed against it would be silently wrong."""
    assert check_mask_binary(strip_indices["test_public"]).ok


def test_missing_mask_is_an_error_not_an_empty_mask(synthetic_root, tmp_path):
    """Treating a missing mask as empty would quietly depress every pixel
    metric, so discovery must refuse to guess."""
    src = tmp_path / "broken"
    shutil.copytree(synthetic_root / "synth_strip", src / "synth_strip")
    gt = src / "synth_strip" / "test_public" / "ground_truth" / "pit"
    next(gt.iterdir()).unlink()

    with pytest.raises(FileNotFoundError, match="no ground-truth mask"):
        discover_split(src, "synth_strip", "test_public", layout=SYNTHETIC)
