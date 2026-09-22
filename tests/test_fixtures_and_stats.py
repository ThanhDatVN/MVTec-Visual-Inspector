"""The fixture generator, and the statistics used for model comparison."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from inspector.fixtures import CATEGORY_SIZES, DEFECT_TYPES, FixtureSpec, generate
from inspector.stats import bootstrap_ci, compare_models, holm_bonferroni, paired_wilcoxon

# --- fixtures ---------------------------------------------------------------


def test_generates_every_split_and_defect_class(synthetic_root, fixture_spec):
    for category, defects in DEFECT_TYPES.items():
        root = synthetic_root / category
        assert len(list((root / "train" / "good").glob("*.png"))) == fixture_spec.n_train
        assert len(list((root / "validation" / "good").glob("*.png"))) == fixture_spec.n_validation
        for defect in defects:
            images = list((root / "test_public" / defect).glob("*.png"))
            masks = list((root / "test_public" / "ground_truth" / defect).glob("*.png"))
            assert len(images) == fixture_spec.n_test_per_defect
            assert len(masks) == len(images)


def test_generation_is_byte_reproducible(tmp_path):
    spec = FixtureSpec(n_train=2, n_validation=2, n_test_good=2, n_test_per_defect=1)
    a = generate(tmp_path / "a", spec=spec, seed=0, overwrite=True)
    b = generate(tmp_path / "b", spec=spec, seed=0, overwrite=True)

    for pa in sorted(a.rglob("*.png")):
        pb = b / pa.relative_to(a)
        assert pa.read_bytes() == pb.read_bytes(), pa


def test_different_seeds_produce_different_pixels(tmp_path):
    spec = FixtureSpec(n_train=2, n_validation=2, n_test_good=2, n_test_per_defect=1)
    a = generate(tmp_path / "a", spec=spec, seed=0, overwrite=True)
    b = generate(tmp_path / "b", spec=spec, seed=1, overwrite=True)
    differing = sum(
        (a / rel).read_bytes() != (b / rel).read_bytes()
        for rel in (p.relative_to(a) for p in a.rglob("*.png"))
    )
    assert differing > 0


def test_masks_are_strictly_binary(synthetic_root):
    for mask_path in synthetic_root.rglob("ground_truth/**/*.png"):
        values = set(np.unique(np.asarray(Image.open(mask_path).convert("L"))).tolist())
        assert values <= {0, 255}, f"{mask_path}: {values}"


def test_every_mask_marks_at_least_one_pixel(synthetic_root):
    for mask_path in synthetic_root.rglob("ground_truth/**/*.png"):
        assert np.asarray(Image.open(mask_path)).sum() > 0, mask_path


def test_defects_are_actually_visible(synthetic_root):
    """A fixture whose defect is invisible would make every model look broken.

    The comparison is against a *local* ring around the defect, not against the
    whole image. Comparing to the whole image is the wrong reference: in
    `synth_grain` most of the frame is dark background outside the part, which
    drags the global mean toward the defect's value and hides a perfectly
    visible crack.
    """
    from scipy import ndimage

    for category, defects in DEFECT_TYPES.items():
        for defect in defects:
            img_dir = synthetic_root / category / "test_public" / defect
            mask_dir = synthetic_root / category / "test_public" / "ground_truth" / defect
            for image_path in sorted(img_dir.glob("*.png")):
                mask = np.asarray(Image.open(mask_dir / f"{image_path.stem}_mask.png")) > 0
                gray = np.asarray(Image.open(image_path).convert("L"), dtype=float)

                ring = ndimage.binary_dilation(mask, iterations=4) & ~mask
                contrast = abs(gray[mask].mean() - gray[ring].mean())
                assert contrast > 8, (
                    f"{category}/{defect}/{image_path.name}: local contrast {contrast:.1f} "
                    "is too low to be detectable"
                )


def test_strip_category_keeps_its_extreme_aspect_ratio():
    """synth_strip stands in for sheet_metal: a blind square resize must be
    visibly destructive, which only holds if the aspect ratio is extreme."""
    w, h = CATEGORY_SIZES["synth_strip"]
    assert w / h >= 4.0


def test_defects_are_small_relative_to_the_image(synthetic_root):
    """The whole resolution axis of the study depends on defects being tiny."""
    mask_dir = synthetic_root / "synth_strip" / "test_public" / "ground_truth" / "pit"
    for mask_path in mask_dir.glob("*.png"):
        fraction = (np.asarray(Image.open(mask_path)) > 0).mean()
        assert fraction < 0.01, f"{mask_path}: defect covers {fraction:.2%} of the image"


# --- statistics -------------------------------------------------------------


def test_bootstrap_ci_brackets_the_point_estimate(rng):
    values = rng.normal(0.5, 0.1, size=100)
    point, low, high = bootstrap_ci(values, seed=0)
    assert low < point < high
    assert point == pytest.approx(values.mean())


def test_bootstrap_ci_is_deterministic_for_a_seed(rng):
    values = rng.normal(size=50)
    assert bootstrap_ci(values, seed=0) == bootstrap_ci(values, seed=0)


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(0)
    _, lo_small, hi_small = bootstrap_ci(rng.normal(size=20), seed=0)
    _, lo_big, hi_big = bootstrap_ci(rng.normal(size=2000), seed=0)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_paired_test_detects_a_consistent_improvement(rng):
    baseline = rng.random(40)
    improved = baseline + 0.05  # uniformly better on every image
    result = paired_wilcoxon(improved, baseline)
    assert result.significant
    assert result.median_difference == pytest.approx(0.05, abs=1e-9)


def test_paired_test_finds_nothing_in_noise(rng):
    a, b = rng.normal(size=60), rng.normal(size=60)
    assert not paired_wilcoxon(a, b).significant


def test_identical_inputs_are_not_significant(rng):
    values = rng.random(20)
    result = paired_wilcoxon(values, values.copy())
    assert result.p_value == 1.0
    assert not result.significant


def test_paired_test_refuses_an_underpowered_sample(rng):
    with pytest.raises(ValueError, match="no useful power"):
        paired_wilcoxon(rng.random(4), rng.random(4))


def test_paired_test_requires_equal_lengths(rng):
    with pytest.raises(ValueError, match="equal lengths"):
        paired_wilcoxon(rng.random(10), rng.random(9))


def test_holm_bonferroni_is_more_conservative_than_raw_p():
    raw = [0.01, 0.02, 0.04]
    adjusted = holm_bonferroni(raw)
    assert all(adj >= p for (adj, _), p in zip(adjusted, raw))
    assert [adj for adj, _ in adjusted] == sorted(adj for adj, _ in adjusted)


def test_holm_bonferroni_kills_the_lone_lucky_result():
    """Running 30 ablations and reporting the one with p < 0.05 is not a result;
    it is the expected number of false positives (DR-2)."""
    p_values = [0.04] + [0.8] * 29
    adjusted = holm_bonferroni(p_values)
    assert not adjusted[0][1]


def test_holm_bonferroni_handles_an_empty_family():
    assert holm_bonferroni([]) == []


def test_compare_models_reports_the_family_size(rng):
    baseline = rng.random(40)
    result = compare_models(baseline + 0.05, baseline, family_size=10)
    assert "family=10" in result.test
    assert result.p_value_corrected >= result.p_value
    assert "median" in str(result)
