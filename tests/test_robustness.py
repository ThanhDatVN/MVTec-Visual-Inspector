"""Corruption suite tests (docs/05 §6).

Corruption code that is subtly wrong invalidates the whole robustness section,
and its bugs are invisible: a corrupted image still looks like a corrupted
image. These tests are the only thing standing between "we measured robustness"
and "we measured a bug".
"""

from __future__ import annotations

import numpy as np
import pytest

from inspector.robustness import (
    CORRUPTIONS,
    MAIN_SUITE,
    SEVERITIES,
    apply_corruption,
    severity_grid,
)

ALL_NAMES = sorted(CORRUPTIONS)


@pytest.fixture
def scene():
    """A textured image with a small bright defect and its exact mask."""
    rng = np.random.default_rng(0)
    image = np.full((96, 96, 3), 70, dtype=np.uint8)
    image = np.clip(image + rng.normal(0, 6, image.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((96, 96), dtype=bool)
    mask[40:48, 40:48] = True
    image[mask] = 210
    return image, mask


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(float) - b.astype(float)) ** 2)))


# --- invariants -------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_NAMES)
def test_severity_zero_is_the_identity(scene, name):
    image, mask = scene
    out_image, out_mask = apply_corruption(image, mask, name, 0)
    assert np.array_equal(out_image, image)
    assert np.array_equal(out_mask, mask)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_output_dtype_and_shape_are_preserved(scene, name):
    image, mask = scene
    out_image, out_mask = apply_corruption(image, mask, name, 3)
    assert out_image.dtype == np.uint8
    assert out_image.shape == image.shape
    assert out_image.min() >= 0 and out_image.max() <= 255
    assert out_mask is not None and out_mask.dtype == bool


@pytest.mark.parametrize("name", ALL_NAMES)
def test_deterministic_for_a_seed(scene, name):
    image, mask = scene
    a, _ = apply_corruption(image, mask, name, 4, seed=7)
    b, _ = apply_corruption(image, mask, name, 4, seed=7)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_severity_is_monotone(scene, name):
    """Distance from the clean image must increase with severity. A scale that
    is not monotone makes every degradation curve uninterpretable."""
    image, mask = scene
    distances = [
        rmse(apply_corruption(image, mask, name, s, seed=0)[0], image) for s in SEVERITIES
    ]
    assert distances[-1] > distances[0], f"{name}: severity 5 is no further than severity 1"
    # Allow small non-monotonicity from JPEG's quantisation, but not a reversal.
    assert distances[-1] >= max(distances) * 0.9


# --- mask handling ----------------------------------------------------------


@pytest.mark.parametrize("name", [n for n in ALL_NAMES if not CORRUPTIONS[n].geometric])
def test_photometric_corruptions_leave_the_mask_untouched(scene, name):
    image, mask = scene
    _, out_mask = apply_corruption(image, mask, name, 5)
    assert np.array_equal(out_mask, mask), f"{name} altered a mask it should not touch"


def test_geometric_corruption_moves_the_mask_with_the_image():
    """The defect and its label must move together, or every pixel metric is
    scored against a ground truth that is somewhere else."""
    from scipy import ndimage

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=bool)
    mask[60:68, 60:68] = True
    image[mask] = 255

    out_image, out_mask = apply_corruption(image, mask, "translate", 5)

    image_centroid = ndimage.center_of_mass(out_image[..., 0] > 128)
    mask_centroid = ndimage.center_of_mass(out_mask)
    assert abs(image_centroid[0] - mask_centroid[0]) < 1.5
    assert abs(image_centroid[1] - mask_centroid[1]) < 1.5


def test_translated_mask_is_actually_displaced():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=bool)
    mask[60:68, 60:68] = True

    _, out_mask = apply_corruption(image, mask, "translate", 5)
    assert not np.array_equal(out_mask, mask)
    assert out_mask.sum() > 0


def test_a_normal_image_gets_no_mask_back(scene):
    image, _ = scene
    out_image, out_mask = apply_corruption(image, None, "gaussian_blur", 3)
    assert out_mask is None
    assert out_image.shape == image.shape


# --- the guard that matters most --------------------------------------------


@pytest.mark.parametrize("name", MAIN_SUITE)
def test_severity_one_does_not_erase_the_defect(scene, name):
    """A calibration error that quietly deletes the thing being detected would
    show up as excellent 'robustness' — the model degrades smoothly because
    there is nothing left to detect. Severity 1 must keep most of the defect's
    contrast against its surroundings."""
    image, mask = scene
    clean_contrast = abs(image[mask].mean() - image[~mask].mean())

    out_image, _ = apply_corruption(image, mask, name, 1)
    contrast = abs(out_image[mask].mean() - out_image[~mask].mean())

    assert contrast > 0.5 * clean_contrast, (
        f"{name} at severity 1 removed {1 - contrast / clean_contrast:.0%} of the "
        "defect contrast; the severity scale is miscalibrated"
    )


def test_severity_scales_with_the_defect_size(scene):
    """Severity is defined relative to the median defect diameter, so a category
    with larger defects must get a proportionally larger kernel. A fixed kernel
    would mean something different on every category."""
    image, mask = scene
    small, _ = apply_corruption(image, mask, "gaussian_blur", 3, median_defect_diameter=4.0)
    large, _ = apply_corruption(image, mask, "gaussian_blur", 3, median_defect_diameter=16.0)
    assert rmse(large, image) > rmse(small, image)


# --- registry ---------------------------------------------------------------


def test_translation_is_excluded_from_the_main_suite():
    """On a fixed rig a geometric shift is out of distribution by design;
    averaging it into a robustness score would misrepresent deployment."""
    assert "translate" not in MAIN_SUITE
    assert not CORRUPTIONS["translate"].in_main_suite


def test_main_suite_spans_the_three_physical_families():
    families = {CORRUPTIONS[n].family for n in MAIN_SUITE}
    assert families == {"optical", "illumination", "transport"}


def test_exposure_directions_are_reported_separately():
    """Over- and under-exposure fail differently; averaging them hides which."""
    assert "exposure_up" in MAIN_SUITE
    assert "exposure_down" in MAIN_SUITE


def test_exposure_directions_move_brightness_opposite_ways(scene):
    image, mask = scene
    up, _ = apply_corruption(image, mask, "exposure_up", 4)
    down, _ = apply_corruption(image, mask, "exposure_down", 4)
    assert up.mean() > image.mean() > down.mean()


def test_grid_size_matches_the_plan():
    grid = severity_grid()
    assert len(grid) == len(MAIN_SUITE) * len(SEVERITIES)


def test_unknown_corruption_is_rejected(scene):
    image, mask = scene
    with pytest.raises(ValueError, match="unknown corruption"):
        apply_corruption(image, mask, "cosmic_rays", 3)


def test_invalid_severity_is_rejected(scene):
    image, mask = scene
    with pytest.raises(ValueError, match="severity must be"):
        apply_corruption(image, mask, "gaussian_blur", 9)


def test_non_uint8_input_is_rejected(scene):
    _, mask = scene
    with pytest.raises(ValueError, match="uint8"):
        apply_corruption(np.zeros((8, 8, 3), dtype=np.float32), mask, "gaussian_blur", 1)
