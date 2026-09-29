"""Object region of interest from training normals (E02b)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from inspector.postproc.roi import fit_background, mask_on_grid, object_mask


def scene(rng, *, board=True, speck=None, size=(200, 300)):
    h, w = size
    img = np.clip(rng.normal(60, 6, (h, w, 3)), 0, 255)  # dark grey felt
    if board:
        img[60:140, 90:210] = (30, 90, 200)  # a blue board in the middle
        img[90:110, 130:170] = (60, 60, 60)  # interior patch that matches the felt
    if speck is not None:
        y, x = speck
        img[y:y + 3, x:x + 3] = 250  # a white fibre on the background
    return img.astype(np.uint8)


@pytest.fixture(scope="module")
def background():
    rng = np.random.default_rng(0)
    return fit_background([scene(rng) for _ in range(10)], work_size=150)


def test_region_covers_the_object_and_excludes_background_debris(background):
    img = scene(np.random.default_rng(1), speck=(10, 10))
    m = object_mask(img, background, margin=0.02)
    s = m.shape[0] / img.shape[0]
    assert m[int(100 * s), int(150 * s)], "board centre (felt-coloured interior) must be inside"
    assert m[int(62 * s), int(92 * s)], "board corner must be inside"
    assert not m[int(11 * s), int(11 * s)], "a background speck must be outside"
    assert m.mean() < 0.4


def test_margin_grows_the_region(background):
    img = scene(np.random.default_rng(2))
    small = object_mask(img, background, margin=0.01).mean()
    large = object_mask(img, background, margin=0.06).mean()
    assert large > small


def test_no_foreground_means_no_restriction(background):
    img = scene(np.random.default_rng(3), board=False)
    assert object_mask(img, background).all()


def test_border_assumption_check(background):
    rng = np.random.default_rng(4)
    assert background.border_foreground_share(scene(rng)) < 0.02
    full = scene(rng)
    full[:, :40] = (30, 90, 200)  # object touching the frame edge
    assert background.border_foreground_share(full) > 0.1


def test_mask_on_grid_is_never_empty():
    assert mask_on_grid(np.zeros((50, 60), bool), (5, 6)).all()
    grid = mask_on_grid(np.pad(np.ones((20, 20), bool), 15), (5, 5))
    assert grid[2, 2] and not grid[0, 0]
