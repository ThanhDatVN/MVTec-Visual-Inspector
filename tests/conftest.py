"""Shared test fixtures.

The synthetic dataset is generated fresh into a tmp dir for each session rather
than committed. That keeps image bytes out of the repository entirely (the
license guard then has nothing to make an exception for) and means the generator
is exercised by every test run instead of rotting as a one-off script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inspector.data import SYNTHETIC, discover_category
from inspector.fixtures import FixtureSpec, generate


@pytest.fixture(scope="session")
def fixture_spec() -> FixtureSpec:
    return FixtureSpec(n_train=12, n_validation=5, n_test_good=5, n_test_per_defect=3)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory, fixture_spec) -> Path:
    root = tmp_path_factory.mktemp("synthetic_dataset")
    return generate(root, spec=fixture_spec, seed=0, overwrite=True)


@pytest.fixture(scope="session")
def strip_indices(synthetic_root):
    """All splits of the high-aspect-ratio, tiny-defect category."""
    return discover_category(synthetic_root, "synth_strip", layout=SYNTHETIC)


@pytest.fixture(scope="session")
def grain_indices(synthetic_root):
    """All splits of the high-normal-variance category."""
    return discover_category(synthetic_root, "synth_grain", layout=SYNTHETIC)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def toy_segmentation():
    """A small, fully-specified localization problem.

    Six anomalous images with two regions each plus two normal images. Small
    enough to reason about by hand, which is what makes the analytic assertions
    in tests/metrics meaningful.
    """
    h = w = 48
    masks, perfect, inverted = [], [], []
    for _ in range(6):
        m = np.zeros((h, w), dtype=bool)
        m[8:14, 8:24] = True
        m[30:34, 34:40] = True
        masks.append(m)
        perfect.append(m.astype(np.float64))
        inverted.append(1.0 - m.astype(np.float64))
    for _ in range(2):
        masks.append(np.zeros((h, w), dtype=bool))
        perfect.append(np.zeros((h, w)))
        inverted.append(np.ones((h, w)))
    return {"masks": masks, "perfect": perfect, "inverted": inverted}
