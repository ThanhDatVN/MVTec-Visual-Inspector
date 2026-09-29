"""AU-PRO against an algorithmically independent oracle (Gate G2, docs/13 item 5).

Our implementation sweeps *score thresholds* and reads FPR and PRO off each one.
The oracle below is parametrized the other way round, by *false-positive rate*:
for each f on a fine grid it takes the threshold at the (1 - f) quantile of the
negative pixels and averages the per-region overlap there — the definition of
Bergmann et al. (IJCV 2021) written without sharing a line of code with
`inspector.metrics.aupro`. The two agree only if the threshold sweep, the FPR
denominator, the connectivity, the per-region averaging and the interpolation
at the integration limit are all right.

A pinned third-party implementation (anomalib's AUPRO) would be stronger still;
it needs torchmetrics, and remains the open half of this gate.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from inspector.metrics.aupro import PROAccumulator, au_pro


def oracle_au_pro(scores, masks, limit: float, grid: int = 20001) -> float:
    negatives = np.sort(np.concatenate([s[~m] for s, m in zip(scores, masks)]))
    regions = []
    for s, m in zip(scores, masks):
        labels, n = ndimage.label(m, structure=np.ones((3, 3), dtype=int))  # 8-connectivity
        regions += [np.sort(s[labels == k]) for k in range(1, n + 1)]

    fprs = np.linspace(0.0, limit, grid)
    pros = np.empty_like(fprs)
    for i, f in enumerate(fprs):
        # the lowest threshold with at most floor(f * N) negatives strictly above it
        k = int(np.floor(f * negatives.size))
        t = negatives[negatives.size - k - 1] if k < negatives.size else -np.inf
        pros[i] = np.mean([1.0 - np.searchsorted(r, t, side="right") / r.size for r in regions])
    return float(np.trapezoid(pros, fprs) / limit)


def dataset(seed: int, n_images: int = 12, size: int = 96):
    """Smooth maps with a signal of varying strength in regions of varying size."""
    rng = np.random.default_rng(seed)
    scores, masks = [], []
    for i in range(n_images):
        noise = ndimage.gaussian_filter(rng.normal(size=(size, size)), 2.0)
        m = np.zeros((size, size), dtype=bool)
        if i % 3:  # a third of the images are normal
            for _ in range(rng.integers(1, 4)):
                r = int(rng.integers(2, 14))
                y, x = rng.integers(r, size - r, size=2)
                yy, xx = np.ogrid[:size, :size]
                m |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        signal = ndimage.gaussian_filter(m.astype(float), 1.5) * rng.uniform(0.1, 0.6)
        scores.append(noise * 0.25 + signal)
        masks.append(m)
    return scores, masks


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("limit", [0.05, 0.30])
def test_exact_path_matches_the_fpr_parametrized_oracle(seed, limit):
    scores, masks = dataset(seed)
    ours = au_pro(scores, masks, integration_limit=limit, num_thresholds=None, max_negative_samples=None)
    oracle = oracle_au_pro(scores, masks, limit)
    assert ours == pytest.approx(oracle, abs=2e-3)


@pytest.mark.parametrize("seed", [3, 4])
def test_production_path_matches_the_oracle(seed):
    """The evaluator's settings: 512 thresholds and sampled negatives."""
    scores, masks = dataset(seed, n_images=18)
    oracle = oracle_au_pro(scores, masks, 0.05)
    acc = PROAccumulator(negative_rate=0.25, seed=0)
    for s, m in zip(scores, masks):
        acc.add(s, m)
    ours = acc.curves([0.05], num_thresholds=512)[0.05].au_pro
    assert ours == pytest.approx(oracle, abs=0.01)


def test_the_oracle_itself_recovers_the_analytic_cases():
    """Guard the guard: perfect = 1, and an uninformative scorer = L/2."""
    _, masks = dataset(5)
    perfect = [m.astype(float) + 1e-9 * np.random.default_rng(0).random(m.shape) for m in masks]
    assert oracle_au_pro(perfect, masks, 0.05) == pytest.approx(1.0, abs=1e-3)
    rng = np.random.default_rng(6)
    uninformative = [rng.random(m.shape) for m in masks]
    assert oracle_au_pro(uninformative, masks, 0.3) == pytest.approx(0.15, abs=0.01)
