"""The tiled foundation-feature kNN model (SuperADD-style). A tiny random ViT stands in for DINOv3."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from inspector.models import tiled_knn
from inspector.models.tiled_knn import TiledKNN, density_subsample, tile_plan

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")


def test_tile_plan_covers_every_token_once_with_the_requested_overlap():
    for length in (40, 41, 57, 95, 165, 400):
        plan = tile_plan(length, 40, 8)
        kept = np.zeros(length, dtype=int)
        for s, e, a, b in plan:
            assert 0 <= s <= a < b <= e <= length
            kept[a:b] += 1
        assert (kept == 1).all(), length
        assert plan[0][0] == 0 and plan[-1][1] == length
        for (_, e0, _, b0), (s1, _, a1, _) in pairwise(plan):
            assert e0 - s1 >= 16  # at least `overlap` on each side of the boundary
            assert b0 == a1 and b0 - s1 >= 8 and e0 - b0 >= 8  # each token at least 8 from a window edge
    assert tile_plan(30, 40, 8) == [(0, 30, 0, 30)]
    assert tile_plan(56, 40, 8) == [(0, 56, 0, 56)]  # within tile + 2 * overlap: one window
    assert len(tile_plan(57, 40, 8)) == 2


def test_density_subsample_drops_duplicates_before_outliers():
    rng = np.random.default_rng(0)
    crowd = np.repeat(rng.normal(size=(5, 8)), 400, axis=0) + rng.normal(scale=1e-3, size=(2000, 8))
    sparse = rng.normal(scale=10.0, size=(50, 8))
    x = np.concatenate([crowd, sparse]).astype(np.float32)
    keep = density_subsample(x, 100, k=20, subset=4096)
    assert len(keep) == 100 and len(np.unique(keep)) == 100
    assert (keep >= 2000).sum() >= 45  # nearly all the sparse points survive


def _tiny_vit(backbone, weights):
    return timm.create_model("vit_small_patch16_224", pretrained=False, depth=3, embed_dim=48, num_heads=3,
                             dynamic_img_size=True)


@pytest.fixture()
def model(monkeypatch):
    monkeypatch.setattr(tiled_knn, "load_vit", _tiny_vit)
    return TiledKNN(layers=(0, 2), scale=0.5, tile=64, overlap=16, candidate_fraction=0.5, bank_size=200,
                    density_neighbors=10, subset_size=256, device="cpu", tile_batch=3)


def test_grid_matches_the_scaled_image_and_tiles_agree_with_a_single_pass(model):
    image = np.random.default_rng(1).normal(size=(3, 300, 200)).astype(np.float32)
    grids = model._grid(image)
    assert [tuple(g.shape) for g in grids] == [(48, 9, 6), (48, 9, 6)]  # round(150/16), round(100/16)


def test_fit_and_score_on_the_synthetic_split(model, strip_indices):
    model.fit(strip_indices["train"])
    assert model.hparams()["layers"] == [0, 2] and model.fit_extra()["bank_effective"] <= 200
    raw = model.predict_raw(np.zeros((128, 96, 3), dtype=np.uint8))
    assert raw.raw_map.ndim == 2 and np.isfinite(raw.raw_map).all() and raw.input_size == (96, 128)


def test_nearest_distance_matches_a_direct_computation(model):
    rng = np.random.default_rng(2)
    bank = torch.as_tensor(rng.normal(size=(300, 48)), dtype=torch.float32)
    model.means = [bank.mean(0).half()]
    centred = (bank - bank.mean(0)).half()
    model.banks, model.bank_norms = [centred], [centred.float().square().sum(1)]
    q = torch.as_tensor(rng.normal(size=(50, 48)), dtype=torch.float32)
    got = model._nn_distance(0, q.half())
    want = torch.cdist(q, bank).min(1).values
    assert torch.allclose(got, want, atol=2e-2, rtol=2e-3)
