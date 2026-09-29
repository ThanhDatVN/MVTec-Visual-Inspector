"""PatchCore regression tests for the defects found in review (docs/13).

The scoring tests use tiny CPU tensors and need no backbone, so they run in
seconds and pin down the *geometry* of each rule exactly. The fit tests use
ResNet18 on synthetic fixtures and are marked `needs_torch`.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inspector.models.patchcore import knn_search, score_patches  # noqa: E402


def t(values) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float32))


# --- F02: the neighbour count must change the score ---------------------------


@pytest.fixture
def line_geometry():
    """One query at the origin; bank entries at distances 1, 2, 3 and 10."""
    queries = t([[0.0, 0.0]])
    bank = t([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [10.0, 0.0]])
    return queries, bank


def test_nearest_reduction_uses_the_first_neighbour(line_geometry):
    queries, bank = line_geometry
    out = score_patches(queries, bank, patch_reduction="nearest", reweight=False)
    assert out.patch_scores.item() == pytest.approx(1.0)


def test_mean_reduction_averages_k_neighbours(line_geometry):
    queries, bank = line_geometry
    out = score_patches(queries, bank, patch_neighbors=3, patch_reduction="mean", reweight=False)
    assert out.patch_scores.item() == pytest.approx(2.0)


def test_kth_reduction_takes_the_kth_distance(line_geometry):
    queries, bank = line_geometry
    out = score_patches(queries, bank, patch_neighbors=3, patch_reduction="kth", reweight=False)
    assert out.patch_scores.item() == pytest.approx(3.0)


def test_k_sweep_changes_scores_under_a_non_nearest_reduction(line_geometry):
    """The review's diagnostic: k = 1, 2, 3 gave identical scores because the
    code always read the first column. Under `mean` they must differ."""
    queries, bank = line_geometry
    scores = [
        score_patches(queries, bank, patch_neighbors=k, patch_reduction="mean", reweight=False)
        .patch_scores.item()
        for k in (1, 2, 3)
    ]
    assert len(set(np.round(scores, 6))) == 3


def test_nearest_reduction_ignores_k_by_design(line_geometry):
    """Documented, not accidental: a k sweep must pair with mean/kth to test
    anything, and this test makes the inert combination explicit."""
    queries, bank = line_geometry
    a = score_patches(queries, bank, patch_neighbors=1, reweight=False).patch_scores.item()
    b = score_patches(queries, bank, patch_neighbors=9, reweight=False).patch_scores.item()
    assert a == b


def test_unknown_reduction_is_rejected(line_geometry):
    queries, bank = line_geometry
    with pytest.raises(ValueError, match="patch_reduction"):
        score_patches(queries, bank, patch_reduction="median")


# --- F03: re-weighting must follow the reference mechanism --------------------


def anomalib_reference(queries: np.ndarray, bank: np.ndarray, n_support: int) -> float:
    """A direct NumPy transcription of anomalib's `compute_anomaly_score`.

    Distances are taken from the *query* peak patch to the support set around
    its nearest bank entry — not among bank entries.
    """
    d = np.linalg.norm(queries[:, None, :] - bank[None, :, :], axis=2)
    patch_scores = d.min(axis=1)
    peak = int(np.argmax(patch_scores))
    nn = int(np.argmin(d[peak]))
    support = np.argsort(np.linalg.norm(bank - bank[nn], axis=1), kind="stable")[:n_support]
    q_to_s = np.linalg.norm(bank[support] - queries[peak], axis=1)
    softmax = np.exp(q_to_s - q_to_s.max())
    softmax /= softmax.sum()
    return float((1.0 - softmax[0]) * patch_scores[peak])


def old_rule(queries: np.ndarray, bank: np.ndarray, n_support: int) -> float:
    """The pre-review rule, weighting by bank-to-bank distances."""
    d = np.linalg.norm(queries[:, None, :] - bank[None, :, :], axis=2)
    patch_scores = d.min(axis=1)
    peak = int(np.argmax(patch_scores))
    nn = int(np.argmin(d[peak]))
    local = np.sort(np.linalg.norm(bank - bank[nn], axis=1))[:n_support]
    softmax = np.exp(local - local.max())
    softmax /= softmax.sum()
    return float((1.0 - softmax[0]) * patch_scores[peak])


@pytest.fixture
def reweight_geometry():
    rng = np.random.default_rng(11)
    bank = rng.normal(size=(40, 6)).astype(np.float32)
    queries = rng.normal(size=(25, 6)).astype(np.float32)
    queries[7] += 4.0  # one clearly anomalous patch
    return queries, bank


def test_reweighting_matches_the_reference_formula(reweight_geometry):
    queries, bank = reweight_geometry
    ours = score_patches(t(queries), t(bank), reweight=True, reweight_neighbors=9).image_score
    assert ours == pytest.approx(anomalib_reference(queries, bank, 9), rel=1e-5)


def test_reference_and_old_rule_genuinely_differ_here(reweight_geometry):
    """Guards the test above against a geometry where both rules coincide and
    the comparison would prove nothing."""
    queries, bank = reweight_geometry
    assert abs(anomalib_reference(queries, bank, 9) - old_rule(queries, bank, 9)) > 1e-3


def test_one_support_neighbour_disables_reweighting(reweight_geometry):
    queries, bank = reweight_geometry
    out = score_patches(t(queries), t(bank), reweight=True, reweight_neighbors=1)
    assert out.image_score == pytest.approx(float(out.patch_scores.max()))


def test_reweighting_leaves_the_anomaly_map_unchanged(reweight_geometry):
    """Re-weighting is an image-score rule. If it moved the map, localization
    metrics would change for a reason that has nothing to do with localization."""
    queries, bank = reweight_geometry
    a = score_patches(t(queries), t(bank), reweight=True).patch_scores
    b = score_patches(t(queries), t(bank), reweight=False).patch_scores
    assert torch.equal(a, b)


# --- F07: chunked exact search must equal the full matrix ---------------------


def test_chunked_search_equals_full_matrix_search():
    rng = np.random.default_rng(3)
    queries, bank = t(rng.normal(size=(1000, 16))), t(rng.normal(size=(300, 16)))
    d_small, i_small = knn_search(queries, bank, 5, chunk=37)
    d_full, i_full = knn_search(queries, bank, 5, chunk=100_000)
    assert torch.allclose(d_small, d_full)
    assert torch.equal(i_small, i_full)


def test_k_larger_than_the_bank_is_clamped():
    rng = np.random.default_rng(4)
    d, _ = knn_search(t(rng.normal(size=(5, 4))), t(rng.normal(size=(3, 4))), 10)
    assert d.shape == (5, 3)


# --- fitting, bank provenance, and save/load parity (needs a backbone) --------


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    from inspector.data import SYNTHETIC, discover_category
    from inspector.data.transforms import ImageTransform
    from inspector.fixtures import FixtureSpec, generate
    from inspector.models.patchcore import PatchCore

    root = generate(
        tmp_path_factory.mktemp("pc"),
        spec=FixtureSpec(n_train=6, n_validation=4, n_test_good=2, n_test_per_defect=2),
        seed=0,
        overwrite=True,
    )
    indices = discover_category(root, "synth_grain", layout=SYNTHETIC)
    kwargs = {
        "backbone": "resnet18", "candidate_fraction": 0.5, "bank_ratio": 0.2,
        "batch_size": 2, "device": "cpu", "seed": 0,
    }
    model = PatchCore(ImageTransform(long_side=64), **kwargs).fit(indices["train"])
    return model, indices, kwargs


@pytest.mark.needs_torch
@pytest.mark.slow
def test_fit_logs_every_bank_size(fitted):
    """F08: total, candidate, requested and effective sizes are all recorded, so
    a sweep whose settings collapse to one bank is visible in the results."""
    model, _, _ = fitted
    stats = model.fit_extra()
    for key in ("patches_total", "candidate_count", "bank_requested", "bank_effective", "bank_capped"):
        assert key in stats
    assert stats["candidate_count"] <= stats["patches_total"]
    assert stats["bank_effective"] == min(stats["bank_requested"], stats["candidate_count"])


@pytest.mark.needs_torch
@pytest.mark.slow
def test_capped_bank_is_flagged(fitted):
    from inspector.data.transforms import ImageTransform
    from inspector.models.patchcore import PatchCore

    _, indices, kwargs = fitted
    model = PatchCore(
        ImageTransform(long_side=64), **{**kwargs, "candidate_fraction": 0.1, "bank_ratio": 0.5}
    ).fit(indices["train"])
    assert model.fit_extra()["bank_capped"] is True


@pytest.mark.needs_torch
@pytest.mark.slow
def test_bank_sources_point_inside_the_training_grid(fitted):
    model, indices, _ = fitted
    sources = model.bank_sources
    assert sources is not None and sources.shape == (model.memory_bank.shape[0], 3)
    assert sources[:, 0].max() < len(indices["train"])
    assert sources[:, 1].max() < model.grid[0] and sources[:, 2].max() < model.grid[1]


@pytest.mark.needs_torch
@pytest.mark.slow
def test_save_load_reproduces_predictions(fitted):
    from inspector.data.transforms import ImageTransform
    from inspector.models.patchcore import PatchCore

    model, indices, kwargs = fitted
    restored = PatchCore(ImageTransform(long_side=64), **kwargs).load_state_dict(model.state_dict())
    sample = indices["test_public"][0]
    a, b = model.predict_sample(sample), restored.predict_sample(sample)
    assert a.score == pytest.approx(b.score, rel=1e-6)
    assert np.allclose(a.anomaly_map, b.anomaly_map)


@pytest.mark.needs_torch
@pytest.mark.slow
def test_load_refuses_mismatched_hyperparameters(fitted):
    from inspector.data.transforms import ImageTransform
    from inspector.models.patchcore import PatchCore

    model, _, kwargs = fitted
    other = PatchCore(ImageTransform(long_side=64), **{**kwargs, "reweight_neighbors": 3})
    with pytest.raises(ValueError, match="hyperparameters differ"):
        other.load_state_dict(model.state_dict())


def test_exactly_one_bank_budget_is_required():
    from inspector.models.patchcore import PatchCore

    with pytest.raises(ValueError, match="exactly one"):
        PatchCore(bank_ratio=0.01, bank_size=100)
    with pytest.raises(ValueError, match="exactly one"):
        PatchCore(bank_ratio=None, bank_size=None)
