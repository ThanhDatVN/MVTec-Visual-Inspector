"""E00, step C1: agreement with a pinned reference on identical tensors.

The oracle is anomalib v2.3.0's PatchCore scoring and k-center loop, excerpted
verbatim into `tests/reference/` (Apache-2.0). Our scoring must reproduce its
patch scores and its re-weighted image score on the same descriptors and bank;
the coreset must reproduce its greedy order up to one documented difference.
Differences that cannot be matched on tensors — feature pooling, candidate
sampling, projection type — are listed in `PatchCore.reference_differences()`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inspector.models.patchcore import PatchCore, greedy_coreset, score_patches  # noqa: E402

_REF = Path(__file__).resolve().parents[1] / "reference" / "anomalib_patchcore.py"
_spec = importlib.util.spec_from_file_location("anomalib_reference", _REF)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)  # type: ignore[union-attr]


def random_case(seed: int, *, patches: int, bank_size: int, dim: int, clustered: bool):
    g = torch.Generator().manual_seed(seed)
    if clustered:
        # A bank with tight clusters: support sets are then dominated by near-
        # duplicates, the regime where the re-weighting actually matters.
        centres = torch.randn(max(2, bank_size // 8), dim, generator=g) * 3
        bank = centres[torch.randint(0, centres.shape[0], (bank_size,), generator=g)]
        bank = bank + 0.05 * torch.randn(bank_size, dim, generator=g)
    else:
        bank = torch.randn(bank_size, dim, generator=g)
    queries = torch.randn(patches, dim, generator=g)
    queries[torch.randint(0, patches, (1,), generator=g)] += 4.0  # one anomalous patch
    return queries, bank


CASES = [
    {"seed": s, "patches": p, "bank_size": m, "dim": d, "clustered": c}
    for s, (p, m, d, c) in enumerate([
        (64, 50, 16, False),
        (64, 50, 16, True),
        (300, 200, 64, True),
        (1500, 400, 32, False),   # more queries than anomalib's 1024 chunk
        (100, 5, 8, True),        # bank smaller than the support size
    ])
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"P{c['patches']}-M{c['bank_size']}-{'clu' if c['clustered'] else 'iid'}")
def test_patch_scores_match_the_reference(case):
    queries, bank = random_case(**case)
    ref_patch, _ = ref.score_images(bank, queries, batch_size=1)
    ours = score_patches(queries, bank, patch_reduction="nearest", reweight=False)
    # anomalib expands |x-y|^2 = |x|^2 - 2xy + |y|^2; cdist is exact. Tolerance
    # is the float32 cancellation error of the expansion, not a modelling gap.
    torch.testing.assert_close(ours.patch_scores, ref_patch[0], rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"P{c['patches']}-M{c['bank_size']}-{'clu' if c['clustered'] else 'iid'}")
@pytest.mark.parametrize("support", [1, 3, 9])
def test_image_score_matches_the_reference(case, support):
    queries, bank = random_case(**case)
    _, ref_score = ref.score_images(bank, queries, batch_size=1, num_neighbors=support)
    ours = score_patches(queries, bank, reweight=True, reweight_neighbors=support)
    assert ours.image_score == pytest.approx(float(ref_score[0]), rel=1e-4, abs=1e-4)


def test_reweighting_is_not_a_no_op_in_the_agreement_cases():
    """Guard against a vacuous pass: the cases must include ones where the
    re-weighted score differs clearly from the raw maximum."""
    changed = 0
    for case in CASES:
        queries, bank = random_case(**case)
        raw = score_patches(queries, bank, reweight=False).image_score
        weighted = score_patches(queries, bank, reweight=True, reweight_neighbors=9).image_score
        changed += abs(weighted - raw) > 1e-3 * raw
    assert changed >= 3


def test_coreset_follows_the_reference_greedy_order():
    """Same start, same features: anomalib uses the random start as a centre
    but does not return it; we return it. So ours = {start} + reference[:n-1]."""
    g = torch.Generator().manual_seed(7)
    features = torch.randn(400, 12, generator=g)
    start, n = 17, 40

    reference = ref.k_center_greedy(features, n, start)
    ours = greedy_coreset(features.numpy(), n, seed=0, device="cpu")
    first = int(np.random.default_rng(0).integers(0, features.shape[0]))  # our seeded start

    reference_from_ours = ref.k_center_greedy(features, n - 1, first)
    assert set(ours.tolist()) == {first, *reference_from_ours}
    assert len(set(reference)) == n  # the oracle itself selected distinct points


def test_the_difference_ledger_records_what_these_tests_found():
    entries = PatchCore.reference_differences()
    assert any(e.startswith("Coreset start:") and "does not return it" in e for e in entries)
    assert any(e.startswith("Scoring:") and "091ca6a" in e for e in entries)
