"""Aggregation and calibration studies on stored predictions (docs/13, E02/E03)."""

from __future__ import annotations

import numpy as np
import pytest

from inspector.analysis import (
    AGGREGATIONS,
    aggregate,
    aggregation_study,
    calibration_study,
    exceedance_p_value,
    mad_threshold,
    normal_shift,
    rank_threshold,
    sigma_threshold,
    summarize_calibration,
)
from inspector.evaluate import SplitPredictions


def make_preds(scores, labels, maps=None, split="test") -> SplitPredictions:
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    maps = maps if maps is not None else [np.full((4, 4), s, dtype=np.float32) for s in scores]
    return SplitPredictions(
        split=split, ids=[f"i{k}" for k in range(n)], labels=np.asarray(labels, dtype=int),
        defect_types=["x"] * n, scores=scores, raw_maps=maps,
        input_sizes=[(4, 4)] * n, native_sizes=[(4, 4)] * n, mask_paths=[None] * n,
    )


# --- aggregation rules --------------------------------------------------------


def test_top_fraction_is_between_mean_and_max():
    rng = np.random.default_rng(0)
    m = rng.random((40, 30)).astype(np.float32)
    top = AGGREGATIONS["top1%"](m)
    assert AGGREGATIONS["mean"](m) <= top <= AGGREGATIONS["max"](m)


def test_top_fraction_keeps_at_least_one_element():
    m = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert AGGREGATIONS["top0.1%"](m) == 11.0


def test_max_catches_a_small_defect_that_mean_dilutes():
    """The mechanism E02 tests: one hot patch moves the max, barely the mean."""
    base = np.zeros((40, 30), dtype=np.float32)
    hot = base.copy()
    hot[5, 5] = 10.0
    assert AGGREGATIONS["max"](hot) - AGGREGATIONS["max"](base) == 10.0
    assert AGGREGATIONS["mean"](hot) - AGGREGATIONS["mean"](base) < 0.01


def test_stored_rule_returns_the_model_score():
    p = make_preds([0.3, 0.9], [0, 1])
    assert np.array_equal(aggregate(p, "stored"), p.scores)


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate(make_preds([0.1], [0]), "median")


def test_aggregation_study_reports_every_rule_and_a_paired_interval():
    rng = np.random.default_rng(1)
    labels = np.array([0] * 20 + [1] * 20)
    runs = []
    for _ in range(2):
        test_maps = [rng.random((6, 6)).astype(np.float32) + (0.5 if y else 0.0) for y in labels]
        val_maps = [rng.random((6, 6)).astype(np.float32) for _ in range(30)]
        test = make_preds([m.mean() for m in test_maps], labels, test_maps)
        val = make_preds([m.mean() for m in val_maps], [0] * 30, val_maps, split="validation")
        runs.append((val, test))
    out = aggregation_study(runs, rules=("stored", "max", "mean"), n_resamples=100)
    assert set(out) == {"stored", "max", "mean"}
    assert out["max"]["d_auroc_ci_low"] <= out["max"]["d_auroc_vs_ref"] <= out["max"]["d_auroc_ci_high"]
    # 'mean' reproduces 'stored' here, so its difference is exactly zero
    assert out["mean"]["d_auroc_vs_ref"] == pytest.approx(0.0)


def test_aggregation_study_accepts_precomputed_rules():
    rng = np.random.default_rng(10)
    labels = np.array([0] * 30 + [1] * 30)
    test = make_preds(np.r_[rng.normal(size=30), rng.normal(1, 1, size=30)], labels)
    val = make_preds(rng.normal(size=40), [0] * 40, split="validation")
    perfect = (val.scores * 0.0, labels.astype(float))
    out = aggregation_study([(val, test)], rules=("stored",), n_resamples=50, extra={"oracle": [perfect]})
    assert out["oracle"]["auroc_mean"] == 1.0
    assert out["oracle"]["d_auroc_ci_low"] > 0
    with pytest.raises(ValueError, match="seeds"):
        aggregation_study([(val, test)], rules=("stored",), extra={"bad": [perfect, perfect]})


def test_aggregation_study_requires_identical_test_images():
    a = (make_preds([0.1] * 5, [0] * 5, split="validation"), make_preds([0.1, 0.9], [0, 1]))
    b = (make_preds([0.1] * 5, [0] * 5, split="validation"), make_preds([0.1, 0.9], [1, 0]))
    with pytest.raises(ValueError, match="same test images"):
        aggregation_study([a, b])


# --- thresholds and the calibration study -------------------------------------


def test_rank_threshold_is_conservative():
    cal = np.arange(136, dtype=float)
    value, effective, met = rank_threshold(cal, 0.01)
    assert value == 135.0 and effective == pytest.approx(1 / 137) and met


def test_rank_threshold_relaxes_and_says_so():
    _, effective, met = rank_threshold(np.arange(20, dtype=float), 0.01)
    assert effective == pytest.approx(1 / 21) and not met


def test_sigma_and_mad_thresholds_are_equivariant():
    """A positive affine transform of scores moves both thresholds the same way,
    so decisions are unchanged — the review's control against 'gains' that are
    really a changed rule."""
    rng = np.random.default_rng(2)
    cal = rng.normal(size=100)
    for fn in (sigma_threshold, mad_threshold):
        assert fn(3.0 * cal + 5.0) == pytest.approx(3.0 * fn(cal) + 5.0)


def test_calibration_study_includes_the_full_pool_once():
    rng = np.random.default_rng(3)
    val = make_preds(rng.normal(size=100), [0] * 100, split="validation")
    test = make_preds(np.r_[rng.normal(size=50), rng.normal(2, 1, size=50)], [0] * 50 + [1] * 50)
    rows = calibration_study(val, test, sizes=(20, 40), targets=(0.05,), draws=10)
    full = [r for r in rows if r["size"] == 100]
    assert len(full) == 3  # one rank row + sigma3 + mad3
    summary = summarize_calibration(rows)
    small = next(s for s in summary if s["rule"] == "rank" and s["size"] == 20)
    assert small["draws"] == 10
    assert 0.0 <= small["fpr_p05"] <= small["fpr_median"] <= small["fpr_p95"] <= 1.0


def test_calibration_summary_groups_rules_without_a_target():
    """sigma3/mad3 have no requested rate. Keying their groups on a fresh NaN
    split every draw into its own group, because NaN hashes by identity."""
    rng = np.random.default_rng(3)
    val = make_preds(rng.normal(size=100), [0] * 100, split="validation")
    test = make_preds(np.r_[rng.normal(size=50), rng.normal(2, 1, size=50)], [0] * 50 + [1] * 50)
    summary = summarize_calibration(calibration_study(val, test, sizes=(20,), targets=(0.05,), draws=10))
    heuristic = [s for s in summary if s["rule"] in ("sigma3", "mad3")]
    assert sorted((s["rule"], s["size"], s["draws"]) for s in heuristic) == [
        ("mad3", 20, 10), ("mad3", 100, 1), ("sigma3", 20, 10), ("sigma3", 100, 1),
    ]
    assert all(np.isnan(s["requested_fpr"]) for s in heuristic)


# --- calibration source -------------------------------------------------------


def _source_rows(val_shift: float, seed: int):
    from inspector.analysis import calibration_source_study

    rng = np.random.default_rng(seed)
    val = make_preds(rng.normal(val_shift, 1, size=150), [0] * 150, split="validation")
    test = make_preds(np.r_[rng.normal(size=100), rng.normal(3, 1, size=100)], [0] * 100 + [1] * 100)
    rows = calibration_source_study(val, test, targets=(0.05,), draws=400, seed=seed)
    return {r["source"]: r for r in rows}


def test_test_half_calibration_hits_the_bound_exactly():
    """The positive control: A and B are exchangeable by construction."""
    rows = _source_rows(val_shift=0.0, seed=6)
    t = rows["test half"]
    assert t["n_calibration"] == 50 and t["rank_k"] == 2
    assert t["fpr_mean"] == pytest.approx(t["bound"], abs=4 * t["fpr_mc_se"] + 1e-3)


def test_lower_scoring_validation_inflates_the_realized_fpr():
    """Validation normals that score lower than test normals — the failure
    mode of calibrating on normals carved from the training pool."""
    rows = _source_rows(val_shift=-0.7, seed=7)
    assert rows["validation"]["fpr_mean"] > 2 * rows["test half"]["fpr_mean"]


def test_train_proximity_detects_near_copies_of_training_images(tmp_path):
    from PIL import Image

    from inspector.analysis import train_proximity
    from inspector.data.core import NORMAL, Sample

    rng = np.random.default_rng(8)

    def sample(name, arr, split):
        p = tmp_path / f"{name}.png"
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(p)
        return Sample(image_path=p, label=NORMAL, defect_type="good", split=split, category="x")

    bases = [rng.integers(0, 255, (32, 32)).astype(float) for _ in range(20)]
    indices = {
        "train": [sample(f"tr{i}", b, "train") for i, b in enumerate(bases)],
        # near-copies of training frames, as a carve from the training pool can be
        "validation": [sample(f"va{i}", b + rng.normal(0, 3, b.shape), "validation")
                       for i, b in enumerate(bases[:10])],
        # genuinely new frames
        "test": [sample(f"te{i}", rng.integers(0, 255, (32, 32)), "test") for i in range(10)],
    }
    out = train_proximity(indices, test_split="test", size=16)
    assert out["auroc"] > 0.95
    assert out["median_val"] < out["median_test"]


# --- the registry driver ------------------------------------------------------


def test_study_pools_seeds_and_separates_configurations(synthetic_root, tmp_path):
    from inspector.analysis import group_runs, render_studies, run_studies
    from inspector.config import load_config
    from inspector.runner import Registry, execute, prepare

    cfg = load_config("configs/data/synth_strip.yaml")
    registry = Registry(tmp_path / "runs")
    for method, seed, overrides in [
        ("random", 0, None), ("random", 1, None),
        ("pixel_pca", 0, {"n_components": 8}), ("pixel_pca", 0, {"n_components": 4}),
    ]:
        spec, model, idx = prepare(cfg, method=method, category="synth_strip", seed=seed,
                                   data_root=synthetic_root, overrides=overrides)
        execute(spec, model, idx, data_root=synthetic_root, registry=registry)

    groups = group_runs(registry.root)
    assert sorted((g.method, len(g.run_ids)) for g in groups) == [
        ("pixel_pca", 1), ("pixel_pca", 1), ("random", 2),
    ]
    assert {g.label for g in groups if g.method == "pixel_pca"} == {
        "pixel_pca n_components=4", "pixel_pca n_components=8",
    }

    study = run_studies(registry.root, sizes=(3,), targets=(0.2,), draws=5, n_resamples=20)
    assert {r["rule"] for r in study["aggregation"]} >= {"stored", "max", "mean"}
    # no patchcore in this registry, so there is no reference to compare against
    assert study["comparisons"] == []
    assert len(study["normal_shift"]) == 4  # one row per run
    assert all(0.0 <= r["auroc"] <= 1.0 for r in study["normal_shift"])
    text = render_studies(study)
    assert "E02" in text and "E03" in text and "pixel_pca n_components=4" in text


def test_method_comparison_is_paired_and_holm_adjusted(tmp_path):
    from inspector.analysis import RunGroup, method_comparisons

    rng = np.random.default_rng(11)
    labels = [0] * 40 + [1] * 40
    ids = [f"img{k}" for k in range(80)]

    def write(run_id, scores):
        d = tmp_path / run_id
        for split, sc, lab in (("validation", rng.normal(size=30), [0] * 30), ("test", scores, labels)):
            p = make_preds(sc, lab, split=split)
            if split == "test":
                p.ids = ids
            p.save(d, with_maps=False)

    base = np.r_[rng.normal(size=40), rng.normal(size=40)]
    write("ref0", base + np.r_[np.zeros(40), np.full(40, 2.0)])
    write("same", base + np.r_[np.zeros(40), np.full(40, 2.0)] + rng.normal(0, 0.01, 80))
    write("worse", base)

    def group(key, method, run):
        return RunGroup(key, "c", method, method, "test", run_ids=[run], seeds=[0])

    rows = method_comparisons([group("a", "patchcore", "ref0"), group("b", "near", "same"),
                               group("c", "chance", "worse")], tmp_path, n_resamples=200)
    by = {r["method"]: r for r in rows}
    assert by["chance"]["d_auroc"] < -0.3 and by["chance"]["significant_holm"]
    assert by["chance"]["p_bootstrap"] > 0  # never exactly zero
    assert abs(by["near"]["d_auroc"]) < 0.02 and not by["near"]["significant_holm"]
    assert all(r["family_size"] == 2 for r in rows)


# --- exchangeability ----------------------------------------------------------


def test_normal_shift_is_half_under_exchangeability():
    rng = np.random.default_rng(4)
    val = rng.normal(size=400)
    test = make_preds(np.r_[rng.normal(size=400), rng.normal(3, 1, size=50)], [0] * 400 + [1] * 50)
    shift = normal_shift(val, test)
    assert shift.auroc == pytest.approx(0.5, abs=0.05)
    assert shift.mannwhitney_p > 0.01


def test_normal_shift_detects_a_shifted_test_set():
    rng = np.random.default_rng(5)
    val = rng.normal(size=150)
    test = make_preds(np.r_[rng.normal(0.8, 1, size=100), rng.normal(3, 1, size=100)], [0] * 100 + [1] * 100)
    shift = normal_shift(val, test)
    assert shift.auroc > 0.65
    assert shift.mannwhitney_p < 1e-3


def test_exceedance_p_value_matches_the_exact_rank_argument():
    # k = 1: all 4 top pooled scores must be test normals — 100*99*98*97 / (236*235*234*233)
    expected = (100 * 99 * 98 * 97) / (236 * 235 * 234 * 233)
    assert exceedance_p_value(4, 100, 136, 1) == pytest.approx(expected)
    assert exceedance_p_value(0, 100, 136, 1) == 1.0


def test_exceedance_p_value_is_calibrated_under_exchangeability():
    """Simulate exchangeable calibration and test normals; the permutation
    p-value must match the simulated tail. The binomial test at k/(n+1) it
    replaced would have been anti-conservative here."""
    rng = np.random.default_rng(9)
    m, n, k, fp = 60, 40, 2, 4
    hits = 0
    trials = 20000
    for _ in range(trials):
        pool = rng.random(m + n)
        thr = np.sort(pool[:m])[::-1][k - 1]
        hits += int(np.sum(pool[m:] > thr) >= fp)
    assert exceedance_p_value(fp, n, m, k) == pytest.approx(hits / trials, abs=0.01)
    from scipy import stats as sps
    assert sps.binom.sf(fp - 1, n, k / (m + 1)) < 0.6 * (hits / trials)
