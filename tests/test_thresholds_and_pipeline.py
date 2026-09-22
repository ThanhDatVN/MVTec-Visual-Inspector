"""Operating points and the end-to-end pipeline.

The threshold tests encode a finding from P2: at MVTec AD 2's validation split
sizes (19-48 normal images), the protocol's nominal 1% false-alarm target is not
statistically achievable. See docs/07, ADR-7.
"""

from __future__ import annotations

import numpy as np
import pytest

from inspector.data.transforms import ImageTransform
from inspector.models import TIER0_MODELS, MeanIntensityScorer, PixelPCAScorer, RandomScorer
from inspector.pipeline import run_experiment
from inspector.postproc import (
    ScoreNormalizer,
    from_validation_fpr,
    from_validation_percentile,
    min_achievable_fpr,
    sigma_from_stats,
)
from inspector.results import COLUMNS, append_results, render_markdown

# --- the achievability floor ------------------------------------------------

#: Validation split sizes for the three study categories, from the AD 2 paper.
AD2_VALIDATION_SIZES = {"sheet_metal": 19, "fruit_jelly": 37, "walnuts": 48}


def test_floor_is_one_over_n_plus_one():
    assert min_achievable_fpr(19) == pytest.approx(1 / 20)
    assert min_achievable_fpr(99) == pytest.approx(1 / 100)


@pytest.mark.parametrize(("category", "n"), sorted(AD2_VALIDATION_SIZES.items()))
def test_one_percent_fpr_is_unreachable_on_every_study_category(category, n):
    """The finding. `OP-FPR1` claims a 1% false-alarm target; none of the three
    categories has enough validation images to calibrate below 2%."""
    assert min_achievable_fpr(n) > 0.01, category


def test_strict_mode_refuses_an_unachievable_target():
    """Refusing beats clamping: a silently clamped target means the report says
    1% while the model delivers 5%, which is the exact failure this prevents."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=r"below the .* floor"):
        from_validation_fpr(rng.normal(size=19), target_fpr=0.01, strict=True)


def test_non_strict_mode_clamps_and_records_what_it_did():
    rng = np.random.default_rng(0)
    threshold = from_validation_fpr(rng.normal(size=19), target_fpr=0.01, strict=False)
    assert threshold.params["target_fpr"] == pytest.approx(0.05)
    assert threshold.params["expected_fpr"] == pytest.approx(0.05)
    assert threshold.params["min_achievable_fpr"] == pytest.approx(0.05)


def test_order_statistic_delivers_its_advertised_rate():
    """Distribution-free claim: the k-th largest of n validation scores is
    exceeded by a fresh normal sample with probability k/(n+1). Verified by
    simulation, and on a skewed distribution to show it does not rely on
    normality the way an interpolated percentile does."""
    rng = np.random.default_rng(7)
    n, target = 48, 0.05
    realized = []
    for _ in range(400):
        validation = rng.lognormal(sigma=1.5, size=n)
        threshold = from_validation_fpr(validation, target_fpr=target)
        fresh = rng.lognormal(sigma=1.5, size=2000)
        realized.append(float((fresh >= threshold.value).mean()))

    expected = np.ceil(target * (n + 1)) / (n + 1)
    assert np.mean(realized) == pytest.approx(expected, abs=0.015)


def test_threshold_is_an_actual_validation_score():
    """An order statistic, not an interpolation between two of them: at these
    sample sizes an interpolated value assumes a tail shape the data cannot
    support."""
    rng = np.random.default_rng(1)
    scores = rng.normal(size=40)
    threshold = from_validation_fpr(scores, target_fpr=0.1)
    assert threshold.value in set(scores.tolist())


def test_percentile_api_delegates_to_the_order_statistic_rule():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=48)
    assert from_validation_percentile(scores, percentile=95.0).value == pytest.approx(
        from_validation_fpr(scores, target_fpr=0.05).value
    )


def test_refuses_an_underpowered_validation_split():
    with pytest.raises(ValueError, match="at least 4"):
        from_validation_fpr(np.array([0.1, 0.2, 0.3]), target_fpr=0.2)


def test_sigma_from_stats_matches_the_official_rule():
    threshold = sigma_from_stats(0.5, 0.1, n_sigma=3.0)
    assert threshold.value == pytest.approx(0.8)
    assert threshold.source_split == "validation"
    assert not threshold.oracle


def test_normalizer_fit_bounds_obeys_the_same_rules():
    with pytest.raises(ValueError, match="rule L3"):
        ScoreNormalizer().fit_bounds(0.0, 1.0, split_name="test_public")

    normalizer = ScoreNormalizer().fit_bounds(0.0, 2.0, split_name="validation")
    with pytest.raises(RuntimeError, match="already fitted"):
        normalizer.fit_bounds(0.0, 1.0, split_name="validation")


# --- the pipeline -----------------------------------------------------------


@pytest.fixture
def small_transform():
    return ImageTransform(mode="aspect_preserving", long_side=128, normalize="imagenet")


def test_pipeline_runs_end_to_end(strip_indices, small_transform, tmp_path):
    model = PixelPCAScorer(small_transform, n_components=4, work_size=32)
    result = run_experiment(model, strip_indices, workdir=tmp_path, seed=0)

    assert result.method == "pixel_pca"
    assert result.category == "synth_strip"
    assert result.n_test == len(strip_indices["test_public"])
    assert 0.0 <= result.image_auroc <= 1.0
    assert 0.0 <= result.aupro_005 <= 1.0
    assert 0.0 <= result.aupro_030 <= 1.0
    assert result.threshold_source == "validation"
    assert not result.oracle_flag


def test_pipeline_thresholds_never_come_from_test(strip_indices, small_transform, tmp_path):
    """The whole ordering of `run_experiment` exists to make this true."""
    result = run_experiment(
        MeanIntensityScorer(small_transform), strip_indices, workdir=tmp_path, seed=0
    )
    assert result.threshold_source == "validation"
    assert result.oracle_flag is False


def test_pipeline_refuses_a_missing_split(strip_indices, small_transform, tmp_path):
    incomplete = {k: v for k, v in strip_indices.items() if k != "validation"}
    with pytest.raises(KeyError, match="missing split"):
        run_experiment(
            MeanIntensityScorer(small_transform), incomplete, workdir=tmp_path, seed=0
        )


def test_random_scorer_lands_near_chance_on_pixels(strip_indices, small_transform, tmp_path):
    """Gate G2's headline control, run through the real pipeline rather than on
    hand-made arrays. Averaged over seeds, because a single 11-image split has
    enormous variance."""
    aurocs, aupro05 = [], []
    for seed in range(8):
        result = run_experiment(
            RandomScorer(small_transform, seed=seed),
            strip_indices,
            workdir=tmp_path / f"s{seed}",
            seed=seed,
        )
        aurocs.append(result.pixel_auroc)
        aupro05.append(result.aupro_005)

    assert np.mean(aurocs) == pytest.approx(0.5, abs=0.08)
    assert np.mean(aupro05) == pytest.approx(0.025, abs=0.035)


def test_model_refuses_to_fit_on_a_test_split(strip_indices, small_transform):
    model = MeanIntensityScorer(small_transform)
    with pytest.raises(ValueError, match="rule L2"):
        model.fit(strip_indices["test_public"])


def test_model_refuses_to_predict_before_fitting(small_transform):
    model = MeanIntensityScorer(small_transform)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_image(np.zeros((32, 32, 3), dtype=np.uint8))


def test_anomaly_map_is_returned_at_native_resolution(strip_indices, small_transform):
    """Scoring a downscaled map against a downscaled mask inflates PRO, because
    a defect occupies proportionally more of a smaller image (protocol §4.1)."""
    model = PixelPCAScorer(small_transform, n_components=4, work_size=16)
    model.fit(strip_indices["train"])

    sample = strip_indices["test_public"].filter(label=1)[0]
    from inspector.data.transforms import load_image

    native = load_image(sample.image_path)
    prediction = model.predict_sample(sample)
    assert prediction.anomaly_map.shape == native.shape[:2]


def test_fit_record_captures_provenance(strip_indices, small_transform):
    model = MeanIntensityScorer(small_transform)
    model.fit(strip_indices["train"])

    assert model.fit_record is not None
    assert model.fit_record.split == "train"
    assert model.fit_record.n_images == len(strip_indices["train"])
    assert len(model.fit_record.manifest_sha256) == 64


def test_every_tier0_model_runs(strip_indices, small_transform, tmp_path):
    for name, cls in TIER0_MODELS.items():
        kwargs = {"seed": 0} if cls.stochastic else {}
        result = run_experiment(
            cls(small_transform, **kwargs),
            strip_indices,
            workdir=tmp_path / name,
            seed=0,
        )
        assert result.method == name
        assert np.isfinite(result.image_auroc)


# --- results table ----------------------------------------------------------


def test_results_csv_roundtrip(strip_indices, small_transform, tmp_path):
    import csv

    result = run_experiment(
        MeanIntensityScorer(small_transform), strip_indices, workdir=tmp_path, seed=0
    )
    path = append_results([result], tmp_path / "results.csv")
    append_results([result], path)  # header written once

    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert list(rows[0]) == COLUMNS


def test_markdown_caption_states_split_and_threshold_source(
    strip_indices, small_transform, tmp_path
):
    """docs/02: a table without split, threshold source and oracle marking is
    not reportable."""
    result = run_experiment(
        MeanIntensityScorer(small_transform), strip_indices, workdir=tmp_path, seed=0
    )
    markdown = render_markdown([result])
    assert "test_public" in markdown
    assert "validation" in markdown
    assert "oracle" in markdown


def test_markdown_of_nothing_is_not_a_crash():
    assert "no results" in render_markdown([])
