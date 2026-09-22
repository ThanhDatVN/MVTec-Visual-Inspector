"""Phase P1 exploratory analysis (docs/04, Gate G1).

The resolution analysis decides the study's most consequential preprocessing
parameter, so it is tested at the *real* scale of MVTec AD 2 rather than at
fixture scale. A synthetic 128x128 fixture cannot demonstrate that a resize
destroys defects, because at that size nothing does.
"""

from __future__ import annotations

import numpy as np
import pytest

from inspector.data.transforms import ImageTransform, target_size
from inspector.eda import (
    DefectRegion,
    analyse_category,
    measure_regions,
    recommend_resolution,
    resolution_impact,
    split_statistics,
    summarize_regions,
)


def sheet_metal_population(n_pits: int = 60, n_scratches: int = 40) -> list[DefectRegion]:
    """A defect population at `sheet_metal`'s real scale: 4224x1056 frames with
    pits of 3-12px diameter and scratches 80-200px long but only 1.5-3.5px thick."""
    rng = np.random.default_rng(0)
    width, height = 4224, 1056
    area = width * height
    regions: list[DefectRegion] = []
    for i in range(n_pits):
        d = rng.uniform(3, 12)
        regions.append(
            DefectRegion(f"pit{i}", "pit", max(1, int(np.pi * (d / 2) ** 2)), area, int(d), int(d), 40.0)
        )
    for i in range(n_scratches):
        length, thickness = rng.uniform(80, 200), rng.uniform(1.5, 3.5)
        regions.append(
            DefectRegion(
                f"scr{i}", "scratch", int(length * thickness), area, int(thickness), int(length), 35.0
            )
        )
    return regions


# --- region measurement -----------------------------------------------------


def test_measures_every_region_in_a_split(strip_indices):
    regions = measure_regions(strip_indices["test_public"])
    assert regions
    assert {r.defect_type for r in regions} == {"pit", "scratch"}
    assert all(r.area_px > 0 for r in regions)
    assert all(0 < r.area_fraction < 1 for r in regions)


def test_equivalent_diameter_matches_a_known_disc():
    """A 100px disc has diameter 2*sqrt(100/pi) ~= 11.28."""
    region = DefectRegion("x", "pit", 100, 10_000, 11, 11, 20.0)
    assert region.equivalent_diameter == pytest.approx(11.28, abs=0.01)


def test_thin_dimension_is_what_a_scratch_loses_first():
    """Equivalent diameter flatters an elongated defect: a 200x2 scratch has a
    diameter of ~22px but survives downscaling only while its 2px thickness
    does. The thin dimension is the quantity that actually decides."""
    scratch = DefectRegion("s", "scratch", 400, 10**6, 2, 200, 30.0)
    assert scratch.equivalent_diameter > 20
    assert scratch.min_extent == 2


def test_summary_reports_low_percentiles():
    """The median defect is rarely the one a model misses, so the summary must
    expose p1/p5, not just central tendency."""
    summary = summarize_regions(sheet_metal_population())
    for key in ("area_px_p1", "area_px_p5", "diameter_px_p1", "min_extent_p1"):
        assert key in summary
    assert summary["area_px_p1"] <= summary["area_px_median"] <= summary["area_px_max"]


def test_summary_of_nothing_is_not_a_crash():
    assert summarize_regions([]) == {"n_regions": 0}


# --- the decisive analysis --------------------------------------------------


def test_naive_256_resize_destroys_sheet_metal_scale_defects():
    """The finding that justifies the study's resolution axis."""
    verdicts = resolution_impact(
        sheet_metal_population(), (4224, 1056), candidates=(256,), include_native=False
    )
    at_256 = verdicts[0]
    assert at_256.verdict == "DESTRUCTIVE"
    assert at_256.frac_regions_below_1px > 0.9
    assert at_256.p5_min_extent_after < 0.2


def test_native_resolution_preserves_them():
    verdicts = resolution_impact(sheet_metal_population(), (4224, 1056), candidates=())
    native = max(verdicts, key=lambda v: v.megapixels)
    assert native.scale == pytest.approx(1.0)
    assert native.frac_regions_below_1px == 0.0
    assert native.verdict in ("SAFE", "MARGINAL")


def test_upscaling_is_labelled_rather_than_rated_safe():
    """Upscaling cannot recover detail the sensor never captured. Left unlabelled
    it sits at the top of the table looking like the safest option, which is how
    a study ends up paying quadratic compute for no information."""
    regions = [DefectRegion(f"r{i}", "pit", 400, 256 * 64, 20, 20, 40.0) for i in range(20)]
    verdicts = resolution_impact(regions, (256, 64), candidates=(512, 1024))
    upscaled = [v for v in verdicts if v.scale > 1.0]

    assert upscaled
    assert all(v.verdict == "UPSCALED" for v in upscaled)


def test_recommendation_never_picks_an_upscale():
    regions = [DefectRegion(f"r{i}", "pit", 400, 256 * 64, 20, 20, 40.0) for i in range(20)]
    verdicts = resolution_impact(regions, (256, 64), candidates=(512, 1024))
    decision = recommend_resolution(verdicts)

    assert decision is not None
    assert decision.chosen.scale <= 1.0


def test_native_is_always_evaluated():
    """Without a native row the table cannot show that no downscale works,
    which for a small-defect category is the actual result."""
    verdicts = resolution_impact(
        sheet_metal_population(), (4224, 1056), candidates=(256, 512)
    )
    assert any(v.scale == pytest.approx(1.0) for v in verdicts)


def test_degradation_is_monotone_in_scale():
    verdicts = resolution_impact(
        sheet_metal_population(), (4224, 1056), candidates=(256, 320, 448, 512, 1024, 2048)
    )
    ordered = sorted(verdicts, key=lambda v: v.scale)
    losses = [v.frac_regions_below_1px for v in ordered]
    assert losses == sorted(losses, reverse=True)


def test_square_resize_is_worse_than_aspect_preserving_on_a_wide_frame():
    """At 4:1, a square resize squashes the short axis hardest, and a defect's
    thin dimension is hit by whichever axis is squashed hardest."""
    regions = sheet_metal_population()
    aspect = resolution_impact(
        regions, (4224, 1056), candidates=(512,), mode="aspect_preserving", include_native=False
    )[0]
    square = resolution_impact(
        regions, (4224, 1056), candidates=(512,), mode="square", include_native=False
    )[0]

    assert square.megapixels > aspect.megapixels  # costs more compute...
    assert square.p5_min_extent_after <= aspect.p5_min_extent_after  # ...and preserves less


def test_recommendation_demands_tiling_when_the_safe_size_exceeds_memory():
    """Preserving defects and fitting in 4 GB are separate constraints. A
    category that satisfies the first only at a size violating the second is
    the definition of one that must be tiled — not a contradiction to paper
    over with the largest candidate."""
    verdicts = resolution_impact(sheet_metal_population(), (4224, 1056), candidates=(256, 512))
    decision = recommend_resolution(verdicts, max_single_pass_megapixels=1.0)

    assert decision is not None
    assert decision.requires_tiling
    assert "tiling" in decision.rationale.lower()


def test_recommendation_takes_the_cheapest_safe_option_when_one_fits():
    """Compute is the scarce resource, so the rule is 'as small as the defects
    permit', not 'as large as fits'."""
    rng = np.random.default_rng(1)
    regions = [
        DefectRegion(f"b{i}", "blob", 4000, 512 * 512, int(rng.uniform(55, 75)), 70, 40.0)
        for i in range(40)
    ]
    verdicts = resolution_impact(regions, (512, 512), candidates=(128, 256, 448))
    decision = recommend_resolution(verdicts, max_single_pass_megapixels=1.0)

    assert decision is not None
    assert not decision.requires_tiling
    assert decision.chosen.megapixels == min(
        v.megapixels for v in verdicts if v.verdict in ("SAFE", "MARGINAL")
    )


def test_recommendation_of_nothing_is_none():
    assert recommend_resolution([]) is None


def test_unknown_preference_is_rejected():
    verdicts = resolution_impact(sheet_metal_population(), (4224, 1056), candidates=(512,))
    with pytest.raises(ValueError, match="unknown preference"):
        recommend_resolution(verdicts, prefer="biggest")


# --- transform policy agrees with the analysis ------------------------------


def test_target_size_preserves_aspect_ratio():
    out_w, out_h = target_size(4224, 1056, mode="aspect_preserving", long_side=512)
    assert out_w == 512
    assert out_h == 128
    assert out_w / out_h == pytest.approx(4224 / 1056, rel=1e-2)


def test_square_mode_does_not_preserve_aspect_ratio():
    assert target_size(4224, 1056, mode="square", long_side=512) == (512, 512)


def test_transform_scale_factor_matches_the_analysis():
    transform = ImageTransform(mode="aspect_preserving", long_side=512)
    assert transform.output_size(4224, 1056) == (512, 128)
    assert transform.scale_factor(4224, 1056) == pytest.approx(512 / 4224, rel=1e-2)


# --- whole-category analysis ------------------------------------------------


def test_analyse_category_runs_end_to_end(strip_indices):
    report = analyse_category(strip_indices, candidates=(64, 128, 256))
    assert report.category == "synth_strip"
    assert report.native_size == (256, 64)
    assert report.aspect_ratio == pytest.approx(4.0)
    assert report.region_summary["n_regions"] > 0
    assert report.recommended is not None
    assert set(report.split_counts) == {"train", "validation", "test_public"}


def test_analyse_category_reports_intensity_shift(strip_indices):
    report = analyse_category(strip_indices, candidates=(128,))
    assert "test_public" in report.shifts
    assert "cohens_d" in report.shifts["test_public"]


def test_split_statistics_records_native_size(strip_indices):
    stats = split_statistics(strip_indices["train"])
    assert (stats.width, stats.height) == (256, 64)
    assert stats.n_images == len(strip_indices["train"])
    assert 0 <= stats.mean <= 255


def test_analyse_category_needs_a_labelled_split(strip_indices):
    with pytest.raises(KeyError, match="no 'test_private'"):
        analyse_category(strip_indices, test_split="test_private")
