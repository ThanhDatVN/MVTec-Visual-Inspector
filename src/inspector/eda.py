"""Exploratory data analysis (docs/04, Phase P1, tasks 1.5-1.7).

The analysis that actually decides something is `resolution_impact`: it measures
the defect-size distribution, then computes what each candidate input resolution
would do to the *smallest* defects. On `sheet_metal` (4224x1056, "extremely small
defects") a reflexive resize to 256x256 plausibly reduces some defects below one
pixel, at which point no model can detect them and the whole experiment measures
the resize rather than the method.

That single analysis justifies the resolution axis of the study, so it is run
before any model is fitted and its output is a required input to Gate G1.

Everything here is torch-free and streams one image at a time, so it runs on the
laptop against multi-megapixel data without exhausting memory.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

from .data.core import ANOMALOUS, DatasetIndex
from .data.transforms import ImageTransform, ResizeMode, load_image, load_mask, target_size

_CONNECTIVITY = np.ones((3, 3), dtype=int)


# ---------------------------------------------------------------------------
# Per-defect-region measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefectRegion:
    """One connected ground-truth region."""

    image_id: str
    defect_type: str
    area_px: int
    image_area_px: int
    bbox_h: int
    bbox_w: int
    local_contrast: float

    @property
    def area_fraction(self) -> float:
        return self.area_px / self.image_area_px

    @property
    def equivalent_diameter(self) -> float:
        """Diameter of a disc with the same area.

        A scale-free size measure that behaves sensibly for both a compact pit
        and an elongated scratch, unlike bbox side length.
        """
        return 2.0 * math.sqrt(self.area_px / math.pi)

    @property
    def min_extent(self) -> int:
        """The thin dimension. A 200x2 scratch survives downscaling only as long
        as *this* stays above a pixel, which is what the equivalent diameter
        hides."""
        return min(self.bbox_h, self.bbox_w)


def measure_regions(index: DatasetIndex, *, with_contrast: bool = True) -> list[DefectRegion]:
    """Measure every connected defect region in a labelled split."""
    regions: list[DefectRegion] = []
    for sample in index:
        if sample.label != ANOMALOUS or sample.mask_path is None:
            continue
        mask = load_mask(sample.mask_path)
        labelled, n = ndimage.label(mask, structure=_CONNECTIVITY)
        if n == 0:
            continue

        gray = None
        if with_contrast:
            gray = load_image(sample.image_path).mean(axis=2)

        image_area = int(mask.size)
        objects = ndimage.find_objects(labelled)
        for region_id in range(1, n + 1):
            region = labelled == region_id
            area = int(region.sum())
            if area == 0:
                continue
            slices = objects[region_id - 1]
            bbox_h = slices[0].stop - slices[0].start
            bbox_w = slices[1].stop - slices[1].start

            contrast = float("nan")
            if gray is not None:
                ring = ndimage.binary_dilation(region, iterations=4) & ~mask
                if ring.any():
                    contrast = float(abs(gray[region].mean() - gray[ring].mean()))

            regions.append(
                DefectRegion(
                    image_id=sample.rel_id(index.root),
                    defect_type=sample.defect_type,
                    area_px=area,
                    image_area_px=image_area,
                    bbox_h=int(bbox_h),
                    bbox_w=int(bbox_w),
                    local_contrast=contrast,
                )
            )
    return regions


def summarize_regions(regions: Sequence[DefectRegion]) -> dict[str, float]:
    """Percentile summary of defect size. The low percentiles are the ones that
    matter: the median defect is rarely the one a model misses."""
    if not regions:
        return {"n_regions": 0}

    areas = np.array([r.area_px for r in regions], dtype=float)
    fractions = np.array([r.area_fraction for r in regions], dtype=float)
    diameters = np.array([r.equivalent_diameter for r in regions], dtype=float)
    extents = np.array([r.min_extent for r in regions], dtype=float)
    contrasts = np.array([r.local_contrast for r in regions], dtype=float)

    return {
        "n_regions": float(len(regions)),
        "n_images": float(len({r.image_id for r in regions})),
        "regions_per_image": float(len(regions) / max(1, len({r.image_id for r in regions}))),
        "area_px_p1": float(np.percentile(areas, 1)),
        "area_px_p5": float(np.percentile(areas, 5)),
        "area_px_median": float(np.median(areas)),
        "area_px_max": float(areas.max()),
        "area_fraction_p1": float(np.percentile(fractions, 1)),
        "area_fraction_p5": float(np.percentile(fractions, 5)),
        "area_fraction_median": float(np.median(fractions)),
        "area_fraction_max": float(fractions.max()),
        "diameter_px_p1": float(np.percentile(diameters, 1)),
        "diameter_px_p5": float(np.percentile(diameters, 5)),
        "diameter_px_median": float(np.median(diameters)),
        "min_extent_p1": float(np.percentile(extents, 1)),
        "min_extent_median": float(np.median(extents)),
        "local_contrast_p5": float(np.nanpercentile(contrasts, 5)),
        "local_contrast_median": float(np.nanmedian(contrasts)),
    }


# ---------------------------------------------------------------------------
# The decisive analysis: what does a resize do to the defects?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionVerdict:
    """What one candidate input resolution does to this category's defects."""

    long_side: int
    mode: str
    out_width: int
    out_height: int
    scale: float
    median_diameter_after: float
    p5_diameter_after: float
    p1_diameter_after: float
    median_min_extent_after: float
    p5_min_extent_after: float
    frac_regions_below_1px: float
    frac_regions_below_3px: float
    megapixels: float

    @property
    def verdict(self) -> str:
        """A blunt label, so the table can be read without interpretation."""
        if self.frac_regions_below_1px > 0.01:
            return "DESTRUCTIVE"
        if self.p5_min_extent_after < 1.0:
            return "LOSSY"
        # Upscaling cannot recover detail the sensor never captured. It costs
        # quadratic compute and memory for no additional information, so it is
        # labelled rather than being allowed to sit in the table looking like
        # the safest option available.
        if self.scale > 1.0:
            return "UPSCALED"
        if self.p5_min_extent_after < 2.0:
            return "MARGINAL"
        return "SAFE"

    def as_dict(self) -> dict:
        return {**asdict(self), "verdict": self.verdict}


def resolution_impact(
    regions: Sequence[DefectRegion],
    native_size: tuple[int, int],
    *,
    candidates: Sequence[int] = (256, 320, 448, 512, 1024),
    mode: ResizeMode = "aspect_preserving",
    include_native: bool = True,
) -> list[ResolutionVerdict]:
    """For each candidate resolution, what happens to the defect population.

    Args:
        regions: measured regions for this category.
        native_size: (width, height) of the source images.
        candidates: long-side targets to evaluate.
        mode: `aspect_preserving` or `square`. Run both to make the cost of a
            blind square resize on a 4:1 frame visible in numbers.

    A region is treated as destroyed when its thin dimension falls below one
    pixel after scaling. That is optimistic — an antialiased sub-pixel feature
    leaves a faint trace — but it is a defensible floor and it is monotone in
    the scale factor, which is what the decision needs.
    """
    width, height = native_size
    diameters = np.array([r.equivalent_diameter for r in regions], dtype=float)
    extents = np.array([r.min_extent for r in regions], dtype=float)

    # Native resolution is always evaluated. Without it the table cannot show
    # that *no* downscale is acceptable, which for a category like sheet_metal
    # is the actual finding.
    targets = list(candidates)
    if include_native and max(width, height) not in targets:
        targets.append(max(width, height))
    targets = sorted(set(targets))

    verdicts: list[ResolutionVerdict] = []
    for long_side in targets:
        out_w, out_h = target_size(width, height, mode=mode, long_side=long_side)

        if mode == "square":
            # Anisotropic: each axis scales differently, and the thin dimension
            # of a defect is hit by whichever axis is squashed hardest.
            sx, sy = out_w / width, out_h / height
            linear = math.sqrt(sx * sy)
            extent_scale = min(sx, sy)
        else:
            linear = out_w / width
            extent_scale = linear

        scaled_diameters = diameters * linear
        scaled_extents = extents * extent_scale

        verdicts.append(
            ResolutionVerdict(
                long_side=long_side,
                mode=mode,
                out_width=out_w,
                out_height=out_h,
                scale=float(linear),
                median_diameter_after=float(np.median(scaled_diameters)),
                p5_diameter_after=float(np.percentile(scaled_diameters, 5)),
                p1_diameter_after=float(np.percentile(scaled_diameters, 1)),
                median_min_extent_after=float(np.median(scaled_extents)),
                p5_min_extent_after=float(np.percentile(scaled_extents, 5)),
                frac_regions_below_1px=float(np.mean(scaled_extents < 1.0)),
                frac_regions_below_3px=float(np.mean(scaled_diameters < 3.0)),
                megapixels=out_w * out_h / 1e6,
            )
        )
    return verdicts


@dataclass(frozen=True)
class ResolutionDecision:
    """The resolution decision for one category, with its justification."""

    chosen: ResolutionVerdict
    requires_tiling: bool
    rationale: str

    @property
    def long_side(self) -> int:
        return self.chosen.long_side

    @property
    def verdict(self) -> str:
        return self.chosen.verdict

    def as_dict(self) -> dict:
        return {
            "chosen": self.chosen.as_dict(),
            "requires_tiling": self.requires_tiling,
            "rationale": self.rationale,
        }


#: Largest single-pass input we assume fits the 4 GB card for a frozen CNN
#: backbone plus its feature maps. A placeholder from the budget in docs/06 §3,
#: to be replaced with the measured value at Gate G5 — the plan requires the
#: maximum feasible resolution to be established empirically, and until it is,
#: this number is an assumption and is labelled as one wherever it is used.
DEFAULT_SINGLE_PASS_MEGAPIXELS = 1.0


def recommend_resolution(
    verdicts: Sequence[ResolutionVerdict],
    *,
    prefer: str = "smallest_safe",
    max_single_pass_megapixels: float = DEFAULT_SINGLE_PASS_MEGAPIXELS,
) -> ResolutionDecision | None:
    """Pick a resolution by a stated rule rather than by habit.

    `smallest_safe` takes the cheapest resolution that is not LOSSY or
    DESTRUCTIVE. Compute is the scarce resource at 4 GB VRAM, so the rule is
    "as small as the defects permit", not "as large as fits".

    When *nothing* qualifies — which is the expected outcome for a category of
    4224x1056 frames carrying 2-pixel scratches — the honest answer is not to
    quietly return the largest candidate and carry on. It is that any single
    downscaled forward pass destroys the defects, so the category must be
    processed by **overlapping tiling at native resolution** (docs/03, Tier 5).
    Returning that as an explicit flag is what stops the pipeline from producing
    a plausible-looking number that measures the resize.
    """
    if not verdicts:
        return None
    if prefer != "smallest_safe":
        raise ValueError(f"unknown preference: {prefer!r}")

    acceptable = [v for v in verdicts if v.verdict in ("SAFE", "MARGINAL")]
    if acceptable:
        chosen = min(acceptable, key=lambda v: v.megapixels)
        # Preserving the defects and fitting in memory are two separate
        # constraints, and a category can satisfy the first only at a size that
        # violates the second. That combination is not a contradiction — it is
        # the definition of a category that must be tiled.
        if chosen.megapixels > max_single_pass_megapixels:
            return ResolutionDecision(
                chosen=chosen,
                requires_tiling=True,
                rationale=(
                    f"the smallest defect-preserving resolution is "
                    f"{chosen.out_width}x{chosen.out_height} ({chosen.megapixels:.2f} MP, rated "
                    f"{chosen.verdict}), which exceeds the assumed {max_single_pass_megapixels:.2f} MP "
                    "single-pass budget for a 4 GB card. Process this category with overlapping "
                    "tiling at that scale: tiles keep the pixel scale that the defects require "
                    "while each forward pass stays within memory. (Budget is an assumption from "
                    "docs/06 §3 until measured at Gate G5.)"
                ),
            )
        return ResolutionDecision(
            chosen=chosen,
            requires_tiling=False,
            rationale=(
                f"cheapest resolution rated {chosen.verdict}: p5 thin dimension "
                f"{chosen.p5_min_extent_after:.2f}px survives, "
                f"{chosen.frac_regions_below_1px:.1%} of regions lost, "
                f"{chosen.megapixels:.2f} MP within the "
                f"{max_single_pass_megapixels:.2f} MP single-pass budget"
            ),
        )

    chosen = max(verdicts, key=lambda v: v.megapixels)
    return ResolutionDecision(
        chosen=chosen,
        requires_tiling=True,
        rationale=(
            "NO candidate resolution preserves the defects: even at "
            f"{chosen.out_width}x{chosen.out_height}, {chosen.frac_regions_below_1px:.1%} of "
            f"regions fall below one pixel (p5 thin dimension {chosen.p5_min_extent_after:.2f}px). "
            "This category requires overlapping tiling at native resolution; a single "
            "downscaled forward pass would measure the resize, not the model."
        ),
    )


# ---------------------------------------------------------------------------
# Distribution shift between splits
# ---------------------------------------------------------------------------


@dataclass
class SplitStats:
    """Global intensity statistics for one split."""

    split: str
    n_images: int
    width: int
    height: int
    mean: float
    std: float
    p1: float
    p99: float
    per_image_means: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        out = asdict(self)
        out.pop("per_image_means")
        return out


def split_statistics(index: DatasetIndex, *, max_images: int | None = 200) -> SplitStats:
    """Intensity statistics, used to detect lighting shift between splits.

    On MVTec AD 2 this is not idle curiosity: the private-mixed split is
    captured under *unseen* lighting, and a visible gap here between train and
    test predicts the threshold drift that Phase P7 measures.
    """
    samples = list(index)[: max_images or len(index)]
    if not samples:
        raise ValueError(f"split {index.split_name!r} has no samples")

    means, all_values = [], []
    width = height = 0
    for sample in samples:
        gray = load_image(sample.image_path).mean(axis=2)
        height, width = gray.shape
        means.append(float(gray.mean()))
        # Subsample pixels: percentiles converge long before we need all 8 MP.
        flat = gray.ravel()
        step = max(1, flat.size // 20_000)
        all_values.append(flat[::step])

    pooled = np.concatenate(all_values)
    return SplitStats(
        split=index.split_name,
        n_images=len(samples),
        width=int(width),
        height=int(height),
        mean=float(pooled.mean()),
        std=float(pooled.std()),
        p1=float(np.percentile(pooled, 1)),
        p99=float(np.percentile(pooled, 99)),
        per_image_means=means,
    )


def lighting_shift(train: SplitStats, other: SplitStats) -> dict[str, float]:
    """How far `other` has drifted from `train` in intensity.

    Cohen's d on per-image means is the headline: it is unit-free, so it is
    comparable across categories with completely different exposure.
    """
    a = np.asarray(train.per_image_means, dtype=float)
    b = np.asarray(other.per_image_means, dtype=float)
    pooled_std = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else 0.0
    cohens_d = float((b.mean() - a.mean()) / pooled_std) if pooled_std > 0 else 0.0
    return {
        "mean_delta": float(other.mean - train.mean),
        "std_ratio": float(other.std / train.std) if train.std else float("nan"),
        "cohens_d": cohens_d,
        "abs_cohens_d": abs(cohens_d),
    }


# ---------------------------------------------------------------------------
# Whole-category analysis
# ---------------------------------------------------------------------------


@dataclass
class CategoryReport:
    category: str
    layout: str
    native_size: tuple[int, int]
    aspect_ratio: float
    split_counts: dict[str, dict[str, int]]
    defect_types: dict[str, int]
    region_summary: dict[str, float]
    resolution_aspect: list[ResolutionVerdict]
    resolution_square: list[ResolutionVerdict]
    recommended: ResolutionDecision | None
    split_stats: dict[str, dict]
    shifts: dict[str, dict[str, float]]

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "layout": self.layout,
            "native_size": list(self.native_size),
            "aspect_ratio": self.aspect_ratio,
            "split_counts": self.split_counts,
            "defect_types": self.defect_types,
            "region_summary": self.region_summary,
            "resolution_aspect_preserving": [v.as_dict() for v in self.resolution_aspect],
            "resolution_square": [v.as_dict() for v in self.resolution_square],
            "recommended": self.recommended.as_dict() if self.recommended else None,
            "split_stats": self.split_stats,
            "lighting_shifts": self.shifts,
        }


def analyse_category(
    indices: dict[str, DatasetIndex],
    *,
    test_split: str = "test_public",
    candidates: Sequence[int] = (256, 320, 448, 512, 1024),
    max_images_for_stats: int | None = 200,
) -> CategoryReport:
    """Run the full P1 analysis for one category."""
    if test_split not in indices:
        raise KeyError(f"no {test_split!r} split to analyse; have {sorted(indices)}")

    test = indices[test_split]
    regions = measure_regions(test)
    if not regions:
        raise ValueError(f"no defect regions found in {test_split!r}")

    stats = {name: split_statistics(idx, max_images=max_images_for_stats) for name, idx in indices.items()}
    native = (stats[test_split].width, stats[test_split].height)

    aspect = resolution_impact(regions, native, candidates=candidates, mode="aspect_preserving")
    square = resolution_impact(regions, native, candidates=candidates, mode="square")

    shifts = {}
    if "train" in stats:
        for name, other in stats.items():
            if name != "train":
                shifts[name] = lighting_shift(stats["train"], other)

    defect_counts: dict[str, int] = {}
    for region in regions:
        defect_counts[region.defect_type] = defect_counts.get(region.defect_type, 0) + 1

    return CategoryReport(
        category=test.category,
        layout=test.layout.name,
        native_size=native,
        aspect_ratio=native[0] / native[1],
        split_counts={
            name: {
                "n": len(idx),
                "normal": idx.n_normal,
                "anomalous": idx.n_anomalous,
            }
            for name, idx in indices.items()
        },
        defect_types=defect_counts,
        region_summary=summarize_regions(regions),
        resolution_aspect=aspect,
        resolution_square=square,
        recommended=recommend_resolution(aspect),
        split_stats={name: s.as_dict() for name, s in stats.items()},
        shifts=shifts,
    )


def verify_transform_policy(
    transform: ImageTransform, report: CategoryReport
) -> tuple[bool, str]:
    """Check a configured transform against the measured defect population.

    This is the guard that turns the EDA from a document into an enforced
    decision: a config whose resize destroys defects fails before it is fitted,
    rather than producing a plausible-looking but meaningless number.
    """
    width, height = report.native_size
    out_w, out_h = transform.output_size(width, height)
    matching = [
        v
        for v in (report.resolution_aspect + report.resolution_square)
        if (v.out_width, v.out_height) == (out_w, out_h)
    ]
    if not matching:
        verdicts = resolution_impact(
            [], report.native_size, candidates=(max(out_w, out_h),)
        )
        return True, f"not evaluated ({out_w}x{out_h}); no defect measurement available: {verdicts}"

    best = matching[0]
    ok = best.verdict in ("SAFE", "MARGINAL")
    return ok, (
        f"{out_w}x{out_h} -> {best.verdict}: p5 thin dimension "
        f"{best.p5_min_extent_after:.2f}px, {best.frac_regions_below_1px:.1%} of regions "
        "below one pixel"
    )
