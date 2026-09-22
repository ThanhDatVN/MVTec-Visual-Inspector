"""Report rendering.

Reports are generated, never hand-written, for the same reason the results table
is: a hand-edited number drifts from the run that produced it, and by the time
anyone notices, nobody remembers which one was right.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .eda import CategoryReport, ResolutionVerdict

ATTRIBUTION_FOOTER = (
    "\n---\n\n"
    "Data: MVTec AD / MVTec AD 2 (c) MVTec Software GmbH, CC BY-NC-SA 4.0. "
    "Non-commercial use only. See ATTRIBUTION.md.\n"
)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _resolution_table(verdicts: list[ResolutionVerdict]) -> str:
    return _table(
        ["long side", "output", "MP", "scale", "median diam.", "p5 thin dim.", "<1px", "verdict"],
        [
            [
                str(v.long_side),
                f"{v.out_width}x{v.out_height}",
                f"{v.megapixels:.2f}",
                f"{v.scale:.3f}",
                f"{v.median_diameter_after:.1f}px",
                f"{v.p5_min_extent_after:.2f}px",
                f"{v.frac_regions_below_1px:.1%}",
                f"**{v.verdict}**",
            ]
            for v in verdicts
        ],
    )


def render_eda_report(reports: list[CategoryReport]) -> str:
    """Render the Phase P1 EDA report (docs/04, Gate G1)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out: list[str] = [
        "# P1 — Dataset Audit and Exploratory Analysis",
        "",
        f"Generated {now} by `inspector eda`. Do not edit by hand.",
        "",
        "This report exists to answer one question before any model is fitted:",
        "**what input resolution do these defects permit?** Everything else here is",
        "supporting evidence for that decision (docs/04, task 1.7).",
        "",
    ]

    out += ["## Summary", ""]
    out.append(
        _table(
            ["category", "native", "aspect", "defect regions", "median area", "recommended"],
            [
                [
                    f"`{r.category}`",
                    f"{r.native_size[0]}x{r.native_size[1]}",
                    f"{r.aspect_ratio:.2f}:1",
                    f"{int(r.region_summary.get('n_regions', 0))}",
                    f"{r.region_summary.get('area_fraction_median', 0):.3%}",
                    (
                        (
                            "**TILING REQUIRED**"
                            if r.recommended.requires_tiling
                            else f"{r.recommended.long_side} ({r.recommended.verdict})"
                        )
                        if r.recommended
                        else "n/a"
                    ),
                ]
                for r in reports
            ],
        )
    )
    out.append("")

    for report in reports:
        out += [f"## `{report.category}`", ""]

        out += ["### Splits", ""]
        out.append(
            _table(
                ["split", "images", "normal", "anomalous"],
                [
                    [name, str(c["n"]), str(c["normal"]), str(c["anomalous"])]
                    for name, c in sorted(report.split_counts.items())
                ],
            )
        )
        out += ["", f"Defect types: {report.defect_types}", ""]

        summary = report.region_summary
        out += ["### Defect size distribution", ""]
        out.append(
            _table(
                ["statistic", "p1", "p5", "median", "max"],
                [
                    [
                        "area (px)",
                        f"{summary.get('area_px_p1', 0):.0f}",
                        f"{summary.get('area_px_p5', 0):.0f}",
                        f"{summary.get('area_px_median', 0):.0f}",
                        f"{summary.get('area_px_max', 0):.0f}",
                    ],
                    [
                        "area (% of image)",
                        f"{summary.get('area_fraction_p1', 0):.4%}",
                        f"{summary.get('area_fraction_p5', 0):.4%}",
                        f"{summary.get('area_fraction_median', 0):.4%}",
                        f"{summary.get('area_fraction_max', 0):.4%}",
                    ],
                    [
                        "equivalent diameter (px)",
                        f"{summary.get('diameter_px_p1', 0):.1f}",
                        f"{summary.get('diameter_px_p5', 0):.1f}",
                        f"{summary.get('diameter_px_median', 0):.1f}",
                        "-",
                    ],
                    [
                        "thin dimension (px)",
                        f"{summary.get('min_extent_p1', 0):.1f}",
                        "-",
                        f"{summary.get('min_extent_median', 0):.1f}",
                        "-",
                    ],
                ],
            )
        )
        out += [
            "",
            f"Regions per anomalous image: {summary.get('regions_per_image', 0):.2f}. "
            f"Local contrast against a 4px ring: median "
            f"{summary.get('local_contrast_median', 0):.1f} grey levels, p5 "
            f"{summary.get('local_contrast_p5', 0):.1f}.",
            "",
        ]

        out += ["### Resolution impact — aspect-preserving", ""]
        out.append(_resolution_table(report.resolution_aspect))
        out += [
            "",
            "`p5 thin dim.` is the 5th-percentile thin dimension of a defect after resizing, "
            "and `<1px` the share of regions whose thin dimension drops below one pixel. "
            "A region below one pixel cannot be detected by any model, so a resolution that "
            "destroys them measures the resize, not the method.",
            "",
        ]

        out += ["### Resolution impact — blind square resize", ""]
        out.append(_resolution_table(report.resolution_square))
        out += [
            "",
            "Shown to quantify the cost of the reflexive `resize(256, 256)`. "
            f"At an aspect ratio of {report.aspect_ratio:.2f}:1 a square resize squashes one "
            "axis disproportionately, and the thin dimension of a defect is hit by whichever "
            "axis is squashed hardest.",
            "",
        ]

        if report.recommended:
            decision = report.recommended
            chosen = decision.chosen
            out += ["### Decision", ""]
            if decision.requires_tiling:
                out += [
                    "**No single-pass resolution is acceptable for this category. "
                    "Overlapping tiling at native resolution is required.**",
                    "",
                    decision.rationale,
                    "",
                    "This is a finding, not a configuration failure: it is the mechanism behind "
                    "the difficulty of the small-defect categories, and it moves tiling "
                    "(docs/03, Tier 5) from an optional refinement to a prerequisite.",
                    "",
                ]
            else:
                out += [
                    f"**Recommended: long side {chosen.long_side} "
                    f"({chosen.out_width}x{chosen.out_height}, {chosen.megapixels:.2f} MP), "
                    f"verdict {chosen.verdict}.**",
                    "",
                    decision.rationale,
                    "",
                    "Selection rule `smallest_safe` (docs/04): the cheapest resolution that is "
                    "not LOSSY or DESTRUCTIVE. Compute is the scarce resource at 4 GB VRAM, so "
                    "the rule is 'as small as the defects permit', not 'as large as fits'.",
                    "",
                ]

        if report.shifts:
            out += ["### Intensity shift relative to `train`", ""]
            out.append(
                _table(
                    ["split", "mean delta", "std ratio", "Cohen's d"],
                    [
                        [
                            name,
                            f"{s['mean_delta']:+.2f}",
                            f"{s['std_ratio']:.3f}",
                            f"{s['cohens_d']:+.3f}",
                        ]
                        for name, s in sorted(report.shifts.items())
                    ],
                )
            )
            worst = max(report.shifts.items(), key=lambda kv: kv[1]["abs_cohens_d"])
            out += [
                "",
                f"Largest shift: `{worst[0]}` at |d| = {worst[1]['abs_cohens_d']:.3f}. "
                "A large value predicts threshold drift in Phase P7: a threshold fitted on "
                "validation lands somewhere else entirely on a shifted split, and the realized "
                "false-alarm rate moves even when AUROC does not.",
                "",
            ]

    out.append(ATTRIBUTION_FOOTER)
    return "\n".join(out)


def write_eda_report(reports: list[CategoryReport], out_dir: str | Path) -> tuple[Path, Path]:
    """Write the markdown report and its machine-readable twin.

    Both, always: the markdown is for a reader, the JSON is what later phases
    and the tests read, so neither has to be parsed out of the other.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "dataset_audit.md"
    md_path.write_text(render_eda_report(reports), encoding="utf-8")

    json_path = out_dir / "dataset_audit.json"
    json_path.write_text(
        json.dumps([r.as_dict() for r in reports], indent=2), encoding="utf-8"
    )
    return md_path, json_path
