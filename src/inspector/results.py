"""Results table generation.

`reports/results.csv` is generated, never hand-edited (docs/06 §4). A
hand-edited results table drifts from the runs that produced it, and by the time
anyone notices, nobody remembers which version was right.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from .pipeline import ExperimentResult

#: Column order for reports/results.csv. `threshold_source` and `oracle_flag`
#: are mandatory, not optional: they are what make the table auditable.
COLUMNS = [
    "run_id", "run_name", "status", "method", "tier", "owner", "category", "dataset",
    "split", "seed", "config_hash", "implementation_id", "git_sha", "dirty", "resolution",
    "n_train", "n_validation", "n_test",
    "image_auroc", "image_aupr", "f1max_oracle", "fpr_at_op1", "recall_at_op1",
    "tp", "fp", "tn", "fn",
    "requested_fpr", "effective_fpr", "calibration_count", "threshold_rank", "target_met",
    "threshold_op1", "threshold_method", "threshold_comparator", "threshold_source",
    "oracle_flag",
    "pixel_auroc", "aupro_005", "aupro_030", "segf1_3sigma", "iou_3sigma",
    "threshold_3sigma", "n_regions", "n_negatives_sampled",
    "smoothing_sigma", "smoothing_units", "metrics_version",
    "fit_seconds", "predict_seconds", "eval_seconds", "e2e_ms_per_image", "peak_vram_mb",
    "escape_by_defect", "fit_extra", "notes",
]

_DISPLAY = [
    ("tier", "tier", "{}"),
    ("method", "method", "{}"),
    ("impl", "owner", "{}"),
    ("category", "category", "{}"),
    ("res", "resolution", "{}"),
    ("I-AUROC", "image_auroc", "{:.4f}"),
    ("I-AUPR", "image_aupr", "{:.4f}"),
    ("F1max (oracle)", "f1max_oracle", "{:.4f}"),
    ("FPR@OP1", "fpr_at_op1", "{:.4f}"),
    ("Recall@OP1", "recall_at_op1", "{:.4f}"),
    ("P-AUROC", "pixel_auroc", "{:.4f}"),
    ("AU-PRO@0.05", "aupro_005", "{:.4f}"),
    ("AU-PRO@0.30", "aupro_030", "{:.4f}"),
    ("SegF1", "segf1_3sigma", "{:.4f}"),
    ("target FPR", "requested_fpr", "{:.2%}"),
    ("eff. FPR", "effective_fpr", "{:.2%}"),
]


def append_results(results: Sequence[ExperimentResult], path: str | Path) -> Path:
    """Append rows, writing the header only when the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for result in results:
            writer.writerow(result.as_row())
    return path


def write_results(results: Sequence[ExperimentResult], path: str | Path) -> Path:
    """Write a complete table (overwriting), sorted for stable diffs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda r: (r.category, r.tier, r.method, r.seed, r.run_id))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for result in ordered:
            writer.writerow(result.as_row())
    return path


def render_markdown(results: Sequence[ExperimentResult]) -> str:
    """A compact markdown view of a set of runs.

    Every caption obligation from docs/02 is met here or not at all: the split,
    the threshold source, and the oracle marking all appear.
    """
    if not results:
        return "_no results_\n"

    splits = sorted({r.split for r in results})
    sources = sorted({r.threshold_source for r in results})
    caption = (
        f"Split: `{', '.join(splits)}`. Thresholded metrics use the "
        f"`{', '.join(sources)}`-derived operating point "
        "(`OP-FPR1` for image, `OP-3SIGMA` for pixel), with strict decisions "
        "(`score > threshold`). `target FPR` is what was requested and `eff. FPR` the "
        "rank rule's finite-sample marginal bound; `FPR@OP1` is what the test normals "
        "actually showed, and may exceed the bound — the bound is marginal, not conditional "
        "on one fitted model, and assumes exchangeable calibration and test normals. "
        "Columns marked (oracle) use test labels and are upper bounds only.\n"
    )

    lines = [
        caption,
        "| " + " | ".join(c[0] for c in _DISPLAY) + " |",
        "|" + "|".join(["---"] * len(_DISPLAY)) + "|",
    ]
    for r in sorted(results, key=lambda x: (x.category, x.tier, x.method)):
        cells = []
        for _, attr, fmt in _DISPLAY:
            value = getattr(r, attr)
            is_nan = isinstance(value, float) and value != value
            cells.append("-" if is_nan else fmt.format(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
