"""Pixel-level metrics: AUROC, SegF1, IoU.

Pixel AUROC is reported because the project scope and the older literature
require it, **not** because it is used for model selection. It is dominated by
the normal-pixel majority and saturates near 1.0 while localization is still
visibly poor. AU-PRO and AUPIMO are the metrics that decide anything here.

Memory note: 150 test images at 8 MP is 1.2e9 pixels. Materialising that as a
float64 array for `sklearn.roc_auc_score` needs ~10 GB. The AUROC here is
therefore computed from score histograms in a single streaming pass, which is
O(bins) in memory and exact to the bin width.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PixelMetrics:
    auroc: float
    seg_f1: float
    seg_precision: float
    seg_recall: float
    iou: float
    threshold: float
    threshold_source: str
    n_positive: int
    n_negative: int

    def as_dict(self) -> dict[str, float | str | int]:
        return asdict(self)


def _score_range(
    scores: Iterable[np.ndarray],
) -> tuple[float, float]:
    lo, hi = np.inf, -np.inf
    for s in scores:
        a = np.asarray(s)
        lo = min(lo, float(a.min()))
        hi = max(hi, float(a.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("anomaly maps contain no finite values")
    return lo, hi


def pixel_auroc(
    scores: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    bins: int = 1 << 16,
) -> float:
    """Streaming, histogram-based pixel AUROC.

    With 65 536 bins the quantization error is far below the run-to-run spread
    of any model we will compare, and memory stays constant regardless of image
    size. Ties inside a bin are handled by the trapezoidal rule, which is the
    standard treatment for tied scores.
    """
    lo, hi = _score_range(scores)
    if hi <= lo:  # constant map: no ranking information at all
        return 0.5
    edges = np.linspace(lo, hi, bins + 1)

    pos = np.zeros(bins, dtype=np.int64)
    neg = np.zeros(bins, dtype=np.int64)
    for score, mask in zip(scores, masks):
        score = np.asarray(score, dtype=np.float64).ravel()
        mask = np.asarray(mask).astype(bool).ravel()
        if score.size != mask.size:
            raise ValueError("score and mask sizes differ")
        idx = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, bins - 1)
        pos += np.bincount(idx[mask], minlength=bins)
        neg += np.bincount(idx[~mask], minlength=bins)

    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("pixel AUROC needs both defect and normal pixels")

    # Sweep from the highest bin down; trapezoid over (FPR, TPR).
    tpr = np.concatenate(([0.0], np.cumsum(pos[::-1]) / n_pos))
    fpr = np.concatenate(([0.0], np.cumsum(neg[::-1]) / n_neg))
    return float(np.trapezoid(tpr, fpr))


def segmentation_metrics(
    scores: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    threshold: float,
    threshold_source: str,
) -> tuple[float, float, float, float, int, int]:
    """Pooled SegF1 / precision / recall / IoU at a fixed pixel threshold.

    Pooled over all pixels of all images, which is the AD 2 benchmark's
    convention. A per-image mean would be a different (and generally higher)
    number, so the two must never be mixed in one table.
    """
    tp = fp = fn = tn = 0
    for score, mask in zip(scores, masks):
        pred = np.asarray(score) >= threshold
        gt = np.asarray(mask).astype(bool)
        if pred.shape != gt.shape:
            raise ValueError(f"score shape {pred.shape} != mask shape {gt.shape}")
        tp += int(np.count_nonzero(pred & gt))
        fp += int(np.count_nonzero(pred & ~gt))
        fn += int(np.count_nonzero(~pred & gt))
        tn += int(np.count_nonzero(~pred & ~gt))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return f1, precision, recall, iou, tp + fn, fp + tn


def compute_pixel_metrics(
    scores: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    threshold: float,
    threshold_source: str,
    bins: int = 1 << 16,
) -> PixelMetrics:
    f1, precision, recall, iou, n_pos, n_neg = segmentation_metrics(
        scores, masks, threshold=threshold, threshold_source=threshold_source
    )
    return PixelMetrics(
        auroc=pixel_auroc(scores, masks, bins=bins),
        seg_f1=f1,
        seg_precision=precision,
        seg_recall=recall,
        iou=iou,
        threshold=float(threshold),
        threshold_source=threshold_source,
        n_positive=n_pos,
        n_negative=n_neg,
    )
