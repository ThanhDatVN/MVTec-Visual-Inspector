"""Metric implementations.

Validated at Gate G2 against a reference implementation before any model number
is trusted. See docs/02-metrics-and-baselines.md for definitions and for which
metric decides what.
"""

from .aupro import PROCurve, au_pro, pro_curve
from .image_level import (
    ImageMetrics,
    compute_image_metrics,
    escape_rate_by_defect,
    f1_max,
    fpr_at_threshold,
    image_aupr,
    image_auroc,
    recall_at_threshold,
)
from .pixel_level import PixelMetrics, compute_pixel_metrics, pixel_auroc, segmentation_metrics

__all__ = [
    "ImageMetrics",
    "PROCurve",
    "PixelMetrics",
    "au_pro",
    "compute_image_metrics",
    "compute_pixel_metrics",
    "escape_rate_by_defect",
    "f1_max",
    "fpr_at_threshold",
    "image_aupr",
    "image_auroc",
    "pixel_auroc",
    "pro_curve",
    "recall_at_threshold",
    "segmentation_metrics",
]
