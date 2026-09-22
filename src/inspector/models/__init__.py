"""Anomaly models, tier 0 upward (docs/03)."""

from .base import AnomalyModel, FitRecord, Prediction
from .trivial import (
    TIER0_MODELS,
    HistogramScorer,
    MeanIntensityScorer,
    PixelPCAScorer,
    RandomScorer,
)

__all__ = [
    "TIER0_MODELS",
    "AnomalyModel",
    "FitRecord",
    "HistogramScorer",
    "MeanIntensityScorer",
    "PixelPCAScorer",
    "Prediction",
    "RandomScorer",
]
