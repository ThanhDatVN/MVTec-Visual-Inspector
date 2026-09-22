"""Anomaly models, tier 0 upward (docs/03).

Tier 0-2 are torch-free; `patchcore` and `autoencoder` import torch lazily so
this package stays importable in the CI path.
"""

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
    "build_model",
]

#: Models that require torch, imported on demand.
TORCH_MODELS = ("patchcore", "cae")


def build_model(name: str, transform=None, **kwargs) -> AnomalyModel:
    """Construct a model by name, importing torch only when one needs it."""
    if name in TIER0_MODELS:
        cls = TIER0_MODELS[name]
        if not cls.stochastic:
            kwargs.pop("seed", None)
        return cls(transform, **kwargs)
    if name == "patchcore":
        from .patchcore import PatchCore

        return PatchCore(transform, **kwargs)
    if name == "cae":
        from .autoencoder import ConvAutoencoder

        return ConvAutoencoder(transform, **kwargs)
    raise KeyError(
        f"unknown model {name!r}; available: {sorted(TIER0_MODELS) + list(TORCH_MODELS)}"
    )
