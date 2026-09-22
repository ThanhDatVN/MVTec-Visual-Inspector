"""Tier 0 — trivial floors (docs/03).

These four models answer two questions that must be settled before any
respectable method is run:

1. **Is my evaluation code correct?** `RandomScorer` must land at AUROC 0.5. If
   it does not, the metric is broken and every number downstream is wrong. This
   is the cheapest insurance in the project.
2. **Is this category trivially separable?** If a global intensity statistic
   already separates normal from defective, then a deep model's success on that
   category measures the lighting, not the method — and any paper-level claim
   built on it is void.

They cost an afternoon and they are the floor row of every results table.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ..data.transforms import ImageTransform
from .base import AnomalyModel


class RandomScorer(AnomalyModel):
    """Uniform random scores. The control.

    Its purpose is to fail: image AUROC must be ~0.5 and AU-PRO@L must be ~L/2.
    A deviation means the metric implementation is wrong, not that random
    guessing works.
    """

    name = "random"
    tier = "T0"
    stochastic = True

    def __init__(self, transform: ImageTransform | None = None, *, seed: int = 0, **kwargs):
        super().__init__(transform, **kwargs)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _fit(self, images: Iterator[np.ndarray]) -> None:
        # Consume the iterator so timing and IO are comparable with real models.
        self._n_train = sum(1 for _ in images)

    def _score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        _, h, w = image.shape
        return float(self._rng.random()), self._rng.random((h, w))

    def fit_extra(self) -> dict[str, Any]:
        return {"seed": self.seed}


class MeanIntensityScorer(AnomalyModel):
    """Gaussian model of the global mean intensity. Four lines of statistics.

    Score is the absolute z-score of the image mean against the training
    distribution. If this separates a category, the category is separable by
    exposure alone and that fact belongs in the report in bold.
    """

    name = "mean_intensity"
    tier = "T0"

    def _fit(self, images: Iterator[np.ndarray]) -> None:
        means = [float(image.mean()) for image in images]
        if not means:
            raise ValueError("no training images")
        self._mean = float(np.mean(means))
        # ddof=1 and a floor: with a handful of images the sample std is a poor
        # estimate, and a zero std would make every z-score infinite.
        self._std = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
        self._std = max(self._std, 1e-6)

    def _score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        z = abs(float(image.mean()) - self._mean) / self._std
        _, h, w = image.shape
        # A global statistic carries no localization information, so the map is
        # constant. Reporting it honestly makes the AU-PRO of this row ~L/2 and
        # shows what a detector without localization is worth.
        return z, np.full((h, w), z, dtype=np.float64)

    def fit_extra(self) -> dict[str, Any]:
        return {"train_mean": self._mean, "train_std": self._std}


class HistogramScorer(AnomalyModel):
    """Chi-square distance between an image's intensity histogram and the mean
    training histogram.

    One step above the global mean: it sees the *distribution* of intensities,
    so it detects a defect that redistributes brightness without changing the
    average. Still global, so still no localization.
    """

    name = "histogram"
    tier = "T0"

    def __init__(self, transform: ImageTransform | None = None, *, bins: int = 64, **kwargs):
        super().__init__(transform, **kwargs)
        self.bins = bins

    def _histogram(self, image: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(image.ravel(), bins=self.bins, range=(-3.0, 3.0))
        total = counts.sum()
        return counts / total if total else counts.astype(np.float64)

    def _fit(self, images: Iterator[np.ndarray]) -> None:
        histograms = [self._histogram(image) for image in images]
        if not histograms:
            raise ValueError("no training images")
        self._reference = np.mean(histograms, axis=0)

    def _score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        observed = self._histogram(image)
        denom = observed + self._reference
        chi2 = 0.5 * np.sum(
            np.where(denom > 0, (observed - self._reference) ** 2 / np.maximum(denom, 1e-12), 0.0)
        )
        _, h, w = image.shape
        return float(chi2), np.full((h, w), float(chi2), dtype=np.float64)

    def fit_extra(self) -> dict[str, Any]:
        return {"bins": self.bins}


class PixelPCAScorer(AnomalyModel):
    """PCA on raw downscaled pixels; score is the reconstruction residual.

    The simplest model in the ladder that produces a *spatial* map, and so the
    first row whose AU-PRO means anything. It is also the honest floor for the
    autoencoder in Tier 1: a conv AE that cannot beat linear PCA on raw pixels
    has learned nothing a linear projection did not already capture, and that
    comparison is the reason this model exists.
    """

    name = "pixel_pca"
    tier = "T0"

    def __init__(
        self,
        transform: ImageTransform | None = None,
        *,
        n_components: int = 16,
        work_size: int = 64,
        **kwargs,
    ):
        super().__init__(transform, **kwargs)
        self.n_components = n_components
        self.work_size = work_size

    def _flatten(self, image: np.ndarray) -> np.ndarray:
        """Downscale to a small grid and flatten. PCA on full-resolution pixels
        would need a covariance the size of the image squared."""
        import cv2

        gray = image.mean(axis=0).astype(np.float32)
        small = cv2.resize(gray, (self.work_size, self.work_size), interpolation=cv2.INTER_AREA)
        return small.ravel()

    def _fit(self, images: Iterator[np.ndarray]) -> None:
        from sklearn.decomposition import PCA

        matrix = np.stack([self._flatten(image) for image in images])
        if matrix.shape[0] < 2:
            raise ValueError("PCA needs at least 2 training images")
        # Cannot ask for more components than samples or features.
        k = min(self.n_components, matrix.shape[0] - 1, matrix.shape[1])
        self._pca = PCA(n_components=k, svd_solver="full", random_state=0).fit(matrix)
        self._k = k

    def _score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        flat = self._flatten(image)[None, :]
        reconstructed = self._pca.inverse_transform(self._pca.transform(flat))
        residual = np.abs(flat - reconstructed).reshape(self.work_size, self.work_size)
        return float(residual.mean()), residual.astype(np.float64)

    def fit_extra(self) -> dict[str, Any]:
        return {
            "n_components": self._k,
            "explained_variance_ratio": float(self._pca.explained_variance_ratio_.sum()),
        }


TIER0_MODELS: dict[str, type[AnomalyModel]] = {
    "random": RandomScorer,
    "mean_intensity": MeanIntensityScorer,
    "histogram": HistogramScorer,
    "pixel_pca": PixelPCAScorer,
}
