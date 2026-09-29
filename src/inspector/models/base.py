"""The anomaly-model interface.

Every model in the ladder — from the four-line intensity baseline to PatchCore —
implements two methods: `_fit` over normal images, and `_score` for one image.
Everything the protocol requires around those two is handled here, once:

* **Rule L2** — `fit` refuses an index whose split is not a fit split. A model
  cannot accidentally be trained on test data, because the check is in the base
  class rather than in each model's own code.
* **§4.1 native-resolution scoring** — subclasses return a map at whatever
  resolution they work in. `postprocess_map` upsamples it to the model's input
  geometry, smooths it there, and upsamples to native. A model that forgot to
  upsample would report an inflated AU-PRO, because a defect occupies
  proportionally more of a smaller image.
* **Smoothing has units.** `smoothing_sigma` is in *input-image* pixels, as in
  anomalib's PatchCore. The previous implementation smoothed at native
  resolution, so sigma=4 meant a filter ~4x narrower than the reference at
  320 px input on a 1404 px image (docs/13, E05).
* **Provenance** — the fitted model records what it was fitted on, so a result
  can always be traced back to exact bytes.

Subclasses therefore cannot get the protocol wrong by omission; they can only
get their own modelling wrong, which is the point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.core import DatasetIndex, Sample
from ..data.transforms import ImageTransform, load_image, smooth_map, upsample_map


@dataclass(frozen=True)
class Prediction:
    """One image's result: a scalar score and a native-resolution map."""

    score: float
    anomaly_map: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.score):
            raise ValueError(f"anomaly score must be finite, got {self.score}")


@dataclass(frozen=True)
class RawPrediction:
    """What a model emits before any post-processing.

    Stored instead of the native map so that smoothing, aggregation and
    calibration studies re-evaluate cached predictions without refitting — the
    native map of one VisA image is 1.5 M floats, the raw PatchCore map 1,200.
    """

    score: float
    raw_map: np.ndarray
    input_size: tuple[int, int]  # (width, height) the model saw
    native_size: tuple[int, int]  # (width, height) of the source image


def postprocess_map(
    raw_map: np.ndarray,
    *,
    input_size: tuple[int, int],
    native_size: tuple[int, int],
    sigma: float,
) -> np.ndarray:
    """Raw map -> input geometry -> Gaussian(sigma, input px) -> native geometry."""
    anomaly_map = np.asarray(raw_map, dtype=np.float64)
    if anomaly_map.ndim != 2:
        raise ValueError(f"expected a 2-d map, got {anomaly_map.ndim}-d")
    anomaly_map = upsample_map(anomaly_map, input_size)
    if sigma > 0:
        anomaly_map = smooth_map(anomaly_map, sigma)
    return upsample_map(anomaly_map, native_size)


@dataclass
class FitRecord:
    """What a fitted model was fitted on."""

    split: str
    category: str
    n_images: int
    manifest_sha256: str
    transform: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class AnomalyModel(ABC):
    """Base class for every one-class anomaly detector in the project."""

    #: Short identifier used in run names and the results table.
    name: str = "unnamed"
    #: Tier in the method ladder (docs/03).
    tier: str = "T?"
    #: `own` for our implementations, `lib` for third-party comparators.
    owner: str = "own"
    #: Whether fitting is stochastic; drives the seeds requirement (§4.2).
    stochastic: bool = False

    def __init__(self, transform: ImageTransform | None = None, *, smoothing_sigma: float = 0.0):
        self.transform = transform or ImageTransform()
        self.smoothing_sigma = smoothing_sigma
        self._fitted = False
        self.fit_record: FitRecord | None = None

    # -- subclass contract -------------------------------------------------
    @abstractmethod
    def _fit(self, images: Iterator[np.ndarray]) -> None:
        """Fit on preprocessed normal images (float32 CxHxW, already resized)."""

    @abstractmethod
    def _score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        """Score one preprocessed image.

        Returns `(image_score, anomaly_map)` where the map may be at any
        resolution; the base class upsamples it to native.
        """

    # -- lifecycle ---------------------------------------------------------
    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, index: DatasetIndex, *, allow_splits: tuple[str, ...] = ()) -> AnomalyModel:
        """Fit on a normal-only split.

        Raises if the split is not a fit split (rule L2) or if it contains
        anomalous samples — a one-class model fitted on labelled defects is no
        longer solving the stated problem.
        """
        permitted = allow_splits or index.layout.fit_splits
        if index.split_name not in permitted:
            raise ValueError(
                f"refusing to fit on split {index.split_name!r}; permitted: {permitted} "
                "(protocol rule L2)"
            )
        if index.n_anomalous:
            raise ValueError(
                f"split {index.split_name!r} contains {index.n_anomalous} anomalous samples; "
                "one-class fitting requires normal-only data"
            )
        if len(index) == 0:
            raise ValueError(f"split {index.split_name!r} is empty")

        self._fit(self._iter_preprocessed(index))
        self._fitted = True
        self.fit_record = FitRecord(
            split=index.split_name,
            category=index.category,
            n_images=len(index),
            manifest_sha256=str(index.manifest()["manifest_sha256"]),
            transform=self.transform.as_dict(),
            extra=self.fit_extra(),
        )
        return self

    def fit_extra(self) -> dict[str, Any]:
        """Model-specific facts worth recording (memory-bank size, epochs run)."""
        return {}

    def hparams(self) -> dict[str, Any]:
        """Every constructor setting that changes behaviour.

        Part of the run identity: a run whose hash omitted a default could be
        silently reused after that default changed (docs/13, F04).
        """
        return {}

    def _iter_preprocessed(self, index: DatasetIndex) -> Iterator[np.ndarray]:
        for sample in index:
            image = load_image(sample.image_path)
            yield self.transform.normalize_image(self.transform.resize_image(image))

    # -- inference ---------------------------------------------------------
    def predict_raw(self, image: np.ndarray) -> RawPrediction:
        """Score a native-resolution HxWx3 uint8 image; no post-processing."""
        if not self._fitted:
            raise RuntimeError(f"{self.name} is not fitted")

        native_h, native_w = image.shape[:2]
        resized = self.transform.resize_image(image)
        input_h, input_w = resized.shape[:2]
        score, raw_map = self._score(self.transform.normalize_image(resized))

        raw = np.asarray(raw_map, dtype=np.float32)
        if raw.ndim != 2:
            raise ValueError(f"{self.name} returned a {raw.ndim}-d map; expected 2-d")
        if not np.isfinite(score):
            raise ValueError(f"{self.name} produced a non-finite image score: {score}")
        return RawPrediction(
            score=float(score),
            raw_map=raw,
            input_size=(input_w, input_h),
            native_size=(native_w, native_h),
        )

    def predict_image(self, image: np.ndarray) -> Prediction:
        """Score a native-resolution image and return a post-processed native map."""
        raw = self.predict_raw(image)
        anomaly_map = postprocess_map(
            raw.raw_map,
            input_size=raw.input_size,
            native_size=raw.native_size,
            sigma=self.smoothing_sigma,
        )
        return Prediction(score=raw.score, anomaly_map=anomaly_map)

    def predict_sample(self, sample: Sample) -> Prediction:
        return self.predict_image(load_image(sample.image_path))

    def predict_index(self, index: DatasetIndex) -> Iterator[tuple[Sample, Prediction]]:
        """Score a whole split, streaming so multi-megapixel maps are never all
        resident at once."""
        for sample in index:
            yield sample, self.predict_sample(sample)

    def image_scores(self, index: DatasetIndex) -> np.ndarray:
        """Just the scalar scores — the cheap path for threshold selection."""
        return np.array([p.score for _, p in self.predict_index(index)], dtype=np.float64)

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not implement state_dict")

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return f"{type(self).__name__}(name={self.name!r}, tier={self.tier!r}, {state})"
