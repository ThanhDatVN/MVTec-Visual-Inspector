"""Thresholds and score normalization, with provenance attached.

Protocol §4.4 defines three operating points, and rules L3/L4 say where their
parameters may come from. Both are enforced by construction here: a `Threshold`
cannot exist without recording which split produced it, and `ScoreNormalizer`
refuses to fit twice. Making the invariant structural beats documenting it,
because the failure it prevents — a threshold or a normalizer quietly fitted on
test data — produces better numbers, not an error, and so is never noticed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

ThresholdMethod = Literal["percentile", "sigma", "f1_max", "manual"]


@dataclass(frozen=True)
class Threshold:
    """A decision threshold that knows where it came from."""

    value: float
    method: ThresholdMethod
    source_split: str
    oracle: bool = False
    #: Free-form detail, e.g. {"percentile": 99} or {"n_sigma": 3}.
    params: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError(f"threshold must be finite, got {self.value}")
        if self.oracle and self.source_split == "validation":
            raise ValueError("a validation-derived threshold is not an oracle")
        if not self.oracle and self.source_split.startswith("test"):
            raise ValueError(
                f"threshold derived from {self.source_split!r} must be marked oracle=True "
                "(protocol rule L4)"
            )

    @property
    def label(self) -> str:
        return f"{self.method}@{self.source_split}" + (" (oracle)" if self.oracle else "")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def apply(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores) >= self.value


def min_achievable_fpr(n_validation: int) -> float:
    """The lowest false-alarm rate calibratable from `n` normal validation images.

    Distribution-free result: if the threshold is the k-th largest of n
    exchangeable validation scores, a fresh normal sample exceeds it with
    probability k/(n+1). The most extreme choice is k = 1, the sample maximum,
    which gives 1/(n+1).

    This has sharp consequences for MVTec AD 2, whose validation splits hold
    19-48 images:

        sheet_metal  n=19  ->  minimum achievable FPR 5.0%
        fruit_jelly  n=37  ->  2.6%
        walnuts      n=48  ->  2.0%

    So `OP-FPR1`'s nominal 1% target is **not reachable** on any of the three
    study categories. Asking numpy for the 99th percentile of 19 numbers does
    not fail; it returns a threshold that will produce roughly 5% false alarms
    while the report claims 1%. Naming the limit is what stops that.
    """
    if n_validation < 1:
        raise ValueError("need at least one validation score")
    return 1.0 / (n_validation + 1)


def from_validation_fpr(
    validation_scores: np.ndarray,
    *,
    target_fpr: float = 0.01,
    split_name: str = "validation",
    strict: bool = True,
) -> Threshold:
    """Threshold targeting `target_fpr` false alarms, via order statistics.

    Picks k = ceil(target_fpr * (n + 1)) and uses the k-th largest validation
    score. The exceedance probability of that order statistic is k/(n+1) for any
    continuous score distribution, so this is distribution-free — unlike a
    linearly-interpolated percentile, which assumes a tail shape the data cannot
    support at these sample sizes.

    Args:
        strict: if True, raise when `target_fpr` is below `min_achievable_fpr`.
            Silently clamping would mean the reported target and the achievable
            rate disagree, which is exactly the failure this function exists to
            prevent.
    """
    scores = np.asarray(validation_scores, dtype=np.float64).ravel()
    n = scores.size
    if n < 4:
        raise ValueError(
            f"need at least 4 validation scores to set an operating point, got {n}"
        )
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")

    floor = min_achievable_fpr(n)
    if target_fpr < floor:
        message = (
            f"target FPR {target_fpr:.1%} is below the {floor:.1%} floor achievable from "
            f"{n} validation images (1/(n+1)). Either accept {floor:.1%}, or obtain more "
            "normal validation images; no estimator can calibrate past this."
        )
        if strict:
            raise ValueError(message)
        target_fpr = floor

    k = max(1, math.ceil(target_fpr * (n + 1)))
    k = min(k, n)
    ordered = np.sort(scores)[::-1]  # descending
    return Threshold(
        value=float(ordered[k - 1]),
        method="percentile",
        source_split=split_name,
        params={
            "target_fpr": float(target_fpr),
            "expected_fpr": float(k / (n + 1)),
            "order_statistic_k": float(k),
            "n_validation": float(n),
            "min_achievable_fpr": float(floor),
        },
    )


def from_validation_percentile(
    validation_scores: np.ndarray,
    *,
    percentile: float = 99.0,
    split_name: str = "validation",
    strict: bool = False,
) -> Threshold:
    """`OP-FPR1` expressed as a percentile; delegates to `from_validation_fpr`.

    Kept because the protocol states the operating point as "the 99th percentile
    of validation scores", but implemented through the order-statistic rule so
    that the achievability floor is enforced rather than silently exceeded.
    """
    return from_validation_fpr(
        validation_scores,
        target_fpr=(100.0 - percentile) / 100.0,
        split_name=split_name,
        strict=strict,
    )


def from_validation_sigma(
    validation_maps: np.ndarray | list[np.ndarray],
    *,
    n_sigma: float = 3.0,
    split_name: str = "validation",
) -> Threshold:
    """`OP-3SIGMA`: mean + n*std of anomaly-map values over validation images.

    This is the MVTec AD 2 benchmark's own rule for the thresholded SegF1
    metric, reproduced exactly so our SegF1 stays comparable to the leaderboard.

    It assumes a roughly normal score distribution. For a heavy-tailed one it
    lands far from any useful operating point — the suspected mechanism behind
    PatchCore's ~3.7% SegF1 on AD 2 against its ~28.8% AU-PRO. Phase P5.8
    investigates; this function is the instrument, not the verdict.
    """
    if isinstance(validation_maps, list):
        flat = np.concatenate([np.asarray(m, dtype=np.float64).ravel() for m in validation_maps])
    else:
        flat = np.asarray(validation_maps, dtype=np.float64).ravel()
    if flat.size == 0:
        raise ValueError("no validation anomaly maps supplied")

    mean, std = float(flat.mean()), float(flat.std())
    return Threshold(
        value=mean + n_sigma * std,
        method="sigma",
        source_split=split_name,
        params={
            "n_sigma": float(n_sigma),
            "mean": mean,
            "std": std,
            "n_pixels": float(flat.size),
        },
    )


def sigma_from_stats(
    mean: float, std: float, *, n_sigma: float = 3.0, n_pixels: float = 0.0,
    split_name: str = "validation",
) -> Threshold:
    """`OP-3SIGMA` from pooled statistics computed in a streaming pass.

    Exists because the anomaly maps of a real validation split do not fit in
    memory: `PredictionStore.map_stats` accumulates mean and std over the maps
    one at a time, and this turns those two numbers into the threshold without
    ever materializing the pixels.
    """
    return Threshold(
        value=mean + n_sigma * std,
        method="sigma",
        source_split=split_name,
        params={"n_sigma": n_sigma, "mean": mean, "std": std, "n_pixels": n_pixels},
    )


def from_test_f1_max(labels, scores, *, split_name: str = "test_public") -> Threshold:
    """`OP-F1MAX`: the F1-maximizing threshold. **Oracle** — uses test labels.

    Reported only as an upper bound, and `Threshold.label` renders the
    "(oracle)" tag so it cannot be printed without the caveat.
    """
    from ..metrics.image_level import f1_max

    _, value = f1_max(labels, scores)
    return Threshold(
        value=value,
        method="f1_max",
        source_split=split_name,
        oracle=True,
        params={"note": float("nan")},
    )


class ScoreNormalizer:
    """Fit-once, then frozen min-max normalizer for anomaly maps (rule L3).

    Min-max normalizing an anomaly map over the *test* set is endemic in public
    repositories. It leaks the test distribution into every score and can move
    pixel AUROC by several points. This class fits on validation and then
    refuses to refit, so the leak cannot happen by accident.
    """

    def __init__(self) -> None:
        self._min: float | None = None
        self._max: float | None = None
        self._source_split: str | None = None

    @property
    def fitted(self) -> bool:
        return self._min is not None

    @property
    def source_split(self) -> str | None:
        return self._source_split

    @property
    def bounds(self) -> tuple[float, float]:
        if self._min is None or self._max is None:
            raise RuntimeError("normalizer is not fitted")
        return self._min, self._max

    def fit(self, maps: np.ndarray | list[np.ndarray], *, split_name: str) -> ScoreNormalizer:
        if self.fitted:
            raise RuntimeError(
                f"normalizer already fitted on {self._source_split!r}; refitting would "
                "let a later split influence the scale (protocol rule L3)"
            )
        if split_name.startswith("test"):
            raise ValueError(
                f"refusing to fit a normalizer on {split_name!r}: normalization statistics "
                "must come from validation (protocol rule L3)"
            )
        if isinstance(maps, list):
            flat = np.concatenate([np.asarray(m, dtype=np.float64).ravel() for m in maps])
        else:
            flat = np.asarray(maps, dtype=np.float64).ravel()
        if flat.size == 0:
            raise ValueError("no maps supplied")

        self._min, self._max = float(flat.min()), float(flat.max())
        self._source_split = split_name
        return self

    def fit_bounds(self, low: float, high: float, *, split_name: str) -> ScoreNormalizer:
        """Fit from bounds already computed in a streaming pass.

        Same rules as `fit`: validation only, once. The streaming path exists
        because a real validation split's anomaly maps do not fit in memory, and
        that must not become a reason to reach for the test split's statistics
        instead.
        """
        if self.fitted:
            raise RuntimeError(
                f"normalizer already fitted on {self._source_split!r}; refitting would "
                "let a later split influence the scale (protocol rule L3)"
            )
        if split_name.startswith("test"):
            raise ValueError(
                f"refusing to fit a normalizer on {split_name!r}: normalization statistics "
                "must come from validation (protocol rule L3)"
            )
        self._min, self._max = float(low), float(high)
        self._source_split = split_name
        return self

    def transform(self, maps):
        """Scale to [0, 1] using the frozen validation bounds.

        Values outside the validation range are *not* clipped: a test score
        above the validation maximum is exactly the signal an anomaly detector
        exists to produce, and clipping it to 1.0 would discard the ranking
        information among the most anomalous samples.
        """
        lo, hi = self.bounds
        span = hi - lo
        if span <= 0:
            raise RuntimeError("degenerate normalizer: validation maps were constant")
        if isinstance(maps, list):
            return [(np.asarray(m, dtype=np.float64) - lo) / span for m in maps]
        return (np.asarray(maps, dtype=np.float64) - lo) / span

    def state_dict(self) -> dict[str, object]:
        return {"min": self._min, "max": self._max, "source_split": self._source_split}

    @classmethod
    def from_state_dict(cls, state: dict) -> ScoreNormalizer:
        obj = cls()
        obj._min, obj._max = state["min"], state["max"]
        obj._source_split = state["source_split"]
        return obj
