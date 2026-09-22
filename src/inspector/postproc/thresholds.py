"""Thresholds and score normalization, with provenance attached.

Protocol §4.4 defines three operating points, and rules L3/L4 say where their
parameters may come from. Both are enforced by construction here: a `Threshold`
cannot exist without recording which split produced it, and `ScoreNormalizer`
refuses to fit twice. Making the invariant structural beats documenting it,
because the failure it prevents — a threshold or a normalizer quietly fitted on
test data — produces better numbers, not an error, and so is never noticed.
"""

from __future__ import annotations

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


def from_validation_percentile(
    validation_scores: np.ndarray,
    *,
    percentile: float = 99.0,
    split_name: str = "validation",
) -> Threshold:
    """`OP-FPR1`: the p-th percentile of scores over normal validation images.

    Targets a (100 - p)% false-alarm rate on normal parts. The *realized* FPR on
    test normals is reported separately, and the gap between the two is the
    calibration error — usually the more interesting number.
    """
    scores = np.asarray(validation_scores, dtype=np.float64).ravel()
    if scores.size < 4:
        raise ValueError(
            f"need at least 4 validation scores to estimate a percentile, got {scores.size}"
        )
    return Threshold(
        value=float(np.percentile(scores, percentile)),
        method="percentile",
        source_split=split_name,
        params={"percentile": float(percentile), "n_validation": float(scores.size)},
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
