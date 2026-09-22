"""Per-Region Overlap (PRO) and AU-PRO.

PRO averages overlap **per connected defect region**, so a 40-pixel scratch
counts as much as a 40 000-pixel stain. That is the whole reason it is the
primary localization metric here: pixel AUROC is dominated by whichever defect
happens to be largest, and by the overwhelming normal-pixel majority.

    PRO(t) = mean over regions R of  |{p in R : score(p) >= t}| / |R|
    FPR(t) = |{p in N : score(p) >= t}| / |N|,  N = all pixels outside any region
    AU-PRO@L = (1/L) * integral_0^L PRO(FPR) d(FPR)

Two conventions differ between public implementations and are worth stating
explicitly, because mixing them moves the number by a few points:

1. **Which pixels count as negatives.** Here: every pixel not inside a
   ground-truth region, including all pixels of normal images. Normal images are
   the main source of false positives in deployment, so excluding them would
   flatter the metric exactly where it matters.
2. **Integration limit.** MVTec AD 2 uses L = 0.05; the classic MVTec AD
   literature uses L = 0.30. Both are supported and the limit is always carried
   in the result, so a 0.05 number can never be printed in a 0.30 column.

Resolved for OD-3 in docs/07 at Gate G2 by comparing against a reference
implementation; see tests/metrics/test_aupro.py.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# 8-connectivity: a diagonal chain of pixels is one scratch, not several.
_CONNECTIVITY = np.ones((3, 3), dtype=int)


@dataclass(frozen=True)
class PROCurve:
    """The sampled PRO-vs-FPR curve and its integral."""

    fpr: np.ndarray
    pro: np.ndarray
    thresholds: np.ndarray
    integration_limit: float
    n_regions: int
    n_negative_pixels: int

    @property
    def au_pro(self) -> float:
        """Normalized area under the curve, in [0, 1]."""
        return float(_trapz_to_limit(self.fpr, self.pro, self.integration_limit))


def _trapz_to_limit(fpr: np.ndarray, pro: np.ndarray, limit: float) -> float:
    """Trapezoidal integral of pro over fpr on [0, limit], divided by limit.

    The curve is interpolated at exactly `limit` rather than truncated at the
    nearest sample: truncating biases the result downward by up to a whole
    sampling interval, which at L=0.05 is not a rounding difference.
    """
    order = np.argsort(fpr, kind="stable")
    fpr, pro = fpr[order], pro[order]

    keep = fpr <= limit
    x = np.concatenate(([0.0], fpr[keep]))
    y = np.concatenate(([0.0], pro[keep]))

    if np.any(~keep):
        i = int(np.argmax(~keep))  # first sample beyond the limit
        x0, x1 = (fpr[i - 1], fpr[i]) if i > 0 else (0.0, fpr[i])
        y0, y1 = (pro[i - 1], pro[i]) if i > 0 else (0.0, pro[i])
        frac = 0.0 if x1 == x0 else (limit - x0) / (x1 - x0)
        x = np.concatenate((x, [limit]))
        y = np.concatenate((y, [y0 + frac * (y1 - y0)]))
    elif x[-1] < limit:
        # Curve never reached the limit: hold the final value flat. This is the
        # saturated case (every negative pixel already scored above threshold).
        x = np.concatenate((x, [limit]))
        y = np.concatenate((y, [y[-1]]))

    return float(np.trapezoid(y, x) / limit)


def _region_scores(
    score: np.ndarray, mask: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    """Split one image's scores into per-region sorted arrays and negatives."""
    labelled, n = ndimage.label(mask, structure=_CONNECTIVITY)
    regions = []
    for region_id in range(1, n + 1):
        values = score[labelled == region_id]
        if values.size:
            regions.append(np.sort(values))
    return regions, score[~mask]


def pro_curve(
    scores: Sequence[np.ndarray] | Iterable[np.ndarray],
    masks: Sequence[np.ndarray] | Iterable[np.ndarray],
    *,
    integration_limit: float = 0.05,
    num_thresholds: int | None = 512,
    max_negative_samples: int | None = 2_000_000,
    seed: int = 0,
) -> PROCurve:
    """Compute the PRO curve.

    Args:
        scores: per-image anomaly maps, float, higher = more anomalous. Must be
            at the image's native resolution (protocol §4.1).
        masks: per-image binary ground truth, same shapes. Normal images pass an
            all-zero mask and contribute only negatives.
        integration_limit: FPR ceiling, 0.05 for AD 2, 0.30 for classic AD.
        num_thresholds: how many points to sample the curve at. Thresholds are
            placed at quantiles of the *negative* score distribution so the FPR
            axis is sampled uniformly on [0, limit] — sampling uniformly in
            score space instead would put almost every point outside the
            integration region and waste the budget. `None` uses every distinct
            score, which is exact but only tractable on small inputs.
        max_negative_samples: cap on negative pixels retained for the FPR axis.
            At 150 images of 8 MP there are ~1.2e9 negatives; a uniform random
            subsample estimates the same quantiles within noise at a fraction of
            the memory. Set None to disable.
        seed: for the negative subsample.

    Returns:
        A `PROCurve`; `.au_pro` is the normalized integral.
    """
    if not 0.0 < integration_limit <= 1.0:
        raise ValueError(f"integration_limit must be in (0, 1], got {integration_limit}")

    regions: list[np.ndarray] = []
    negative_chunks: list[np.ndarray] = []
    n_negatives_total = 0

    for score, mask in zip(scores, masks):
        score = np.asarray(score, dtype=np.float64)
        mask = np.asarray(mask).astype(bool)
        if score.shape != mask.shape:
            raise ValueError(f"score shape {score.shape} != mask shape {mask.shape}")
        img_regions, negatives = _region_scores(score, mask)
        regions.extend(img_regions)
        n_negatives_total += negatives.size
        negative_chunks.append(negatives)

    if not regions:
        raise ValueError("no ground-truth regions found: AU-PRO is undefined")
    if n_negatives_total == 0:
        raise ValueError("no negative pixels found: FPR is undefined")

    negatives = np.concatenate(negative_chunks)
    del negative_chunks
    if max_negative_samples is not None and negatives.size > max_negative_samples:
        rng = np.random.default_rng(seed)
        negatives = rng.choice(negatives, size=max_negative_samples, replace=False)
    negatives.sort()

    thresholds = _pick_thresholds(
        negatives, regions, integration_limit, num_thresholds
    )

    n_neg = negatives.size
    fpr = np.empty(thresholds.size)
    pro = np.empty(thresholds.size)
    region_sizes = np.array([r.size for r in regions], dtype=np.float64)

    for i, t in enumerate(thresholds):
        # count of values >= t  ==  n - searchsorted(sorted_values, t, 'left')
        fpr[i] = (n_neg - np.searchsorted(negatives, t, side="left")) / n_neg
        overlaps = np.fromiter(
            (r.size - np.searchsorted(r, t, side="left") for r in regions),
            dtype=np.float64,
            count=len(regions),
        )
        pro[i] = float(np.mean(overlaps / region_sizes))

    return PROCurve(
        fpr=fpr,
        pro=pro,
        thresholds=thresholds,
        integration_limit=integration_limit,
        n_regions=len(regions),
        n_negative_pixels=n_neg,
    )


def _pick_thresholds(
    negatives: np.ndarray,
    regions: list[np.ndarray],
    limit: float,
    num_thresholds: int | None,
) -> np.ndarray:
    """Threshold grid, dense where the integration happens."""
    if num_thresholds is None:
        allv = np.concatenate([negatives, *regions])
        return np.unique(allv)[::-1]

    # Quantiles of the negatives corresponding to FPR in [0, limit], plus a
    # little headroom so the curve brackets the limit for interpolation.
    fpr_targets = np.linspace(0.0, min(1.0, limit * 1.2), num_thresholds)
    quantiles = np.clip(1.0 - fpr_targets, 0.0, 1.0)
    thresholds = np.quantile(negatives, quantiles)

    # A threshold strictly above every score yields FPR=0, anchoring the curve.
    top = max(float(negatives[-1]), max(float(r[-1]) for r in regions))
    thresholds = np.concatenate(([np.nextafter(top, np.inf)], thresholds))
    return np.unique(thresholds)[::-1]


def au_pro(
    scores: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    integration_limit: float = 0.05,
    num_thresholds: int | None = 512,
    max_negative_samples: int | None = 2_000_000,
    seed: int = 0,
) -> float:
    """Normalized AU-PRO. Convenience wrapper over `pro_curve`."""
    return pro_curve(
        scores,
        masks,
        integration_limit=integration_limit,
        num_thresholds=num_thresholds,
        max_negative_samples=max_negative_samples,
        seed=seed,
    ).au_pro
