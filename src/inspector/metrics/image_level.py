"""Image-level detection metrics.

The pair that matters operationally is `fpr_at_threshold` / `recall_at_threshold`
evaluated at a *validation-derived* threshold. AUROC says the model can rank;
those two say whether it can be deployed at an alarm rate somebody has to staff.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class ImageMetrics:
    """Image-level results at one operating point."""

    auroc: float
    aupr: float
    f1_max_oracle: float
    threshold_f1_max_oracle: float
    threshold: float
    threshold_source: str
    fpr: float
    recall: float
    precision: float
    f1: float
    n_normal: int
    n_anomalous: int

    def as_dict(self) -> dict[str, float | str | int]:
        return asdict(self)


def _validate(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape != scores.shape:
        raise ValueError(f"labels {labels.shape} and scores {scores.shape} differ in length")
    if labels.size == 0:
        raise ValueError("empty input")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 (normal) or 1 (anomalous)")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or inf")
    return labels, scores


def image_auroc(labels, scores) -> float:
    labels, scores = _validate(labels, scores)
    if len(np.unique(labels)) < 2:
        raise ValueError("AUROC needs both classes present")
    return float(roc_auc_score(labels, scores))


def image_aupr(labels, scores) -> float:
    """Average precision. Honest under the class imbalance of a real line."""
    labels, scores = _validate(labels, scores)
    return float(average_precision_score(labels, scores))


def f1_max(labels, scores) -> tuple[float, float]:
    """Best achievable F1 over all thresholds, and the threshold achieving it.

    **Oracle**: it reads test labels. Reported only as an upper bound and always
    labelled as such (protocol §4.4, OP-F1MAX).
    """
    labels, scores = _validate(labels, scores)
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    s = scores[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    fn = y.sum() - tp
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)

    best = int(np.argmax(f1))
    return float(f1[best]), float(s[best])


def fpr_at_threshold(labels, scores, threshold: float) -> float:
    """Realized false-positive rate on normals at a fixed threshold."""
    labels, scores = _validate(labels, scores)
    normals = scores[labels == 0]
    if normals.size == 0:
        return float("nan")
    return float(np.mean(normals >= threshold))


def recall_at_threshold(labels, scores, threshold: float) -> float:
    """Defect catch rate at a fixed threshold. 1 - recall is the escape rate."""
    labels, scores = _validate(labels, scores)
    anomalous = scores[labels == 1]
    if anomalous.size == 0:
        return float("nan")
    return float(np.mean(anomalous >= threshold))


def compute_image_metrics(
    labels, scores, *, threshold: float, threshold_source: str
) -> ImageMetrics:
    """All image-level metrics at one operating point.

    `threshold_source` is mandatory and is carried into the results table. A
    threshold whose provenance nobody recorded is the single most common
    production failure in anomaly detection.
    """
    labels, scores = _validate(labels, scores)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    best_f1, best_t = f1_max(labels, scores)

    return ImageMetrics(
        auroc=image_auroc(labels, scores) if n_pos and n_neg else float("nan"),
        aupr=image_aupr(labels, scores),
        f1_max_oracle=best_f1,
        threshold_f1_max_oracle=best_t,
        threshold=float(threshold),
        threshold_source=threshold_source,
        fpr=fpr_at_threshold(labels, scores, threshold),
        recall=recall,
        precision=precision,
        f1=f1,
        n_normal=n_neg,
        n_anomalous=n_pos,
    )


def escape_rate_by_defect(
    labels, scores, defect_types, threshold: float
) -> dict[str, float]:
    """Per-defect-type miss rate at a fixed threshold.

    A scalar recall hides that one defect class is missed entirely, which is
    precisely what a process engineer needs to know.
    """
    labels, scores = _validate(labels, scores)
    defect_types = np.asarray(defect_types).ravel()
    out: dict[str, float] = {}
    for dtype in sorted(set(defect_types[labels == 1].tolist())):
        sel = (labels == 1) & (defect_types == dtype)
        out[dtype] = float(np.mean(scores[sel] < threshold))
    return out
