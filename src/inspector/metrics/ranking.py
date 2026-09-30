"""Exact ROC-AUC and average precision, without scikit-learn.

scikit-learn was the only dependency in the torch-free core, and it dragged in
pyarrow, whose DLL Windows Smart App Control blocks on this machine. sklearn's
own guard is `except ModuleNotFoundError`, which does not catch the
`ImportError` a blocked DLL raises, so the whole test suite became unimportable
because of a package nothing here uses.

Two functions did not justify that surface. Both are implemented directly and
both are **exact**, including the tie handling that a naive implementation gets
wrong:

* AUROC via the Mann-Whitney U statistic on mid-ranks. A tied pair contributes
  exactly 0.5, which is why an all-tied scorer scores 0.5 rather than 1.0.
* Average precision as the step-wise sum used by `sklearn.average_precision_score`,
  not the interpolated area under the PR curve — the two differ, and mixing them
  is a reporting bug.

Validated against the analytic cases in `tests/metrics/`, and cross-checked
against the histogram pixel AUROC in `pixel_level.py`, which was itself verified
against sklearn to 1e-4 while sklearn was still importable.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve for binary labels.

    Uses the Mann-Whitney identity: AUROC is the probability that a randomly
    chosen positive outranks a randomly chosen negative, with ties counted as
    half. Computing it from mid-ranks makes that exact in one pass, rather than
    approximating it by sweeping thresholds.
    """
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC needs both classes present")

    # `rankdata` assigns the mean rank to tied values, which is precisely the
    # half-credit rule the metric requires.
    ranks = rankdata(scores, method="average")
    rank_sum_pos = float(ranks[labels == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision: sum of precision at each threshold, weighted by the
    recall gained there.

    This is the step-wise definition, matching
    `sklearn.metrics.average_precision_score`. It is deliberately *not* the
    interpolated area under the precision-recall curve: the interpolated version
    is optimistically biased, and the two names are used interchangeably in the
    literature often enough that the choice has to be stated.

    Tied scores are handled by advancing through the whole tie group before
    recording a point; otherwise the ordering within a tie would change the
    result.
    """
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)

    # Keep only the last index of each tie group: a threshold cannot separate
    # equal scores, so intermediate positions inside a group are not reachable.
    distinct = np.r_[np.nonzero(np.diff(sorted_scores))[0], sorted_labels.size - 1]
    tp, fp = tp[distinct], fp[distinct]

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_gain = np.diff(np.r_[0.0, recall])
    return float(np.sum(precision * recall_gain))


def pca_fit(matrix: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, float]:
    """PCA by SVD on the centred matrix.

    Returns `(mean, components, explained_variance_ratio)` where `components`
    has shape (k, n_features). Replaces `sklearn.decomposition.PCA` for the
    Tier 0 pixel baseline; `full_matrices=False` keeps the decomposition to the
    rank the data actually has.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")

    n_samples, n_features = matrix.shape
    k = max(1, min(n_components, n_samples - 1, n_features))

    mean = matrix.mean(axis=0)
    centred = matrix - mean
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)

    variance = singular**2
    total = float(variance.sum())
    ratio = float(variance[:k].sum() / total) if total > 0 else 0.0
    return mean, vt[:k], ratio


def pca_reconstruct(
    vector: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """Project onto the principal subspace and back."""
    centred = np.asarray(vector, dtype=np.float64) - mean
    return mean + (centred @ components.T) @ components


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of y over x.

    Written out rather than calling `np.trapezoid`, which exists only from NumPy
    2.0; hosted notebook images (Kaggle) may still ship NumPy 1.26.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])) / 2.0)
