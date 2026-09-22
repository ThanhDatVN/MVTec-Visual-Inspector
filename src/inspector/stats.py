"""Statistics for model comparison (docs/02 §1.4).

Decision rule DR-2: never declare a method better on a mean alone. A comparison
needs the paired test, the effect size, and the per-category table. These
helpers provide the first two; the results generator provides the third.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of a paired comparison between two models."""

    n: int
    median_difference: float
    mean_difference: float
    statistic: float
    p_value: float
    p_value_corrected: float | None
    test: str
    significant: bool

    def __str__(self) -> str:
        p = self.p_value_corrected if self.p_value_corrected is not None else self.p_value
        tag = "significant" if self.significant else "not significant"
        return (
            f"{self.test}: n={self.n}, median Δ={self.median_difference:+.4f}, "
            f"p={p:.4g} ({tag})"
        )


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic=np.mean,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI over images.

    Returns (point_estimate, low, high). Uses the percentile interval rather
    than BCa: with 80-150 test images the bias correction is itself noisy, and a
    plain percentile interval is the more honest summary at this sample size.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("empty input")
    point = float(statistic(values))
    if values.size == 1:
        return point, point, point

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    resampled = values[idx]

    # np.mean/np.median take an axis argument, so the common cases vectorize.
    # apply_along_axis is a Python-level loop and is ~50x slower here, which
    # matters when every results table triggers hundreds of these.
    if statistic in (np.mean, np.median):
        samples = statistic(resampled, axis=1)
    else:
        samples = np.apply_along_axis(statistic, 1, resampled)

    alpha = (1.0 - confidence) / 2.0
    return point, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def paired_wilcoxon(
    a: np.ndarray, b: np.ndarray, *, alpha: float = 0.05
) -> ComparisonResult:
    """Paired Wilcoxon signed-rank test on per-image scores.

    Paired by image, so it controls for the fact that some images are simply
    harder. Makes no normality assumption, which matters because per-image
    metric distributions are routinely skewed and bimodal.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired test needs equal lengths, got {a.shape} and {b.shape}")
    if a.size < 6:
        raise ValueError(
            f"paired Wilcoxon on {a.size} pairs has no useful power; "
            "report the raw differences instead of a p-value"
        )

    diff = a - b
    if np.allclose(diff, 0):
        return ComparisonResult(
            n=a.size,
            median_difference=0.0,
            mean_difference=0.0,
            statistic=float("nan"),
            p_value=1.0,
            p_value_corrected=None,
            test="wilcoxon-signed-rank",
            significant=False,
        )

    result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return ComparisonResult(
        n=a.size,
        median_difference=float(np.median(diff)),
        mean_difference=float(np.mean(diff)),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        p_value_corrected=None,
        test="wilcoxon-signed-rank",
        significant=bool(result.pvalue < alpha),
    )


def holm_bonferroni(p_values: list[float], *, alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Holm-Bonferroni step-down correction.

    Returns [(adjusted_p, reject), ...] in the input order.

    DR-2 requires this across the comparisons in one results table, and the
    family size must be stated. Running 30 ablations and reporting the one with
    p < 0.05 is not a result — it is the expected number of false positives.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])

    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)  # enforce monotonicity
        adjusted[idx] = running
    return [(adjusted[i], adjusted[i] < alpha) for i in range(m)]


def compare_models(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    family_size: int = 1,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Paired comparison with the correction applied for a family of `family_size`.

    `family_size` is the number of comparisons in the table this result will
    appear in — not the number you happened to find interesting afterwards.
    """
    result = paired_wilcoxon(scores_a, scores_b, alpha=alpha)
    corrected = min(1.0, result.p_value * family_size)
    return ComparisonResult(
        n=result.n,
        median_difference=result.median_difference,
        mean_difference=result.mean_difference,
        statistic=result.statistic,
        p_value=result.p_value,
        p_value_corrected=corrected,
        test=f"{result.test} (Holm-Bonferroni, family={family_size})",
        significant=bool(corrected < alpha),
    )
