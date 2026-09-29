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
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    family_size: int = 1,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Paired Wilcoxon with a **Bonferroni** adjustment for `family_size` tests.

    This multiplies one p-value by the family size. That is Bonferroni, and it
    was previously labelled Holm-Bonferroni (docs/13, F10). Holm is a step-down
    procedure over the *whole* family and needs every p-value at once: use
    `holm_bonferroni` for a prespecified family instead.

    Compare paired *performance* values — per-image errors or per-image metric
    values — never raw anomaly scores from two models, whose scales are
    arbitrary and unrelated.
    """
    result = paired_wilcoxon(values_a, values_b, alpha=alpha)
    corrected = min(1.0, result.p_value * family_size)
    return ComparisonResult(
        n=result.n,
        median_difference=result.median_difference,
        mean_difference=result.mean_difference,
        statistic=result.statistic,
        p_value=result.p_value,
        p_value_corrected=corrected,
        test=f"{result.test} (Bonferroni, family={family_size})",
        significant=bool(corrected < alpha),
    )


@dataclass(frozen=True)
class PairedBootstrap:
    """A paired, class-stratified bootstrap of a metric difference."""

    estimate: float
    low: float
    high: float
    n_resamples: int
    confidence: float
    metric: str

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:
        pct = round(self.confidence * 100)
        return f"d{self.metric} = {self.estimate:+.4f}  [{pct}% CI {self.low:+.4f}, {self.high:+.4f}]"


def paired_bootstrap_difference(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    name: str = "",
) -> PairedBootstrap:
    """Bootstrap CI for `metric(labels, a) - metric(labels, b)` on the same images.

    Paired: each resample draws the same images for both methods, so the
    interval reflects the difference, not two independent uncertainties.
    Stratified by class: normals and anomalies are resampled separately, so
    every resample keeps the original class counts and AUROC stays defined
    (docs/13 §5.4).

    This quantifies finite-evaluation-sample uncertainty for *one* fitted model
    per method. It says nothing about fitting-seed variability, which the
    seed spread reports separately.
    """
    labels = np.asarray(labels).astype(int).ravel()
    a = np.asarray(scores_a, dtype=np.float64).ravel()
    b = np.asarray(scores_b, dtype=np.float64).ravel()
    if not (labels.shape == a.shape == b.shape):
        raise ValueError("labels and both score arrays must have the same length")

    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("both classes are required")

    estimate = float(metric(labels, a) - metric(labels, b))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = np.concatenate(
            [rng.choice(pos, size=pos.size, replace=True), rng.choice(neg, size=neg.size, replace=True)]
        )
        y = labels[idx]
        diffs[i] = metric(y, a[idx]) - metric(y, b[idx])
    tail = (1.0 - confidence) / 2.0
    return PairedBootstrap(
        estimate=estimate,
        low=float(np.quantile(diffs, tail)),
        high=float(np.quantile(diffs, 1.0 - tail)),
        n_resamples=n_resamples,
        confidence=confidence,
        metric=name,
    )


def binomial_upper_bound(failures: int, trials: int, *, confidence: float = 0.95) -> float:
    """One-sided exact (Clopper-Pearson) upper bound on a rate.

    Zero false alarms among 60 normals does not mean an FPR of 0%: the 95%
    upper bound is about 4.9% (docs/13 §5.4). Every reported realized FPR and
    escape rate carries this bound alongside it.
    """
    if trials <= 0:
        return float("nan")
    if failures >= trials:
        return 1.0
    return float(stats.beta.ppf(confidence, failures + 1, trials - failures))
