"""The experiment pipeline.

`run_experiment` executes the protocol in the only order it permits:

    fit on train  ->  predict validation  ->  predict test  ->  evaluate

Every threshold and normalization statistic is derived inside `evaluate` from
the validation predictions, so no code path lets the test split reach an
operating point (rules L3, L4). All models go through this function, which is
what keeps comparisons between them controlled.

Timing note (docs/13, F12): `e2e_ms_per_image` is end-to-end — it includes
reading the image from disk and resizing it. It is **not** a model latency
distribution; p50/p95 model latency is a separate benchmark.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data.core import DatasetIndex
from .evaluate import EvalConfig, SplitPredictions, evaluate
from .models.base import AnomalyModel
from .utils.memory import PeakRSS, PeakVRAM

NAN = float("nan")


@dataclass
class ExperimentResult:
    """One row of the canonical results table (docs/02 §4, extended by docs/13 §9)."""

    # identity
    run_name: str
    method: str
    tier: str
    owner: str
    category: str
    dataset: str
    split: str
    seed: int
    run_id: str = ""
    config_hash: str = ""
    implementation_id: str = ""
    git_sha: str = ""
    dirty: bool = True
    status: str = "completed"

    # configuration
    resolution: str = ""
    n_train: int = 0
    n_validation: int = 0
    n_test: int = 0

    # image level, at the validation-calibrated operating point
    image_auroc: float = NAN
    image_aupr: float = NAN
    f1max_oracle: float = NAN
    fpr_at_op1: float = NAN
    recall_at_op1: float = NAN
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    # the operating point itself: requested and effective are different numbers
    requested_fpr: float = NAN
    effective_fpr: float = NAN
    calibration_count: int = 0
    threshold_rank: int = 0
    target_met: bool = False
    threshold_op1: float = NAN
    threshold_method: str = ""
    threshold_comparator: str = ">"
    threshold_source: str = "validation"
    oracle_flag: bool = False

    # pixel level
    pixel_auroc: float = NAN
    aupro_005: float = NAN
    aupro_030: float = NAN
    segf1_3sigma: float = NAN
    iou_3sigma: float = NAN
    threshold_3sigma: float = NAN
    n_regions: int = 0
    n_negatives_sampled: int = 0
    smoothing_sigma: float = NAN
    smoothing_units: str = ""
    metrics_version: str = ""

    # resources and time
    fit_seconds: float = NAN
    predict_seconds: float = NAN
    eval_seconds: float = NAN
    e2e_ms_per_image: float = NAN
    peak_vram_mb: float = NAN
    # per stage: host RSS is the binding constraint here, and one end-of-run
    # peak cannot say which stage caused it (docs/13, F07)
    peak_rss_fit_mb: float = NAN
    peak_rss_predict_mb: float = NAN
    peak_rss_eval_mb: float = NAN
    peak_vram_fit_mb: float = NAN
    peak_vram_predict_mb: float = NAN

    # diagnostics
    escape_by_defect: dict[str, float] = field(default_factory=dict)
    fit_extra: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["escape_by_defect"] = ";".join(
            f"{k}={v:.3f}" for k, v in sorted(self.escape_by_defect.items())
        )
        row["fit_extra"] = ";".join(f"{k}={v}" for k, v in sorted(self.fit_extra.items()))
        return row

    def summary(self) -> str:
        return (
            f"{self.method:<16s} {self.category:<11s} s{self.seed} "
            f"I-AUROC={self.image_auroc:.4f}  AU-PRO@5%={self.aupro_005:.4f}  "
            f"P-AUROC={self.pixel_auroc:.4f}  SegF1={self.segf1_3sigma:.4f}  "
            f"FPR={self.fpr_at_op1:.3f} (target {self.requested_fpr:.2%}, "
            f"effective {self.effective_fpr:.2%})  recall={self.recall_at_op1:.3f}"
        )


def _max_or_nan(*values: float) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return max(finite) if finite else NAN


def run_with_predictions(
    model: AnomalyModel,
    indices: dict[str, DatasetIndex],
    *,
    seed: int = 0,
    test_split: str = "test_public",
    eval_config: EvalConfig | None = None,
) -> tuple[ExperimentResult, SplitPredictions, SplitPredictions]:
    """Fit, predict and evaluate; also return the raw predictions for storage."""
    for required in ("train", "validation", test_split):
        if required not in indices:
            raise KeyError(f"missing split {required!r}; have {sorted(indices)}")
    cfg = eval_config or EvalConfig()
    train, validation, test = indices["train"], indices["validation"], indices[test_split]

    with PeakRSS() as rss_fit, PeakVRAM() as vram_fit:
        t0 = time.perf_counter()
        model.fit(train)
        fit_seconds = time.perf_counter() - t0

    with PeakRSS() as rss_predict, PeakVRAM() as vram_predict:
        t0 = time.perf_counter()
        val_preds = SplitPredictions.from_model(model, validation)
        test_preds = SplitPredictions.from_model(model, test)
        predict_seconds = time.perf_counter() - t0

    with PeakRSS() as rss_eval:
        t0 = time.perf_counter()
        metrics = evaluate(val_preds, test_preds, cfg)
        eval_seconds = time.perf_counter() - t0

    oracle_threshold = metrics.pop("oracle_f1max_threshold", None)
    escape = metrics.pop("escape_by_defect", {})
    extra = dict(model.fit_record.extra) if model.fit_record else {}
    if oracle_threshold is not None:
        extra["oracle_f1max_threshold"] = f"{oracle_threshold:.6g}"

    known = set(ExperimentResult.__dataclass_fields__)
    result = ExperimentResult(
        run_name=f"{model.name}-{test.category}-s{seed}",
        method=model.name,
        tier=model.tier,
        owner=model.owner,
        category=test.category,
        dataset=test.layout.name,
        split=test.split_name,
        seed=seed,
        resolution=f"{model.transform.long_side or 'native'}",
        n_train=len(train),
        n_validation=len(validation),
        n_test=len(test),
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        eval_seconds=eval_seconds,
        e2e_ms_per_image=1000.0 * predict_seconds / max(1, len(validation) + len(test)),
        peak_vram_mb=_max_or_nan(vram_fit.peak_mb, vram_predict.peak_mb),
        peak_rss_fit_mb=rss_fit.peak_mb,
        peak_rss_predict_mb=rss_predict.peak_mb,
        peak_rss_eval_mb=rss_eval.peak_mb,
        peak_vram_fit_mb=vram_fit.peak_mb,
        peak_vram_predict_mb=vram_predict.peak_mb,
        escape_by_defect=escape,
        fit_extra=extra,
        **{k: v for k, v in metrics.items() if k in known},
    )
    return result, val_preds, test_preds


def run_experiment(
    model: AnomalyModel,
    indices: dict[str, DatasetIndex],
    *,
    workdir: str | Path | None = None,
    seed: int = 0,
    test_split: str = "test_public",
    fpr1_percentile: float = 99.0,
    sigma_n: float = 3.0,
    aupro_limits: tuple[float, ...] = (0.05, 0.30),
    aupro_num_thresholds: int | None = 512,
    smoothing_sigma: float | None = None,
    threshold_policy: str = "relax",
    report_oracle: bool = True,
    keep_predictions: bool = False,
) -> ExperimentResult:
    """Fit, calibrate on validation, evaluate on test, return one results row.

    `smoothing_sigma` defaults to the model's own setting so direct callers keep
    their behaviour; the runner always passes it explicitly from the config, so
    two methods in one table cannot silently differ in post-processing — which
    is exactly what happened to the first VisA comparison (Tier 0 at sigma 4
    through the CLI, PatchCore at sigma 0 through an inline script).
    """
    cfg = EvalConfig(
        smoothing_sigma=model.smoothing_sigma if smoothing_sigma is None else smoothing_sigma,
        target_fpr=(100.0 - fpr1_percentile) / 100.0,
        threshold_policy=threshold_policy,
        sigma_n=sigma_n,
        aupro_limits=tuple(aupro_limits),
        aupro_num_thresholds=aupro_num_thresholds or 512,
        report_oracle=report_oracle,
    )
    result, val_preds, test_preds = run_with_predictions(
        model, indices, seed=seed, test_split=test_split, eval_config=cfg
    )
    if keep_predictions and workdir is not None:
        val_preds.save(workdir)
        test_preds.save(workdir)
    return result


def aggregate_seeds(results: list[ExperimentResult], metric: str) -> tuple[float, float, int]:
    """(mean, sample std, n) of one metric across seeds."""
    values = np.asarray([getattr(r, metric) for r in results], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return NAN, NAN, 0
    std = float(values.std(ddof=1)) if values.size > 1 else NAN
    return float(values.mean()), std, int(values.size)
