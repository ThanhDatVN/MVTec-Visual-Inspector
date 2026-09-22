"""The experiment pipeline.

One function, `run_experiment`, executes the protocol end to end in the only
order the protocol permits:

    fit on train  ->  score validation  ->  freeze normalizer and thresholds
                  ->  score test        ->  compute metrics  ->  emit one row

The ordering is the point. Thresholds and normalization statistics are fixed
*before* a single test image is scored, so there is no code path in which the
test split can influence them (rules L3, L4). Every model in the ladder goes
through this same function, which is also what makes the comparison between
them controlled.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data.core import DatasetIndex
from .metrics import compute_image_metrics, compute_pixel_metrics, escape_rate_by_defect, pro_curve
from .models.base import AnomalyModel
from .postproc import ScoreNormalizer, from_validation_percentile
from .postproc.thresholds import from_test_f1_max, sigma_from_stats
from .predictions import PredictionStore


@dataclass
class ExperimentResult:
    """One row of the canonical results table (docs/02 §4)."""

    # identity
    run_name: str
    method: str
    tier: str
    owner: str
    category: str
    dataset: str
    split: str
    seed: int
    config_hash: str = ""
    git_sha: str = ""
    dirty: bool = True

    # configuration
    resolution: str = ""
    n_train: int = 0
    n_validation: int = 0
    n_test: int = 0

    # image level
    image_auroc: float = float("nan")
    image_aupr: float = float("nan")
    f1max_oracle: float = float("nan")
    fpr_at_op1: float = float("nan")
    recall_at_op1: float = float("nan")

    # pixel level
    pixel_auroc: float = float("nan")
    aupro_005: float = float("nan")
    aupro_030: float = float("nan")
    segf1_3sigma: float = float("nan")
    iou_3sigma: float = float("nan")

    # provenance of the operating points
    threshold_op1: float = float("nan")
    threshold_3sigma: float = float("nan")
    threshold_source: str = "validation"
    oracle_flag: bool = False

    # systems
    fit_seconds: float = float("nan")
    predict_seconds: float = float("nan")
    latency_per_image_ms: float = float("nan")

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
            f"{self.method:<16s} {self.category:<14s} "
            f"I-AUROC={self.image_auroc:.4f}  AU-PRO@5%={self.aupro_005:.4f}  "
            f"AU-PRO@30%={self.aupro_030:.4f}  P-AUROC={self.pixel_auroc:.4f}  "
            f"SegF1={self.segf1_3sigma:.4f}  FPR@OP1={self.fpr_at_op1:.3f}  "
            f"recall@OP1={self.recall_at_op1:.3f}"
        )


def run_experiment(
    model: AnomalyModel,
    indices: dict[str, DatasetIndex],
    *,
    workdir: str | Path,
    seed: int = 0,
    test_split: str = "test_public",
    fpr1_percentile: float = 99.0,
    sigma_n: float = 3.0,
    aupro_limits: tuple[float, ...] = (0.05, 0.30),
    aupro_num_thresholds: int | None = 512,
    report_oracle: bool = True,
    keep_predictions: bool = False,
) -> ExperimentResult:
    """Fit, calibrate, evaluate, and return one results row."""
    for required in ("train", "validation", test_split):
        if required not in indices:
            raise KeyError(f"missing split {required!r}; have {sorted(indices)}")

    train, validation, test = indices["train"], indices["validation"], indices[test_split]
    workdir = Path(workdir)

    # -- 1. fit, on train only ------------------------------------------
    t0 = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - t0

    # -- 2. calibrate, on validation only -------------------------------
    # Everything that defines an operating point is frozen here, before any
    # test image has been seen. This ordering is the enforcement of L3/L4.
    with PredictionStore(workdir / "validation", keep=False) as val_store:
        for sample, prediction in model.predict_index(validation):
            val_store.add(sample, prediction, validation.root)

        threshold_op1 = from_validation_percentile(
            val_store.score_array, percentile=fpr1_percentile
        )
        # Pooled map statistics are accumulated in a streaming pass; a real
        # validation split's maps do not fit in memory.
        val_stats = val_store.map_stats()
        threshold_sigma = sigma_from_stats(
            val_stats["mean"], val_stats["std"], n_sigma=sigma_n, n_pixels=val_stats["n"]
        )
        normalizer = ScoreNormalizer().fit_bounds(
            val_stats["min"], val_stats["max"], split_name="validation"
        )

    # -- 3. score the test split ----------------------------------------
    t0 = time.perf_counter()
    test_store = PredictionStore(workdir / test_split, keep=keep_predictions)
    for sample, prediction in model.predict_index(test):
        test_store.add(sample, prediction, test.root)
    predict_seconds = time.perf_counter() - t0

    try:
        result = _evaluate(
            model=model,
            store=test_store,
            train=train,
            validation=validation,
            test=test,
            seed=seed,
            threshold_op1=threshold_op1,
            threshold_sigma=threshold_sigma,
            aupro_limits=aupro_limits,
            aupro_num_thresholds=aupro_num_thresholds,
            report_oracle=report_oracle,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
        )
        result.fit_extra["normalizer_bounds"] = f"{normalizer.bounds[0]:.4g}..{normalizer.bounds[1]:.4g}"
        if keep_predictions:
            test_store.save_index()
        return result
    finally:
        if not keep_predictions:
            test_store.cleanup()


def _evaluate(
    *,
    model: AnomalyModel,
    store: PredictionStore,
    train: DatasetIndex,
    validation: DatasetIndex,
    test: DatasetIndex,
    seed: int,
    threshold_op1,
    threshold_sigma,
    aupro_limits: tuple[float, ...],
    aupro_num_thresholds: int | None,
    report_oracle: bool,
    fit_seconds: float,
    predict_seconds: float,
) -> ExperimentResult:
    labels = store.label_array
    scores = store.score_array

    image_metrics = compute_image_metrics(
        labels, scores, threshold=threshold_op1.value, threshold_source=threshold_op1.source_split
    )

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
        image_auroc=image_metrics.auroc,
        image_aupr=image_metrics.aupr,
        f1max_oracle=image_metrics.f1_max_oracle if report_oracle else float("nan"),
        fpr_at_op1=image_metrics.fpr,
        recall_at_op1=image_metrics.recall,
        threshold_op1=threshold_op1.value,
        threshold_3sigma=threshold_sigma.value,
        threshold_source=threshold_op1.source_split,
        oracle_flag=False,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        latency_per_image_ms=1000.0 * predict_seconds / max(1, len(store)),
        fit_extra=dict(model.fit_record.extra) if model.fit_record else {},
    )

    result.escape_by_defect = escape_rate_by_defect(
        labels, scores, np.asarray(store.defect_types), threshold_op1.value
    )

    # Pixel metrics need both classes of pixel present.
    if any(label == 1 for label in store.labels):
        pixel_metrics = compute_pixel_metrics(
            list(store.maps()),
            list(store.masks()),
            threshold=threshold_sigma.value,
            threshold_source=threshold_sigma.source_split,
        )
        result.pixel_auroc = pixel_metrics.auroc
        result.segf1_3sigma = pixel_metrics.seg_f1
        result.iou_3sigma = pixel_metrics.iou

        for limit in aupro_limits:
            curve = pro_curve(
                list(store.maps()),
                list(store.masks()),
                integration_limit=limit,
                num_thresholds=aupro_num_thresholds,
            )
            if abs(limit - 0.05) < 1e-9:
                result.aupro_005 = curve.au_pro
            elif abs(limit - 0.30) < 1e-9:
                result.aupro_030 = curve.au_pro

    if report_oracle:
        oracle = from_test_f1_max(labels, scores, split_name=test.split_name)
        result.fit_extra["oracle_f1max_threshold"] = f"{oracle.value:.6g}"

    return result
