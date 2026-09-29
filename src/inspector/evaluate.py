"""The shared evaluator.

One code path turns stored predictions into every reported number, for clean
test data and for corrupted test data alike. Before this module existed the
robustness notebook computed AU-PRO over anomalous images only while the main
pipeline included normal images — two different FPR denominators reported under
one column name (docs/13, F09). Now a corrupted split is simply another
`SplitPredictions`, and severity 0 reproduces the clean numbers exactly, which a
test asserts.

Design:

* **Predictions are stored raw** — the model's own map at its own resolution
  plus the image score. Smoothing, upsampling and thresholding happen here, so
  a smoothing ablation, an aggregation study or a calibration study re-evaluates
  cached predictions without refitting (docs/13, E02/E03/E05).
* **One streaming pass per split.** Validation: pooled mean/std of the native
  maps for the pixel threshold. Test: the pixel-AUROC histogram, the confusion
  counts at the pixel threshold, and the AU-PRO region/negative samples, all
  accumulated together. No split's native maps are ever resident at once.
* **Thresholds are validation-derived and strict** (`score > threshold`), with
  requested and effective targets recorded separately (docs/13, F01).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data.core import ANOMALOUS, DatasetIndex, Sample
from .data.transforms import load_image, load_mask
from .metrics.aupro import PROAccumulator
from .metrics.image_level import compute_image_metrics, escape_rate_by_defect
from .metrics.pixel_level import PixelHistogram
from .models.base import AnomalyModel, postprocess_map
from .postproc.thresholds import from_test_f1_max, from_validation_fpr, sigma_from_stats

#: Bump when any metric definition, convention or default changes. Recorded with
#: every result so numbers computed under different conventions never share a
#: table unnoticed.
METRICS_VERSION = "2"

#: (image, mask, sample) -> (image, mask). Used to corrupt a split on the fly.
ImageFn = Callable[[np.ndarray, "np.ndarray | None", Sample], "tuple[np.ndarray, np.ndarray | None]"]


@dataclass
class SplitPredictions:
    """Raw model outputs for one split, in dataset order."""

    split: str
    ids: list[str]
    labels: np.ndarray
    defect_types: list[str]
    scores: np.ndarray
    raw_maps: list[np.ndarray]
    input_sizes: list[tuple[int, int]]
    native_sizes: list[tuple[int, int]]
    mask_paths: list[str | None]
    #: Masks transformed by a geometric corruption; None means "load from path".
    mask_overrides: list[np.ndarray | None] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def from_model(
        cls,
        model: AnomalyModel,
        index: DatasetIndex,
        *,
        image_fn: ImageFn | None = None,
    ) -> SplitPredictions:
        ids, labels, types, scores = [], [], [], []
        maps, in_sizes, nat_sizes, mask_paths, overrides = [], [], [], [], []
        for sample in index:
            image = load_image(sample.image_path)
            override = None
            if image_fn is not None:
                mask = load_mask(sample.mask_path) if sample.mask_path else None
                image, new_mask = image_fn(image, mask, sample)
                if mask is not None and new_mask is not None and not np.array_equal(mask, new_mask):
                    override = new_mask.astype(bool)
            raw = model.predict_raw(image)
            ids.append(sample.rel_id(index.root))
            labels.append(int(sample.label))
            types.append(sample.defect_type)
            scores.append(raw.score)
            maps.append(raw.raw_map)
            in_sizes.append(raw.input_size)
            nat_sizes.append(raw.native_size)
            mask_paths.append(str(sample.mask_path) if sample.mask_path else None)
            overrides.append(override)
        return cls(
            split=index.split_name,
            ids=ids,
            labels=np.asarray(labels, dtype=int),
            defect_types=types,
            scores=np.asarray(scores, dtype=np.float64),
            raw_maps=maps,
            input_sizes=in_sizes,
            native_sizes=nat_sizes,
            mask_paths=mask_paths,
            mask_overrides=overrides,
        )

    def with_scores(self, scores: np.ndarray) -> SplitPredictions:
        """The same maps with a different image score — for aggregation studies."""
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != self.scores.shape:
            raise ValueError(f"expected {self.scores.shape} scores, got {scores.shape}")
        return SplitPredictions(
            split=self.split,
            ids=self.ids,
            labels=self.labels,
            defect_types=self.defect_types,
            scores=scores,
            raw_maps=self.raw_maps,
            input_sizes=self.input_sizes,
            native_sizes=self.native_sizes,
            mask_paths=self.mask_paths,
            mask_overrides=self.mask_overrides,
        )

    # -- lazy native-resolution views ----------------------------------------
    def native_maps(self, sigma: float) -> Iterator[np.ndarray]:
        for raw, inp, nat in zip(self.raw_maps, self.input_sizes, self.native_sizes):
            yield postprocess_map(raw, input_size=inp, native_size=nat, sigma=sigma)

    def masks(self) -> Iterator[np.ndarray]:
        """Ground truth aligned with `native_maps`; normal images get all-zero
        masks, so they contribute negatives to every FPR denominator."""
        overrides = self.mask_overrides or [None] * len(self.ids)
        for path, size, override in zip(self.mask_paths, self.native_sizes, overrides):
            if override is not None:
                yield override
            elif path is None:
                yield np.zeros((size[1], size[0]), dtype=bool)
            else:
                yield load_mask(path)

    def raw_range(self) -> tuple[float, float]:
        lo = min(float(m.min()) for m in self.raw_maps)
        hi = max(float(m.max()) for m in self.raw_maps)
        return lo, hi

    # -- persistence ---------------------------------------------------------
    def save(
        self,
        directory: str | Path,
        *,
        with_maps: bool = True,
        relative_to: str | Path | None = None,
    ) -> None:
        """Write predictions as JSON (+ raw maps as float16 npz).

        Mask paths are stored relative to `relative_to` when given, so a saved
        run does not bake one machine's absolute paths into the repository.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        def rel(path: str | None) -> str | None:
            if path is None or relative_to is None:
                return path
            try:
                return Path(path).resolve().relative_to(Path(relative_to).resolve()).as_posix()
            except ValueError:
                return path

        records = [
            {
                "id": i,
                "label": int(y),
                "defect_type": d,
                "score": float(s),
                "input_size": list(inp),
                "native_size": list(nat),
                "mask": rel(m),
            }
            for i, y, d, s, inp, nat, m in zip(
                self.ids, self.labels, self.defect_types, self.scores,
                self.input_sizes, self.native_sizes, self.mask_paths,
            )
        ]
        (directory / f"{self.split}.predictions.json").write_text(
            json.dumps(records, separators=(",", ":")), encoding="utf-8"
        )
        if with_maps:
            arrays: dict[str, Any] = {
                f"m{i}": m.astype(np.float16) for i, m in enumerate(self.raw_maps)
            }
            np.savez_compressed(directory / f"{self.split}.maps.npz", **arrays)

    @classmethod
    def load(
        cls, directory: str | Path, split: str, *, root: str | Path | None = None
    ) -> SplitPredictions:
        """Load saved predictions; relative mask paths resolve against `root`."""
        directory = Path(directory)
        records = json.loads((directory / f"{split}.predictions.json").read_text(encoding="utf-8"))
        maps_path = directory / f"{split}.maps.npz"
        maps: list[np.ndarray] = []
        if maps_path.is_file():
            with np.load(maps_path) as bundle:
                maps = [bundle[f"m{i}"].astype(np.float32) for i in range(len(records))]
        return cls(
            split=split,
            ids=[r["id"] for r in records],
            labels=np.asarray([r["label"] for r in records], dtype=int),
            defect_types=[r["defect_type"] for r in records],
            scores=np.asarray([r["score"] for r in records], dtype=np.float64),
            raw_maps=maps,
            input_sizes=[tuple(r["input_size"]) for r in records],
            native_sizes=[tuple(r["native_size"]) for r in records],
            mask_paths=[
                None if r["mask"] is None
                else (str(Path(root) / r["mask"]) if root and not Path(r["mask"]).is_absolute() else r["mask"])
                for r in records
            ],
            mask_overrides=[],
        )

    def digest(self) -> str:
        """Hash of ids, labels and scores — identifies a prediction artifact."""
        h = hashlib.sha256()
        h.update(json.dumps(self.ids).encode())
        h.update(self.labels.tobytes())
        h.update(np.round(self.scores, 10).tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class EvalConfig:
    """Everything that changes a metric value. Part of every run's identity."""

    smoothing_sigma: float = 4.0
    smoothing_units: str = "input_px"
    target_fpr: float = 0.01
    threshold_policy: str = "relax"
    sigma_n: float = 3.0
    aupro_limits: tuple[float, ...] = (0.05, 0.30)
    aupro_num_thresholds: int = 512
    max_negative_samples: int = 2_000_000
    sampling_seed: int = 0
    pixel_auroc_bins: int = 1 << 16
    report_oracle: bool = True
    metrics_version: str = METRICS_VERSION

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["aupro_limits"] = list(self.aupro_limits)
        return out


def image_metrics(
    val: SplitPredictions, test: SplitPredictions, cfg: EvalConfig
) -> dict[str, Any]:
    """Image-level metrics at a validation-calibrated operating point."""
    threshold = from_validation_fpr(
        val.scores, target_fpr=cfg.target_fpr, policy=cfg.threshold_policy  # type: ignore[arg-type]
    )
    params = threshold.params or {}
    m = compute_image_metrics(
        test.labels, test.scores, threshold=threshold.value, threshold_source=threshold.source_split
    )
    decisions = test.scores > threshold.value
    out: dict[str, Any] = {
        "image_auroc": m.auroc,
        "image_aupr": m.aupr,
        "f1max_oracle": m.f1_max_oracle if cfg.report_oracle else float("nan"),
        "fpr_at_op1": m.fpr,
        "recall_at_op1": m.recall,
        "tp": int(np.sum(decisions & (test.labels == 1))),
        "fp": int(np.sum(decisions & (test.labels == 0))),
        "tn": int(np.sum(~decisions & (test.labels == 0))),
        "fn": int(np.sum(~decisions & (test.labels == 1))),
        "threshold_op1": threshold.value,
        "threshold_method": threshold.method,
        "threshold_comparator": threshold.comparator,
        "threshold_source": threshold.source_split,
        "requested_fpr": params.get("requested_fpr", float("nan")),
        "effective_fpr": params.get("effective_fpr", float("nan")),
        "calibration_count": int(params.get("n_calibration", len(val))),
        "threshold_rank": int(params.get("order_rank_k", 0)),
        "target_met": bool(params.get("target_met", 0.0)),
        "escape_by_defect": escape_rate_by_defect(
            test.labels, test.scores, np.asarray(test.defect_types), threshold.value
        ),
    }
    if cfg.report_oracle and len(np.unique(test.labels)) == 2:
        out["oracle_f1max_threshold"] = from_test_f1_max(
            test.labels, test.scores, split_name=test.split
        ).value
    return out


def pixel_metrics(
    val: SplitPredictions, test: SplitPredictions, cfg: EvalConfig
) -> dict[str, Any]:
    """Pixel-level metrics in one streaming pass per split."""
    # Validation pass: pooled statistics of native maps for the 3-sigma rule.
    total = total_sq = count = 0.0
    for native in val.native_maps(cfg.smoothing_sigma):
        flat = native.ravel()
        total += float(flat.sum())
        total_sq += float(np.square(flat).sum())
        count += flat.size
    mean = total / count
    std = float(np.sqrt(max(0.0, total_sq / count - mean**2)))
    tau = sigma_from_stats(mean, std, n_sigma=cfg.sigma_n, n_pixels=count)

    # Test pass: histogram, confusion counts and AU-PRO samples together.
    lo, hi = test.raw_range()
    hist = PixelHistogram(lo, hi, bins=cfg.pixel_auroc_bins)
    est_negatives = float(sum(w * h for w, h in test.native_sizes))
    rate = min(1.0, cfg.max_negative_samples / max(est_negatives, 1.0))
    pro = PROAccumulator(negative_rate=rate, seed=cfg.sampling_seed)
    tp = fp = fn = tn = 0

    for native, mask in zip(test.native_maps(cfg.smoothing_sigma), test.masks()):
        if native.shape != mask.shape:
            raise ValueError(f"map {native.shape} and mask {mask.shape} differ")
        hist.add(native, mask)
        pro.add(native, mask)
        pred = native > tau.value
        tp += int(np.count_nonzero(pred & mask))
        fp += int(np.count_nonzero(pred & ~mask))
        fn += int(np.count_nonzero(~pred & mask))
        tn += int(np.count_nonzero(~pred & ~mask))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    out: dict[str, Any] = {
        "threshold_3sigma": tau.value,
        "pixel_auroc": hist.auroc() if (tp + fn) and (fp + tn) else float("nan"),
        "segf1_3sigma": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "iou_3sigma": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
        "pixel_tp": tp,
        "pixel_fp": fp,
        "pixel_fn": fn,
        "negative_sample_rate": rate,
    }
    if pro.regions:
        curves = pro.curves(cfg.aupro_limits, num_thresholds=cfg.aupro_num_thresholds)
        for limit, curve in curves.items():
            out[f"aupro_{round(limit * 100):03d}"] = curve.au_pro
        out["n_regions"] = len(pro.regions)
        out["n_negatives_sampled"] = int(pro.negatives().size)
    return out


def evaluate(val: SplitPredictions, test: SplitPredictions, cfg: EvalConfig) -> dict[str, Any]:
    """Every reported metric for one (validation, test) pair."""
    if not len(val) or not len(test):
        raise ValueError("empty validation or test predictions")
    out = image_metrics(val, test, cfg)
    if test.raw_maps and val.raw_maps and np.any(test.labels == ANOMALOUS):
        out.update(pixel_metrics(val, test, cfg))
    out["metrics_version"] = cfg.metrics_version
    out["smoothing_sigma"] = cfg.smoothing_sigma
    out["smoothing_units"] = cfg.smoothing_units
    return out
