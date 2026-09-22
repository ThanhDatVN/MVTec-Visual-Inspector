"""On-disk prediction store.

A test split of 150 images at 8 MP is 1.2e9 pixels. Held as float64 anomaly maps
that is ~10 GB, and the metric suite needs several passes over them (AU-PRO,
pixel AUROC, SegF1 each sweep the data once). Recomputing the model for each
pass would triple inference cost; holding them in RAM is not possible on the
laptop.

So predictions are written once to a directory of `.npy` files and replayed
lazily. Cost: one float32 copy on disk. Benefit: the metric code can take as
many passes as it needs and stays a simple function over sequences.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from .data.core import DatasetIndex, Sample
from .data.transforms import load_mask
from .models.base import AnomalyModel, Prediction


class PredictionStore:
    """Anomaly maps on disk, plus the scalar scores and labels in memory."""

    def __init__(self, root: str | Path, *, keep: bool = False) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.ids: list[str] = []
        self.scores: list[float] = []
        self.labels: list[int] = []
        self.defect_types: list[str] = []
        self._mask_paths: list[Path | None] = []
        self._map_paths: list[Path] = []
        self._native_sizes: list[tuple[int, int]] = []

    # -- writing -----------------------------------------------------------
    def add(self, sample: Sample, prediction: Prediction, dataset_root: Path) -> None:
        index = len(self.ids)
        path = self.root / f"{index:05d}.npy"
        # float32 halves the footprint against float64 and is far finer than any
        # difference between the models being compared.
        np.save(path, prediction.anomaly_map.astype(np.float32))

        self.ids.append(sample.rel_id(dataset_root))
        self.scores.append(float(prediction.score))
        self.labels.append(int(sample.label))
        self.defect_types.append(sample.defect_type)
        self._mask_paths.append(sample.mask_path)
        self._map_paths.append(path)
        self._native_sizes.append(prediction.anomaly_map.shape[:2])

    @classmethod
    def from_model(
        cls,
        model: AnomalyModel,
        index: DatasetIndex,
        root: str | Path,
        *,
        keep: bool = False,
    ) -> PredictionStore:
        store = cls(root, keep=keep)
        for sample, prediction in model.predict_index(index):
            store.add(sample, prediction, index.root)
        return store

    # -- reading -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.ids)

    def maps(self) -> Iterator[np.ndarray]:
        """Replay the anomaly maps, one at a time."""
        for path in self._map_paths:
            yield np.load(path).astype(np.float64)

    def masks(self) -> Iterator[np.ndarray]:
        """Ground-truth masks aligned with `maps()`.

        A normal image has no mask file; it yields an all-zero mask so it
        contributes negatives to the FPR denominator, which is what makes
        AU-PRO reflect the false alarms deployment actually cares about.
        """
        for mask_path, size in zip(self._mask_paths, self._native_sizes):
            if mask_path is None:
                yield np.zeros(size, dtype=bool)
            else:
                yield load_mask(mask_path)

    def anomalous_pairs(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Maps and masks for anomalous images only, materialized.

        Used where a metric needs random access rather than a single pass. Only
        the anomalous subset, which is the minority, so this stays affordable.
        """
        maps, masks = [], []
        for i, (amap, mask) in enumerate(zip(self.maps(), self.masks())):
            if self.labels[i] == 1:
                maps.append(amap)
                masks.append(mask)
        return maps, masks

    @property
    def score_array(self) -> np.ndarray:
        return np.asarray(self.scores, dtype=np.float64)

    @property
    def label_array(self) -> np.ndarray:
        return np.asarray(self.labels, dtype=int)

    def map_stats(self) -> dict[str, float]:
        """Pooled statistics over all maps, needed for the mean+3sigma rule."""
        total = count = 0.0
        total_sq = 0.0
        lo, hi = np.inf, -np.inf
        for amap in self.maps():
            flat = amap.ravel()
            total += float(flat.sum())
            total_sq += float(np.square(flat).sum())
            count += flat.size
            lo = min(lo, float(flat.min()))
            hi = max(hi, float(flat.max()))
        mean = total / count
        # Two-pass would be more numerically stable, but the maps are bounded
        # and float64 accumulation over ~1e9 values is comfortably adequate here.
        variance = max(0.0, total_sq / count - mean**2)
        return {"mean": mean, "std": float(np.sqrt(variance)), "min": lo, "max": hi, "n": count}

    # -- lifecycle ---------------------------------------------------------
    def save_index(self) -> Path:
        path = self.root / "index.json"
        path.write_text(
            json.dumps(
                {
                    "ids": self.ids,
                    "scores": self.scores,
                    "labels": self.labels,
                    "defect_types": self.defect_types,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def cleanup(self) -> None:
        if not self.keep and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> PredictionStore:
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()


def subset_scores(store: PredictionStore, label: int) -> np.ndarray:
    return store.score_array[store.label_array == label]


def select_case_ids(store: PredictionStore, n: int = 10) -> Sequence[str]:
    """A fixed, deterministic image subset for per-run artifact logging.

    Logging the same images every run gives a visual diff across the whole
    project history for free, and is how you notice that a two-point
    'improvement' was actually a normalization change.
    """
    return sorted(store.ids)[:n]
