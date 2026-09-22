"""PatchCore — the project's core model (docs/03, Tier 3).

Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.

The idea in one sentence: store patch descriptors of normal images in a memory
bank, subsample it to a coreset that preserves its coverage, and score a test
patch by its distance to the nearest stored normal patch.

Why it is the right upgrade target here: there is **no training**. The whole
model is a frozen backbone plus a stored matrix, so it is trivially
reproducible, has no training noise to confound ablations, and is inherently
explainable — every score points at a specific nearest normal patch, which is
the most convincing explanation available to a non-ML audience.

Memory (docs/06 §3 — the binding constraint at 4 GB):
* descriptors are stored fp16 and the coreset search runs in GPU-sized tiles;
* a random pre-subsample bounds the greedy k-center search, which is otherwise
  O(N·k) over 1.77 M vectors for `walnuts` at 512²;
* distances accumulate in fp32 regardless of storage dtype.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ..data.transforms import ImageTransform
from .base import AnomalyModel
from .features import FeatureExtractor, FeatureSpec, batched, resolve_device


def greedy_coreset(
    features: np.ndarray,
    n_select: int,
    *,
    seed: int = 0,
    device: str = "auto",
    tile: int = 8192,
) -> np.ndarray:
    """Greedy k-center coreset selection; returns the selected row indices.

    Iteratively picks the point furthest from everything already chosen, which
    is the standard k-center greedy and gives a 2-approximation to the minimax
    facility-location objective. In this setting that means the coreset covers
    the *spread* of normal appearance rather than its density — important,
    because a rare-but-normal configuration is exactly what a density-based
    subsample would drop and then flag as anomalous at test time.

    Only the running minimum-distance vector (N floats) stays resident, so peak
    memory is O(N + tile·D) rather than O(N²).
    """
    import torch

    n = features.shape[0]
    n_select = max(1, min(int(n_select), n))
    dev = torch.device(resolve_device(device))

    data = torch.as_tensor(features, device=dev, dtype=torch.float32)
    squared_norms = (data * data).sum(dim=1)

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    selected = [first]

    min_dist = _squared_distances(data, squared_norms, data[first], tile)
    min_dist[first] = -1.0

    for _ in range(n_select - 1):
        nxt = int(torch.argmax(min_dist).item())
        selected.append(nxt)
        new_dist = _squared_distances(data, squared_norms, data[nxt], tile)
        torch.minimum(min_dist, new_dist, out=min_dist)
        min_dist[nxt] = -1.0

    del data, min_dist
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(sorted(selected), dtype=np.int64)


def _squared_distances(data, squared_norms, centre, tile: int):
    """||x - c||^2 for every row, computed in tiles to bound peak memory."""
    import torch

    out = torch.empty(data.shape[0], device=data.device, dtype=torch.float32)
    centre_norm = (centre * centre).sum()
    for start in range(0, data.shape[0], tile):
        stop = min(start + tile, data.shape[0])
        block = data[start:stop]
        out[start:stop] = squared_norms[start:stop] - 2.0 * (block @ centre) + centre_norm
    return out.clamp_(min=0.0)


class PatchCore(AnomalyModel):
    """PatchCore with a coreset-subsampled memory bank."""

    name = "patchcore"
    tier = "T3"
    owner = "own"
    #: The coreset start point is drawn at random, so the fit is stochastic and
    #: the protocol's 3-seed rule applies (§4.2).
    stochastic = True

    def __init__(
        self,
        transform: ImageTransform | None = None,
        *,
        backbone: str = "wide_resnet50_2",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        patch_pool: int = 3,
        projection_dim: int = 1024,
        coreset_ratio: float = 0.01,
        coreset_presubsample: float = 0.1,
        k: int = 1,
        image_score_reweight: bool = True,
        batch_size: int = 4,
        feature_dtype: str = "float16",
        device: str = "auto",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(transform, **kwargs)
        self.spec = FeatureSpec(
            backbone=backbone,
            layers=tuple(layers),
            patch_pool=patch_pool,
            projection_dim=projection_dim,
            device=device,
            dtype=feature_dtype,
        )
        self.coreset_ratio = coreset_ratio
        self.coreset_presubsample = coreset_presubsample
        self.k = k
        self.image_score_reweight = image_score_reweight
        self.batch_size = batch_size
        self.seed = seed

        self.extractor = FeatureExtractor(self.spec)
        self.memory_bank: np.ndarray | None = None
        self.grid: tuple[int, int] = (0, 0)
        self._stats: dict[str, Any] = {}

    # -- fitting -----------------------------------------------------------
    def _fit(self, images: Iterator[np.ndarray]) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        features, grid = self.extractor.embed_all(batched(images, self.batch_size))
        self.grid = grid
        total = features.shape[0]

        rng = np.random.default_rng(self.seed)
        # Pre-subsample before the greedy search. Greedy k-center is O(N·k); at
        # 1.77 M vectors the search, not the extraction, becomes the bottleneck.
        # Random pre-subsampling preserves the distribution's support in
        # expectation, which is what the subsequent coverage-greedy step needs.
        working = features
        presubsample_n = total
        if 0 < self.coreset_presubsample < 1.0 and total > 50_000:
            presubsample_n = max(1, int(total * self.coreset_presubsample))
            keep = rng.choice(total, size=presubsample_n, replace=False)
            working = features[np.sort(keep)]

        n_select = max(1, round(total * self.coreset_ratio))
        n_select = min(n_select, working.shape[0])

        if n_select >= working.shape[0]:
            selected = np.arange(working.shape[0], dtype=np.int64)
        else:
            selected = greedy_coreset(
                working, n_select, seed=self.seed, device=self.spec.device
            )

        self.memory_bank = np.ascontiguousarray(working[selected])
        self._stats = {
            "patches_total": int(total),
            "patches_after_presubsample": int(presubsample_n),
            "memory_bank_size": int(self.memory_bank.shape[0]),
            "descriptor_dim": int(self.memory_bank.shape[1]),
            "memory_bank_mb": round(self.memory_bank.nbytes / 1024**2, 2),
            "feature_grid": f"{grid[0]}x{grid[1]}",
            "peak_vram_mb": round(self.extractor.peak_vram_mb(), 1),
        }

    def fit_extra(self) -> dict[str, Any]:
        return {**self._stats, **self.spec.as_dict(), "coreset_ratio": self.coreset_ratio, "k": self.k}

    # -- scoring -----------------------------------------------------------
    def _score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        import torch

        if self.memory_bank is None:
            raise RuntimeError("memory bank is empty; fit() first")

        descriptors, grid = self.extractor.embed_batch(image[None, ...])
        descriptors = descriptors[0].float()

        dev = descriptors.device
        bank = torch.as_tensor(self.memory_bank, device=dev, dtype=torch.float32)

        # cdist in fp32 even though the bank is stored fp16: accumulating 1024
        # squared differences in fp16 loses precision that the kNN ordering
        # actually depends on.
        distances = torch.cdist(descriptors, bank)
        knn, indices = torch.topk(distances, k=min(self.k, bank.shape[0]), dim=1, largest=False)
        patch_scores = knn[:, 0]

        image_score = self._image_score(patch_scores, descriptors, bank, indices)
        anomaly_map = patch_scores.reshape(grid).cpu().numpy().astype(np.float64)

        del bank, distances
        return float(image_score), anomaly_map

    def _image_score(self, patch_scores, descriptors, bank, indices) -> float:
        """The image score for the most anomalous patch.

        PatchCore re-weights the maximum patch distance by how *isolated* its
        nearest memory entry is: if that entry sits in a dense region of the
        bank, the patch is near well-supported normal appearance and the score
        is damped; if the entry is itself an outlier among normals, the evidence
        is weaker and the score is boosted. Without this the image score is just
        a max over patches and inherits all of its noise.
        """
        import torch

        peak = int(torch.argmax(patch_scores).item())
        max_distance = patch_scores[peak]

        if not self.image_score_reweight or bank.shape[0] < 2:
            return float(max_distance)

        nearest_index = int(indices[peak, 0])
        nearest = bank[nearest_index : nearest_index + 1]

        neighbours = min(9, bank.shape[0])
        bank_distances = torch.cdist(nearest, bank)[0]
        local, _ = torch.topk(bank_distances, k=neighbours, largest=False)

        weight = 1.0 - torch.softmax(local, dim=0)[0]
        return float(weight * max_distance)

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        if self.memory_bank is None:
            raise RuntimeError("model is not fitted")
        return {
            "memory_bank": self.memory_bank,
            "grid": self.grid,
            "spec": self.spec.as_dict(),
            "coreset_ratio": self.coreset_ratio,
            "k": self.k,
            "transform": self.transform.as_dict(),
            "stats": self._stats,
        }
