"""PatchCore — the project's core retrieval model (docs/03, Tier 3).

Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.

Store patch descriptors of normal images in a memory bank, subsample the bank to
a coreset that preserves its coverage, and score a test patch by its distance to
the stored normal patches. The bank is fitted to data — "no training" means no
gradient steps, not no data-dependent fitting.

This version corrects four defects found in review (docs/13):

* **F02 — the `k` setting did nothing.** Patch scores always used the nearest
  distance regardless of `k`. Neighbour count and reduction are now separate,
  explicit settings (`patch_neighbors`, `patch_reduction`), and a behavioural
  test checks that changing them changes the scores.
* **F03 — image-score re-weighting did not match the reference.** It weighted by
  distances *among bank entries*; the reference (anomalib's
  `compute_anomaly_score`) weights by distances *from the anomalous query patch*
  to the support set around its nearest bank entry. Now reference-faithful, with
  the support size a separate setting (`reweight_neighbors`, default 9 as in
  anomalib).
* **F07 — memory was bounded only in parts.** Candidates are now sampled per
  image *during* extraction, the bank is converted to a device tensor once, and
  nearest-neighbour search is chunked over query patches.
* **F08 — coreset settings could collapse to the same bank.** The requested bank
  size is capped by the candidate pool, so a 10% and a 25% bank were identical
  under a 10% candidate pool. Total, candidate, requested and effective sizes
  are all logged, and `bank_capped` flags the collapse.

Remaining, recorded differences from the author code are listed in
`reference_differences()` so a reproduction report can cite them rather than
rediscover them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..data.transforms import ImageTransform
from .base import AnomalyModel
from .features import FeatureExtractor, FeatureSpec, batched, resolve_device

PatchReduction = Literal["nearest", "mean", "kth"]


# ---------------------------------------------------------------------------
# Coreset selection
# ---------------------------------------------------------------------------


def random_projection(dim_in: int, dim_out: int, seed: int) -> np.ndarray:
    """A seeded Gaussian projection matrix (dim_in, dim_out).

    The author repository projects descriptors to 128 dimensions before greedy
    coreset selection (Johnson-Lindenstrauss): distances are approximately
    preserved, and the candidate matrix shrinks 8x — which is also what keeps it
    inside 4 GB of VRAM at higher resolutions.
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((dim_in, dim_out)) / np.sqrt(dim_out)).astype(np.float32)


def greedy_coreset(
    features: np.ndarray,
    n_select: int,
    *,
    seed: int = 0,
    device: str = "auto",
    tile: int = 16384,
) -> np.ndarray:
    """Greedy k-center selection; returns sorted row indices of the coreset.

    Repeatedly picks the point furthest from everything already chosen — a
    2-approximation to the k-center objective. The coreset therefore covers the
    *spread* of normal appearance rather than its density, so a rare but normal
    configuration is kept instead of being flagged at test time.

    Only the running minimum-distance vector (N floats) and one tile are
    resident beyond the input matrix.
    """
    import torch

    n = features.shape[0]
    n_select = max(1, min(int(n_select), n))
    dev = torch.device(resolve_device(device))

    data = torch.as_tensor(np.ascontiguousarray(features), device=dev, dtype=torch.float32)
    squared_norms = (data * data).sum(dim=1)

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    selected = [first]

    min_dist = _squared_distances(data, squared_norms, data[first], tile)
    min_dist[first] = -1.0
    for _ in range(n_select - 1):
        nxt = int(torch.argmax(min_dist).item())
        selected.append(nxt)
        torch.minimum(min_dist, _squared_distances(data, squared_norms, data[nxt], tile), out=min_dist)
        min_dist[nxt] = -1.0

    del data, min_dist, squared_norms
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(sorted(selected), dtype=np.int64)


def _squared_distances(data, squared_norms, centre, tile: int):
    import torch

    out = torch.empty(data.shape[0], device=data.device, dtype=torch.float32)
    centre_norm = (centre * centre).sum()
    for start in range(0, data.shape[0], tile):
        stop = min(start + tile, data.shape[0])
        out[start:stop] = squared_norms[start:stop] - 2.0 * (data[start:stop] @ centre) + centre_norm
    return out.clamp_(min=0.0)


# ---------------------------------------------------------------------------
# Scoring — a pure function, testable without a backbone
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchScores:
    patch_scores: Any  # torch.Tensor (P,)
    image_score: float
    nearest_index: Any  # torch.Tensor (P,) — nearest bank entry per patch
    peak_patch: int


def knn_search(queries, bank, k: int, *, chunk: int = 4096):
    """Exact k-nearest-neighbour distances and indices, chunked over queries.

    The full (queries x bank) matrix is never formed: at native resolution that
    is 23,584 x ~20,000 floats per image. Chunking keeps the peak at
    `chunk x bank`, and the result is identical to the full computation — a test
    asserts it.
    """
    import torch

    k = max(1, min(int(k), bank.shape[0]))
    dists, idx = [], []
    for start in range(0, queries.shape[0], chunk):
        block = torch.cdist(queries[start : start + chunk], bank)
        d, i = torch.topk(block, k=k, dim=1, largest=False)
        dists.append(d)
        idx.append(i)
    return torch.cat(dists), torch.cat(idx)


def score_patches(
    descriptors,
    bank,
    *,
    patch_neighbors: int = 1,
    patch_reduction: PatchReduction = "nearest",
    reweight: bool = True,
    reweight_neighbors: int = 9,
    chunk: int = 4096,
) -> PatchScores:
    """Patch anomaly scores and the image score for one image.

    Args:
        descriptors: (P, D) float32 tensor — the query image's patches.
        bank: (M, D) float32 tensor on the same device.
        patch_neighbors: how many nearest bank entries define a patch score.
        patch_reduction: `nearest` (distance to the 1st neighbour — the paper),
            `mean` (mean of the k nearest) or `kth` (distance to the k-th).
            `nearest` ignores `patch_neighbors`, so a `k` sweep must pair with a
            non-`nearest` reduction to test anything.
        reweight: apply the reference image-score re-weighting.
        reweight_neighbors: size of the support set around the nearest bank
            entry of the peak patch (anomalib default 9; 1 disables it, as there
            is nothing to weight against).

    Image score, following anomalib's `compute_anomaly_score`:

        1. peak patch = argmax of patch scores
        2. m* = the bank entry nearest to the peak patch
        3. support = the `reweight_neighbors` bank entries nearest to m*
        4. d_j = distance from the **query peak patch** to each support entry
        5. weight = 1 - softmax(d)[0];  image score = weight * peak score
    """
    import torch

    if patch_reduction not in ("nearest", "mean", "kth"):
        raise ValueError(f"unknown patch_reduction {patch_reduction!r}")

    k_needed = 1 if patch_reduction == "nearest" else max(1, int(patch_neighbors))
    dists, indices = knn_search(descriptors, bank, k_needed, chunk=chunk)
    k_eff = dists.shape[1]

    if patch_reduction == "nearest":
        patch_scores = dists[:, 0]
    elif patch_reduction == "mean":
        patch_scores = dists[:, :k_eff].mean(dim=1)
    else:  # kth
        patch_scores = dists[:, k_eff - 1]

    nearest_index = indices[:, 0]
    peak = int(torch.argmax(patch_scores).item())
    peak_score = patch_scores[peak]

    n_support = min(int(reweight_neighbors), bank.shape[0])
    if not reweight or n_support < 2:
        return PatchScores(patch_scores, float(peak_score), nearest_index, peak)

    m_star = bank[nearest_index[peak]].unsqueeze(0)
    _, support = torch.topk(torch.cdist(m_star, bank)[0], k=n_support, largest=False)
    query = descriptors[peak].unsqueeze(0)
    query_to_support = torch.cdist(query, bank[support])[0]
    weight = 1.0 - torch.softmax(query_to_support, dim=0)[0]
    return PatchScores(patch_scores, float(weight * peak_score), nearest_index, peak)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class PatchCore(AnomalyModel):
    """PatchCore with streaming candidate sampling and a greedy coreset bank."""

    name = "patchcore"
    tier = "T3"
    owner = "own"
    #: Candidate sampling, the projection and the coreset start point are all
    #: seeded draws, so the fit is stochastic and the 3-seed rule applies.
    stochastic = True

    def __init__(
        self,
        transform: ImageTransform | None = None,
        *,
        backbone: str = "wide_resnet50_2",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        patch_pool: int = 3,
        projection_dim: int = 1024,
        candidate_fraction: float = 0.1,
        bank_ratio: float | None = 0.01,
        bank_size: int | None = None,
        coreset_projection_dim: int | None = 128,
        patch_neighbors: int = 1,
        patch_reduction: PatchReduction = "nearest",
        reweight: bool = True,
        reweight_neighbors: int = 9,
        query_chunk: int = 4096,
        batch_size: int = 4,
        feature_dtype: str = "float16",
        device: str = "auto",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(transform, **kwargs)
        if (bank_ratio is None) == (bank_size is None):
            raise ValueError("set exactly one of bank_ratio or bank_size")
        if not 0.0 < candidate_fraction <= 1.0:
            raise ValueError(f"candidate_fraction must be in (0, 1], got {candidate_fraction}")
        if patch_reduction not in ("nearest", "mean", "kth"):
            raise ValueError(f"unknown patch_reduction {patch_reduction!r}")

        self.spec = FeatureSpec(
            backbone=backbone,
            layers=tuple(layers),
            patch_pool=patch_pool,
            projection_dim=projection_dim,
            device=device,
            dtype=feature_dtype,
        )
        self.candidate_fraction = candidate_fraction
        self.bank_ratio = bank_ratio
        self.bank_size = bank_size
        self.coreset_projection_dim = coreset_projection_dim
        self.patch_neighbors = patch_neighbors
        self.patch_reduction: PatchReduction = patch_reduction
        self.reweight = reweight
        self.reweight_neighbors = reweight_neighbors
        self.query_chunk = query_chunk
        self.batch_size = batch_size
        self.seed = seed

        self.extractor = FeatureExtractor(self.spec)
        self.memory_bank: np.ndarray | None = None
        #: (image_index, row, col) of every bank entry in the training split —
        #: what makes a nearest-normal retrieval panel reconstructable.
        self.bank_sources: np.ndarray | None = None
        self.grid: tuple[int, int] = (0, 0)
        self._bank_tensor: Any = None
        self._stats: dict[str, Any] = {}

    # -- fitting -----------------------------------------------------------
    def _fit(self, images: Iterator[np.ndarray]) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        rng = np.random.default_rng(self.seed)
        kept: list[np.ndarray] = []
        sources: list[np.ndarray] = []
        total = 0
        grid = (0, 0)

        for image_idx, (desc, grid) in enumerate(
            self.extractor.iter_embeddings(batched(images, self.batch_size))
        ):
            n_patches = desc.shape[0]
            total += n_patches
            if self.candidate_fraction >= 1.0:
                pick = np.arange(n_patches)
            else:
                n_pick = max(1, round(n_patches * self.candidate_fraction))
                pick = np.sort(rng.choice(n_patches, size=n_pick, replace=False))
            kept.append(np.asarray(desc[pick]))
            rows, cols = np.divmod(pick, grid[1])
            sources.append(np.stack([np.full_like(pick, image_idx), rows, cols], axis=1))

        if not kept:
            raise ValueError("no training images")
        candidates = np.concatenate(kept)
        candidate_sources = np.concatenate(sources)
        del kept, sources
        self.grid = grid

        requested = (
            int(self.bank_size) if self.bank_size is not None
            else max(1, round(total * float(self.bank_ratio or 0.0)))
        )
        effective = min(requested, candidates.shape[0])

        if effective >= candidates.shape[0]:
            selected = np.arange(candidates.shape[0], dtype=np.int64)
        else:
            space = candidates.astype(np.float32)
            if self.coreset_projection_dim and self.coreset_projection_dim < space.shape[1]:
                proj = random_projection(space.shape[1], self.coreset_projection_dim, self.seed)
                space = space @ proj
            selected = greedy_coreset(space, effective, seed=self.seed, device=self.spec.device)
            del space

        self.memory_bank = np.ascontiguousarray(candidates[selected])
        self.bank_sources = candidate_sources[selected]
        self._bank_tensor = None

        self._stats = {
            "patches_total": int(total),
            "candidate_count": int(candidates.shape[0]),
            "bank_requested": int(requested),
            "bank_effective": int(self.memory_bank.shape[0]),
            "bank_capped": bool(requested > candidates.shape[0]),
            "descriptor_dim": int(self.memory_bank.shape[1]),
            "bank_mb": round(self.memory_bank.nbytes / 1024**2, 2),
            "feature_grid": f"{grid[0]}x{grid[1]}",
            "fit_peak_vram_mb": round(self.extractor.peak_vram_mb(), 1),
        }

    def hparams(self) -> dict[str, Any]:
        """Every setting that changes behaviour — part of the run identity."""
        return {
            **self.spec.as_dict(),
            "candidate_fraction": self.candidate_fraction,
            "bank_ratio": self.bank_ratio,
            "bank_size": self.bank_size,
            "coreset_projection_dim": self.coreset_projection_dim,
            "patch_neighbors": self.patch_neighbors,
            "patch_reduction": self.patch_reduction,
            "reweight": self.reweight,
            "reweight_neighbors": self.reweight_neighbors,
            "seed": self.seed,
        }

    def fit_extra(self) -> dict[str, Any]:
        return {**self._stats, **self.hparams()}

    # -- scoring -----------------------------------------------------------
    def _bank(self, device):
        import torch

        if self.memory_bank is None:
            raise RuntimeError("memory bank is empty; fit() first")
        if self._bank_tensor is None or self._bank_tensor.device != device:
            # Converted once and kept resident, not per image (F07).
            self._bank_tensor = torch.as_tensor(self.memory_bank, device=device, dtype=torch.float32)
        return self._bank_tensor

    def _score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        descriptors, grid = self.extractor.embed_batch(image[None, ...])
        queries = descriptors[0].float()
        result = score_patches(
            queries,
            self._bank(queries.device),
            patch_neighbors=self.patch_neighbors,
            patch_reduction=self.patch_reduction,
            reweight=self.reweight,
            reweight_neighbors=self.reweight_neighbors,
            chunk=self.query_chunk,
        )
        anomaly_map = result.patch_scores.reshape(grid).cpu().numpy().astype(np.float64)
        return result.image_score, anomaly_map

    @staticmethod
    def reference_differences() -> list[str]:
        """Known, deliberate or unresolved differences from the author code.

        A reproduction report should cite these rather than rediscover them.
        """
        return [
            "Scoring: patch scores and the re-weighted image score agree with anomalib "
            "v2.3.0 (commit 091ca6a) on identical tensors to float32 tolerance "
            "(tests/models/test_reference_agreement.py).",
            "Coreset start: anomalib v2.3.0 uses its random start as a centre but does not "
            "return it, so its coreset is the next n greedy picks; ours returns the start "
            "plus n-1 greedy picks. Same greedy order otherwise (tested).",
            "Coreset projection: exact greedy k-center on a 128-d Gaussian projection, as in "
            "the author code's dimension; anomalib v2.3.0 uses a sparse random projection "
            "with eps=0.9 (dimension from the Johnson-Lindenstrauss bound), and the author "
            "code an approximate greedy with 10 random starting points.",
            "Candidates: a uniform per-image random sample (candidate_fraction) precedes the "
            "coreset, to bound memory; the author code and anomalib select from all patches.",
            "Descriptor: channel-wise adaptive average pooling to projection_dim after 3x3 "
            "local aggregation, following the author code; anomalib v2.3.0 concatenates the "
            "pooled layers without channel reduction (1536-d for WRN50 layer2+3).",
            "Smoothing: the evaluator upsamples the patch grid to the model input size and "
            "blurs there (sigma=4 input px), the same order and sigma as anomalib's anomaly "
            "map generator, then resizes to native for evaluation. Interpolation and kernel "
            "truncation are not compared element-wise, nor against the author repository.",
        ]

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        if self.memory_bank is None or self.bank_sources is None:
            raise RuntimeError("model is not fitted")
        return {
            "memory_bank": self.memory_bank,
            "bank_sources": self.bank_sources,
            "grid": np.asarray(self.grid),
            "hparams": self.hparams(),
            "transform": self.transform.as_dict(),
            "stats": self._stats,
        }

    def load_state_dict(self, state: dict[str, Any]) -> PatchCore:
        """Restore a fitted bank. Hyperparameters must match the constructor."""
        if state["hparams"] != self.hparams():
            raise ValueError(
                "hyperparameters differ from the saved model; construct the model with the "
                f"saved settings. saved={state['hparams']}, current={self.hparams()}"
            )
        self.memory_bank = np.asarray(state["memory_bank"])
        self.bank_sources = np.asarray(state["bank_sources"])
        self.grid = (int(state["grid"][0]), int(state["grid"][1]))
        self._stats = dict(state.get("stats", {}))
        self._bank_tensor = None
        self._fitted = True
        return self
