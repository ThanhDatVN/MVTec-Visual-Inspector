"""Tiled foundation-feature nearest-neighbour model (R1 + R8, after SuperADD).

PatchCore's idea — a memory bank of normal patch features, a test patch scored by its distance
to the nearest one — with the three changes that took training-free memory-bank methods to the
best published segmentation on MVTec AD 2 (SuperAD, arXiv:2505.19750; SuperADD,
arXiv:2605.14808), reimplemented here from their descriptions:

* **Self-supervised ViT features** (DINOv3 through timm), several transformer blocks, each with
  its own memory bank; the per-block distance maps are divided by the feature dimension and
  averaged.
* **Overlapping tiles at near-native resolution.** The image is scaled by ``scale`` (0.625, so a
  2448 px side becomes 1530 px) and covered by ``tile`` px windows overlapping by at least
  ``overlap`` px on each side; each token is taken from the window in which it is most central.
  A 640 px window over a 1530 px image is what the E19 resolution finding asks for, without the
  host-memory limit of embedding the whole image at once.
* **A bank chosen for coverage.** Candidates are sampled per image, then reduced per block to
  ``bank_size`` by density: in random subsets, a point with few neighbours closer than the mean
  k-NN distance is kept before one in a crowded region (near-duplicate background patches are
  dropped first).

Training images are also seen at a random brightness factor in ``brightness`` — the lighting
augmentation of E14d/SuperADD. Calibration and test images are never altered.

Distances are computed as |q|^2 + |k|^2 - 2 q.k with fp16 operands and fp32 accumulation and
output (`torch.mm(..., out_dtype=float32)`), after centring both sides on the bank mean, which
leaves distances unchanged and keeps the cross term small.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..data.transforms import ImageTransform, load_image
from .base import AnomalyModel


def tile_plan(length: int, tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Windows along one axis, in tokens: (start, end, keep_from, keep_to).

    Windows of `tile` tokens, evenly spread so the first starts at 0 and the last ends at
    `length`, with at least `overlap` tokens shared on each side of an interior boundary. Each
    token is kept from the window in which it lies furthest from the edge: the boundary between
    two windows is the middle of their overlap. An axis shorter than `tile` is one window.
    """
    if length <= tile:
        return [(0, length, 0, length)]
    step = tile - 2 * overlap
    if step <= 0:
        raise ValueError(f"tile {tile} must exceed twice the overlap {overlap}")
    n = math.ceil((length - tile) / step) + 1
    starts = [round(i * (length - tile) / (n - 1)) for i in range(n)]
    cuts = [0] + [(starts[i] + starts[i - 1] + tile) // 2 for i in range(1, n)] + [length]
    return [(s, s + tile, cuts[i], cuts[i + 1]) for i, s in enumerate(starts)]


def load_vit(backbone: str, weights: str):
    """timm's pretrained `<backbone>.<weights>` at any input size (downloaded on first use)."""
    import timm

    return timm.create_model(f"{backbone}.{weights}", pretrained=True, dynamic_img_size=True)


class TiledKNN(AnomalyModel):
    name = "tiled_knn"
    tier = "T3"
    owner = "own"
    stochastic = True

    def __init__(
        self,
        transform: ImageTransform | None = None,
        *,
        backbone: str = "vit_large_patch16_dinov3",
        weights: str = "lvd1689m",
        layers: tuple[int, ...] = (5, 11, 17, 23),
        scale: float = 0.625,
        tile: int = 640,
        overlap: int = 128,
        brightness: tuple[float, float] = (0.8, 1.2),
        candidate_fraction: float = 0.1,
        bank_size: int = 100_000,
        density_neighbors: int = 100,
        subset_size: int = 4096,
        image_score: str = "max",
        tile_batch: int = 2,
        query_chunk: int = 4096,
        bank_chunk: int = 32768,
        device: str = "auto",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(transform or ImageTransform(mode="none", long_side=None), **kwargs)
        if not 0.0 < candidate_fraction <= 1.0:
            raise ValueError(f"candidate_fraction must be in (0, 1], got {candidate_fraction}")
        if image_score not in ("max",):
            raise ValueError(f"unknown image_score {image_score!r}")
        self.backbone, self.weights = backbone, weights
        self.layers = tuple(int(i) for i in layers)
        self.scale, self.tile, self.overlap = float(scale), int(tile), int(overlap)
        self.brightness = (float(brightness[0]), float(brightness[1]))
        self.candidate_fraction, self.bank_size = float(candidate_fraction), int(bank_size)
        self.density_neighbors, self.subset_size = int(density_neighbors), int(subset_size)
        self.image_score = image_score
        self.tile_batch, self.query_chunk, self.bank_chunk = int(tile_batch), int(query_chunk), int(bank_chunk)
        self.device_request, self.seed = device, int(seed)
        self._net = None
        self._device: str | None = None
        self.banks: list[Any] = []      # per layer, device fp16 (n, C), centred
        self.means: list[Any] = []      # per layer, device fp16 (C,)
        self.bank_norms: list[Any] = []  # per layer, device fp32 (n,)
        self._stats: dict[str, Any] = {}

    # -- backbone ---------------------------------------------------------
    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device_request == "auto" \
                else self.device_request
        return self._device

    @property
    def net(self):
        if self._net is None:
            import torch

            net = load_vit(self.backbone, self.weights).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            depth = len(net.blocks)
            if any(not 0 <= i < depth for i in self.layers):
                raise ValueError(f"layers {self.layers} outside the {depth} blocks of {self.backbone}")
            self._net = net.to(self.device, dtype=torch.float16 if self.device != "cpu" else torch.float32)
        return self._net

    @property
    def patch(self) -> int:
        return int(self.net.patch_embed.patch_size[0])

    def _grid(self, image: np.ndarray) -> list[Any]:
        """Per-layer feature grids (C, Ht, Wt) of one normalized CHW image, tiled and stitched."""
        import torch
        import torch.nn.functional as F

        net, p = self.net, self.patch
        dtype = next(net.parameters()).dtype
        x = torch.as_tensor(image, device=self.device)[None]
        h, w = x.shape[-2:]
        th, tw = max(p, round(h * self.scale / p) * p), max(p, round(w * self.scale / p) * p)
        x = F.interpolate(x, size=(th, tw), mode="bicubic", align_corners=False, antialias=True).to(dtype)
        ht, wt = th // p, tw // p
        ty = tile_plan(ht, self.tile // p, self.overlap // p)
        tx = tile_plan(wt, self.tile // p, self.overlap // p)
        windows = [(a, b) for a in ty for b in tx]
        grids: list[Any] = [None] * len(self.layers)
        with torch.inference_mode():
            for i in range(0, len(windows), self.tile_batch):
                batch = windows[i:i + self.tile_batch]
                crops = torch.stack([x[0, :, ys * p:ye * p, xs * p:xe * p] for (ys, ye, _, _), (xs, xe, _, _) in batch])
                feats = net.forward_intermediates(crops, indices=list(self.layers), norm=False,
                                                  output_fmt="NCHW", intermediates_only=True)
                for li, f in enumerate(feats):
                    if grids[li] is None:
                        grids[li] = torch.empty((f.shape[1], ht, wt), device=self.device, dtype=torch.float16)
                    for bi, ((ys, _, ky0, ky1), (xs, _, kx0, kx1)) in enumerate(batch):
                        grids[li][:, ky0:ky1, kx0:kx1] = f[bi, :, ky0 - ys:ky1 - ys, kx0 - xs:kx1 - xs].half()
        self._stats.setdefault("tiles_per_image", len(windows))
        return grids

    # -- fitting ----------------------------------------------------------
    def _iter_preprocessed(self, index) -> Iterator[np.ndarray]:
        rng = np.random.default_rng((self.seed, 7))
        lo, hi = self.brightness
        for sample in index:
            image = load_image(sample.image_path).astype(np.float32)
            if (lo, hi) != (1.0, 1.0):
                image = np.clip(image * rng.uniform(lo, hi), 0.0, 255.0)
            yield self.transform.normalize_image(self.transform.resize_image(image))

    def _fit(self, images: Iterator[np.ndarray]) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rng = np.random.default_rng(self.seed)
        pools: list[list[np.ndarray]] = [[] for _ in self.layers]
        total = 0
        for image in images:
            grids = self._grid(image)
            n = grids[0].shape[1] * grids[0].shape[2]
            total += n
            pick = torch.as_tensor(np.sort(rng.choice(n, max(1, round(n * self.candidate_fraction)), replace=False)),
                                   device=self.device)
            for li, g in enumerate(grids):
                pools[li].append(g.reshape(g.shape[0], -1)[:, pick].T.contiguous().cpu().numpy())
            self._stats["feature_grid"] = f"{grids[0].shape[1]}x{grids[0].shape[2]}"
        if not total:
            raise ValueError("no training images")
        self.banks, self.means, self.bank_norms = [], [], []
        for li, pool in enumerate(pools):
            cand = np.concatenate(pool)
            pools[li] = []
            keep = density_subsample(cand, self.bank_size, k=self.density_neighbors, subset=self.subset_size,
                                     device=self.device, seed=self.seed + li)
            bank = torch.as_tensor(cand[keep], device=self.device)
            mean = bank.float().mean(0)
            bank = (bank.float() - mean).half()
            self.banks.append(bank)
            self.means.append(mean.half())
            self.bank_norms.append(bank.float().square().sum(1))
            self._stats[f"candidates_l{self.layers[li]}"] = len(cand)
            del cand
        self._stats.update({
            "patches_total": total, "bank_effective": int(self.banks[0].shape[0]),
            "descriptor_dims": [int(b.shape[1]) for b in self.banks],
            "fit_peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0.0,
        })

    # -- scoring ----------------------------------------------------------
    def _nn_distance(self, li: int, queries) -> Any:
        """Euclidean distance of each query (N, C) fp16 to its nearest bank entry, fp32."""
        import torch

        bank, norms = self.banks[li], self.bank_norms[li]
        q = queries - self.means[li]
        qn = q.float().square().sum(1)
        best = torch.full((q.shape[0],), float("inf"), device=q.device)
        for a in range(0, q.shape[0], self.query_chunk):
            qa = q[a:a + self.query_chunk]
            for b in range(0, bank.shape[0], self.bank_chunk):
                cross = torch.mm(qa, bank[b:b + self.bank_chunk].T, out_dtype=torch.float32) if q.is_cuda \
                    else qa.float() @ bank[b:b + self.bank_chunk].float().T
                d2 = qn[a:a + self.query_chunk, None] + norms[None, b:b + self.bank_chunk] - 2.0 * cross
                best[a:a + self.query_chunk] = torch.minimum(best[a:a + self.query_chunk], d2.min(1).values)
        return best.clamp_min(0).sqrt()

    def _score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        import torch

        if not self.banks:
            raise RuntimeError("memory bank is empty; fit() first")
        grids = self._grid(image)
        _, ht, wt = grids[0].shape
        with torch.inference_mode():
            maps = [self._nn_distance(li, g.reshape(g.shape[0], -1).T).reshape(ht, wt) / g.shape[0]
                    for li, g in enumerate(grids)]
            anomaly_map = torch.stack(maps).mean(0).cpu().numpy().astype(np.float64)
        return float(anomaly_map.max()), anomaly_map

    # -- identity ---------------------------------------------------------
    def hparams(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone, "weights": self.weights, "layers": list(self.layers),
            "scale": self.scale, "tile": self.tile, "overlap": self.overlap, "brightness": list(self.brightness),
            "candidate_fraction": self.candidate_fraction, "bank_size": self.bank_size,
            "density_neighbors": self.density_neighbors, "subset_size": self.subset_size,
            "image_score": self.image_score, "dtype": "float16", "seed": self.seed,
        }

    def fit_extra(self) -> dict[str, Any]:
        return {**self._stats, **self.hparams()}


def density_subsample(x: np.ndarray, n: int, *, k: int = 100, subset: int = 4096, device: str = "cpu",
                      seed: int = 0) -> np.ndarray:
    """Indices of `n` rows of `x` that cover it, preferring points in sparse regions.

    The rows are split into random subsets of about `subset`. Within each, a point's crowding is
    the number of its `k` nearest neighbours closer than the subset's mean k-NN distance; each
    subset keeps its share of `n` least-crowded points (ties broken at random). Near-duplicates —
    the thousand patches of one uniform background — share a high crowding and are dropped first.
    """
    import torch

    m = len(x)
    if m <= n:
        return np.arange(m)
    rng = np.random.default_rng(seed)
    groups = np.array_split(rng.permutation(m), math.ceil(m / subset))
    quota = np.floor(n * np.array([len(g) for g in groups]) / m).astype(int)
    quota[: n - quota.sum()] += 1  # distribute the remainder
    kept = []
    for g, q in zip(groups, quota):
        t = torch.as_tensor(x[g], device=device).float()
        d = torch.cdist(t, t)
        d.fill_diagonal_(float("inf"))
        kd = d.topk(min(k, len(g) - 1), largest=False).values
        crowd = (kd < kd.mean()).sum(1).cpu().numpy() + rng.random(len(g))
        kept.append(g[np.argsort(crowd)[:q]])
    return np.sort(np.concatenate(kept))


__all__ = ["TiledKNN", "density_subsample", "tile_plan"]
