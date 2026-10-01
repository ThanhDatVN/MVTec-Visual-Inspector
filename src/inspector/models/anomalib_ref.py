"""anomalib's PatchCore as a comparator inside the shared runner (E00, step C2; owner=lib).

Step C1 showed our scoring matches anomalib v2.3.0 on identical tensors. C2 asks what the
remaining, deliberate differences cost on real images: anomalib keeps the 1536-d concatenated
descriptor (we pool to 1024), selects the coreset from *all* patches with a sparse random
projection (we sample 10% of patches per image first and project to 128-d), and loads timm's
backbone weights (we load torchvision's). Running anomalib's own pipeline (vendored at the
pinned commit, `models/reference/`) through the same runner, inputs and evaluator makes the
comparison about the models and nothing else.

`weights` selects the timm pretrained tag: None is timm's default for the backbone (what
anomalib loads), `"tv_in1k"` is the torchvision ImageNet weights our PatchCore uses — the
switch that separates "different weights" from "different pipeline".

Memory: all patches are kept before the coreset (fp32, 1536-d): ~5.9 GB of host RAM and the
same on the GPU at 320 px for 768 training images. Fine on a 16 GB Kaggle GPU; not on 4 GB.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ..data.transforms import ImageTransform
from .base import AnomalyModel
from .features import batched, resolve_device


class AnomalibPatchCore(AnomalyModel):
    name = "anomalib_patchcore"
    tier = "T3"
    owner = "lib"

    def __init__(
        self,
        transform: ImageTransform | None = None,
        *,
        backbone: str = "wide_resnet50_2",
        weights: str | None = None,
        layers: tuple[str, ...] = ("layer2", "layer3"),
        coreset_sampling_ratio: float = 0.01,
        num_neighbors: int = 9,
        batch_size: int = 4,
        device: str = "auto",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(transform, **kwargs)
        self.backbone = backbone
        self.weights = weights
        self.layers = tuple(layers)
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.seed = seed
        self._model: Any = None
        self._stats: dict[str, Any] = {}

    def hparams(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone,
            "weights": self.weights,
            "layers": list(self.layers),
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "num_neighbors": self.num_neighbors,
            "reference": "anomalib v2.3.0 @ 091ca6a (vendored)",
            "seed": self.seed,
        }

    def fit_extra(self) -> dict[str, Any]:
        return {**self._stats, **self.hparams()}

    def _fit(self, images: Iterator[np.ndarray]) -> None:
        import torch

        from .reference.anomalib_v2_3_0 import PatchcoreModel

        # anomalib draws the coreset start from torch's RNG and the projection from NumPy's
        # global state; seeding both makes a run reproducible without touching its code.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)  # noqa: NPY002 - anomalib draws from the global state
        name = self.backbone if not self.weights else f"{self.backbone}.{self.weights}"
        model = PatchcoreModel(layers=list(self.layers), backbone=name, pre_trained=True,
                               num_neighbors=self.num_neighbors).to(self.device)
        model.train()
        total = 0
        for batch in batched(images, self.batch_size):
            embedding = model(torch.as_tensor(batch, device=self.device))
            # Keep the store in host memory; anomalib keeps it on the device it ran on.
            model.embedding_store[-1] = embedding.detach().cpu()
            total += int(embedding.shape[0])
        model.subsample_embedding(self.coreset_sampling_ratio, device=self.device)
        model.eval()
        self._model = model
        self._stats = {
            "patches_total": total,
            "bank_effective": int(model.memory_bank.shape[0]),
            "descriptor_dim": int(model.memory_bank.shape[1]),
            "bank_mb": round(model.memory_bank.numel() * 4 / 1024**2, 2),
        }

    def _score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        import torch

        with torch.no_grad():
            pred_score, patch_scores = self._model(torch.as_tensor(image[None], device=self.device))
        return float(pred_score[0]), patch_scores[0, 0].float().cpu().numpy()
