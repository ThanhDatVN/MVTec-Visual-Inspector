"""Frozen backbone feature extraction.

Shared by every Tier 2 and Tier 3 method. The design is dictated by the 4 GB
card (docs/06 §3):

* features leave the GPU as **fp16** and are accumulated on the host, because
  `walnuts` at 512² produces 1.77 M patch vectors and 7.2 GB in fp32;
* distances are nonetheless accumulated in **fp32**, since fp16 accumulation
  over 1024 dimensions loses real precision;
* extraction is chunked, so peak VRAM depends on batch size rather than on the
  size of the training set.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Layer names understood for the torchvision ResNet family.
RESNET_LAYERS = ("layer1", "layer2", "layer3", "layer4")

BACKBONES = {
    "wide_resnet50_2": "WideResNet50-2, ImageNet. PatchCore's published default.",
    "resnet50": "ResNet50, ImageNet.",
    "resnet18": "ResNet18, ImageNet. The cheap option for the 4 GB card.",
}


@dataclass(frozen=True)
class FeatureSpec:
    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    #: 3x3 average pooling over the feature map. This is PatchCore's "local
    #: neighbourhood aggregation": each patch descriptor sees its neighbours, so
    #: a defect smaller than the receptive field still shifts a descriptor.
    patch_pool: int = 3
    #: Target descriptor dimension after adaptive pooling across channels.
    projection_dim: int = 1024
    device: str = "auto"
    dtype: str = "float16"

    def as_dict(self) -> dict:
        return {
            "backbone": self.backbone,
            "layers": list(self.layers),
            "patch_pool": self.patch_pool,
            "projection_dim": self.projection_dim,
            "dtype": self.dtype,
        }


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class FeatureExtractor:
    """A frozen CNN that returns patch descriptors for a batch of images."""

    def __init__(self, spec: FeatureSpec | None = None):
        self.spec = spec or FeatureSpec()
        self.device = resolve_device(self.spec.device)
        self._model = None
        # Values are torch Tensors. Annotated loosely because torch must not
        # appear in a module-level annotation: this module is imported by the
        # torch-free CI path.
        self._features: dict[str, Any] = {}

    def _build(self):
        import torchvision

        if self.spec.backbone not in BACKBONES:
            raise ValueError(
                f"unknown backbone {self.spec.backbone!r}; known: {sorted(BACKBONES)}"
            )
        for layer in self.spec.layers:
            if layer not in RESNET_LAYERS:
                raise ValueError(f"unknown layer {layer!r}; known: {RESNET_LAYERS}")

        factory = getattr(torchvision.models, self.spec.backbone)
        model = factory(weights="IMAGENET1K_V1")
        model.eval().to(self.device)
        for param in model.parameters():
            param.requires_grad_(False)

        # Forward hooks rather than a rebuilt truncated network: hooks cannot
        # get the layer wiring subtly wrong, and they keep the published
        # architecture intact.
        def make_hook(name: str):
            def hook(_module, _inputs, output):
                self._features[name] = output

            return hook

        for layer in self.spec.layers:
            getattr(model, layer).register_forward_hook(make_hook(layer))

        self._model = model
        return model

    @property
    def model(self):
        return self._model if self._model is not None else self._build()

    def embed_batch(self, batch: np.ndarray):
        """Patch descriptors for one batch.

        Args:
            batch: float32 array of shape (B, 3, H, W), already normalized.

        Returns:
            (descriptors, grid) where `descriptors` is a torch tensor of shape
            (B, n_patches, D) and `grid` is the (h, w) feature-map size that
            `n_patches` unflattens to.
        """
        import torch
        import torch.nn.functional as F

        model = self.model
        tensor = torch.as_tensor(batch, dtype=torch.float32, device=self.device)

        self._features.clear()
        with torch.no_grad():
            model(tensor)

        maps: list[Any] = [self._features[layer] for layer in self.spec.layers]
        # The first requested layer defines the output grid. Higher layers are
        # upsampled to it: the study's whole point is that the *finest* grid is
        # what small defects need, so coarsening to the smallest would discard
        # exactly the information under investigation.
        target = maps[0].shape[-2:]
        aligned = [
            m if m.shape[-2:] == target else F.interpolate(m, size=target, mode="bilinear", align_corners=False)
            for m in maps
        ]

        pooled: list[Any] = []
        for m in aligned:
            if self.spec.patch_pool > 1:
                m = F.avg_pool2d(m, kernel_size=self.spec.patch_pool, stride=1,
                                 padding=self.spec.patch_pool // 2)
            pooled.append(m)

        stacked = torch.cat(pooled, dim=1)  # (B, C_total, h, w)
        b, c, h, w = stacked.shape
        flat = stacked.permute(0, 2, 3, 1).reshape(b, h * w, c)

        if self.spec.projection_dim and self.spec.projection_dim != c:
            # Adaptive average pooling over the channel axis, as in PatchCore:
            # a fixed descriptor size independent of the backbone, with no
            # learned parameters to fit and therefore nothing to leak.
            flat = F.adaptive_avg_pool1d(flat, self.spec.projection_dim)

        out_dtype = torch.float16 if self.spec.dtype == "float16" else torch.float32
        return flat.to(out_dtype), (h, w)

    def embed_all(
        self, batches: Iterable[np.ndarray], *, progress=None
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Descriptors for every image, concatenated on the host as fp16."""
        chunks: list[np.ndarray] = []
        grid: tuple[int, int] = (0, 0)
        seen = 0
        for batch in batches:
            descriptors, grid = self.embed_batch(batch)
            chunks.append(descriptors.reshape(-1, descriptors.shape[-1]).cpu().numpy())
            seen += batch.shape[0]
            if progress:
                progress(seen)
        if not chunks:
            raise ValueError("no images supplied")
        return np.concatenate(chunks, axis=0), grid

    def peak_vram_mb(self) -> float:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024**2


def batched(images: Iterable[np.ndarray], batch_size: int) -> Iterator[np.ndarray]:
    """Group preprocessed (3, H, W) images into (B, 3, H, W) batches."""
    buffer: list[np.ndarray] = []
    for image in images:
        buffer.append(image)
        if len(buffer) == batch_size:
            yield np.stack(buffer)
            buffer = []
    if buffer:
        yield np.stack(buffer)
