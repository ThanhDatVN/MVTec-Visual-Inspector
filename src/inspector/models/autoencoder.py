"""Convolutional autoencoder baseline (docs/03, Tier 1).

The scope's required from-scratch baseline. Its job is not to win — it is to
quantify, in points, what the pretrained prior in Tier 2/3 is actually worth. A
project that starts at PatchCore cannot make that measurement.

Two losses, because the comparison between them teaches more than either alone:

* **L2** blurs. A blurry reconstruction produces error *everywhere*, not at the
  defect, so the anomaly map has poor contrast — the classic failure.
* **SSIM** compares local structure (luminance, contrast, correlation) instead of
  per-pixel intensity, which is much closer to what "looks wrong" means to an
  inspector. Expect a clear gain; it is the most instructive ablation in Tier 1.

Model selection uses validation reconstruction error on **normal images only**.
There is no anomalous validation data, so early stopping cannot peek at the
quantity we ultimately report (protocol §3.4).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ..data.transforms import ImageTransform
from .base import AnomalyModel
from .features import resolve_device


def _build_network(in_channels: int, base: int, depth: int, latent: int):
    """Symmetric conv encoder/decoder. Built lazily so torch stays optional."""
    import torch
    from torch import nn

    class ConvAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            encoder: list[nn.Module] = []
            channels = in_channels
            for level in range(depth):
                out_channels = base * (2**level)
                encoder += [
                    nn.Conv2d(channels, out_channels, 4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
                channels = out_channels
            encoder += [nn.Conv2d(channels, latent, 3, stride=1, padding=1)]
            self.encoder = nn.Sequential(*encoder)

            decoder: list[nn.Module] = [
                nn.Conv2d(latent, channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            for level in reversed(range(depth)):
                out_channels = base * (2 ** max(0, level - 1)) if level else in_channels
                decoder += [nn.ConvTranspose2d(channels, out_channels, 4, stride=2, padding=1)]
                if level:
                    decoder += [nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True)]
                channels = out_channels
            self.decoder = nn.Sequential(*decoder)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))

    return ConvAE()


def ssim_map(a, b, *, window: int = 11, sigma: float = 1.5, data_range: float = 2.0):
    """Per-pixel SSIM between two (B, C, H, W) tensors.

    Returned as a map rather than a scalar: the same computation then serves as
    the training loss (1 - mean) and as the anomaly map (1 - map), which
    guarantees the map a model is scored on is the quantity it was trained to
    minimize.
    """
    import torch
    import torch.nn.functional as F

    channels = a.shape[1]
    coords = torch.arange(window, dtype=a.dtype, device=a.device) - window // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] @ gauss[None, :]).expand(channels, 1, window, window).contiguous()

    def blur(x):
        return F.conv2d(x, kernel, padding=window // 2, groups=channels)

    mu_a, mu_b = blur(a), blur(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = blur(a * a) - mu_a2
    sigma_b2 = blur(b * b) - mu_b2
    sigma_ab = blur(a * b) - mu_ab

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return numerator / (denominator + 1e-12)


class ConvAutoencoder(AnomalyModel):
    """Conv autoencoder scored by reconstruction residual."""

    name = "cae"
    tier = "T1"
    owner = "own"
    stochastic = True

    def __init__(
        self,
        transform: ImageTransform | None = None,
        *,
        loss: str = "l2",
        latent_dim: int = 128,
        base_channels: int = 32,
        depth: int = 4,
        epochs: int = 60,
        lr: float = 2e-4,
        batch_size: int = 8,
        val_fraction: float = 0.1,
        early_stopping_patience: int = 12,
        residual: str = "multiscale",
        device: str = "auto",
        seed: int = 0,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(transform, **kwargs)
        if loss not in ("l2", "ssim", "l2+ssim"):
            raise ValueError(f"unknown loss {loss!r}; use l2, ssim or l2+ssim")
        if residual not in ("raw", "multiscale"):
            raise ValueError(f"unknown residual {residual!r}; use raw or multiscale")

        self.loss_name = loss
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.patience = early_stopping_patience
        self.residual = residual
        self.device = resolve_device(device)
        self.seed = seed
        self.verbose = verbose

        self.network: Any = None
        self.history: list[dict[str, float]] = []
        self._stats: dict[str, Any] = {}
        self.name = f"cae_{loss.replace('+', '_')}"

    # -- training ----------------------------------------------------------
    def _loss(self, output, target):
        import torch.nn.functional as F

        if self.loss_name == "l2":
            return F.mse_loss(output, target)
        structural = 1.0 - ssim_map(output, target).mean()
        if self.loss_name == "ssim":
            return structural
        return F.mse_loss(output, target) + structural

    def _fit(self, images: Iterator[np.ndarray]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        stacked = np.stack(list(images))
        if stacked.shape[0] < 4:
            raise ValueError(f"need at least 4 training images, got {stacked.shape[0]}")

        # A held-out slice of the *training* normals for early stopping. The
        # protocol's validation split is reserved for threshold selection; using
        # it here as well would tune two things on one sample.
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(stacked.shape[0])
        n_val = max(2, int(stacked.shape[0] * self.val_fraction))
        val_data = stacked[order[:n_val]]
        train_data = stacked[order[n_val:]]

        self.network = _build_network(
            stacked.shape[1], self.base_channels, self.depth, self.latent_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        loader = DataLoader(
            TensorDataset(torch.as_tensor(train_data, dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_tensor = torch.as_tensor(val_data, dtype=torch.float32, device=self.device)

        best = float("inf")
        best_state = None
        stale = 0

        for epoch in range(self.epochs):
            self.network.train()
            running = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(self.network(batch), batch)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * batch.shape[0]
            train_loss = running / max(1, len(train_data))

            self.network.eval()
            with torch.no_grad():
                val_loss = float(self._loss(self.network(val_tensor), val_tensor).item())

            self.history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
            if self.verbose:
                print(f"  epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}")

            if val_loss < best - 1e-6:
                best, stale = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.network.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()

        self._stats = {
            "epochs_run": len(self.history),
            "best_val_loss": round(best, 6),
            "n_train": int(train_data.shape[0]),
            "n_holdout": int(val_data.shape[0]),
            "parameters": sum(p.numel() for p in self.network.parameters()),
            "peak_vram_mb": round(
                torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0, 1
            ),
        }

    def fit_extra(self) -> dict[str, Any]:
        return {
            **self._stats,
            "loss": self.loss_name,
            "latent_dim": self.latent_dim,
            "residual": self.residual,
        }

    # -- scoring -----------------------------------------------------------
    def _score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        import torch
        import torch.nn.functional as F

        if self.network is None:
            raise RuntimeError("model is not fitted")

        tensor = torch.as_tensor(image[None, ...], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            reconstruction = self.network(tensor)
            if reconstruction.shape[-2:] != tensor.shape[-2:]:
                reconstruction = F.interpolate(
                    reconstruction, size=tensor.shape[-2:], mode="bilinear", align_corners=False
                )

            if self.loss_name in ("ssim", "l2+ssim"):
                # Score with the same quantity that was optimized, rather than
                # training on structure and then scoring on intensity.
                residual = (1.0 - ssim_map(reconstruction, tensor)).mean(dim=1, keepdim=True)
            else:
                residual = (reconstruction - tensor).pow(2).mean(dim=1, keepdim=True)

            if self.residual == "multiscale":
                # Raw per-pixel residuals are dominated by edge misalignment;
                # averaging across scales suppresses that while keeping a
                # genuinely anomalous region, which is wrong at every scale.
                scales = [residual]
                for factor in (2, 4):
                    small = F.avg_pool2d(residual, factor)
                    scales.append(
                        F.interpolate(small, size=residual.shape[-2:], mode="bilinear", align_corners=False)
                    )
                residual = torch.stack(scales).mean(dim=0)

        anomaly_map = residual[0, 0].cpu().numpy().astype(np.float64)
        # Max, not mean: a small defect barely moves an image-wide mean, which
        # is the aggregation failure documented in reports/P2-tier0-findings.md.
        # Top-k mean is the ablation to run against this.
        return float(anomaly_map.max()), anomaly_map

    def state_dict(self) -> dict[str, Any]:
        if self.network is None:
            raise RuntimeError("model is not fitted")
        return {
            "network": {k: v.cpu().numpy() for k, v in self.network.state_dict().items()},
            "loss": self.loss_name,
            "latent_dim": self.latent_dim,
            "transform": self.transform.as_dict(),
            "stats": self._stats,
            "history": self.history,
        }
