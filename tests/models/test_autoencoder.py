"""Autoencoder regression tests (docs/13, F06).

A 1500x1000 VisA image at long side 256 is 256x171. A depth-4 network halves
the grid four times, so it cannot round-trip 171 rows: the output came back
160 rows high, and training compared that directly with the 171-row input.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inspector.data.transforms import ImageTransform  # noqa: E402
from inspector.models.autoencoder import ConvAutoencoder, _build_network  # noqa: E402


@pytest.fixture
def model():
    m = ConvAutoencoder(ImageTransform(long_side=256), depth=4, latent_dim=16, base_channels=8, device="cpu")
    m.network = _build_network(3, m.base_channels, m.depth, m.latent_dim)
    return m


@pytest.mark.parametrize("shape", [(2, 3, 171, 256), (1, 3, 97, 131), (2, 3, 64, 64)])
def test_reconstruction_keeps_the_input_extent(model, shape):
    out = model._reconstruct(torch.randn(*shape))
    assert out.shape == shape


def test_non_square_training_step_runs(model):
    """Forward, loss and backward on the exact failing shape from the review."""
    batch = torch.randn(2, 3, 171, 256)
    loss = model._loss(model._reconstruct(batch), batch)
    loss.backward()
    grads = [p.grad for p in model.network.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_ssim_data_range_follows_the_normalization():
    imagenet = ConvAutoencoder(ImageTransform(normalize="imagenet"), loss="ssim", device="cpu")
    unit = ConvAutoencoder(ImageTransform(normalize="unit"), loss="ssim", device="cpu")
    assert imagenet._data_range == pytest.approx(1 / 0.224, rel=1e-3)
    assert unit._data_range == 1.0


def test_combined_loss_scoring_choice_is_explicit_and_recorded():
    """L2+SSIM training scored with SSIM alone was implicit; it is now a
    recorded setting, and changing it changes the run identity."""
    default = ConvAutoencoder(loss="l2+ssim", device="cpu")
    summed = ConvAutoencoder(loss="l2+ssim", score_map="l2+ssim", device="cpu")
    assert default.hparams()["score_map"] == "ssim"
    assert summed.hparams()["score_map"] == "l2+ssim"
    assert default.hparams() != summed.hparams()


@pytest.mark.slow
def test_fit_and_score_on_non_square_fixture(tmp_path):
    from inspector.data import SYNTHETIC, discover_category
    from inspector.fixtures import FixtureSpec, generate

    root = generate(tmp_path, spec=FixtureSpec(n_train=8, n_validation=4, n_test_good=2, n_test_per_defect=1), overwrite=True)
    indices = discover_category(root, "synth_strip", layout=SYNTHETIC)  # 256x64: 4:1
    m = ConvAutoencoder(ImageTransform(long_side=200), epochs=2, batch_size=4, latent_dim=16,
                        base_channels=8, device="cpu").fit(indices["train"])
    raw = m.predict_raw(np.asarray(__import__("PIL.Image").Image.open(indices["test_public"][0].image_path).convert("RGB")))
    assert raw.raw_map.shape == (raw.input_size[1], raw.input_size[0])
    assert np.isfinite(raw.score)
