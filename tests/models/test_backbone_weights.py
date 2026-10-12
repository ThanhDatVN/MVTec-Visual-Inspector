"""Backbone weights (torchvision names or a timm tag), and the extractor's lifetime."""

from __future__ import annotations

import pytest

from inspector.models import features
from inspector.models.features import FeatureExtractor, FeatureSpec

torch = pytest.importorskip("torch")


def test_weights_are_part_of_the_run_identity():
    from inspector.models.patchcore import PatchCore

    default = PatchCore(bank_ratio=None, bank_size=10)
    timm_w = PatchCore(bank_ratio=None, bank_size=10, weights="timm:racm_in1k")
    assert default.hparams()["weights"] == "IMAGENET1K_V1"
    assert timm_w.hparams()["weights"] == "timm:racm_in1k"
    assert default.hparams() != timm_w.hparams()


def test_timm_resnets_keep_torchvision_parameter_names():
    timm = pytest.importorskip("timm")
    import torchvision

    for arch in ("wide_resnet50_2", "resnet50", "resnet18"):
        tv = torchvision.models.__dict__[arch](weights=None).state_dict()
        tm = timm.create_model(arch, pretrained=False).state_dict()
        assert tv.keys() == tm.keys(), arch
        assert all(tv[k].shape == tm[k].shape for k in tv), arch


def test_a_timm_tag_is_loaded_into_the_torchvision_model(monkeypatch):
    timm = pytest.importorskip("timm")
    reference = timm.create_model("resnet18", pretrained=False).state_dict()
    calls = []

    def fake(backbone, tag):  # stands in for the download
        calls.append((backbone, tag))
        return reference

    monkeypatch.setattr(features, "timm_state_dict", fake)
    extractor = FeatureExtractor(FeatureSpec(backbone="resnet18", weights="timm:some_tag", device="cpu"))
    model = extractor.model
    assert calls == [("resnet18", "some_tag")]
    built = model.state_dict()
    assert all(torch.equal(built[k].cpu(), reference[k]) for k in reference)


def test_a_mismatched_state_dict_fails_loudly(monkeypatch):
    monkeypatch.setattr(features, "timm_state_dict", lambda backbone, tag: {"fc.weight": torch.zeros(1)})
    extractor = FeatureExtractor(FeatureSpec(backbone="resnet18", weights="timm:broken", device="cpu"))
    with pytest.raises(RuntimeError):
        extractor.model  # noqa: B018


def test_a_dropped_extractor_is_freed_without_the_cycle_collector():
    """Forward hooks closing over the extractor made a reference cycle: every finished run kept
    its backbone (and on a GPU ~300 MB) until Python's cycle collector happened to run."""
    import gc
    import weakref

    import numpy as np
    import torchvision  # noqa: F401  (imported once, as in any session after its first run)

    gc.disable()
    try:
        extractor = FeatureExtractor(FeatureSpec(backbone="resnet18", device="cpu"))
        extractor.embed_batch(np.zeros((1, 3, 64, 64), np.float32))
        ref = weakref.ref(extractor)
        del extractor
        assert ref() is None
    finally:
        gc.enable()
