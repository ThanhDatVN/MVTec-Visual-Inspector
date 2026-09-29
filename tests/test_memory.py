"""Stage-wise peak memory measurement (docs/13, F07)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from inspector.utils.memory import PeakRSS, PeakVRAM


def test_peak_rss_sees_a_transient_allocation():
    pytest.importorskip("psutil")
    with PeakRSS(interval=0.005) as before:
        pass
    with PeakRSS(interval=0.005) as during:
        block = np.ones(300 * 1024**2 // 8)  # 300 MB, then released
        block.sum()
        del block
    assert during.peak_mb - before.peak_mb > 200


def test_peak_rss_is_nan_without_psutil(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with PeakRSS() as m:
        pass
    assert math.isnan(m.peak_mb)


def test_peak_vram_is_nan_or_nonnegative():
    with PeakVRAM() as m:
        pass
    assert math.isnan(m.peak_mb) or m.peak_mb >= 0


def test_pipeline_records_stage_peaks(synthetic_root):
    pytest.importorskip("psutil")
    from inspector.config import load_config
    from inspector.pipeline import run_with_predictions
    from inspector.runner import prepare

    cfg = load_config("configs/data/synth_strip.yaml")
    spec, model, idx = prepare(cfg, method="pixel_pca", category="synth_strip", seed=0,
                               data_root=synthetic_root)
    result, _, _ = run_with_predictions(model, idx, test_split=spec.dataset["test_split"])
    for stage in ("fit", "predict", "eval"):
        assert getattr(result, f"peak_rss_{stage}_mb") > 0
