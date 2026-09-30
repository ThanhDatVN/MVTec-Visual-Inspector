"""A fitted detector as a deployable artifact.

An artifact is a directory with the memory bank (`.npy`) and one `meta.json`: the
model's hyperparameters and input transform, the operating threshold frozen on
clean validation normals, and provenance. No pickle — loading an artifact
executes no code from it.

The threshold travels with the model because the two are one decision rule:
the conservative rank threshold (ADR-9) only means what it says for the model
and calibration normals it was computed from.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..data.transforms import ImageTransform
from ..models.base import postprocess_map
from ..models.patchcore import PatchCore
from ..postproc.thresholds import from_validation_fpr
from .guards import InputGuards

ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class Prediction:
    score: float
    is_anomalous: bool
    threshold: float
    anomaly_map: np.ndarray  # native resolution, float32
    peak_yx: tuple[int, int]
    latency_ms: float
    input_ok: bool = True
    input_issues: tuple[str, ...] = ()

    @property
    def decision(self) -> str:
        """`refused` when the image is outside the conditions the threshold was set in (E08)."""
        if not self.input_ok:
            return "refused"
        return "anomalous" if self.is_anomalous else "normal"


def save_artifact(
    model: PatchCore,
    calibration_scores: np.ndarray,
    directory: str | Path,
    *,
    target_fpr: float = 0.01,
    category: str = "",
    smoothing_sigma: float = 4.0,
    guards: InputGuards | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a fitted PatchCore and its frozen operating threshold."""
    from ..runner import implementation_id
    from ..utils.env import git_info

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    np.save(directory / "memory_bank.npy", np.asarray(state["memory_bank"]))
    np.save(directory / "bank_sources.npy", np.asarray(state["bank_sources"]))

    scores = np.asarray(calibration_scores, dtype=np.float64)
    threshold = from_validation_fpr(scores, target_fpr=target_fpr, policy="relax")
    info = git_info()
    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "model": "patchcore",
        "category": category,
        "hparams": state["hparams"],
        "transform": state["transform"],
        "grid": [int(g) for g in state["grid"]],
        "stats": state["stats"],
        "smoothing_sigma": smoothing_sigma,
        "threshold": {
            "value": float(threshold.value),
            "method": threshold.method,
            "comparator": threshold.comparator,
            "params": threshold.params,
        },
        "calibration": {
            "n": int(scores.size),
            "max": float(scores.max()),
            "median": float(np.median(scores)),
        },
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "implementation_id": implementation_id(),
            "git_sha": info["git_sha"],
            "dirty": info["dirty"],
            **(extra or {}),
        },
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if guards is not None:
        (directory / "guards.json").write_text(json.dumps(guards.as_dict(), indent=2), encoding="utf-8")
    return directory


class Inspector:
    """Loaded artifact: image in, score + decision + native-resolution map out."""

    def __init__(self, model: PatchCore, meta: dict[str, Any], guards: InputGuards | None = None) -> None:
        self.model = model
        self.meta = meta
        self.guards = guards
        self.threshold = float(meta["threshold"]["value"])
        self.sigma = float(meta.get("smoothing_sigma", 4.0))

    def predict(self, image: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        check = self.guards.check(image) if self.guards is not None else {"ok": True, "reasons": []}
        raw = self.model.predict_raw(image)
        native = postprocess_map(raw.raw_map, input_size=raw.input_size,
                                 native_size=raw.native_size, sigma=self.sigma).astype(np.float32)
        peak = np.unravel_index(int(np.argmax(native)), native.shape)
        return Prediction(
            score=float(raw.score),
            is_anomalous=bool(raw.score > self.threshold),  # strict, as calibrated (ADR-9)
            threshold=self.threshold,
            anomaly_map=native,
            peak_yx=(int(peak[0]), int(peak[1])),
            latency_ms=1000.0 * (time.perf_counter() - t0),
            input_ok=bool(check["ok"]),
            input_issues=tuple(check["reasons"]),  # type: ignore[arg-type]
        )


def load_artifact(directory: str | Path, *, device: str = "auto") -> Inspector:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if meta.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(f"unsupported artifact version {meta.get('artifact_version')!r}")
    hp = meta["hparams"]
    model = PatchCore(
        ImageTransform(**meta["transform"]),
        backbone=hp["backbone"], layers=tuple(hp["layers"]), patch_pool=hp["patch_pool"],
        projection_dim=hp["projection_dim"], feature_dtype=hp["dtype"],
        candidate_fraction=hp["candidate_fraction"], bank_ratio=hp["bank_ratio"],
        bank_size=hp["bank_size"], coreset_projection_dim=hp["coreset_projection_dim"],
        patch_neighbors=hp["patch_neighbors"], patch_reduction=hp["patch_reduction"],
        reweight=hp["reweight"], reweight_neighbors=hp["reweight_neighbors"],
        seed=hp["seed"], device=device,
    )
    model.load_state_dict({
        "memory_bank": np.load(directory / "memory_bank.npy"),
        "bank_sources": np.load(directory / "bank_sources.npy"),
        "grid": meta["grid"],
        "hparams": hp,
        "stats": meta.get("stats", {}),
    })
    guards_file = directory / "guards.json"
    guards = InputGuards(**json.loads(guards_file.read_text(encoding="utf-8"))) if guards_file.is_file() else None
    return Inspector(model, meta, guards)


def overlay(image: np.ndarray, anomaly_map: np.ndarray, threshold: float) -> np.ndarray:
    """Heatmap over the image on a scale fixed by the threshold, not by the image.

    Stretching each map to its own maximum makes a normal image look as alarming
    as a defective one; here 1.0 on the colour scale is the operating threshold.
    """
    import cv2

    level = np.clip(anomaly_map / max(threshold, 1e-12) / 1.5, 0.0, 1.0)
    colour = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    return (0.55 * image.astype(np.float32) + 0.45 * colour.astype(np.float32)).astype(np.uint8)
