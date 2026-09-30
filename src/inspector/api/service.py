"""HTTP inspection service over one exported artifact.

    inspector serve --artifact artifacts/pcb1 --port 8000

Endpoints:
    GET  /health   model, category, threshold and its provenance
    POST /predict  multipart image -> score, decision, peak location, latency;
                   `?heatmap=true` adds a PNG overlay (base64) on a threshold-fixed scale

The decision is the artifact's frozen rule (`score > threshold`), never
re-derived per request. The response states the requested and effective
false-alarm rates so a caller cannot mistake one for the other.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from .. import __version__
from .artifact import Inspector, load_artifact, overlay


def _decode(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _png(image: np.ndarray, max_side: int = 1024) -> str:
    """Base64 PNG, downscaled so a response stays a few hundred kB, not megabytes."""
    from PIL import Image

    img = Image.fromarray(image)
    img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_app(artifact: str | Path | Inspector) -> FastAPI:
    # FastAPI is imported at module level on purpose: with postponed annotations it
    # resolves `UploadFile` from the module's globals, not from a function scope.
    inspector = artifact if isinstance(artifact, Inspector) else load_artifact(artifact)
    meta = inspector.meta
    params = meta["threshold"].get("params") or {}
    app = FastAPI(title="MVTec Visual Inspector", version=__version__,
                  description="One-class visual anomaly detection (PatchCore).")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": meta["model"],
            "category": meta.get("category"),
            "input_long_side": meta["transform"].get("long_side"),
            "threshold": inspector.threshold,
            "requested_fpr": params.get("requested_fpr"),
            "effective_fpr": params.get("effective_fpr"),
            "calibration_normals": meta["calibration"]["n"],
            "provenance": meta.get("provenance", {}),
        }

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),  # noqa: B008 — FastAPI's declared-parameter idiom
        heatmap: bool = False,
    ) -> dict[str, Any]:
        data = await file.read()
        try:
            image = _decode(data)
        except Exception as exc:  # unreadable upload is the client's error, not ours
            raise HTTPException(status_code=400, detail=f"not a readable image: {exc}") from exc
        pred = inspector.predict(image)
        body: dict[str, Any] = {
            "filename": file.filename,
            "score": pred.score,
            "is_anomalous": pred.is_anomalous,
            "threshold": pred.threshold,
            "decision_rule": "score > threshold",
            "effective_fpr": params.get("effective_fpr"),
            "peak_yx": list(pred.peak_yx),
            "latency_ms": round(pred.latency_ms, 1),
        }
        if heatmap:
            body["heatmap_png_base64"] = _png(overlay(image, pred.anomaly_map, pred.threshold))
        return body

    return app
