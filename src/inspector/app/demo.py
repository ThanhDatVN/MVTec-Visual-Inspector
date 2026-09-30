"""Gradio demo over one exported artifact.

    inspector demo --artifact artifacts/pcb1

Upload an image; the demo shows the anomaly overlay on a colour scale fixed by
the operating threshold (so a normal image looks calm), the score, and the
frozen decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..api.artifact import load_artifact, overlay


def build_demo(artifact: str | Path) -> Any:
    import gradio as gr

    inspector = load_artifact(artifact)
    meta = inspector.meta
    eff = (meta["threshold"].get("params") or {}).get("effective_fpr")

    def run(image: np.ndarray) -> tuple[np.ndarray, str]:
        pred = inspector.predict(image)
        verdict = pred.decision.upper()
        if pred.input_issues:
            verdict += f" ({', '.join(pred.input_issues)})"
        text = (f"**{verdict}** — score {pred.score:.3f} vs threshold {pred.threshold:.3f} "
                f"(effective false-alarm bound {eff:.2%}); {pred.latency_ms:.0f} ms")
        return overlay(image, pred.anomaly_map, pred.threshold), text

    return gr.Interface(
        fn=run,
        inputs=gr.Image(type="numpy", label="image"),
        outputs=[gr.Image(type="numpy", label="anomaly overlay (1.0 = threshold)"), gr.Markdown()],
        title=f"MVTec Visual Inspector — {meta.get('category', '')}",
        description="PatchCore, fitted on normal images only. Threshold frozen on validation normals.",
        flagging_mode="never",
    )
