"""Input guards: refuse to decide on images outside the conditions the model was fitted in.

E08 showed the frozen operating point breaks under a quarter to two thirds of a stop of
exposure change or a 2-4 px blur, while image AUROC barely moves — the anomaly score cannot
see the drift that invalidates its own threshold. Two cheap image statistics can:

* **exposure** — mean grey level of the downscaled image;
* **focus** — variance of its Laplacian (drops under blur and defocus).

Each accepts the range [min, max] observed on the training normals. For a new clean image that
is exchangeable with them, falling outside that range has probability at most 2/(n+1) — the
same rank argument as the operating threshold (ADR-9), with no distribution assumed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

WORK_SIDE = 512


def image_stats(image: np.ndarray) -> dict[str, float]:
    import cv2

    h, w = image.shape[:2]
    scale = WORK_SIDE / max(h, w)
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    small = cv2.resize(grey, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return {
        "exposure": float(small.mean()),
        "focus": float(cv2.Laplacian(small.astype(np.float32), cv2.CV_32F).var()),
    }


@dataclass(frozen=True)
class InputGuards:
    exposure_min: float
    exposure_max: float
    focus_min: float
    focus_max: float
    n_reference: int

    @classmethod
    def fit(cls, images: Iterable[np.ndarray]) -> InputGuards:
        stats = [image_stats(im) for im in images]
        exp = [s["exposure"] for s in stats]
        foc = [s["focus"] for s in stats]
        return cls(min(exp), max(exp), min(foc), max(foc), len(stats))

    @property
    def false_flag_bound(self) -> float:
        """Marginal probability that a clean exchangeable image trips one guard, per guard."""
        return 2.0 / (self.n_reference + 1)

    def check(self, image: np.ndarray) -> dict[str, object]:
        s = image_stats(image)
        reasons = []
        if s["exposure"] < self.exposure_min:
            reasons.append("under-exposed")
        elif s["exposure"] > self.exposure_max:
            reasons.append("over-exposed")
        if s["focus"] < self.focus_min:
            reasons.append("out of focus / blurred")
        elif s["focus"] > self.focus_max:
            reasons.append("sharper or noisier than any training image")
        return {"ok": not reasons, "reasons": reasons, **s}

    def as_dict(self) -> dict[str, float]:
        return asdict(self)
