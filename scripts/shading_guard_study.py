"""R10 (docs/14): a lighting-structure guard, against the exposure/focus guards, on MVTec AD 2.

The deployed guards (`inspector.api.guards`) watch mean grey level and Laplacian variance. On AD 2
`test_public` they refuse many over- and under-exposed captures, which the model scores like the
training lighting, and accept most changed-lighting (`shift_k`) captures, which carry the excess
false alarms (`scripts/ad2_condition_study.py`). A guard should watch what breaks the model.

The shading guard describes an image by its low-frequency lighting field: the image reduced to a
coarse grid (8 cells on the short side), each channel divided by its own mean (so a global
exposure change cancels) and logged. An image is refused when its field is farther from every
training field than any training field is from its nearest other one (the rank rule at k = 1 on
leave-one-out distances, as the exposure guard's [min, max] range).

Per capture condition: share of test normals refused by each guard, and the false alarms of the
stored v1 operating point among the normals each guard accepts. Exploratory, development data.

    python scripts/shading_guard_study.py --data-root data/raw/mvtec_ad2 --out reports/studies/ad2-conditions
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from ad2_condition_study import CONDITIONS, NAME
from global_branch_study import reference_runs

from inspector.api.guards import InputGuards
from inspector.data.transforms import load_image
from inspector.postproc.thresholds import from_validation_fpr

GRID_SHORT = 8


def lighting_field(image: np.ndarray) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    gh, gw = (GRID_SHORT, max(GRID_SHORT, round(GRID_SHORT * w / h))) if h <= w else \
        (max(GRID_SHORT, round(GRID_SHORT * h / w)), GRID_SHORT)
    small = cv2.resize(image.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) + 1.0
    return np.log(small / small.reshape(-1, small.shape[-1]).mean(axis=0)).ravel()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--implementation", default="af89b5297f4d")
    ap.add_argument("--out", default="reports/studies/ad2-conditions")
    args = ap.parse_args()
    root = Path(args.data_root)
    refs = reference_runs(Path(args.registry), args.implementation, "confirmation")
    rows = []
    for category in sorted(refs):
        train = [load_image(p) for p in sorted((root / category / "train" / "good").glob("*.png"))]
        guards = InputGuards.fit(train)
        fields = np.stack([lighting_field(im) for im in train])
        del train
        d = np.sqrt(((fields[:, None, :] - fields[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        limit = float(d.min(axis=1).max())

        decisions = {}
        for folder in ("good", "bad"):
            for path in sorted((root / category / "test_public" / folder).glob("*.png")):
                image = load_image(path)
                nn = float(np.sqrt(((fields - lighting_field(image)) ** 2).sum(-1)).min())
                decisions[f"{category}/test_public/{folder}/{path.name}"] = {
                    "exposure_focus": not guards.check(image)["ok"], "shading": nn > limit}
        for cond in CONDITIONS:
            row = {"category": category, "condition": cond}
            for guard in ("exposure_focus", "shading"):
                refused, fa_accepted, n_accepted = [], [], []
                for _, (val, test) in sorted(refs[category].items()):
                    thr = from_validation_fpr(val, target_fpr=0.01, policy="relax").value
                    normals = [r for r in test if r["label"] == 0 and _cond(r["id"]) == cond]
                    ref = [decisions[r["id"]][guard] for r in normals]
                    acc = [r for r, x in zip(normals, ref) if not x]
                    refused.append(np.mean(ref))
                    fa_accepted.append(sum(r["score"] > thr for r in acc))
                    n_accepted.append(len(acc))
                row[f"{guard}_refused_normals"] = float(np.mean(refused))
                row[f"{guard}_fa_among_accepted"] = float(np.sum(fa_accepted) / max(1, np.sum(n_accepted)))
            rows.append(row)
        print(category, "  ".join(f"{r['condition']}: refuse {r['exposure_focus_refused_normals']:.0%}/"
                                  f"{r['shading_refused_normals']:.0%}" for r in rows[-len(CONDITIONS):]), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "shading_guard.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for cond in CONDITIONS:
        block = [r for r in rows if r["condition"] == cond]
        print(f"{cond:<13} refused exposure/focus {np.mean([r['exposure_focus_refused_normals'] for r in block]):.0%}"
              f"  shading {np.mean([r['shading_refused_normals'] for r in block]):.0%}  |  FA among accepted: "
              f"exposure/focus {np.mean([r['exposure_focus_fa_among_accepted'] for r in block]):.1%}"
              f"  shading {np.mean([r['shading_fa_among_accepted'] for r in block]):.1%}")
    return 0


def _cond(rel_id: str) -> str:
    c = NAME.match(rel_id.split("/")[-1]).group(2)
    return "shift" if c.startswith("shift_") else c


if __name__ == "__main__":
    raise SystemExit(main())
