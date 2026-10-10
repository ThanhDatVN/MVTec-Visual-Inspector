"""G4 diagnostic: is the translation breakdown the shift itself, or the replicated border strip?

`translate` (robustness/corruptions.py) shifts the image by (dx, dy) and fills the uncovered top
and left strips by edge replication. For each severity this excludes the feature cells whose
native extent touches that strip, plus a margin of two cells for receptive-field spill; the masked
image score is the maximum raw patch score over the remaining cells. Its threshold is recalibrated
on *clean* validation normals with the same mask (k = floor(alpha(n+1)), strict >), so clean and
shifted test normals are compared like for like. Uses the exported artifacts (`inspector export`).

    python scripts/translate_border_study.py --data-root <VisA>
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from inspector.api.artifact import load_artifact
from inspector.config import deep_merge, load_config
from inspector.data import ensure_validation, load_category
from inspector.data.transforms import load_image
from inspector.postproc.thresholds import from_validation_fpr
from inspector.robustness import SEVERITIES, apply_corruption

MARGIN = 2  # feature cells beyond the strip (receptive-field spill)
RECIPE = "configs/recipes/confirmation-v1-patchcore-640.yaml"
FRACS = dict(zip(SEVERITIES, np.linspace(0.002, 0.02, len(SEVERITIES))))  # as in `translate`


def maps(insp, images, sev):
    """(image score, raw patch map) per image, translated at `sev` (0 = clean)."""
    res = []
    for im in images:
        x = apply_corruption(im, None, "translate", sev)[0] if sev else im
        raw = insp.model.predict_raw(x)
        res.append((float(raw.score), np.asarray(raw.raw_map, dtype=np.float32)))
    return res


def keep_mask(shape, hw, sev):
    """Cells outside the replicated strip and its margin; also the excluded row/column counts."""
    (gh, gw), (h, w) = shape, hw
    dy, dx = round(h * FRACS[sev]), round(w * FRACS[sev])
    ny = int(((np.arange(gh) * h / gh) < dy).sum()) + MARGIN
    nx = int(((np.arange(gw) * w / gw) < dx).sum()) + MARGIN
    keep = np.ones(shape, dtype=bool)
    keep[:ny, :] = False
    keep[:, :nx] = False
    return keep, ny, nx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--categories", nargs="+", default=["pcb1", "macaroni2", "capsules"])
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="reports/studies/e08/translate_border.csv")
    args = ap.parse_args()
    cfg = deep_merge(load_config("configs/data/visa_pcb1.yaml"), load_config(RECIPE))
    data = cfg["data"]
    rows = []
    for cat in args.categories:
        insp = load_artifact(Path(args.artifacts) / cat)
        idx = load_category(args.data_root, cat, layout=data["layout"])
        idx, _ = ensure_validation(idx, val_fraction=float(data["val_carve_fraction"]),
                                   seed=int(data["val_carve_seed"]))
        val = [load_image(s.image_path) for s in idx["validation"]]
        normals = [load_image(s.image_path) for s in idx[data["test_split"]] if s.is_normal]
        hw = normals[0].shape[:2]
        val_maps = maps(insp, val, 0)
        clean_maps = maps(insp, normals, 0)
        for sev in SEVERITIES:
            keep, ny, nx = keep_mask(val_maps[0][1].shape, hw, sev)
            t = from_validation_fpr(np.array([m[keep].max() for _, m in val_maps]),
                                    target_fpr=0.01, policy="relax").value
            shifted = maps(insp, normals, sev)
            row = {
                "category": cat, "severity": sev, "n_normals": len(normals),
                "fpr_full": float(np.mean([s > insp.threshold for s, _ in shifted])),
                "fpr_masked_clean": float(np.mean([m[keep].max() > t for _, m in clean_maps])),
                "fpr_masked_shifted": float(np.mean([m[keep].max() > t for _, m in shifted])),
                "peak_in_strip": float(np.mean([not keep.flat[int(np.argmax(m))] for _, m in shifted])),
                "excluded_rows": ny, "excluded_cols": nx,
            }
            rows.append(row)
            print(row, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
