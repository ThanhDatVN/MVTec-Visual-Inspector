"""Build the two-input hybrid submission from what the v5 export kept (CPU only).

`export_ad2_submission.py --keep-components` stored, per category and image, each v5
component's map at quarter resolution divided by its validation 95th percentile. The two-input
head (`fusion_head_study.category_heads`, trained on the E23 features of held-out training
images and synthetic defects) turns the pair into a probability map, written as
`anomaly_images/<cat>/<split>/<name>.tiff` at native resolution; v5's masks are copied unchanged
to `anomaly_images_thresholded`, so SegF1 and ClassF1 are v5's. `scripts/hybrid_maps_check.py`
scores this exact construction on `test_public`.

    python scripts/export_hybrid_from_components.py --features C:/Users/<you>/mvi_head_features \
        --components C:/Users/<you>/mvi_v5_components --masks <v5 submission dir> --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from fusion_head_study import category_heads, features2

CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
SPLITS = ("test_private", "test_private_mixed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--components", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    args = ap.parse_args()
    import tifffile
    from PIL import Image

    from inspector.models.base import postprocess_map

    out, masks, comps = Path(args.out), Path(args.masks), Path(args.components)
    log = []
    for cat in args.categories:
        t0 = time.perf_counter()
        head = category_heads(Path(args.features) / cat, kinds=("lr2",),
                              groups_to_load=("head", "synthetic", "validation"))["heads"]["lr2"]
        counts = {}
        for split in SPLITS:
            maps_dir, masks_dir = out / "anomaly_images" / cat / split, out / "anomaly_images_thresholded" / cat / split
            maps_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            stems = sorted(p.stem for p in (masks / "anomaly_images_thresholded" / cat / split).glob("*.png"))
            for stem in stems:
                qd = np.load(comps / cat / "0" / split / f"{stem}.npy")
                qw = np.load(comps / cat / "1" / split / f"{stem}.npy")
                prob = head(features2(np.stack([qd, qw]), np.ones(2))).reshape(qd.shape)
                mask_src = masks / "anomaly_images_thresholded" / cat / split / f"{stem}.png"
                with Image.open(mask_src) as im:
                    size = im.size
                native = postprocess_map(prob, input_size=size, native_size=size, sigma=0.0)
                tifffile.imwrite(maps_dir / f"{stem}.tiff", native.astype(np.float16), compression="zlib")
                shutil.copyfile(mask_src, masks_dir / f"{stem}.png")
            counts[split] = len(stems)
        log.append({"category": cat, "counts": counts, "seconds": time.perf_counter() - t0})
        print(log[-1], flush=True)
    (out.parent / f"{out.name}.export_log.json").write_text(
        json.dumps({"components": str(comps), "masks_from": str(masks), "features": args.features,
                    "head": "two-input logistic (fusion_head_study.category_heads, kind lr2)", "categories": log},
                   indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
