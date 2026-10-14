"""Export a hybrid MVTec AD 2 submission: learned-head maps for AU-PRO, an existing submission's masks.

The server scores the continuous maps (`anomaly_images`, AU-PRO) and the masks
(`anomaly_images_thresholded`, SegF1 and ClassF1) from separate files. E23 found the learned fusion
head (`scripts/fusion_head_study.py`) raising AU-PRO while v5's fixed rule gave the better masks.
This writes, per category:

* `anomaly_images/<cat>/<split>/<name>.tiff` — the head's probability map, at native resolution
  (float16): both v5 components refitted on seven eighths of `train/good` exactly as
  `head_features.py` does, their per-block maps run through the head trained from the stored E23
  features (`category_heads`, same seed, same samples);
* `anomaly_images_thresholded/<cat>/<split>/<name>.png` — copied unchanged from `--masks`.

    python scripts/export_head_maps.py --data-root data/raw/mvtec_ad2 --features C:/Users/<you>/mvi_head_features \
        --masks <v5 submission dir> --out <new submission dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from fusion_head_study import Record, category_heads, features, quarter_channels
from head_features import CATEGORIES, DINO, WRN, dino_maps, fit_component, release, wrn_maps

SPLITS = ("test_private", "test_private_mixed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--features", required=True, help="the E23 feature directory (head training data)")
    ap.add_argument("--masks", required=True, help="submission directory whose masks are kept")
    ap.add_argument("--out", required=True)
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--negatives", type=int, default=10)
    ap.add_argument("--test-images", type=int, default=None, help="smoke tests: first N images per split")
    ap.add_argument("--dino-recipe", default=None, help="smoke tests: the component the features were made with")
    ap.add_argument("--device", default=None, help="smoke tests: cpu")
    ap.add_argument("--limit", type=int, default=None, help="smoke tests: as head_features --limit")
    args = ap.parse_args()
    import tifffile

    from inspector.data.transforms import load_image
    from inspector.models.base import postprocess_map

    out, masks = Path(args.out), Path(args.masks)
    log_path = out.parent / f"{out.name}.export_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))["categories"] if log_path.is_file() else []
    for cat in args.categories:
        if any(e["category"] == cat for e in log):
            print(f"{cat}: already exported", flush=True)
            continue
        t0 = time.perf_counter()
        fitted = category_heads(Path(args.features) / cat, seed=args.seed, negatives=args.negatives, kinds=("lr",),
                                groups_to_load=("head", "synthetic", "validation"))
        head, scales = fitted["heads"]["lr"], fitted["scales"]
        # two passes, one component on the card at a time (see head_features.fit_component)
        kw = {"seed": args.seed, "device": args.device, "limit": args.limit}
        model, idx, _ = fit_component("dino", cat, args.data_root, recipe=args.dino_recipe or DINO, **kw)
        first = next(iter(idx["validation"]))
        stored = np.load(Path(args.features) / cat / "validation" /
                         f"{first.image_path.stem}_{first.label}_{first.image_path.parent.name}.npz")
        drift = {}

        def rel(a, b):
            a, b = a.astype(np.float32), b.astype(np.float32)
            return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-9))

        drift["layers"] = rel(dino_maps(model, load_image(first.image_path))["layers"], stored["layers"])
        samples = {split: list(idx[split])[: args.test_images] if args.test_images else list(idx[split])
                   for split in SPLITS}
        layers = {(split, s.image_path.stem): dino_maps(model, load_image(s.image_path))
                  for split in SPLITS for s in samples[split]}
        release(model)
        model, idx, _ = fit_component("wrn", cat, args.data_root, recipe=WRN, **kw)
        drift["wrn"] = rel(wrn_maps(model, load_image(first.image_path))["wrn"], stored["wrn"])
        print(f"{cat}: refit drift against the stored features {drift}", flush=True)
        if max(drift.values()) > 0.05:
            raise RuntimeError(f"{cat}: the refitted components do not reproduce the E23 features: {drift}")
        counts = {}
        for split in SPLITS:
            maps_dir, masks_dir = out / "anomaly_images" / cat / split, out / "anomaly_images_thresholded" / cat / split
            maps_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            for sample in samples[split]:
                stem = Path(sample.image_path).stem
                z = {**layers[(split, stem)], **wrn_maps(model, load_image(sample.image_path))}
                rec = Record(quarter_channels(z), tuple(int(v) for v in z["native"]), None, 0, "", None)
                prob = head(features(rec, scales)).reshape(rec.channels.shape[1:])
                native = postprocess_map(prob, input_size=rec.native, native_size=rec.native, sigma=0.0)
                tifffile.imwrite(maps_dir / f"{stem}.tiff", native.astype(np.float16), compression="zlib")
                shutil.copyfile(masks / "anomaly_images_thresholded" / cat / split / f"{stem}.png", masks_dir / f"{stem}.png")
            counts[split] = len(samples[split])
        release(model)
        log.append({"category": cat, "refit_drift": drift, "counts": counts, "head_positives": fitted["positives"],
                    "scales": [float(x) for x in scales], "seconds": time.perf_counter() - t0,
                    "masks_from": str(masks), "test_images": args.test_images})
        log_path.write_text(json.dumps({"categories": log}, indent=1), encoding="utf-8")
        print(log[-1], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
