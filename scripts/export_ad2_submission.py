"""Export an MVTec AD 2 submission for the official evaluation server (ADR-13 §6, ADR-15).

For each category: fit the recipe (seed 0) on `train/good`, derive the pixel threshold from the
native-resolution maps of `validation/good` (normal images only), then write for `test_private`
and `test_private_mixed`:

    <out>/anomaly_images/<category>/<split>/<name>.tiff               float16, native resolution
    <out>/anomaly_images_thresholded/<category>/<split>/<name>.png    {0, 255}

The model is the recipe's `model.name`. How maps become masks is the recipe's `submission:`
block, frozen with it:

    submission:
      threshold: 3sigma | p95      # p95: the validation pixels' 95th percentile times `gain`
      gain: 1.35
      mask: plain | morph          # morph: inspector.postproc.binarize.MaskRule (its defaults
      mask_rule: {...}             #        unless overridden here)

Without the block: `3sigma` and `plain`, the S1 submission's rule (ADR-13 §6). Check the result
with MVTec's `check_and_prepare_data_for_upload.py` before uploading; the account and the upload
are the repository owner's.

    python scripts/export_ad2_submission.py --data-root data/raw/mvtec_ad2 \
        --recipe configs/recipes/<recipe>.yaml --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
SPLITS = ("test_private", "test_private_mixed")
DATA_CONFIG = "configs/data/mvtec_ad2.yaml"


def mask_policy(cfg: dict):
    """(threshold name, MaskRule) from the recipe's `submission:` block."""
    from inspector.postproc.binarize import PLAIN, MaskRule

    sub = cfg.get("submission") or {}
    kind = sub.get("threshold", "3sigma")
    if kind == "3sigma":
        name = "3sigma"
    elif kind == "p95":
        name = f"p95x{float(sub['gain']):g}"
    else:
        raise ValueError(f"unknown submission threshold {kind!r}")
    mask = sub.get("mask", "plain")
    if mask == "plain":
        rule = PLAIN
    elif mask == "morph":
        rule = MaskRule(**(sub.get("mask_rule") or {}))
    else:
        raise ValueError(f"unknown submission mask {mask!r}")
    gains = (float(sub["gain"]),) if kind == "p95" else ()
    return name, gains, rule


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    ap.add_argument("--train-limit", type=int, default=None,
                    help="testing the export only; a real submission fits on the full training split")
    args = ap.parse_args()
    try:
        import tifffile
    except ImportError:
        print("error: needs tifffile (pip install -e '.[benchmark]')", file=sys.stderr)
        return 2
    from PIL import Image

    from inspector.config import deep_merge, load_config
    from inspector.data.transforms import load_image
    from inspector.evaluate import SplitPredictions
    from inspector.models.base import postprocess_map
    from inspector.postproc.binarize import validation_thresholds
    from inspector.runner import eval_config_from_config, prepare

    out = Path(args.out)
    log_path = out / "export_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))["categories"] if log_path.is_file() else []
    done = {entry["category"] for entry in log}
    for cat in args.categories:
        if cat in done:  # resumable: a category is logged only after all its files are written
            print(f"{cat}: already exported", flush=True)
            continue
        cfg = deep_merge(load_config(DATA_CONFIG), load_config(args.recipe))
        if args.train_limit:
            cfg = deep_merge(cfg, {"data": {"train_limit": args.train_limit}})
        ev = eval_config_from_config(cfg)
        name, gains, rule = mask_policy(cfg)
        spec, model, idx = prepare(cfg, method=cfg["model"]["name"], category=cat, seed=args.seed,
                                   data_root=args.data_root, role="confirmation")
        t0 = time.perf_counter()
        model.fit(idx["train"])
        val = SplitPredictions.from_model(model, idx["validation"])
        lo, hi = val.raw_range()
        tau = validation_thresholds(val.native_maps(ev.smoothing_sigma), lo, hi, n_sigma=ev.sigma_n, gains=gains)[name]
        counts, rejected = {}, {}
        for split in SPLITS:
            if split not in idx:
                print(f"error: {cat} has no {split} split under {args.data_root}", file=sys.stderr)
                return 1
            maps_dir = out / "anomaly_images" / cat / split
            masks_dir = out / "anomaly_images_thresholded" / cat / split
            maps_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            n_rejected = 0
            for sample in idx[split]:
                raw = model.predict_raw(load_image(sample.image_path))
                native = postprocess_map(raw.raw_map, input_size=raw.input_size,
                                         native_size=raw.native_size, sigma=ev.smoothing_sigma)
                stem = Path(sample.image_path).stem
                tifffile.imwrite(maps_dir / f"{stem}.tiff", native.astype(np.float16), compression="zlib")
                mask = rule.apply(native, tau)
                n_rejected += bool(mask.any())
                Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(masks_dir / f"{stem}.png")
            counts[split], rejected[split] = len(idx[split]), n_rejected
        log.append({"category": cat, "run_spec_id": spec.run_id, "method": cfg["model"]["name"], "seed": args.seed,
                    "threshold_rule": name, "pixel_threshold": tau, "mask_rule": rule.as_dict(),
                    "counts": counts, "images_with_mask": rejected, "seconds": time.perf_counter() - t0,
                    "train_limit": args.train_limit})
        print(log[-1], flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({"recipe": args.recipe, "categories": log}, indent=1), encoding="utf-8")
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
