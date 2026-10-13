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
    ap.add_argument("--keep-components", action="store_true",
                    help="keep each component's normalized quarter-resolution maps (for re-processing)")
    ap.add_argument("--test-images", type=int, default=None,
                    help="testing the export only: the first N images of each split")
    args = ap.parse_args()

    def first(index):
        return list(index)[: args.test_images] if args.test_images else list(index)
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
        recipe = load_config(args.recipe)
        name, gains, rule = mask_policy(recipe)
        components = (recipe.get("submission") or {}).get("components")
        t0 = time.perf_counter()

        def fitted(recipe_cfg, cat=cat):
            cfg = deep_merge(load_config(DATA_CONFIG), recipe_cfg)
            if args.train_limit:
                cfg = deep_merge(cfg, {"data": {"train_limit": args.train_limit}})
            spec, model, idx = prepare(cfg, method=cfg["model"]["name"], category=cat, seed=args.seed,
                                       data_root=args.data_root, role="confirmation")
            model.fit(idx["train"])
            return cfg, spec, model, idx

        def native_map(model, sample, sigma):
            raw = model.predict_raw(load_image(sample.image_path))
            return postprocess_map(raw.raw_map, input_size=raw.input_size, native_size=raw.native_size, sigma=sigma)

        if not components:  # one model: its own maps, as S1
            cfg, spec, model, idx = fitted(recipe)
            ev = eval_config_from_config(cfg)
            val = SplitPredictions.from_model(model, idx["validation"])
            lo, hi = val.raw_range()
            tau = validation_thresholds(val.native_maps(ev.smoothing_sigma), lo, hi, n_sigma=ev.sigma_n,
                                        gains=gains)[name]

            def test_maps(split, model=model, idx=idx, sigma=ev.smoothing_sigma):
                for sample in first(idx[split]):
                    yield sample, native_map(model, sample, sigma)

            provenance = {"run_spec_id": spec.run_id, "method": cfg["model"]["name"]}
        else:  # fusion (scripts/fusion_study.py): each component's map / its validation P95, at 1/4 res
            from fusion_study import p95, quarter

            store = out / "_components" / cat
            norms, ids = [], []
            for k, path in enumerate(components):
                cfg, spec, model, idx = fitted(load_config(path))
                sigma = eval_config_from_config(cfg).smoothing_sigma
                val = SplitPredictions.from_model(model, idx["validation"])
                norms.append(p95(val, sigma))
                ids.append(spec.run_id)
                for split in ("validation", *SPLITS):
                    (store / str(k) / split).mkdir(parents=True, exist_ok=True)
                    for sample in (idx[split] if split == "validation" else first(idx[split])):
                        np.save(store / str(k) / split / f"{Path(sample.image_path).stem}.npy",
                                quarter(native_map(model, sample, sigma)) / norms[-1])
                del model, val
            ev = eval_config_from_config(deep_merge(load_config(DATA_CONFIG), load_config(components[0])))

            def fused(split, stem, store=store, components=components, recipe=recipe):
                maps = [np.load(store / str(k) / split / f"{stem}.npy") for k in range(len(components))]
                mode = recipe["submission"].get("fusion", "max")
                return np.maximum.reduce(maps) if mode == "max" else np.mean(maps, axis=0)

            val_maps, val_sizes = [], []
            for sample in idx["validation"]:
                val_maps.append(fused("validation", Path(sample.image_path).stem))
                with Image.open(sample.image_path) as im:
                    val_sizes.append(im.size)
            lo, hi = min(float(m.min()) for m in val_maps), max(float(m.max()) for m in val_maps)
            tau = validation_thresholds(
                (postprocess_map(m, input_size=s, native_size=s, sigma=0.0) for m, s in zip(val_maps, val_sizes)),
                lo, hi, n_sigma=ev.sigma_n, gains=gains)[name]

            def test_maps(split, idx=idx):
                for sample in first(idx[split]):
                    with Image.open(sample.image_path) as im:
                        size = im.size
                    m = fused(split, Path(sample.image_path).stem)
                    yield sample, postprocess_map(m, input_size=size, native_size=size, sigma=0.0)

            provenance = {"component_run_spec_ids": ids, "components": components,
                          "fusion": recipe["submission"].get("fusion", "max"), "component_p95": norms}

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
            for sample, native in test_maps(split):
                stem = Path(sample.image_path).stem
                tifffile.imwrite(maps_dir / f"{stem}.tiff", native.astype(np.float16), compression="zlib")
                mask = rule.apply(native, tau)
                n_rejected += bool(mask.any())
                Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(masks_dir / f"{stem}.png")
            counts[split], rejected[split] = len(first(idx[split])), n_rejected
        log.append({"category": cat, **provenance, "seed": args.seed, "threshold_rule": name, "pixel_threshold": tau,
                    "mask_rule": rule.as_dict(), "counts": counts, "images_with_mask": rejected,
                    "seconds": time.perf_counter() - t0, "train_limit": args.train_limit,
                    "test_images": args.test_images})
        print(log[-1], flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({"recipe": args.recipe, "categories": log}, indent=1), encoding="utf-8")
        if components and not args.keep_components:
            import shutil

            shutil.rmtree(out / "_components" / cat, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
