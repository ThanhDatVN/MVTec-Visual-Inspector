"""E23, step 1 (GPU): per-block maps of the v5 components for a learned fusion head.

Per category, both v5 components are fitted on seven eighths of `train/good` (every training
image whose index is not a multiple of 8); the held-out eighth never enters a memory bank. Each
model then scores:

* `head/`       — the held-out training images, clean (normal pixels for the head);
* `synthetic/`  — `--variants` synthetic defects painted on each of them (positives, with masks),
                  `inspector.data.synthetic_defects`;
* `validation/` — `validation/good` (thresholds, as everywhere);
* `test_public/`— for development scoring only.

Stored per image (outside the repository; AD 2 is CC BY-NC-SA and derived maps stay local):
the ViT-H+ per-block distance grids, WRN50's raw map and input size, the native size, and the
synthetic mask at quarter resolution. Nothing here sees a test label.

    python scripts/head_features.py --data-root data/raw/mvtec_ad2 --out C:/Users/<you>/mvi_head_features
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
DATA_CONFIG = "configs/data/mvtec_ad2.yaml"
DINO = "configs/recipes/candidate-v5-component-dinov3hplus.yaml"
WRN = "configs/recipes/candidate-v3-patchcore-1280.yaml"


def quarter_mask(mask: np.ndarray) -> np.ndarray:
    import cv2

    h, w = mask.shape
    return cv2.resize(mask.astype(np.float32), (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA) > 0.25


def fit_component(name: str, cat: str, data_root: str, *, seed: int = 0, recipe: str, device: str | None = None,
                  limit: int | None = None):
    """One v5 component fitted on seven eighths of `train/good` (indices not a multiple of 8).
    The components are fitted and run one after the other: ViT-H+ with its four 100k banks and
    WRN50 together exceed the 4 GB card, and the spill to shared memory made E23's first pass
    four to five times slower. Returns (model, the category's indices, the training list)."""
    from inspector.config import deep_merge, load_config
    from inspector.data.core import DatasetIndex
    from inspector.runner import prepare

    cfg = deep_merge(load_config(DATA_CONFIG), load_config(recipe))
    if device:
        cfg = deep_merge(cfg, {"model": {"device": device}})
    _, model, idx = prepare(cfg, method=cfg["model"]["name"], category=cat, seed=seed,
                            data_root=data_root, role="development")
    train = list(idx["train"])[: (8 * limit if limit else None)]
    bank = DatasetIndex([s for i, s in enumerate(train) if i % 8], root=idx["train"].root, category=cat,
                        split_name="train", layout=idx["train"].layout)
    model.fit(bank)
    return model, idx, train


def dino_maps(model, image: np.ndarray) -> dict:
    layers = model.layer_maps(model.transform.normalize_image(model.transform.resize_image(image)))
    return {"layers": layers.astype(np.float16)}


def wrn_maps(model, image: np.ndarray) -> dict:
    raw = model.predict_raw(image)
    return {"wrn": raw.raw_map.astype(np.float16), "wrn_input": np.array(raw.input_size),
            "native": np.array(image.shape[1::-1])}


def release(model) -> None:
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dino-recipe", default=DINO)
    ap.add_argument("--wrn-recipe", default=WRN)
    ap.add_argument("--device", default=None, help="override both models' device (smoke tests on cpu)")
    ap.add_argument("--limit", type=int, default=None, help="smoke tests: first N images of every set")
    args = ap.parse_args()

    from inspector.data.synthetic_defects import synthetic_defect
    from inspector.data.transforms import load_image

    out = Path(args.out)
    for ci, cat in enumerate(args.categories):
        done = out / cat / "done.json"
        if done.is_file():
            print(f"{cat}: done", flush=True)
            continue
        t0 = time.perf_counter()

        def items(idx, train, ci=ci):
            """(group, stem, image, extra) for every image to score; synthetic defects are redrawn
            from their seeds, so both passes see the same pixels."""
            held_out = [s for i, s in enumerate(train) if i % 8 == 0]
            for i, sample in enumerate(held_out):
                image = load_image(sample.image_path)
                yield "head", sample.image_path.stem, image, {}
                for k in range(args.variants):
                    defective, mask = synthetic_defect(image, np.random.default_rng((args.seed, ci, i, k)))
                    yield "synthetic", f"{sample.image_path.stem}_{k}", defective, {"mask": quarter_mask(mask)}
            for split in ("validation", "test_public"):
                samples = list(idx[split])[: args.limit] if args.limit else list(idx[split])
                for sample in samples:
                    yield (split, f"{sample.image_path.stem}_{sample.label}_{sample.image_path.parent.name}",
                           load_image(sample.image_path),
                           {"label": np.array(sample.label), "id": np.array(sample.rel_id(idx["train"].root)),
                            "mask_path": np.array(str(Path(sample.mask_path).resolve()) if sample.mask_path else "")})

        model, idx, train = fit_component("dino", cat, args.data_root, seed=args.seed, recipe=args.dino_recipe,
                                          device=args.device, limit=args.limit)
        layers = {(g, stem): dino_maps(model, image) for g, stem, image, _ in items(idx, train)}
        release(model)
        model, idx, train = fit_component("wrn", cat, args.data_root, seed=args.seed, recipe=args.wrn_recipe,
                                          device=args.device, limit=args.limit)
        n_held = 0
        for g, stem, image, extra in items(idx, train):
            (out / cat / g).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / cat / g / f"{stem}.npz", **layers[(g, stem)], **wrn_maps(model, image), **extra)
            n_held += g == "head"
        release(model)
        done.write_text(json.dumps({"category": cat, "held_out": n_held, "variants": args.variants,
                                    "seconds": time.perf_counter() - t0}), encoding="utf-8")
        print(f"{cat}: {n_held} held-out images, {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
