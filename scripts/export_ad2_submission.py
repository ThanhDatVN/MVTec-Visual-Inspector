"""Export an MVTec AD 2 submission for the official evaluation server (ADR-13 §6).

For each category: fit the selected recipe (seed 0) on `train/good`, set the pixel threshold at
the validation mean + 3 sigma of the native-resolution maps (`OP-3SIGMA`, exactly as
`evaluate.pixel_metrics` computes it, and the baseline MVTec's utilities suggest), then write for
`test_private` and `test_private_mixed`:

    <out>/anomaly_images/<category>/<split>/<name>.tiff               float16, native resolution
    <out>/anomaly_images_thresholded/<category>/<split>/<name>.png    {0, 255}

Check the result with MVTec's `check_and_prepare_data_for_upload.py` (MVTecAD2_public_code_utils)
before uploading; the account and the upload are the repository owner's.

    python scripts/export_ad2_submission.py --data-root data/raw/mvtec_ad2 \
        --recipe configs/recipes/candidate-v2-patchcore-layer2-960.yaml --out data/ad2_submission
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


def three_sigma(val, smoothing_sigma: float, n_sigma: float) -> float:
    """The validation mean + n_sigma standard deviations of native maps, as `evaluate.pixel_metrics` pools them."""
    from inspector.postproc.thresholds import sigma_from_stats

    total = total_sq = count = 0.0
    for native in val.native_maps(smoothing_sigma):
        flat = native.ravel()
        total += float(flat.sum())
        total_sq += float(np.square(flat).sum())
        count += flat.size
    mean = total / count
    std = float(np.sqrt(max(0.0, total_sq / count - mean**2)))
    return float(sigma_from_stats(mean, std, n_sigma=n_sigma, n_pixels=count).value)


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
    from inspector.runner import eval_config_from_config, prepare

    out = Path(args.out)
    log = []
    for cat in args.categories:
        cfg = deep_merge(load_config(DATA_CONFIG), load_config(args.recipe))
        if args.train_limit:
            cfg = deep_merge(cfg, {"data": {"train_limit": args.train_limit}})
        ev = eval_config_from_config(cfg)
        spec, model, idx = prepare(cfg, method="patchcore", category=cat, seed=args.seed,
                                   data_root=args.data_root, role="confirmation")
        t0 = time.perf_counter()
        model.fit(idx["train"])
        tau = three_sigma(SplitPredictions.from_model(model, idx["validation"]), ev.smoothing_sigma, ev.sigma_n)
        counts = {}
        for split in SPLITS:
            if split not in idx:
                print(f"error: {cat} has no {split} split under {args.data_root}", file=sys.stderr)
                return 1
            maps_dir = out / "anomaly_images" / cat / split
            masks_dir = out / "anomaly_images_thresholded" / cat / split
            maps_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            for sample in idx[split]:
                raw = model.predict_raw(load_image(sample.image_path))
                native = postprocess_map(raw.raw_map, input_size=raw.input_size,
                                         native_size=raw.native_size, sigma=ev.smoothing_sigma)
                stem = Path(sample.image_path).stem
                tifffile.imwrite(maps_dir / f"{stem}.tiff", native.astype(np.float16), compression="zlib")
                Image.fromarray(np.where(native > tau, 255, 0).astype(np.uint8)).save(masks_dir / f"{stem}.png")
            counts[split] = len(idx[split])
        log.append({"category": cat, "run_spec_id": spec.run_id, "seed": args.seed, "pixel_threshold": tau,
                    "counts": counts, "seconds": time.perf_counter() - t0, "train_limit": args.train_limit})
        print(log[-1], flush=True)
        del model
    (out / "export_log.json").write_text(json.dumps({"recipe": args.recipe, "categories": log}, indent=1),
                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
