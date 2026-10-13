"""Fusing two models' anomaly maps, evaluated on `test_public` from stored predictions (no refit).

The tiled DINOv3 model and WRN50 PatchCore see different things: DINOv3 tokens cover 25 native px
at scale 0.625, WRN50's `layer2` at 1280 px about 14, and E14-E20 showed WRN50's maps carry the
`can` defects that DINOv3's miss. Each map is brought to a common scale by dividing it by the 95th
percentile of its own model's validation pixels (normal images only), at a quarter of native
resolution; the fused map is their mean or maximum. The fused maps then go through the same
threshold and mask rules as `mask_rule_study.py`, with the same server-style metrics on `all`
images and on `regular` ones. Development evidence only.

    python scripts/fusion_study.py --data-root data/raw/mvtec_ad2 --a <DINOv3 run ids> --b <WRN50 run ids> \
        --mode mean --label fuse-mean --out reports/studies/fusion
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from mask_rule_study import evaluate_rules, focus_args, gains_args

from inspector.evaluate import SplitPredictions
from inspector.postproc.binarize import validation_thresholds


def quarter(native: np.ndarray) -> np.ndarray:
    import cv2

    h, w = native.shape
    return cv2.resize(native.astype(np.float32), (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)


def load(run_dir: Path, data_root: str):
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    val = SplitPredictions.load(run_dir, "validation", root=data_root)
    test = SplitPredictions.load(run_dir, spec["dataset"]["test_split"], root=data_root)
    return spec, val, test


def p95(split: SplitPredictions, sigma: float) -> float:
    lo, hi = split.raw_range()
    return validation_thresholds(split.native_maps(sigma), lo, hi, gains=(1.0,))["p95x1"]


def image_norm(m: np.ndarray, kind: str) -> np.ndarray:
    """Per-image background normalization (E22): a changed light raises the whole map of a normal
    image, so each map is referred to its own robust background level before thresholding."""
    if kind == "none":
        return m
    if kind == "sub-median":
        return m - float(np.median(m))
    if kind == "sub-q75":
        return m - float(np.quantile(m, 0.75))
    if kind == "div-median":
        return m / max(float(np.median(m)), 1e-6)
    raise ValueError(f"unknown image normalization {kind!r}")


def highlights(path: str, shape: tuple[int, int]) -> np.ndarray:
    """Specular highlights at the map's quarter resolution (E22): bright *local* peaks — white
    top-hat (grey minus its opening by a 15 px disc) above 0.15 on a pixel brighter than 0.8 —
    grown by one pixel. An absolute level would take the backlit backgrounds of `vial` and
    `fruit_jelly` (60-73% of their pixels exceed 0.92); the top-hat marks 0.2-3% of any category."""
    import cv2

    from inspector.data.transforms import load_image

    grey = cv2.cvtColor(load_image(path), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    small = cv2.resize(grey, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    tophat = small - cv2.morphologyEx(small, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    peaks = ((tophat > 0.15) & (small > 0.8)).astype(np.uint8)
    return cv2.dilate(peaks, np.ones((3, 3), np.uint8)).astype(bool)


def fuse(splits: list[SplitPredictions], sigmas: list[float], norms: list[float], mode: str,
         norm: str = "none", specular: float = 0.0, root: str | None = None, paths: list[str] | None = None):
    """Fused quarter-resolution maps of the same images from several models. With `specular` > 0,
    scores on near-saturated pixels of the image are multiplied by (1 - specular)."""
    if any(s.ids != splits[0].ids for s in splits):
        raise ValueError("the runs were not evaluated on the same images")
    maps = []
    for natives in zip(*(s.native_maps(sg) for s, sg in zip(splits, sigmas))):
        q = [quarter(m) / n for m, n in zip(natives, norms)]
        m = np.maximum.reduce(q) if mode == "max" else np.mean(q, axis=0)
        if specular > 0:
            m = np.where(highlights(str(Path(root) / paths[len(maps)]), m.shape), m * (1.0 - specular), m)
        maps.append(image_norm(m, norm))
    return replace(splits[0], raw_maps=maps, input_sizes=list(splits[0].native_sizes),
                   scores=np.array([float(m.max()) for m in maps]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--a", nargs="+", required=True, help="run ids of model A, one per category")
    ap.add_argument("--b", nargs="+", required=True, help="run ids of model B, matched by category")
    ap.add_argument("--c", nargs="+", default=None, help="run ids of an optional model C, matched by category")
    ap.add_argument("--mode", choices=("mean", "max"), default="mean")
    ap.add_argument("--image-norm", choices=("none", "sub-median", "sub-q75", "div-median"), default="none")
    ap.add_argument("--specular", type=float, default=0.0, help="attenuation of scores on specular highlights")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="reports/studies/fusion")
    ap.add_argument("--focus", action="store_true", help="the P95 band and variations of the mask rule")
    ap.add_argument("--gains", action="store_true", help="the morphological rule over P95 gains 1.1-1.6 only")
    ap.add_argument("--dump", default=None, help="directory for per-image counts (split-half analyses)")
    args = ap.parse_args()
    reg = Path(args.registry)
    others = []
    for ids in (args.b, args.c or []):
        if not ids:
            continue
        by_cat = {}
        for rid in ids:
            spec = json.loads((reg / rid / "spec.json").read_text(encoding="utf-8"))
            by_cat[spec["dataset"]["category"]] = rid
        others.append(by_cat)
    rows = []
    for rid in args.a:
        spec_a = json.loads((reg / rid / "spec.json").read_text(encoding="utf-8"))
        cat = spec_a["dataset"]["category"]
        run_ids = [rid, *(o[cat] for o in others)]
        loaded = [load(reg / r, args.data_root) for r in run_ids]
        sigmas = [sp["evaluation"]["smoothing_sigma"] for sp, _, _ in loaded]
        norms = [p95(val, sg) for (_, val, _), sg in zip(loaded, sigmas)]
        val_s, test_s = [v for _, v, _ in loaded], [t for _, _, t in loaded]
        val_f = fuse(val_s, sigmas, norms, args.mode, args.image_norm, args.specular, args.data_root, val_s[0].ids)
        test_f = fuse(test_s, sigmas, norms, args.mode, args.image_norm, args.specular, args.data_root, test_s[0].ids)
        ev = {**spec_a["evaluation"], "smoothing_sigma": 0.0}
        got = evaluate_rules(val_f, test_f, ev, "+".join(run_ids), cat, **(focus_args() if args.focus else gains_args() if args.gains else {}),
                             dump=Path(args.dump) if args.dump else None)
        rows += [{"label": args.label, **r} for r in got]
        best = max((r for r in got if r["subset"] == "all"), key=lambda r: r["segf1"])
        print(f"{cat:<12} AU-PRO all {best['aupro_005']:.3f}  best {best['threshold']}/{best['post']} "
              f"SegF1 {best['segf1']:.3f}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"rules_{args.label}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = sorted({(r["threshold"], r["post"]) for r in rows})

    def macro(key, sub, metric):
        return float(np.mean([r[metric] for r in rows if (r["threshold"], r["post"]) == key and r["subset"] == sub]))

    print(f"\n{args.label}: macro over {len(args.a)} categories; AU-PRO@0.05 regular "
          f"{macro(keys[0], 'regular', 'aupro_005'):.4f} all {macro(keys[0], 'all', 'aupro_005'):.4f}")
    for key in sorted(keys, key=lambda k: -macro(k, "all", "segf1"))[:10]:
        print(f"  {key[0]:<9} {key[1]:<14} SegF1 defective-only {macro(key, 'defective', 'segf1'):.4f} regular {macro(key, 'regular', 'segf1'):.4f} "
              f"all {macro(key, 'all', 'segf1'):.4f}  ClassF1 all {macro(key, 'all', 'classf1'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
