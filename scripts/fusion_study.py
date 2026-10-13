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
from mask_rule_study import evaluate_rules, focus_args

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


def fuse(a: SplitPredictions, b: SplitPredictions, sa: float, sb: float, na: float, nb: float, mode: str):
    if a.ids != b.ids:
        raise ValueError("the two runs were not evaluated on the same images")
    maps = []
    for ma, mb in zip(a.native_maps(sa), b.native_maps(sb)):
        qa, qb = quarter(ma) / na, quarter(mb) / nb
        maps.append(np.maximum(qa, qb) if mode == "max" else 0.5 * (qa + qb))
    return replace(a, raw_maps=maps, input_sizes=list(a.native_sizes),
                   scores=np.array([float(m.max()) for m in maps]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--a", nargs="+", required=True, help="run ids of model A, one per category")
    ap.add_argument("--b", nargs="+", required=True, help="run ids of model B, matched by category")
    ap.add_argument("--mode", choices=("mean", "max"), default="mean")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="reports/studies/fusion")
    ap.add_argument("--focus", action="store_true", help="the P95 band and variations of the mask rule")
    args = ap.parse_args()
    reg = Path(args.registry)
    by_cat = {}
    for rid in args.b:
        spec = json.loads((reg / rid / "spec.json").read_text(encoding="utf-8"))
        by_cat[spec["dataset"]["category"]] = rid
    rows = []
    for rid in args.a:
        spec_a, val_a, test_a = load(reg / rid, args.data_root)
        cat = spec_a["dataset"]["category"]
        spec_b, val_b, test_b = load(reg / by_cat[cat], args.data_root)
        sa, sb = spec_a["evaluation"]["smoothing_sigma"], spec_b["evaluation"]["smoothing_sigma"]
        na, nb = p95(val_a, sa), p95(val_b, sb)
        val_f = fuse(val_a, val_b, sa, sb, na, nb, args.mode)
        test_f = fuse(test_a, test_b, sa, sb, na, nb, args.mode)
        ev = {**spec_a["evaluation"], "smoothing_sigma": 0.0}
        got = evaluate_rules(val_f, test_f, ev, f"{rid}+{by_cat[cat]}", cat, **(focus_args() if args.focus else {}))
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
