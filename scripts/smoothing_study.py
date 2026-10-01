"""E05 (smoothing half): how much does the map smoothing choose the localization numbers?

The evaluator smooths each raw patch map with a Gaussian in input pixels before upsampling to
native resolution (sigma 4 by default, as in anomalib). The image score does not depend on it,
but every pixel metric does. Re-evaluating the stored predictions of the frozen recipe at several
sigmas isolates that choice without refitting anything.

    python scripts/smoothing_study.py --runs <run ids...> --out reports/studies/e05
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from inspector.evaluate import EvalConfig, SplitPredictions, evaluate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 2.0, 4.0, 8.0])
    ap.add_argument("--out", default="reports/studies/e05")
    args = ap.parse_args()

    rows = []
    for run_id in args.runs:
        run_dir = Path(args.registry) / run_id
        spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
        ev = spec["evaluation"]
        base = EvalConfig(**{**ev, "aupro_limits": tuple(ev["aupro_limits"])})
        val = SplitPredictions.load(run_dir, "validation", root=args.data_root)
        test = SplitPredictions.load(run_dir, spec["dataset"]["test_split"], root=args.data_root)
        for sigma in args.sigmas:
            m = evaluate(val, test, replace(base, smoothing_sigma=sigma))
            rows.append({"run_id": run_id, "category": spec["dataset"]["category"], "seed": spec["seed"],
                         "sigma": sigma, **{k: m.get(k) for k in
                                            ("aupro_005", "aupro_030", "pixel_auroc", "segf1_3sigma", "image_auroc")}})
            print(f"{spec['dataset']['category']:<10} s{spec['seed']} sigma={sigma:<4} "
                  f"AU-PRO@0.05={m.get('aupro_005'):.4f} @0.30={m.get('aupro_030'):.4f} "
                  f"P-AUROC={m.get('pixel_auroc'):.4f} SegF1={m.get('segf1_3sigma'):.4f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoothing.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
