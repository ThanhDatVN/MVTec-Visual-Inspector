"""docs/14 B3: the 3-sigma pixel threshold against the validation pixel quantile, on stored maps.

Re-evaluates stored predictions with the current evaluator (no refit), which reports SegF1 at both
segmentation thresholds: `3sigma` (mean + 3 SD of the validation pixels, the MVTec AD 2 benchmark
rule) and `valq` (the validation pixel quantile at a 0.1% normal-pixel rate). Image metrics and
every threshold-free pixel metric must come back unchanged; the script checks that, to 1e-3,
because the registry stores maps in float16 and the original evaluation saw them in float32
(the drift is ~1e-5).

    python scripts/ad2_b3_pixel_threshold.py --implementation af89b5297f4d --role confirmation \
        --data-root data/raw/mvtec_ad2 --out reports/studies/pixel-threshold
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from inspector.evaluate import EvalConfig, SplitPredictions, evaluate

UNCHANGED = ("image_auroc", "pixel_auroc", "aupro_005", "segf1_3sigma")
KEYS = ("segf1_3sigma", "segf1_valq", "iou_3sigma", "iou_valq", "threshold_3sigma", "threshold_valq",
        "pixel_fpr_3sigma", "pixel_fpr_valq", "validation_pixel_fpr_valq")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--implementation", required=True)
    ap.add_argument("--role", default="confirmation")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="reports/studies/pixel-threshold")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(Path(args.registry).glob("*/result.json")):
        record = json.loads(f.read_text(encoding="utf-8"))
        spec = json.loads(f.with_name("spec.json").read_text(encoding="utf-8"))
        if (record.get("status") != "completed" or spec["model"]["name"] != "patchcore"
                or spec.get("role", "development") != args.role
                or not spec["implementation_id"].startswith(args.implementation)):
            continue
        ev = spec["evaluation"]
        cfg = EvalConfig(**{**{k: v for k, v in ev.items() if k != "metrics_version"},
                            "aupro_limits": tuple(ev["aupro_limits"])})
        val = SplitPredictions.load(f.parent, "validation", root=args.data_root)
        test = SplitPredictions.load(f.parent, spec["dataset"]["test_split"], root=args.data_root)
        m = evaluate(val, test, cfg)
        stored = record["result"]
        drift = {k: abs(m[k] - stored[k]) for k in UNCHANGED}
        if max(drift.values()) > 1e-3:
            raise RuntimeError(f"{f.parent.name}: re-evaluation does not reproduce the stored metrics: {drift}")
        hp = spec["model"]["hparams"]
        rows.append({"run_id": f.parent.name, "category": spec["dataset"]["category"], "seed": spec["seed"],
                     "recipe": f"{spec['transform']['long_side']}px,{'+'.join(hp.get('layers', []))}",
                     **{k: m[k] for k in KEYS}})
        r = rows[-1]
        print(f"{r['category']:<12} {r['recipe']:<20} s{r['seed']}  SegF1 3sigma {r['segf1_3sigma']:.3f} "
              f"(thr {r['threshold_3sigma']:.2f}, test normal-pixel FPR {r['pixel_fpr_3sigma']:.4f})  "
              f"valq {r['segf1_valq']:.3f} (thr {r['threshold_valq']:.2f}, FPR {r['pixel_fpr_valq']:.4f})",
              flush=True)
    rows.sort(key=lambda r: (r["recipe"], r["category"], r["seed"]))
    with open(out / "pixel_threshold.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
