"""What MVTec's server should report for recipe v1, predicted from `test_public` before the result.

`test_private` holds the training lighting only (file names `<n>_regular.png`) and
`test_private_mixed` the varied lighting (`<n>_mixed.png`). `test_public` holds both (docs/14 §0),
so splitting its stored E13 predictions by capture condition gives a development estimate of each
private split: `regular` captures for `test_private`, the other captures for `test_private_mixed`.
Written and committed after the S1 upload and before any server result, so the comparison with
the server is an honest out-of-sample check of the development numbers.

    python scripts/server_s1_expectation.py --data-root data/raw/mvtec_ad2 --out reports/studies/ad2-server
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

import numpy as np

from inspector.evaluate import EvalConfig, SplitPredictions, evaluate

E13_IMPLEMENTATION = "af89b5297f4d"
KEYS = ("aupro_005", "segf1_3sigma", "segf1_valq", "pixel_auroc", "image_auroc")


def subset(pred: SplitPredictions, keep: np.ndarray) -> SplitPredictions:
    idx = np.flatnonzero(keep)
    pick = lambda xs: [xs[i] for i in idx]  # noqa: E731
    return replace(pred, ids=pick(pred.ids), labels=pred.labels[idx], defect_types=pick(pred.defect_types),
                   scores=pred.scores[idx], raw_maps=pick(pred.raw_maps), input_sizes=pick(pred.input_sizes),
                   native_sizes=pick(pred.native_sizes), mask_paths=pick(pred.mask_paths), mask_overrides=[])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="reports/studies/ad2-server")
    args = ap.parse_args()
    rows = []
    for f in sorted(Path(args.registry).glob("*/spec.json")):
        spec = json.loads(f.read_text(encoding="utf-8"))
        hp = spec["model"]["hparams"]
        if (not spec["implementation_id"].startswith(E13_IMPLEMENTATION) or spec.get("role") != "confirmation"
                or spec["model"]["name"] != "patchcore" or spec["transform"]["long_side"] != 640
                or hp.get("layers") != ["layer2", "layer3"]):
            continue
        ev = spec["evaluation"]
        cfg = EvalConfig(**{**{k: v for k, v in ev.items() if k != "metrics_version"},
                            "aupro_limits": tuple(ev["aupro_limits"])})
        val = SplitPredictions.load(f.parent, "validation", root=args.data_root)
        test = SplitPredictions.load(f.parent, spec["dataset"]["test_split"], root=args.data_root)
        regular = np.array([i.rsplit("/", 1)[-1].endswith("_regular.png") for i in test.ids])
        for name, keep in (("private_estimate", regular), ("mixed_estimate", ~regular), ("all", np.ones_like(regular))):
            m = evaluate(val, subset(test, keep), cfg)
            rows.append({"category": spec["dataset"]["category"], "seed": spec["seed"], "split": name,
                         "images": int(keep.sum()), **{k: m[k] for k in KEYS}})
        print(spec["dataset"]["category"], spec["seed"], {r["split"]: round(r["aupro_005"], 3) for r in rows[-3:]},
              flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("private_estimate", "mixed_estimate", "all"):
        per_cat = {}
        for cat in sorted({r["category"] for r in rows}):
            got = [r for r in rows if r["category"] == cat and r["split"] == split]
            per_cat[cat] = {k: statistics.mean(r[k] for r in got) for k in KEYS}
        summary[split] = {"per_category": per_cat,
                          "macro": {k: statistics.mean(v[k] for v in per_cat.values()) for k in KEYS}}
    (out / "expectation_v1.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=1),
                                            encoding="utf-8")
    for split, s in summary.items():
        print(split, {k: round(v, 3) for k, v in s["macro"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
