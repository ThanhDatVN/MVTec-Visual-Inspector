"""E23: the two-input hybrid's maps, built exactly as the hybrid export builds them, scored on test_public.

The v5 export kept each component's map as `quarter(native map) / its validation P95` (the
fusion of `scripts/ad2_e21_fusion.py`). The two-input head (`ad2_e23_fusion_head.category_heads`,
trained on the E23 features) is applied to those normalized maps with unit scales. This script
rebuilds the same maps from the full-bank development runs v5 was selected on and scores their
AU-PRO against v5's rule (the maximum of the same two maps), so the number used for the decision
is the number of the pipeline that will be submitted.

    python scripts/ad2_e23_hybrid_check.py --features C:/Users/<you>/mvi_head_features --dino <H+ runs> --wrn <WRN runs>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from ad2_e21_fusion import load, p95, quarter
from ad2_e21_mask_rules import evaluate_rules
from ad2_e23_fusion_head import EV, category_heads, features2

from inspector.postproc.binarize import MaskRule


def normalized(split, sigma: float, scale: float) -> list[np.ndarray]:
    return [quarter(m) / scale for m in split.native_maps(sigma)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dino", nargs="+", required=True)
    ap.add_argument("--wrn", nargs="+", required=True)
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--data-root", default="data/raw/mvtec_ad2")
    ap.add_argument("--out", default="reports/studies/fusion-head/hybrid_check.json")
    args = ap.parse_args()
    reg = Path(args.registry)

    def by_cat(ids):
        return {json.loads((reg / r / "spec.json").read_text(encoding="utf-8"))["dataset"]["category"]: r for r in ids}

    dino, wrn = by_cat(args.dino), by_cat(args.wrn)
    results = {}
    for cat in sorted(dino):
        head = category_heads(Path(args.features) / cat, kinds=("lr2",),
                              groups_to_load=("head", "synthetic", "validation"))["heads"]["lr2"]
        (sd, vd, td), (sw, vw, tw) = load(reg / dino[cat], args.data_root), load(reg / wrn[cat], args.data_root)
        sig_d, sig_w = sd["evaluation"]["smoothing_sigma"], sw["evaluation"]["smoothing_sigma"]
        nd, nw = p95(vd, sig_d), p95(vw, sig_w)
        out = {}
        for name in ("rule", "hybrid"):
            built = {}
            for split, (a, b) in (("validation", (vd, vw)), ("test_public", (td, tw))):
                maps = []
                for qd, qw in zip(normalized(a, sig_d, nd), normalized(b, sig_w, nw)):
                    two = np.stack([qd, qw])
                    maps.append(np.maximum(qd, qw) if name == "rule"
                                else head(features2(two, np.ones(2))).reshape(qd.shape))
                built[split] = replace(a, raw_maps=maps, input_sizes=list(a.native_sizes),
                                       scores=np.array([float(m.max()) for m in maps]))
            got = evaluate_rules(built["validation"], built["test_public"], EV, name, cat,
                                 posts={"morph": MaskRule()}, keep={"p95x1.3"})
            out[name] = {sub: next(r["aupro_005"] for r in got if r["subset"] == sub) for sub in ("regular", "all")}
        results[cat] = out
        print(f"{cat:<12} rule {out['rule']['regular']:.3f}/{out['rule']['all']:.3f}  "
              f"hybrid {out['hybrid']['regular']:.3f}/{out['hybrid']['all']:.3f}", flush=True)
    macro = {name: {sub: float(np.mean([results[c][name][sub] for c in results])) for sub in ("regular", "all")}
             for name in ("rule", "hybrid")}
    print(json.dumps(macro, indent=1))
    Path(args.out).write_text(json.dumps({"categories": results, "macro": macro}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
