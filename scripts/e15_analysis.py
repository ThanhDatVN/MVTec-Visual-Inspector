"""E15 screen on MVTec AD 2 `test_public`: lighting remedies (development, docs/11 Stage H⁗).

Written with the plan, before any E15 run. Each variant of recipe v1 against the E15 reference
(v1 at the same implementation, seed 0), per category:

* image AUROC, paired on the same images, with the **object** bootstrap — every capture of an
  object resampled together, class-stratified, 10,000 resamples — and Holm over the 8 categories
  of that variant (`ad2_condition_study.cluster_paired`; the image bootstrap is reported beside);
* false alarms of the validation operating point among the `shift` normals (changed light
  direction and colour, the condition that carries AD 2's excess false alarms) and among all;
* image AUROC by capture condition.

Carry rule (docs/11): seeds 1-2 if pooled `shift` false alarms fall by at least half with no
category significantly worse, or image AUROC is significantly better on >= 3 categories and worse
on none. The reference must reproduce E13's seed-0 v1 predictions.

    python scripts/e15_analysis.py --implementation <E15 id> --out reports/studies/e15
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ad2_condition_study import by_condition, cluster_paired, read_split

from inspector.stats import holm_bonferroni

CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
E13_IMPLEMENTATION = "af89b5297f4d"


def variant_of(spec: dict) -> str:
    hp, tr = spec["model"]["hparams"], spec["transform"]
    parts = []
    if tr.get("normalize", "imagenet") != "imagenet":
        parts.append(f"normalize={tr['normalize']}")
    if hp.get("bank_exposures"):
        parts.append(f"bank_exposures={hp['bank_exposures']}")
    if hp.get("bank_relight"):
        parts.append(f"bank_relight={hp['bank_relight']}")
    if tr["long_side"] != 640 or hp.get("layers") != ["layer2", "layer3"]:
        parts.append(f"{tr['long_side']}px,{'+'.join(hp.get('layers', []))}")
    return ",".join(parts) or "reference"


def load(registry: Path, implementation: str, role: str, seed: int) -> dict:
    runs = {}
    for f in sorted(registry.glob("*/result.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        spec = json.loads(f.with_name("spec.json").read_text(encoding="utf-8"))
        if (rec.get("status") != "completed" or spec["dataset"].get("layout") != "mvtec_ad2"
                or spec["model"]["name"] != "patchcore" or spec.get("role", "development") != role
                or int(spec["seed"]) != seed or not spec["implementation_id"].startswith(implementation)):
            continue
        val = np.array([r["score"] for r in read_split(f.parent, "validation")])
        test = read_split(f.parent, spec["dataset"]["test_split"])
        runs[(spec["dataset"]["category"], variant_of(spec))] = {
            "seeds": {seed: (val, test)}, "result": rec["result"], "digests": rec.get("prediction_digests")}
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--implementation", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/studies/e15")
    args = ap.parse_args()
    reg = Path(args.registry)
    runs = load(reg, args.implementation, "development", args.seed)
    e13 = load(reg, E13_IMPLEMENTATION, "confirmation", args.seed)
    reproduced = {c: bool(runs.get((c, "reference"), {}).get("digests"))
                  and runs[(c, "reference")]["digests"] == e13.get((c, "reference"), {}).get("digests")
                  for c in CATEGORIES}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    variants = sorted({v for (_, v) in runs if v != "reference"})
    rows, decisions = [], []
    for vi, v in enumerate(variants):
        block = []
        for ci, c in enumerate(CATEGORIES):
            ref, var = runs.get((c, "reference")), runs.get((c, v))
            if not ref or not var:
                continue
            cond_ref = {r["condition"]: r for r in by_condition(ref["seeds"])}
            cond_var = {r["condition"]: r for r in by_condition(var["seeds"])}
            row = {"variant": v, "category": c,
                   **{f"auroc_{k}_ref": cond_ref[k]["auroc"] for k in cond_ref},
                   **{f"auroc_{k}_variant": cond_var[k]["auroc"] for k in cond_var},
                   "fa_shift_ref": round(cond_ref["shift"]["false_alarms_mean"]),
                   "fa_shift_variant": round(cond_var["shift"]["false_alarms_mean"]),
                   "fa_all_ref": round(cond_ref["all"]["false_alarms_mean"]),
                   "fa_all_variant": round(cond_var["all"]["false_alarms_mean"]),
                   "normals_shift": cond_ref["shift"]["normals"], "normals_all": cond_ref["all"]["normals"],
                   "aupro_ref": ref["result"]["aupro_005"], "aupro_variant": var["result"]["aupro_005"],
                   "fit_s_ref": ref["result"]["fit_seconds"], "fit_s_variant": var["result"]["fit_seconds"],
                   **cluster_paired(var["seeds"], ref["seeds"], seed=100 * vi + ci)}
            block.append(row)
        for unit in ("object", "image"):
            for row, (adj, sig) in zip(block, holm_bonferroni([r[f"p_{unit}"] for r in block])):
                row[f"p_holm_{unit}"], row[f"significant_{unit}"] = adj, sig
        rows += block
        fa_ref, fa_var = sum(r["fa_shift_ref"] for r in block), sum(r["fa_shift_variant"] for r in block)
        gains = sum(r["significant_object"] and r["estimate"] > 0 for r in block)
        losses = sum(r["significant_object"] and r["estimate"] < 0 for r in block)
        decisions.append({
            "variant": v, "categories": len(block),
            "shift_false_alarms_ref": fa_ref, "shift_false_alarms_variant": fa_var,
            "shift_normals": sum(r["normals_shift"] for r in block),
            "all_false_alarms_ref": sum(r["fa_all_ref"] for r in block),
            "all_false_alarms_variant": sum(r["fa_all_variant"] for r in block),
            "significant_gains": gains, "significant_losses": losses,
            "macro_auroc_ref": sum(r["auroc_all_ref"] for r in block) / max(1, len(block)),
            "macro_auroc_variant": sum(r["auroc_all_variant"] for r in block) / max(1, len(block)),
            "macro_auroc_shift_ref": sum(r["auroc_shift_ref"] for r in block) / max(1, len(block)),
            "macro_auroc_shift_variant": sum(r["auroc_shift_variant"] for r in block) / max(1, len(block)),
            "macro_auroc_regular_ref": sum(r["auroc_regular_ref"] for r in block) / max(1, len(block)),
            "macro_auroc_regular_variant": sum(r["auroc_regular_variant"] for r in block) / max(1, len(block)),
            "carry_to_three_seeds": bool(len(block) == len(CATEGORIES) and (
                (fa_var <= 0.5 * fa_ref and losses == 0) or (gains >= 3 and losses == 0))),
        })

    for name, data in (("screen.csv", rows), ("decisions.csv", decisions)):
        if data:
            with open(out / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    summary = {"implementation": args.implementation, "seed": args.seed,
               "reference_reproduces_e13": reproduced, "decisions": decisions}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for v in variants:
        print(f"\n{v}")
        for r in (r for r in rows if r["variant"] == v):
            print(f"  {r['category']:<12} AUROC {r['auroc_all_ref']:.3f} -> {r['auroc_all_variant']:.3f} "
                  f"({r['estimate']:+.3f}, object [{r['ci_low_object']:+.3f}, {r['ci_high_object']:+.3f}]"
                  f"{' *' if r['significant_object'] else ''})  shift AUROC {r['auroc_shift_ref']:.3f} -> "
                  f"{r['auroc_shift_variant']:.3f}  shift FA {r['fa_shift_ref']} -> {r['fa_shift_variant']}"
                  f" / {r['normals_shift']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
