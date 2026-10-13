"""E19 (docs/11 Stage J): whole-image resolution for the thin-defect AD 2 categories (R1, A2).

Each (category, resolution) run of recipe v1 against E18-ref (640 px, same implementation, seed 0):
image AUROC paired on the same images with the object bootstrap (`ad2_condition_study`), Holm
over the five comparisons as registered; AU-PRO@0.05, false alarms, fit time. Reading: a
resolution goes to seeds 1-2 for its category if image AUROC is significantly better and AU-PRO
is not lower. For a carried resolution with seeds 0-2 at both sides, the seed-averaged comparison
(object bootstrap over the three seeds' scores together) is reported, descriptively.

    python scripts/e19_analysis.py --implementation 5ab1a8956c23 --out reports/studies/e19

`--gate` applies ADR-14's development gate instead: v3 (1280 px) on all eight categories against
the 640 px reference, Holm over 8. v3 goes to the server if no category is significantly worse in
image AUROC and macro AU-PRO@0.05 is higher.

    python scripts/e19_analysis.py --implementation 5ab1a8956c23 --gate --out reports/studies/e20
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ad2_condition_study import by_condition, cluster_paired
from e15_analysis import load

from inspector.stats import holm_bonferroni

CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
PLAN = (("sheet_metal", 1280), ("sheet_metal", 1920), ("sheet_metal", 2816), ("fabric", 1280), ("rice", 1280))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--implementation", required=True)
    ap.add_argument("--out", default="reports/studies/e19")
    ap.add_argument("--gate", action="store_true", help="ADR-14's gate: v3 on all eight categories")
    args = ap.parse_args()
    runs = load(Path(args.registry), args.implementation, "development", 0)
    rows = []
    plan = tuple((c, 1280) for c in CATEGORIES) if args.gate else PLAN
    for i, (category, side) in enumerate(plan):
        ref, var = runs.get((category, "reference")), runs.get((category, f"{side}px,layer2+layer3"))
        if not ref or not var:
            print(f"missing: {category} {side}")
            continue
        cond_ref = {r["condition"]: r for r in by_condition(ref["seeds"])}
        cond_var = {r["condition"]: r for r in by_condition(var["seeds"])}
        rr, rv = ref["result"], var["result"]
        rows.append({
            "category": category, "long_side": side,
            "auroc_ref": cond_ref["all"]["auroc"], "auroc_variant": cond_var["all"]["auroc"],
            "auroc_regular_ref": cond_ref["regular"]["auroc"], "auroc_regular_variant": cond_var["regular"]["auroc"],
            "aupro_ref": rr["aupro_005"], "aupro_variant": rv["aupro_005"],
            "pixel_auroc_ref": rr["pixel_auroc"], "pixel_auroc_variant": rv["pixel_auroc"],
            "fa_ref": round(cond_ref["all"]["false_alarms_mean"]), "fa_variant": round(cond_var["all"]["false_alarms_mean"]),
            "normals": cond_ref["all"]["normals"],
            "fit_s_ref": rr["fit_seconds"], "fit_s_variant": rv["fit_seconds"],
            "peak_rss_fit_mb_variant": rv.get("peak_rss_fit_mb"),
            **cluster_paired(var["seeds"], ref["seeds"], seed=1900 + i),
        })
    for row, (adj, sig) in zip(rows, holm_bonferroni([r["p_object"] for r in rows])):
        row["p_holm_object"], row["significant_object"] = adj, sig
        row["carry_to_three_seeds"] = bool(sig and row["estimate"] > 0 and row["aupro_variant"] >= row["aupro_ref"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out / "resolution.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (out / "summary.json").write_text(json.dumps(
            [{k: r[k] for k in ("category", "long_side", "auroc_ref", "auroc_variant", "estimate", "ci_low_object",
                                "ci_high_object", "p_holm_object", "aupro_ref", "aupro_variant", "fa_ref",
                                "fa_variant", "carry_to_three_seeds")} for r in rows], indent=1), encoding="utf-8")
    if args.gate:
        complete = len(rows) == len(CATEGORIES)
        losses = [r["category"] for r in rows if r["significant_object"] and r["estimate"] < 0]
        aupro_ref = sum(r["aupro_ref"] for r in rows) / max(1, len(rows))
        aupro_v3 = sum(r["aupro_variant"] for r in rows) / max(1, len(rows))
        gate = {"categories": len(rows), "significant_losses": losses,
                "significant_gains": [r["category"] for r in rows if r["significant_object"] and r["estimate"] > 0],
                "macro_auroc_ref": sum(r["auroc_ref"] for r in rows) / max(1, len(rows)),
                "macro_auroc_v3": sum(r["auroc_variant"] for r in rows) / max(1, len(rows)),
                "macro_aupro_ref": aupro_ref, "macro_aupro_v3": aupro_v3,
                "false_alarms_ref": sum(r["fa_ref"] for r in rows), "false_alarms_v3": sum(r["fa_variant"] for r in rows),
                "submit_v3": bool(complete and not losses and aupro_v3 > aupro_ref)}
        (out / "gate.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")
        print(json.dumps(gate, indent=1))
    three = []
    for r in (r for r in rows if r["carry_to_three_seeds"] and not args.gate):
        ref_s, var_s = {}, {}
        for seed in (0, 1, 2):
            by_seed = load(Path(args.registry), args.implementation, "development", seed)
            ref, var = by_seed.get((r["category"], "reference")), by_seed.get((r["category"], f"{r['long_side']}px,layer2+layer3"))
            if ref and var:
                ref_s.update(ref["seeds"])
                var_s.update(var["seeds"])
        if len(ref_s) == len(var_s) == 3:
            res = cluster_paired(var_s, ref_s, seed=1950)
            aupro = {}
            for name, seeds in (("ref", ref_s), ("variant", var_s)):
                aupro[name] = [load(Path(args.registry), args.implementation, "development", s)[
                    (r["category"], "reference" if name == "ref" else f"{r['long_side']}px,layer2+layer3")]["result"]["aupro_005"]
                    for s in seeds]
            three.append({"category": r["category"], "long_side": r["long_side"], **res,
                          "aupro_ref_mean": sum(aupro["ref"]) / 3, "aupro_variant_mean": sum(aupro["variant"]) / 3})
    if three:
        (out / "three_seeds.json").write_text(json.dumps(three, indent=1), encoding="utf-8")
        for t in three:
            print(f"three seeds: {t['category']} {t['long_side']}px  {t['estimate']:+.3f} object "
                  f"[{t['ci_low_object']:+.3f}, {t['ci_high_object']:+.3f}] p {t['p_object']:.4f}  "
                  f"AU-PRO {t['aupro_ref_mean']:.3f} -> {t['aupro_variant_mean']:.3f}")
    for r in rows:
        print(f"{r['category']:<12} {r['long_side']:>5}px  AUROC {r['auroc_ref']:.3f} -> {r['auroc_variant']:.3f} "
              f"({r['estimate']:+.3f}, object [{r['ci_low_object']:+.3f}, {r['ci_high_object']:+.3f}], Holm p "
              f"{r['p_holm_object']:.3f})  regular {r['auroc_regular_ref']:.3f} -> {r['auroc_regular_variant']:.3f}  "
              f"AU-PRO {r['aupro_ref']:.3f} -> {r['aupro_variant']:.3f}  FA {r['fa_ref']} -> {r['fa_variant']}/{r['normals']}  "
              f"fit {r['fit_s_ref']:.0f} -> {r['fit_s_variant']:.0f} s  carry {r['carry_to_three_seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
