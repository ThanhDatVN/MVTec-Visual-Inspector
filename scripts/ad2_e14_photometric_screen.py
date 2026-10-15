"""E14 screen on MVTec AD 2 `test_public` (development, docs/11 Stage H‴).

Each variant of recipe v1 against the E14 reference (v1 at the same implementation, same seed),
per category: image AUROC paired on the same test images (class-stratified bootstrap, 10,000
resamples, Holm over the 8 categories of that variant), AU-PRO@0.05, and the operating point's
false alarms against the expected count. The carry rule is the one written in docs/11 before the
runs. The reference must reproduce E13's seed-0 v1 predictions exactly.

    python scripts/ad2_e14_photometric_screen.py --implementation 54752852d47e --out reports/studies/e14
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from inspector.evaluate import SplitPredictions
from inspector.metrics.ranking import roc_auc
from inspector.stats import holm_bonferroni

CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
E13_IMPLEMENTATION = "af89b5297f4d"
N_BOOT = 10_000


def variant_of(spec: dict) -> str:
    normalize = spec["transform"].get("normalize", "imagenet")
    exposures = spec["model"]["hparams"].get("bank_exposures") or []
    parts = []
    if normalize != "imagenet":
        parts.append(f"normalize={normalize}")
    if exposures:
        parts.append(f"bank_exposures={exposures}")
    if spec["transform"]["long_side"] != 640 or spec["model"]["hparams"].get("layers") != ["layer2", "layer3"]:
        parts.append("other")
    return ",".join(parts) or "reference"


def load(registry: Path, implementation: str, role: str) -> dict:
    runs: dict[tuple[str, str, int], dict] = {}
    for f in sorted(registry.glob("*/result.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        spec = json.loads(f.with_name("spec.json").read_text(encoding="utf-8"))
        if (rec.get("status") != "completed" or spec["dataset"].get("layout") != "mvtec_ad2"
                or spec["model"]["name"] != "patchcore" or spec.get("role", "development") != role
                or not spec["implementation_id"].startswith(implementation)):
            continue
        key = (spec["dataset"]["category"], variant_of(spec), int(spec["seed"]))
        runs[key] = {"dir": f.parent, "result": rec["result"], "digests": rec.get("prediction_digests")}
    return runs


def paired(a: SplitPredictions, b: SplitPredictions, seed: int) -> dict:
    if a.ids != b.ids:
        raise ValueError("runs were not evaluated on the same images")
    y = a.labels
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    est = roc_auc(y, a.scores) - roc_auc(y, b.scores)
    rng = np.random.default_rng(seed)
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        d[i] = roc_auc(y[idx], a.scores[idx]) - roc_auc(y[idx], b.scores[idx])
    tail = min(int(np.sum(d <= 0)), int(np.sum(d >= 0)))
    return {"d_auroc": float(est), "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
            "p_boot": float(min(1.0, 2 * (tail + 1) / (N_BOOT + 1)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--implementation", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/studies/e14")
    args = ap.parse_args()
    reg = Path(args.registry)
    runs = load(reg, args.implementation, "development")
    e13 = load(reg, E13_IMPLEMENTATION, "confirmation")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reproduced = {c: (runs.get((c, "reference", args.seed), {}).get("digests") is not None
                      and runs[(c, "reference", args.seed)]["digests"] == e13.get((c, "reference", args.seed), {}).get("digests"))
                  for c in CATEGORIES}
    variants = sorted({v for (_, v, s) in runs if v != "reference" and s == args.seed})
    rows = []
    for vi, v in enumerate(variants):
        block = []
        for ci, c in enumerate(CATEGORIES):
            ref, var = runs.get((c, "reference", args.seed)), runs.get((c, v, args.seed))
            if not ref or not var:
                continue
            split = "test_public"
            a, b = SplitPredictions.load(var["dir"], split), SplitPredictions.load(ref["dir"], split)
            rv, rr = var["result"], ref["result"]
            block.append({"variant": v, "category": c,
                          "auroc_ref": rr["image_auroc"], "auroc_variant": rv["image_auroc"],
                          "aupro_ref": rr["aupro_005"], "aupro_variant": rv["aupro_005"],
                          "pixel_auroc_ref": rr["pixel_auroc"], "pixel_auroc_variant": rv["pixel_auroc"],
                          "fp_ref": rr["fp"], "fp_variant": rv["fp"], "normals": rr["fp"] + rr["tn"],
                          "bound": rr["effective_fpr"],
                          "recall_ref": rr["recall_at_op1"], "recall_variant": rv["recall_at_op1"],
                          "fit_s_variant": rv["fit_seconds"], "fit_s_ref": rr["fit_seconds"],
                          **paired(a, b, seed=100 * vi + ci)})
        for row, (adj, sig) in zip(block, holm_bonferroni([r["p_boot"] for r in block])):
            row.update({"p_holm": adj, "significant": sig})
        rows += block

    decisions = []
    for v in variants:
        block = [r for r in rows if r["variant"] == v]
        fa_ref, fa_var = sum(r["fp_ref"] for r in block), sum(r["fp_variant"] for r in block)
        gains = sum(r["significant"] and r["d_auroc"] > 0 for r in block)
        losses = sum(r["significant"] and r["d_auroc"] < 0 for r in block)
        carry = (fa_var <= 0.5 * fa_ref and losses == 0) or (gains >= 3 and losses == 0)
        decisions.append({"variant": v, "categories": len(block), "false_alarms_ref": fa_ref,
                          "false_alarms_variant": fa_var,
                          "expected": float(sum(r["bound"] * r["normals"] for r in block)),
                          "significant_gains": gains, "significant_losses": losses,
                          "macro_auroc_ref": float(np.mean([r["auroc_ref"] for r in block])),
                          "macro_auroc_variant": float(np.mean([r["auroc_variant"] for r in block])),
                          "macro_aupro_ref": float(np.mean([r["aupro_ref"] for r in block])),
                          "macro_aupro_variant": float(np.mean([r["aupro_variant"] for r in block])),
                          "carry_to_three_seeds": carry})

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
    by = defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)
    for v, block in by.items():
        print(f"\n{v}")
        for r in block:
            print(f"  {r['category']:<12} AUROC {r['auroc_ref']:.3f} -> {r['auroc_variant']:.3f} "
                  f"({r['d_auroc']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]{' *' if r['significant'] else ''})  "
                  f"AU-PRO {r['aupro_ref']:.3f} -> {r['aupro_variant']:.3f}  FP {r['fp_ref']} -> {r['fp_variant']} / {r['normals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
