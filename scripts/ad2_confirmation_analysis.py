"""The pre-registered analysis of ADR-13: candidate recipe v2 against v1 on MVTec AD 2.

Committed together with ADR-13, before any MVTec AD 2 image was downloaded, so the analysis —
not only the recipes — was fixed before the data was seen. Running it again on the same
registry reproduces the report's numbers exactly.

    python scripts/ad2_confirmation_analysis.py --registry reports/runs --out reports/studies/ad2-confirmation

Hypotheses (ADR-13 §3):
  H3   image AUROC: mean over seeds 0-2 of v2 minus that of v1, per category; paired,
       class-stratified image bootstrap (10,000 resamples), two-sided (count + 1) / (n + 1)
       p-value, Holm over the 8 categories. Confirmed if >= 5 of 8 are significantly positive
       and none significantly negative.
  H3b  AU-PRO@0.05: v2's mean over seeds exceeds v1's on >= 7 of 8 categories (one-sided exact
       sign test, p = 9/256 = 0.035 at 7 of 8).
  v2 is CONFIRMED if H3 and H3b both hold, PARTIAL if one does, NOT CONFIRMED otherwise.
Calibration and the floors are descriptive. A category without all arms is reported as missing
and counts as not significant.

`--dry-run-development` exercises the script on the E8 development runs (VisA `pcb1`,
`macaroni2`, `capsules`), which contain the same two recipes; its output is never reported.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from inspector.analysis import exceedance_p_value
from inspector.config import deep_merge, load_config
from inspector.evaluate import SplitPredictions
from inspector.metrics.ranking import roc_auc
from inspector.models import build_model
from inspector.postproc.thresholds import conformal_rank
from inspector.runner import model_kwargs, transform_from_config
from inspector.stats import holm_bonferroni

CATEGORIES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
DEV_CATEGORIES = ["pcb1", "macaroni2", "capsules"]
RECIPES = {"v1": "configs/recipes/confirmation-v1-patchcore-640.yaml",
           "v2": "configs/recipes/candidate-v2-patchcore-layer2-960.yaml"}
DATA_CONFIG = "configs/data/mvtec_ad2.yaml"
N_RESAMPLES = 10_000
ALPHA = 0.05
H3_MIN_POSITIVE = 5
H3B_MIN_CATEGORIES = 7
TARGET_FPR = 0.01


def recipe_signature(recipe: str) -> tuple[dict, int]:
    """The recipe's effective PatchCore hyperparameters (defaults included) and long side."""
    cfg = deep_merge(load_config(DATA_CONFIG), load_config(recipe))
    transform = transform_from_config(cfg)
    model = build_model("patchcore", transform, seed=0, **model_kwargs(cfg, "patchcore"))
    hp = {k: v for k, v in model.hparams().items() if k != "seed"}
    return json.loads(json.dumps(hp)), int(cfg["preprocess"]["long_side"])


def arm_of(spec: dict, signatures: dict[str, tuple[dict, int]]) -> str:
    name = spec["model"]["name"]
    if name != "patchcore":
        return name
    hp = {k: v for k, v in spec["model"]["hparams"].items() if k != "seed"}
    hp.setdefault("weights", "IMAGENET1K_V1")
    for arm, (sig, side) in signatures.items():
        if hp == sig and spec["transform"]["long_side"] == side:
            return arm
    return "patchcore-other"


def load(registry: Path, role: str, layout: str, signatures, implementation: str | None):
    """(category, arm) -> completed runs, sorted by seed."""
    runs: dict[tuple[str, str], list[dict]] = {}
    for f in sorted(registry.glob("*/result.json")):
        record = json.loads(f.read_text(encoding="utf-8"))
        spec_file = f.with_name("spec.json")
        if record.get("status") != "completed" or not spec_file.is_file():
            continue
        spec = json.loads(spec_file.read_text(encoding="utf-8"))
        if spec.get("role", "development") != role or spec["dataset"].get("layout") != layout:
            continue
        if implementation and not spec["implementation_id"].startswith(implementation):
            continue
        key = (spec["dataset"]["category"], arm_of(spec, signatures))
        runs.setdefault(key, []).append({
            "dir": f.parent, "seed": int(spec["seed"]), "split": spec["dataset"]["test_split"],
            "result": record["result"], "implementation_id": spec["implementation_id"],
        })
    for v in runs.values():
        v.sort(key=lambda r: r["seed"])
    return runs


def paired(labels: np.ndarray, a: list[np.ndarray], b: list[np.ndarray], seed: int) -> dict:
    """Seed-averaged AUROC(a) - AUROC(b) with its class-stratified bootstrap."""
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    estimate = float(np.mean([roc_auc(labels, s) for s in a]) - np.mean([roc_auc(labels, s) for s in b]))
    rng = np.random.default_rng(seed)
    diffs = np.empty(N_RESAMPLES)
    for i in range(N_RESAMPLES):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        y = labels[idx]
        diffs[i] = np.mean([roc_auc(y, s[idx]) for s in a]) - np.mean([roc_auc(y, s[idx]) for s in b])
    tail = min(int(np.sum(diffs <= 0.0)), int(np.sum(diffs >= 0.0)))
    return {"estimate": estimate, "ci_low": float(np.quantile(diffs, 0.025)),
            "ci_high": float(np.quantile(diffs, 0.975)),
            "p": float(min(1.0, 2.0 * (tail + 1) / (N_RESAMPLES + 1))), "draws": diffs}


def sign_test_p(successes: int, n: int) -> float:
    """One-sided exact binomial P(X >= successes) at p = 1/2."""
    return sum(math.comb(n, k) for k in range(successes, n + 1)) / 2**n


def mean_sd(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows]
    return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--out", default="reports/studies/ad2-confirmation")
    ap.add_argument("--implementation", help="id prefix; default: every completed run of the role")
    ap.add_argument("--dry-run-development", action="store_true",
                    help="exercise the script on the E8 development runs (never reported)")
    args = ap.parse_args()
    signatures = {arm: recipe_signature(path) for arm, path in RECIPES.items()}
    categories, role, layout = CATEGORIES, "confirmation", "mvtec_ad2"
    if args.dry_run_development:
        categories, role, layout = DEV_CATEGORIES, "development", "visa"
    runs = load(Path(args.registry), role, layout, signatures, args.implementation)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    needed = ("v1", "v2")
    missing = [(c, a) for c in categories for a in needed if len(runs.get((c, a), [])) < 3]
    tests, calibration, descriptive, pro_rows = [], [], [], []
    draws: list[np.ndarray] = []
    for c_index, cat in enumerate(categories):
        if any(m[0] == cat for m in missing):
            continue
        r1, r2 = runs[(cat, "v1")], runs[(cat, "v2")]
        split = r1[0]["split"]
        t1 = [SplitPredictions.load(r["dir"], split) for r in r1]
        t2 = [SplitPredictions.load(r["dir"], split) for r in r2]
        if any(t.ids != t1[0].ids for t in t1 + t2):
            raise ValueError(f"{cat}: arms were not evaluated on the same images")
        labels = t1[0].labels
        res = paired(labels, [t.scores for t in t2], [t.scores for t in t1], seed=1000 * c_index + 3)
        draws.append(res.pop("draws"))
        tests.append({"category": cat, **res})

        pro1 = statistics.mean(r["result"]["aupro_005"] for r in r1)
        pro2 = statistics.mean(r["result"]["aupro_005"] for r in r2)
        pro_rows.append({"category": cat, "aupro_v1": pro1, "aupro_v2": pro2, "v2_higher": pro2 > pro1})

        for arm, rr, tt in (("v1", r1, t1), ("v2", r2, t2)):
            seed0 = next(t for r, t in zip(rr, tt) if r["seed"] == 0)
            val0 = SplitPredictions.load(next(r for r in rr if r["seed"] == 0)["dir"], "validation")
            n_val = int(val0.scores.size)
            k = conformal_rank(n_val, TARGET_FPR) or 1  # policy relax: the sample maximum
            thr = float(np.sort(val0.scores)[::-1][k - 1])
            normals = seed0.labels == 0
            fp = int(np.sum(seed0.scores[normals] > thr))
            calibration.append({
                "category": cat, "recipe": arm, "n_validation": n_val, "k": k, "bound": k / (n_val + 1),
                "test_normals": int(normals.sum()), "fp_seed0": fp,
                "tail_p": exceedance_p_value(fp, int(normals.sum()), n_val, k),
                "fp_all_seeds": [int(r["result"]["fp"]) for r in rr],
            })

        floors = {name: runs.get((cat, name), [{}])[0].get("result", {}) for name in ("pixel_pca", "histogram")}
        res1, res2 = [r["result"] for r in r1], [r["result"] for r in r2]
        descriptive.append({
            "category": cat,
            **{f"{m}_{arm}": mean_sd(rows, key) for arm, rows in (("v1", res1), ("v2", res2))
               for m, key in (("auroc", "image_auroc"), ("aupro005", "aupro_005"), ("aupro030", "aupro_030"),
                              ("pixel_auroc", "pixel_auroc"), ("segf1", "segf1_3sigma"),
                              ("fpr", "fpr_at_op1"), ("recall", "recall_at_op1"))},
            "effective_fpr": res1[0]["effective_fpr"],
            "auroc_pixel_pca": floors["pixel_pca"].get("image_auroc", float("nan")),
            "auroc_histogram": floors["histogram"].get("image_auroc", float("nan")),
            "fit_s_v1": statistics.mean(x["fit_seconds"] for x in res1),
            "fit_s_v2": statistics.mean(x["fit_seconds"] for x in res2),
            "ms_per_image_v1": statistics.mean(x["e2e_ms_per_image"] for x in res1),
            "ms_per_image_v2": statistics.mean(x["e2e_ms_per_image"] for x in res2),
        })

    for row, (adj, reject) in zip(tests, holm_bonferroni([t["p"] for t in tests], alpha=ALPHA)):
        row.update({"p_holm": adj, "significant": reject})
    for row, (adj, reject) in zip(calibration, holm_bonferroni([c["tail_p"] for c in calibration], alpha=ALPHA)):
        row.update({"tail_p_holm": adj, "tail_significant": reject})

    positive = sum(t["significant"] and t["estimate"] > 0 for t in tests)
    negative = sum(t["significant"] and t["estimate"] < 0 for t in tests)
    h3 = positive >= H3_MIN_POSITIVE and negative == 0
    pro_wins = sum(r["v2_higher"] for r in pro_rows)
    h3b = pro_wins >= H3B_MIN_CATEGORIES
    overall = "CONFIRMED" if (h3 and h3b) else "PARTIAL" if (h3 or h3b) else "NOT CONFIRMED"
    macro = {}
    if draws:
        stacked = np.mean(np.stack(draws), axis=0)  # categories resampled independently
        macro = {"estimate": float(np.mean([t["estimate"] for t in tests])),
                 "ci_low": float(np.quantile(stacked, 0.025)), "ci_high": float(np.quantile(stacked, 0.975))}
    pooled = {arm: {"fp": sum(sum(c["fp_all_seeds"]) for c in calibration if c["recipe"] == arm),
                    "expected": sum(c["bound"] * c["test_normals"] * len(c["fp_all_seeds"])
                                    for c in calibration if c["recipe"] == arm)} for arm in needed}
    implementation = sorted({r["implementation_id"][:12] for v in runs.values() for r in v})

    summary = {"missing": missing, "h3_tests": tests, "h3": h3, "h3_positive": positive, "h3_negative": negative,
               "h3b_rows": pro_rows, "h3b": h3b, "h3b_wins": pro_wins,
               "h3b_sign_p": sign_test_p(pro_wins, len(pro_rows)) if pro_rows else None,
               "verdict": overall, "macro_auroc_difference": macro, "calibration": calibration,
               "pooled_false_alarms": pooled, "descriptive": descriptive,
               "implementation_ids": implementation, "n_resamples": N_RESAMPLES,
               "dry_run_development": args.dry_run_development}
    (out / "confirmation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Confirmation v2 on MVTec AD 2 — pre-registered analysis output (ADR-13)", "",
             f"Generated by `scripts/ad2_confirmation_analysis.py`. Implementation ids: {', '.join(implementation)}."
             + (" **DRY RUN on development data — not a result.**" if args.dry_run_development else ""),
             f"Missing (category, recipe): {missing or 'none'}.", "",
             f"## H3 — image AUROC, v2 minus v1 (Holm over {len(tests)}, {N_RESAMPLES:,} resamples)", "",
             "| category | Δ AUROC [95% CI] | p (Holm) |", "|---|---|---|"]
    for t in tests:
        lines.append(f"| `{t['category']}` | {t['estimate']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] | "
                     f"{t['p_holm']:.3g}{' *' if t['significant'] else ''} |")
    lines += ["", f"- **H3: {'holds' if h3 else 'does not hold'}** ({positive}/{len(categories)} significantly "
              f"positive, {negative} negative; rule ≥ {H3_MIN_POSITIVE} and none)."
              + (f" Macro {macro['estimate']:+.3f} [{macro['ci_low']:+.3f}, {macro['ci_high']:+.3f}]." if macro else ""),
              "", "## H3b — AU-PRO@0.05, mean over seeds", "",
              "| category | v1 | v2 | v2 higher |", "|---|---|---|---|"]
    for r in pro_rows:
        lines.append(f"| `{r['category']}` | {r['aupro_v1']:.3f} | {r['aupro_v2']:.3f} | {'yes' if r['v2_higher'] else 'no'} |")
    lines += ["", f"- **H3b: {'holds' if h3b else 'does not hold'}** (v2 higher on {pro_wins}/{len(categories)}; "
              f"rule ≥ {H3B_MIN_CATEGORIES}; one-sided sign test p = "
              f"{summary['h3b_sign_p'] if summary['h3b_sign_p'] is None else round(summary['h3b_sign_p'], 4)}).",
              "", f"## Verdict: **{overall}**", "",
              "## Calibration (descriptive)", "",
              "| category | recipe | n val | k | bound | test normals | FP seed 0 | tail p (Holm) | FP seeds 0/1/2 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in calibration:
        lines.append(f"| `{c['category']}` | {c['recipe']} | {c['n_validation']} | {c['k']} | {c['bound']:.2%} | "
                     f"{c['test_normals']} | {c['fp_seed0']} | {c['tail_p_holm']:.3g}"
                     f"{' *' if c['tail_significant'] else ''} | {'/'.join(map(str, c['fp_all_seeds']))} |")
    lines += [""] + [f"- Pooled false alarms, {arm}: {p['fp']} observed, {p['expected']:.1f} expected at the bounds."
                     for arm, p in pooled.items()]
    lines += ["", "## Descriptive", "",
              "| category | I-AUROC v1 | I-AUROC v2 | AU-PRO@0.05 v1 | v2 | AU-PRO@0.30 v1 | v2 | SegF1 v1 | v2 | "
              "FPR v1 | v2 (bound) | recall v1 | v2 | pixel PCA | histogram | ms/img v1 | v2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in descriptive:
        lines.append(f"| `{d['category']}` | {d['auroc_v1']} | {d['auroc_v2']} | {d['aupro005_v1']} | {d['aupro005_v2']} | "
                     f"{d['aupro030_v1']} | {d['aupro030_v2']} | {d['segf1_v1']} | {d['segf1_v2']} | {d['fpr_v1']} | "
                     f"{d['fpr_v2']} ({d['effective_fpr']:.2%}) | {d['recall_v1']} | {d['recall_v2']} | "
                     f"{d['auroc_pixel_pca']:.3f} | {d['auroc_histogram']:.3f} | {d['ms_per_image_v1']:.0f} | "
                     f"{d['ms_per_image_v2']:.0f} |")
    (out / "confirmation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
