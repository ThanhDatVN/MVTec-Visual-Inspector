"""The pre-registered confirmation analysis of ADR-11.

Committed together with ADR-11, before any confirmation run existed, so that the
analysis — not only the recipe — was fixed before the data was seen. Running it
again on the same registry reproduces the report's numbers exactly.

    python scripts/confirmation_analysis.py --registry reports/runs \
        --out reports/studies/confirmation-v1

Hypotheses (ADR-11 §4), one family of 18 tests, Holm-adjusted together:
  H1  AUROC(PatchCore 640 px, seed 0) - AUROC(PatchCore 320 px, seed 0) > 0
  H2  mean AUROC(PatchCore 640 px, seeds 0-2) - AUROC(pixel PCA) > 0
Each is a paired, class-stratified image bootstrap with a two-sided
(count + 1) / (n + 1) p-value. A hypothesis is *confirmed* at dataset level if
at least 6 of 9 categories are significant in the positive direction and none
in the negative direction.

Calibration (ADR-11 §5) is a separate, descriptive family of 9 tail tests.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from inspector.analysis import exceedance_p_value
from inspector.evaluate import SplitPredictions
from inspector.metrics.ranking import roc_auc
from inspector.postproc.thresholds import conformal_rank
from inspector.stats import holm_bonferroni

CATEGORIES = ["candle", "cashew", "chewinggum", "fryum", "macaroni1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
N_RESAMPLES = 10_000
ALPHA = 0.05
CONFIRM_MIN_CATEGORIES = 6
TARGET_FPR = 0.01


def arm_of(spec: dict) -> str:
    """The pre-declared arms; any other PatchCore setting gets a key no test reads."""
    name = spec["model"]["name"]
    if name == "patchcore":
        fixed = spec["model"]["hparams"].get("bank_size") == 10_000
        return f"patchcore{spec['transform']['long_side']}" + ("" if fixed else "-other-bank")
    return name


def load(registry: Path, role: str = "confirmation") -> dict[tuple[str, str], list[dict]]:
    """(category, arm) -> completed runs of `role`, sorted by seed."""
    runs: dict[tuple[str, str], list[dict]] = {}
    for f in sorted(registry.glob("*/result.json")):
        record = json.loads(f.read_text(encoding="utf-8"))
        spec_file = f.with_name("spec.json")
        if record.get("status") != "completed" or not spec_file.is_file():
            continue
        spec = json.loads(spec_file.read_text(encoding="utf-8"))
        if spec.get("role") != role:
            continue
        key = (spec["dataset"]["category"], arm_of(spec))
        runs.setdefault(key, []).append({
            "dir": f.parent, "seed": int(spec["seed"]), "split": spec["dataset"]["test_split"],
            "result": record["result"], "implementation_id": spec["implementation_id"],
        })
    for v in runs.values():
        v.sort(key=lambda r: r["seed"])
    return runs


def predictions(run: dict, split: str) -> SplitPredictions:
    return SplitPredictions.load(run["dir"], split)


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
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p": float(min(1.0, 2.0 * (tail + 1) / (N_RESAMPLES + 1))),
        "draws": diffs,
    }


def verdict(rows: list[dict]) -> str:
    positive = sum(r["significant"] and r["estimate"] > 0 for r in rows)
    negative = sum(r["significant"] and r["estimate"] < 0 for r in rows)
    ok = positive >= CONFIRM_MIN_CATEGORIES and negative == 0
    return (f"{'CONFIRMED' if ok else 'NOT CONFIRMED'} "
            f"({positive}/{len(rows)} significant positive, {negative} negative)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--out", default="reports/studies/confirmation-v1")
    ap.add_argument("--dry-run-development", action="store_true",
                    help="smoke-test this script on the development categories (never reported)")
    args = ap.parse_args()
    categories = CATEGORIES
    role = "confirmation"
    if args.dry_run_development:
        categories, role = ["pcb1", "macaroni2", "capsules"], "development"
    runs = load(Path(args.registry), role)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    missing = [(c, a) for c in categories for a in ("patchcore640", "patchcore320", "pixel_pca")
               if (c, a) not in runs]
    tests: list[dict] = []
    draws: dict[str, list[np.ndarray]] = {"H1": [], "H2": []}
    calibration: list[dict] = []
    descriptive: list[dict] = []

    for c_index, cat in enumerate(categories):
        if any(m[0] == cat for m in missing):
            continue
        r640, r320, pca = runs[(cat, "patchcore640")], runs[(cat, "patchcore320")], runs[(cat, "pixel_pca")]
        split = r640[0]["split"]
        t640 = [predictions(r, split) for r in r640]
        t320 = predictions(r320[0], split)
        tpca = predictions(pca[0], split)
        labels = t640[0].labels
        for other in (*t640, t320, tpca):
            if other.ids != t640[0].ids:
                raise ValueError(f"{cat}: arms were not evaluated on the same images")

        seed0 = next(t for r, t in zip(r640, t640) if r["seed"] == 0)
        for name, a, b in (("H1", [seed0.scores], [t320.scores]),
                           ("H2", [t.scores for t in t640], [tpca.scores])):
            res = paired(labels, a, b, seed=1000 * c_index + (1 if name == "H1" else 2))
            draws[name].append(res.pop("draws"))
            tests.append({"hypothesis": name, "category": cat, **res})

        # calibration, seed 0 for the test; all seeds for the realized rates
        val0 = predictions(next(r for r in r640 if r["seed"] == 0), "validation")
        n_val = int(val0.scores.size)
        k = conformal_rank(n_val, TARGET_FPR) or 1
        thr = float(np.sort(val0.scores)[::-1][k - 1])
        normals = seed0.labels == 0
        fp = int(np.sum(seed0.scores[normals] > thr))
        calibration.append({
            "category": cat, "n_validation": n_val, "k": k, "bound": k / (n_val + 1),
            "test_normals": int(normals.sum()), "fp_seed0": fp,
            "tail_p": exceedance_p_value(fp, int(normals.sum()), n_val, k),
            "fp_all_seeds": [int(r["result"]["fp"]) for r in r640],
            "flagged_seed0": [i for i, s, y in zip(seed0.ids, seed0.scores, seed0.labels) if y == 0 and s > thr],
        })

        res = [r["result"] for r in r640]
        def ms(key: str, rows: list[dict] = res) -> str:
            vals = [x[key] for x in rows]
            return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"
        hist = runs.get((cat, "histogram"), [{}])[0].get("result", {})
        rand = runs.get((cat, "random"), [{}])[0].get("result", {})
        descriptive.append({
            "category": cat,
            "auroc_640": ms("image_auroc"), "aupro005_640": ms("aupro_005"), "aupro030_640": ms("aupro_030"),
            "recall_640": ms("recall_at_op1"), "fpr_640": ms("fpr_at_op1"),
            "effective_fpr": res[0]["effective_fpr"],
            "auroc_320": r320[0]["result"]["image_auroc"], "aupro005_320": r320[0]["result"]["aupro_005"],
            "auroc_pixel_pca": pca[0]["result"]["image_auroc"],
            "auroc_histogram": hist.get("image_auroc", float("nan")),
            "auroc_random": rand.get("image_auroc", float("nan")),
            "fit_s_640": statistics.mean(x["fit_seconds"] for x in res),
            "peak_rss_fit_mb_640": max(x.get("peak_rss_fit_mb", float("nan")) for x in res),
            "peak_vram_mb_640": max(x.get("peak_vram_mb", float("nan")) for x in res),
        })

    for row, (adj, reject) in zip(tests, holm_bonferroni([t["p"] for t in tests], alpha=ALPHA)):
        row.update({"p_holm": adj, "significant": reject})
    for row, (adj, reject) in zip(calibration, holm_bonferroni([c["tail_p"] for c in calibration], alpha=ALPHA)):
        row.update({"tail_p_holm": adj, "tail_significant": reject})

    macro = {}
    for name in ("H1", "H2"):
        if draws[name]:
            stacked = np.mean(np.stack(draws[name]), axis=0)  # independent resampling per category
            est = float(np.mean([t["estimate"] for t in tests if t["hypothesis"] == name]))
            macro[name] = {"estimate": est, "ci_low": float(np.quantile(stacked, 0.025)),
                           "ci_high": float(np.quantile(stacked, 0.975))}
    verdicts = {name: verdict([t for t in tests if t["hypothesis"] == name]) for name in ("H1", "H2")
                if any(t["hypothesis"] == name for t in tests)}

    pooled_fp = sum(sum(c["fp_all_seeds"]) for c in calibration)
    pooled_expected = sum(c["bound"] * c["test_normals"] * len(c["fp_all_seeds"]) for c in calibration)
    implementation = sorted({r["implementation_id"][:12] for v in runs.values() for r in v})

    summary = {"missing": missing, "tests": tests, "macro": macro, "verdicts": verdicts,
               "calibration": calibration, "pooled_fp": pooled_fp, "pooled_expected_fp": pooled_expected,
               "descriptive": descriptive, "implementation_ids": implementation,
               "family_size": len(tests), "n_resamples": N_RESAMPLES}
    (out / "confirmation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Confirmation v1 — pre-registered analysis output (ADR-11)", "",
             f"Generated by `scripts/confirmation_analysis.py`. Implementation ids: {', '.join(implementation)}.",
             f"Missing (category, arm): {missing or 'none'}.", "",
             "## Confirmatory tests", "",
             f"Family of {len(tests)} tests, Holm at alpha = {ALPHA}; bootstrap {N_RESAMPLES:,} resamples.", "",
             "| hypothesis | category | d AUROC [95% CI] | p (Holm) |", "|---|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['hypothesis']} | `{t['category']}` | {t['estimate']:+.3f} [{t['ci_low']:+.3f}, "
                     f"{t['ci_high']:+.3f}] | {t['p_holm']:.3g}{' *' if t['significant'] else ''} |")
    lines += [""]
    for name, v in verdicts.items():
        m = macro[name]
        lines.append(f"- **{name}: {v}.** Macro-mean difference {m['estimate']:+.3f} "
                     f"[{m['ci_low']:+.3f}, {m['ci_high']:+.3f}].")
    lines += ["", "## Calibration (descriptive family)", "",
              "| category | n val | k | bound | test normals | FP seed 0 | tail p (Holm) | FP seeds 0/1/2 |",
              "|---|---|---|---|---|---|---|---|"]
    for c in calibration:
        lines.append(f"| `{c['category']}` | {c['n_validation']} | {c['k']} | {c['bound']:.2%} | {c['test_normals']} | "
                     f"{c['fp_seed0']} | {c['tail_p_holm']:.3g}{' *' if c['tail_significant'] else ''} | "
                     f"{'/'.join(map(str, c['fp_all_seeds']))} |")
    lines += ["", f"Pooled false alarms over all categories and seeds: {pooled_fp} observed, "
              f"{pooled_expected:.1f} expected at the effective bounds.", "",
              "## Descriptive", "",
              "| category | I-AUROC 640 | AU-PRO@0.05 640 | AU-PRO@0.30 640 | recall@OP | FPR@OP (bound) | "
              "I-AUROC 320 | pixel PCA | histogram | random | fit s | peak RSS MB |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in descriptive:
        lines.append(f"| `{d['category']}` | {d['auroc_640']} | {d['aupro005_640']} | {d['aupro030_640']} | "
                     f"{d['recall_640']} | {d['fpr_640']} ({d['effective_fpr']:.2%}) | {d['auroc_320']:.3f} | "
                     f"{d['auroc_pixel_pca']:.3f} | {d['auroc_histogram']:.3f} | {d['auroc_random']:.3f} | "
                     f"{d['fit_s_640']:.0f} | {d['peak_rss_fit_mb_640']:.0f} |")
    (out / "confirmation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
