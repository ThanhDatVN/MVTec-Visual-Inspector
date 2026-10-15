"""Analysis of the Kaggle sessions (sweep, Gate E follow-up, CAE, C2, MPDD) at one implementation id.

Reads the registry only (`reports/runs`), never the images. Image-AUROC differences are paired
on the same test images with a class-stratified bootstrap; for the sweep (single seed) the
bootstrap p-values are Holm-adjusted over every variant x category comparison. AU-PRO
differences are point estimates, read against the reference's seed spread.

Variants run at three seeds on every category form the Gate E follow-up family (docs/11, Stage E,
E8): seed-averaged paired comparisons with the reference, Holm over that family, and the DR-1
decision exactly as it was written down before the runs.

    python scripts/visa_k1_kaggle_studies.py --implementation e53218f1371e --out reports/studies/kaggle

DR-1's VRAM criterion should come from `--vram-csv` (scripts/deploy_bench_inference.py, one fresh
process per configuration): until the FeatureExtractor fix, the runner's per-run VRAM peaks include the
backbones of earlier runs in the same session (see reports/K1), so the registry value is used only
as a fallback and flagged as such.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from inspector.config import deep_merge, load_config
from inspector.evaluate import SplitPredictions
from inspector.metrics.ranking import roc_auc
from inspector.models import build_model
from inspector.runner import model_kwargs, transform_from_config
from inspector.stats import holm_bonferroni

CATS = ("pcb1", "macaroni2", "capsules")
MPDD = ("bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes")
RECIPE = ("configs/data/visa_pcb1.yaml", "configs/recipes/confirmation-v1-patchcore-640.yaml")
# Enough resamples that a Holm-adjusted bootstrap p-value can fall below 0.05 for a family
# of 51 (the smallest attainable p is 2 / (N_BOOT + 1)).
N_BOOT = 10000
#: DR-1 (4): peak VRAM at inference, at the deployment resolution, in MB.
VRAM_BUDGET_MB = 3500


def load(registry: Path, impl: str) -> list[dict]:
    runs = []
    for f in sorted(registry.glob("*/result.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        spec = json.loads(f.with_name("spec.json").read_text(encoding="utf-8"))
        if rec.get("status") != "completed" or not spec["implementation_id"].startswith(impl):
            continue
        r = rec["result"]
        hp = {k: v for k, v in spec["model"]["hparams"].items() if k != "seed"}
        if spec["model"]["name"] == "patchcore":
            hp.setdefault("weights", "IMAGENET1K_V1")  # recorded only since the weights axis
        # The spec's fields win over the result's (`seed`, `method` and `split` appear in both).
        runs.append({**r, "id": rec["run_id"], "dir": f.parent, "method": spec["model"]["name"],
                     "role": spec.get("role", "development"), "cat": spec["dataset"]["category"],
                     "seed": spec["seed"], "res": spec["transform"]["long_side"], "hp": hp,
                     "split": spec["dataset"]["test_split"]})
    return runs


def recipe_hparams() -> dict:
    """The frozen recipe's effective PatchCore hyperparameters, defaults included."""
    cfg = deep_merge(load_config(RECIPE[0]), load_config(RECIPE[1]))
    model = build_model("patchcore", transform_from_config(cfg), seed=0, **model_kwargs(cfg, "patchcore"))
    return json.loads(json.dumps({k: v for k, v in model.hparams().items() if k != "seed"}))


def variant_of(run: dict, ref_hp: dict) -> str:
    """How a PatchCore run differs from the recipe; "" for the recipe itself."""
    diff = {k: v for k, v in run["hp"].items() if v != ref_hp.get(k)}
    if run["res"] != 640:
        diff["long_side"] = run["res"]
    return ",".join(f"{k}={v}" for k, v in diff.items())


def scores(run: dict) -> SplitPredictions:
    return SplitPredictions.load(run["dir"], run["split"])


def paired(a_runs: list[dict], b_runs: list[dict], seed: int = 0) -> dict:
    """Seed-averaged image AUROC of a minus b, bootstrapped over the same test images."""
    a = [scores(r) for r in a_runs]
    b = [scores(r) for r in b_runs]
    if any(p.ids != a[0].ids for p in a + b):
        raise ValueError("runs were not evaluated on the same images")
    y = a[0].labels
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    def diff(idx):
        return (np.mean([roc_auc(y[idx], p.scores[idx]) for p in a])
                - np.mean([roc_auc(y[idx], p.scores[idx]) for p in b]))

    est = float(diff(np.arange(y.size)))
    rng = np.random.default_rng(seed)
    d = np.array([diff(np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)]))
                  for _ in range(N_BOOT)])
    tail = min(int(np.sum(d <= 0)), int(np.sum(d >= 0)))
    return {"d_auroc": est, "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
            "p_boot": float(min(1.0, 2 * (tail + 1) / (N_BOOT + 1)))}


def dr1(rows: list[dict], vram_mb: float) -> str:
    """DR-1 as written in docs/11 (Stage E, E8) before the follow-up ran.

    (1) mean AU-PRO@0.05 improves on >= 2 of 3 categories and drops by <= 0.01 on the third;
    (2) the seed-averaged image-AUROC difference is significantly positive after Holm on
    >= 2 categories and significantly negative on none; (4) peak inference VRAM < 3.5 GB.
    (1)-(2) without (4) is an accuracy-only variant. No latency budget is set for this phase.
    """
    d_pro = [r["d_aupro"] for r in rows]
    accuracy = (sum(d > 0 for d in d_pro) >= 2 and min(d_pro) >= -0.01
                and sum(bool(r["significant_holm"]) and r["d_auroc"] > 0 for r in rows) >= 2
                and not any(bool(r["significant_holm"]) and r["d_auroc"] < 0 for r in rows))
    if not accuracy:
        return "not adopted"
    if vram_mb >= VRAM_BUDGET_MB:
        return "accuracy-only variant"
    return "adopted (development)"


def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(st.mean(vals)) if vals else float("nan")


def sd(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(st.stdev(vals)) if len(vals) > 1 else 0.0


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--implementation", required=True)
    ap.add_argument("--out", default="reports/studies/kaggle")
    ap.add_argument("--vram-csv", help="latency.csv from scripts/deploy_bench_inference.py (DR-1 criterion 4)")
    args = ap.parse_args()
    out = Path(args.out)
    runs = load(Path(args.registry), args.implementation)
    dev = [r for r in runs if r["role"] == "development"]

    # --- the reference: the frozen recipe at 640 px, three seeds per category ---------------
    ref_hp = recipe_hparams()
    pc640 = [r for r in dev if r["method"] == "patchcore" and r["res"] == 640]
    ref = {c: [r for r in pc640 if r["hp"] == ref_hp and r["cat"] == c] for c in CATS}
    if not all(len({r["seed"] for r in ref[c]}) >= 3 for c in CATS):
        counts = {c: len(ref[c]) for c in CATS}
        raise SystemExit(f"no 3-seed reference at this implementation: {counts}")
    ref0 = {c: next(r for r in ref[c] if r["seed"] == 0) for c in CATS}
    ref_rows = [{"category": c, "seeds": len(ref[c]),
                 **{f"{k}_mean": mean(ref[c], k) for k in ("image_auroc", "aupro_005", "fpr_at_op1", "recall_at_op1")},
                 **{f"{k}_sd": sd(ref[c], k) for k in ("image_auroc", "aupro_005")},
                 "fit_s": mean(ref[c], "fit_seconds"), "rss_fit_mb": mean(ref[c], "peak_rss_fit_mb"),
                 "e2e_ms": mean(ref[c], "e2e_ms_per_image")} for c in CATS]
    write(out / "reference.csv", ref_rows)

    # --- recipe variants: three seeds on every category = follow-up, otherwise sweep --------
    variants: dict[str, list[dict]] = defaultdict(list)
    for r in dev:
        if r["method"] == "patchcore" and r["res"] != 320:
            v = variant_of(r, ref_hp)
            if v:
                variants[v].append(r)
    followup_vs = sorted(v for v, rs in variants.items()
                         if all(len({x["seed"] for x in rs if x["cat"] == c}) >= 3 for c in CATS))

    # --- sweep and resolution: single seed against the seed-0 reference ---------------------
    sweep = []
    for v, rs in variants.items():
        if v in followup_vs:
            continue
        for r in (x for x in rs if x["seed"] == 0):
            b = ref0[r["cat"]]
            sweep.append({"variant": v, "category": r["cat"],
                          "image_auroc": r["image_auroc"], "d_aupro": r["aupro_005"] - b["aupro_005"],
                          "d_fpr": r["fpr_at_op1"] - b["fpr_at_op1"],
                          "d_recall": r["recall_at_op1"] - b["recall_at_op1"],
                          "fit_ratio": r["fit_seconds"] / b["fit_seconds"],
                          "rss_fit_mb": r["peak_rss_fit_mb"],
                          "e2e_ratio": r["e2e_ms_per_image"] / b["e2e_ms_per_image"],
                          "bank_effective": r["fit_extra"].get("bank_effective"),
                          "bank_capped": r["fit_extra"].get("bank_capped"),
                          **paired([r], [b])})
    for row, (adj, sig) in zip(sweep, holm_bonferroni([s["p_boot"] for s in sweep])):
        row.update({"p_holm": adj, "significant_holm": sig})
    sweep.sort(key=lambda s: (s["variant"], CATS.index(s["category"])))
    if sweep:
        write(out / "sweep.csv", sweep)

    # --- Gate E follow-up: seed-averaged, Holm over the family, DR-1 ----------------------
    follow = []
    for v in followup_vs:
        for c in CATS:
            rs = [r for r in variants[v] if r["cat"] == c]
            follow.append({"variant": v, "category": c, "seeds": len(rs),
                           "image_auroc": mean(rs, "image_auroc"), "image_auroc_sd": sd(rs, "image_auroc"),
                           "aupro_005": mean(rs, "aupro_005"), "aupro_005_sd": sd(rs, "aupro_005"),
                           "d_aupro": mean(rs, "aupro_005") - mean(ref[c], "aupro_005"),
                           "d_fpr": mean(rs, "fpr_at_op1") - mean(ref[c], "fpr_at_op1"),
                           "d_recall": mean(rs, "recall_at_op1") - mean(ref[c], "recall_at_op1"),
                           "e2e_ms": mean(rs, "e2e_ms_per_image"),
                           "e2e_ratio": mean(rs, "e2e_ms_per_image") / mean(ref[c], "e2e_ms_per_image"),
                           "vram_predict_mb": max(r["peak_vram_predict_mb"] for r in rs),
                           "rss_fit_mb": max(r["peak_rss_fit_mb"] for r in rs),
                           **paired(rs, ref[c])})
    for row, (adj, sig) in zip(follow, holm_bonferroni([f["p_boot"] for f in follow])):
        row.update({"p_holm": adj, "significant_holm": sig})
    bench_vram: dict[str, float] = {}
    if args.vram_csv:
        with open(args.vram_csv, encoding="utf-8") as fh:
            bench_vram = {row["variant"]: float(row["peak_vram_alloc_mb"])
                          for row in csv.DictReader(fh) if row["device"] == "cuda"}
    decisions = []
    for v in followup_vs:
        rows = [f for f in follow if f["variant"] == v]
        vram = bench_vram.get(v, max(f["vram_predict_mb"] for f in rows))
        decisions.append({"variant": v, "decision": dr1(rows, vram), "inference_vram_mb": vram,
                          "vram_source": "bench, fresh process" if v in bench_vram
                          else "registry, may include earlier runs' backbones"})
    if follow:
        write(out / "followup.csv", follow)
        write(out / "followup_decisions.csv", decisions)

    # --- CAE ------------------------------------------------------------------------------
    cae = []
    for r in sorted((r for r in dev if r["method"].startswith("cae")),
                    key=lambda r: (r["hp"]["loss"], CATS.index(r["cat"]))):
        cae.append({"loss": r["hp"]["loss"], "category": r["cat"], "epochs_run": r["fit_extra"].get("epochs_run"),
                    "epochs_max": r["hp"]["epochs"], "best_val_loss": r["fit_extra"].get("best_val_loss"),
                    "image_auroc": r["image_auroc"], "aupro_005": r["aupro_005"], "pixel_auroc": r["pixel_auroc"],
                    "fpr_at_op1": r["fpr_at_op1"], "recall_at_op1": r["recall_at_op1"],
                    "patchcore_auroc": mean(ref[r["cat"]], "image_auroc"),
                    "patchcore_aupro": mean(ref[r["cat"]], "aupro_005"),
                    "fit_min": r["fit_seconds"] / 60, **paired(ref[r["cat"]], [r])})
    if cae:
        write(out / "cae.csv", cae)

    # --- C2: ours vs anomalib v2.3.0 at 320 px ---------------------------------------------
    arms = {"ours": lambda r: r["method"] == "patchcore" and r["res"] == 320,
            "anomalib_tv": lambda r: r["method"] == "anomalib_patchcore" and r["hp"].get("weights") == "tv_in1k",
            "anomalib_default": lambda r: r["method"] == "anomalib_patchcore" and r["hp"].get("weights") != "tv_in1k"}
    c2_runs = {(a, c): [r for r in dev if f(r) and r["cat"] == c] for a, f in arms.items() for c in CATS}
    c2 = []
    for c in CATS:
        for a in arms:
            rs = c2_runs[(a, c)]
            if rs:
                c2.append({"category": c, "arm": a, "seeds": len(rs),
                           **{f"{k}_mean": mean(rs, k) for k in ("image_auroc", "aupro_005", "pixel_auroc", "fpr_at_op1")},
                           "image_auroc_sd": sd(rs, "image_auroc"), "aupro_005_sd": sd(rs, "aupro_005"),
                           "bank": mean([{"b": r["fit_extra"].get("bank_effective")} for r in rs], "b"),
                           "descriptor_dim": rs[0]["fit_extra"].get("descriptor_dim"),
                           "fit_s": mean(rs, "fit_seconds"), "rss_fit_mb": mean(rs, "peak_rss_fit_mb"),
                           "e2e_ms": mean(rs, "e2e_ms_per_image")})
    c2_pairs = []
    for c in CATS:
        for a, b, what in (("anomalib_tv", "ours", "pipeline"), ("anomalib_default", "anomalib_tv", "weights"),
                           ("anomalib_default", "ours", "total")):
            if c2_runs[(a, c)] and c2_runs[(b, c)]:
                c2_pairs.append({"category": c, "comparison": what, "a": a, "b": b,
                                 **paired(c2_runs[(a, c)], c2_runs[(b, c)])})
    for row, (adj, sig) in zip(c2_pairs, holm_bonferroni([p["p_boot"] for p in c2_pairs])):
        row.update({"p_holm": adj, "significant_holm": sig})
    if c2:
        write(out / "c2.csv", c2)
    if c2_pairs:
        write(out / "c2_pairs.csv", c2_pairs)

    # --- MPDD (ADR-12): reported, not tested -----------------------------------------------
    conf = [r for r in runs if r["role"] == "confirmation"]
    mp = []
    for c in MPDD:
        rec = [r for r in conf if r["cat"] == c and r["method"] == "patchcore"]
        if not rec:
            continue
        floor = {r["method"]: r for r in conf if r["cat"] == c and r["method"] != "patchcore"}
        n_norm = sum(1 for y in scores(rec[0]).labels if y == 0)
        fp = sum(round(r["fpr_at_op1"] * n_norm) for r in rec)
        mp.append({"category": c, "seeds": len(rec), "n_train": rec[0]["n_train"],
                   "n_calibration": rec[0]["calibration_count"], "n_test_normal": n_norm,
                   "n_test_defect": len(scores(rec[0]).labels) - n_norm,
                   "image_auroc_mean": mean(rec, "image_auroc"), "image_auroc_sd": sd(rec, "image_auroc"),
                   "aupro_005_mean": mean(rec, "aupro_005"), "pixel_auroc_mean": mean(rec, "pixel_auroc"),
                   "fpr_mean": mean(rec, "fpr_at_op1"), "effective_fpr": rec[0]["effective_fpr"],
                   "target_met": rec[0]["target_met"], "fp_total": fp,
                   "fp_expected": len(rec) * n_norm * rec[0]["effective_fpr"],
                   "recall_mean": mean(rec, "recall_at_op1"),
                   "pixel_pca_auroc": floor.get("pixel_pca", {}).get("image_auroc"),
                   "histogram_auroc": floor.get("histogram", {}).get("image_auroc"),
                   **({f"vs_pca_{k}": v for k, v in paired(rec, [floor["pixel_pca"]]).items()}
                      if "pixel_pca" in floor else {})})
    if mp:
        write(out / "mpdd.csv", mp)

    summary = {"implementation": args.implementation, "runs": len(runs), "reference_hparams": ref_hp,
               "sweep_comparisons": len(sweep), "followup_comparisons": len(follow),
               "followup_decisions": {d["variant"]: d["decision"] for d in decisions},
               "c2_comparisons": len(c2_pairs)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
