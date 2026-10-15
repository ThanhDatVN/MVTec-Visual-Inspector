"""E17 (docs/11 Stage J): recalibration under a new lighting, with and without adapting the bank.

E16 added new-lighting normals to the bank but kept the threshold from training-lighting
validation normals, which no longer bounds the false-alarm rate under the new lighting (the
calibration and test normals are not exchangeable). The fair comparison is at a *valid*
operating point: calibrate both the unadapted and the adapted model on normals under the new
lighting that neither bank saw.

Same folds as E16 (`ad2_e16_fewshot_adaptation.folds`). For fold f, the other fold's normal objects O are
split in order: the first ceil(|O|/2) go into the bank (adapted arm only), the rest are
calibration objects whose every capture joins `validation/good` in the calibration set. Arms,
evaluated on fold f's images:

    R      reference (E15-ref), threshold from validation           — E16's reference
    R+cal  reference, threshold from validation + calibration objects
    A      adapted bank, threshold from validation
    A+cal  adapted bank, threshold from validation + calibration objects

All thresholds are the conservative rank rule at a 1% request (relaxed to the maximum at these
sizes). Reading, written with the plan: recalibration restores the bound if R+cal's pooled false
alarms are no more than its expected count (sum of bound x normals) plus two Poisson SDs;
adaptation pays at a valid operating point if A+cal's macro recall exceeds R+cal's (object
bootstrap 95% CI above zero) with pooled false alarms no higher.

    python scripts/ad2_e17_recalibration.py --data-root data/raw/mvtec_ad2 --out reports/studies/e17
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from ad2_e16_fewshot_adaptation import (
    CATEGORIES,
    RECIPE,
    REFERENCE_IMPLEMENTATION,
    SEED,
    condition_of,
    folds,
    object_of,
)
from ad2_r4_global_branch import reference_runs

from inspector.config import deep_merge, load_config
from inspector.data import discover_category
from inspector.data.core import DatasetIndex
from inspector.evaluate import SplitPredictions
from inspector.models import build_model
from inspector.postproc.thresholds import from_validation_fpr
from inspector.runner import implementation_id, model_kwargs, transform_from_config
from inspector.utils.env import git_info

ARMS = ("R", "R+cal", "A", "A+cal")
N_BOOT = 10_000


def subset(test: DatasetIndex, samples: list, name: str) -> DatasetIndex:
    return DatasetIndex(samples, root=test.root, category=test.category, split_name=name, layout=test.layout)


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = deep_merge(load_config(RECIPE[0]), load_config(RECIPE[1]))
    transform = transform_from_config(cfg)
    impl = implementation_id()
    if not impl.startswith(REFERENCE_IMPLEMENTATION):
        raise SystemExit(f"implementation {impl[:12]} differs from E15-ref's {REFERENCE_IMPLEMENTATION}")
    for category in args.categories:
        path_done = [out / f"{category}_fold{f}.json" for f in (0, 1)]
        if all(p.exists() for p in path_done):
            continue
        idx = discover_category(args.data_root, category, layout="mvtec_ad2",
                                splits=("train", "validation", "test_public"))
        test = idx["test_public"]
        parts = folds(test)
        for f in (0, 1):
            path = out / f"{category}_fold{f}.json"
            if path.exists():
                continue
            other = parts[(0, 1 - f)]
            cut = math.ceil(len(other) / 2)
            bank_objects, cal_objects = other[:cut], other[cut:]
            bank = [s for s in test if s.label == 0 and object_of(s) in bank_objects]
            cal = [s for s in test if s.label == 0 and object_of(s) in cal_objects]
            keep = set(parts[(0, f)]) | {f"a{o}" for o in parts[(1, f)]}
            evaluate = [s for s in test if (object_of(s) if s.label == 0 else f"a{object_of(s)}") in keep]

            model = build_model("patchcore", transform, seed=SEED, **model_kwargs(cfg, "patchcore"))
            t0 = time.perf_counter()
            model.fit(subset(test, [*idx["train"], *bank], "train_adapted"), allow_splits=("train_adapted",))
            fit_s = time.perf_counter() - t0
            val = SplitPredictions.from_model(model, idx["validation"])
            cal_pred = SplitPredictions.from_model(model, subset(test, cal, "test_public"))
            ev = SplitPredictions.from_model(model, subset(test, evaluate, "test_public"))
            record = {
                "category": category, "fold": f, "bank_objects": bank_objects, "calibration_objects": cal_objects,
                "n_bank_images": len(bank), "n_calibration_images": len(cal), "fit_seconds": fit_s,
                "implementation_id": impl, **git_info(),
                "validation_scores": val.scores.tolist(),
                "calibration": [{"id": i, "score": float(s)} for i, s in zip(cal_pred.ids, cal_pred.scores)],
                "test": [{"id": i, "label": int(y), "score": float(s)} for i, y, s in zip(ev.ids, ev.labels, ev.scores)],
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            print(f"{category} fold {f}: bank +{len(bank)}, calibration +{len(cal)} images, fit {fit_s:.0f} s", flush=True)
            del model


def threshold(scores) -> tuple[float, float]:
    t = from_validation_fpr(np.asarray(scores, dtype=np.float64), target_fpr=0.01, policy="relax")
    return t.value, float(t.params["effective_fpr"])


def analyze(args) -> None:
    out = Path(args.out)
    refs = reference_runs(Path(args.registry), REFERENCE_IMPLEMENTATION, "development")
    per_cat = {}
    for category in CATEGORIES:
        recs = [out / f"{category}_fold{f}.json" for f in (0, 1)]
        if not all(p.exists() for p in recs) or category not in refs:
            continue
        ref_val, ref_test = refs[category][SEED]
        ref_score = {t["id"]: t["score"] for t in ref_test}
        images = []  # one row per evaluated image: id, label, condition, anomalous object, alarm per arm
        expected = dict.fromkeys(ARMS, 0.0)
        for p in recs:
            r = json.loads(p.read_text(encoding="utf-8"))
            cal_ids = [c["id"] for c in r["calibration"]]
            thr = {
                "R": threshold(ref_val),
                "R+cal": threshold([*ref_val, *(ref_score[i] for i in cal_ids)]),
                "A": threshold(r["validation_scores"]),
                "A+cal": threshold([*r["validation_scores"], *(c["score"] for c in r["calibration"])]),
            }
            for t in r["test"]:
                scores = {"R": ref_score[t["id"]], "R+cal": ref_score[t["id"]], "A": t["score"], "A+cal": t["score"]}
                images.append({"id": t["id"], "label": t["label"], "condition": condition_of(t["id"]),
                               "object": t["id"].split("/")[-1].split("_")[0],
                               **{a: scores[a] > thr[a][0] for a in ARMS}})
                if t["label"] == 0:
                    for a in ARMS:
                        expected[a] += thr[a][1]
        per_cat[category] = {"images": images, "expected": expected}

    rows = []
    for category, d in per_cat.items():
        imgs = d["images"]
        normals = [i for i in imgs if i["label"] == 0]
        anomalies = [i for i in imgs if i["label"] == 1]
        row = {"category": category, "normals": len(normals), "anomalies": len(anomalies),
               "normals_shift": sum(i["condition"] == "shift" for i in normals)}
        for a in ARMS:
            row[f"fa_{a}"] = sum(i[a] for i in normals)
            row[f"fa_shift_{a}"] = sum(i[a] for i in normals if i["condition"] == "shift")
            row[f"expected_{a}"] = d["expected"][a]
            row[f"recall_{a}"] = float(np.mean([i[a] for i in anomalies]))
        rows.append(row)

    # object bootstrap of the macro recall difference A+cal - R+cal (anomalous objects resampled per category)
    rng = np.random.default_rng(17)
    groups = {}
    for category, d in per_cat.items():
        by_obj = {}
        for i in d["images"]:
            if i["label"] == 1:
                by_obj.setdefault(i["object"], []).append(i)
        groups[category] = [np.array([[x["A+cal"], x["R+cal"]] for x in v], dtype=float) for v in by_obj.values()]
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        per = []
        for objs in groups.values():
            pick = np.concatenate([objs[j] for j in rng.integers(0, len(objs), len(objs))])
            per.append(pick[:, 0].mean() - pick[:, 1].mean())
        diffs[b] = np.mean(per)

    summary = {"categories": len(rows)}
    for a in ARMS:
        fa = sum(r[f"fa_{a}"] for r in rows)
        exp = sum(r[f"expected_{a}"] for r in rows)
        summary[a] = {"false_alarms": fa, "expected": exp, "bound_held": bool(fa <= exp + 2 * math.sqrt(exp)),
                      "shift_false_alarms": sum(r[f"fa_shift_{a}"] for r in rows),
                      "macro_recall": float(np.mean([r[f"recall_{a}"] for r in rows]))}
    summary["recall_A+cal_minus_R+cal"] = {
        "estimate": summary["A+cal"]["macro_recall"] - summary["R+cal"]["macro_recall"],
        "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975))}
    summary["recalibration_restores_bound"] = summary["R+cal"]["bound_held"]
    summary["adaptation_pays_at_valid_op"] = bool(
        summary["recall_A+cal_minus_R+cal"]["ci_low"] > 0
        and summary["A+cal"]["false_alarms"] <= summary["R+cal"]["false_alarms"])
    with open(out / "recalibration.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(f"{r['category']:<12} " + "  ".join(
            f"{a}: FA {r[f'fa_{a}']}/{r['normals']} (exp {r[f'expected_{a}']:.1f}) rec {r[f'recall_{a}']:.2f}" for a in ARMS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    ap.add_argument("--out", default="reports/studies/e17")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        run(args)
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
