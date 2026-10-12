"""E16 (docs/11 Stage H⁗, E16): few-shot adaptation of recipe v1 to changed lighting on MVTec AD 2.

What a line does when its lighting changes: photograph a few good parts under the new lighting
and add them to the memory bank. Here, with `test_public`'s captures (docs/14 §0):

* the objects of each category are split into two folds by a seeded permutation, separately for
  normal and anomalous objects;
* for fold f, recipe v1 (seed 0) is fitted on `train/good` plus every capture of `n` normal
  objects of the *other* fold (n = 1, and n = all of that fold), and evaluated on fold f's images,
  so no evaluated object was seen and every test image is scored once over the two folds;
* the operating point stays calibrated on `validation/good` (training lighting only): the
  question is whether the adapted bank brings changed-lighting normals back under it.

The reference is E15-ref (v1, same implementation, no adaptation) on the same images. Reading,
written with the plan: adaptation with n objects works if the pooled false alarms among `shift`
normals fall by at least half with no category significantly worse in image AUROC (object
bootstrap, Holm over 8). A measurement for deployment guidance, not a recipe change.

    python scripts/e16_adaptation.py --data-root data/raw/mvtec_ad2 --out reports/studies/e16
    python scripts/e16_adaptation.py --data-root data/raw/mvtec_ad2 --out reports/studies/e16 --analyze-only
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from ad2_condition_study import CONDITIONS, NAME, cluster_paired
from global_branch_study import reference_runs

from inspector.config import deep_merge, load_config
from inspector.data import discover_category
from inspector.data.core import DatasetIndex
from inspector.evaluate import SplitPredictions
from inspector.metrics.ranking import roc_auc
from inspector.models import build_model
from inspector.postproc.thresholds import from_validation_fpr
from inspector.runner import implementation_id, model_kwargs, transform_from_config
from inspector.stats import holm_bonferroni
from inspector.utils.env import git_info

CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
RECIPE = ("configs/data/mvtec_ad2.yaml", "configs/recipes/confirmation-v1-patchcore-640.yaml")
REFERENCE_IMPLEMENTATION = "67985a22598d"  # E15-ref
SEED = 0


def object_of(sample) -> str:
    return NAME.match(sample.image_path.name).group(1)


def condition_of(rel_id: str) -> str:
    c = NAME.match(rel_id.split("/")[-1]).group(2)
    return "shift" if c.startswith("shift_") else c


def folds(test: DatasetIndex) -> dict[tuple[int, int], list[str]]:
    """(label, fold) -> objects, from a seeded permutation of each class's objects."""
    out: dict[tuple[int, int], list[str]] = {}
    for label in (0, 1):
        objects = sorted({object_of(s) for s in test if s.label == label})
        order = [objects[i] for i in np.random.default_rng(SEED).permutation(len(objects))]
        for f in (0, 1):
            out[(label, f)] = order[f::2]
    return out


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = deep_merge(load_config(RECIPE[0]), load_config(RECIPE[1]))
    transform = transform_from_config(cfg)
    impl = implementation_id()
    if not impl.startswith(REFERENCE_IMPLEMENTATION):
        raise SystemExit(f"implementation {impl[:12]} differs from E15-ref's {REFERENCE_IMPLEMENTATION}")
    for category in args.categories:
        idx = discover_category(args.data_root, category, layout="mvtec_ad2",
                                splits=("train", "validation", "test_public"))
        test = idx["test_public"]
        parts = folds(test)
        for f in (0, 1):
            other = parts[(0, 1 - f)]
            levels = sorted({1, len(other)} if not args.levels else set(args.levels))
            for n in levels:
                path = out / f"{category}_fold{f}_n{n}.json"
                if path.exists():
                    continue
                adapt_objects = other[:n]
                adapt = [s for s in test if s.label == 0 and object_of(s) in adapt_objects]
                fit_index = DatasetIndex([*idx["train"], *adapt], root=test.root, category=category,
                                         split_name="train_adapted", layout=test.layout)
                keep = set(parts[(0, f)]) | {f"a{o}" for o in parts[(1, f)]}
                evaluate = [s for s in test if (object_of(s) if s.label == 0 else f"a{object_of(s)}") in keep]
                eval_index = DatasetIndex(evaluate, root=test.root, category=category,
                                          split_name="test_public", layout=test.layout)
                model = build_model("patchcore", transform, seed=SEED, **model_kwargs(cfg, "patchcore"))
                t0 = time.perf_counter()
                model.fit(fit_index, allow_splits=("train_adapted",))
                fit_s = time.perf_counter() - t0
                val = SplitPredictions.from_model(model, idx["validation"])
                ev = SplitPredictions.from_model(model, eval_index)
                record = {
                    "category": category, "fold": f, "n_objects": n, "adapt_objects": adapt_objects,
                    "n_adapt_images": len(adapt), "eval_objects_normal": parts[(0, f)],
                    "eval_objects_anomalous": parts[(1, f)], "fit_seconds": fit_s,
                    "implementation_id": impl, **git_info(),
                    "validation_scores": val.scores.tolist(),
                    "test": [{"id": i, "label": int(y), "score": float(s)}
                             for i, y, s in zip(ev.ids, ev.labels, ev.scores)],
                }
                path.write_text(json.dumps(record), encoding="utf-8")
                print(f"{category} fold {f} n={n}: {len(adapt)} adaptation images, fit {fit_s:.0f} s", flush=True)
                del model


def analyze(args) -> None:
    out = Path(args.out)
    refs = reference_runs(Path(args.registry), REFERENCE_IMPLEMENTATION, "development")
    records = defaultdict(list)
    for p in sorted(out.glob("*_fold*_n*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        records[(r["category"], r["n_objects"] if r["n_objects"] == 1 else "all")].append(r)
    rows = []
    for level in (1, "all"):
        block = []
        for ci, category in enumerate(CATEGORIES):
            got = records.get((category, level), [])
            if len(got) != 2 or category not in refs:
                continue
            ref_val, ref_test = refs[category][SEED]
            ref_thr = from_validation_fpr(ref_val, target_fpr=0.01, policy="relax").value
            adapted, thr_of = {}, {}
            for r in got:
                thr = from_validation_fpr(np.array(r["validation_scores"]), target_fpr=0.01, policy="relax").value
                for t in r["test"]:
                    adapted[t["id"]] = t
                    thr_of[t["id"]] = thr
            if set(adapted) != {t["id"] for t in ref_test}:
                raise ValueError(f"{category}: the folds do not partition the test set")
            ordered = [adapted[t["id"]] for t in ref_test]
            row = {"level": level, "category": category,
                   "adapt_images": sum(r["n_adapt_images"] for r in got) / 2}
            for name, recs, thr in (("ref", ref_test, lambda i, t=ref_thr: t), ("adapted", ordered, thr_of.get)):
                y = np.array([t["label"] for t in recs])
                s = np.array([t["score"] for t in recs])
                cond = np.array([condition_of(t["id"]) for t in recs])
                alarms = np.array([t["score"] > thr(t["id"]) for t in recs])
                row[f"fa_shift_{name}"] = int(np.sum(alarms & (y == 0) & (cond == "shift")))
                row[f"fa_all_{name}"] = int(np.sum(alarms & (y == 0)))
                row[f"recall_{name}"] = float(np.mean(alarms[y == 1]))
                for c in ("all", *CONDITIONS):
                    keep = np.ones(y.size, bool) if c == "all" else cond == c
                    row[f"auroc_{c}_{name}"] = roc_auc(y[keep], s[keep])
            row["normals_shift"] = int(sum(t["label"] == 0 and condition_of(t["id"]) == "shift" for t in ref_test))
            row.update(cluster_paired({SEED: (None, ordered)}, {SEED: (ref_val, ref_test)}, seed=2000 + ci))
            block.append(row)
        for row, (adj, sig) in zip(block, holm_bonferroni([r["p_object"] for r in block])):
            row["p_holm_object"], row["significant_object"] = adj, sig
        rows += block
    if not rows:
        print("no complete category yet")
        return
    with open(out / "adaptation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    decisions = []
    for level in (1, "all"):
        block = [r for r in rows if r["level"] == level]
        fa_ref, fa_ad = sum(r["fa_shift_ref"] for r in block), sum(r["fa_shift_adapted"] for r in block)
        losses = sum(r["significant_object"] and r["estimate"] < 0 for r in block)
        decisions.append({
            "level": level, "categories": len(block),
            "shift_false_alarms_ref": fa_ref, "shift_false_alarms_adapted": fa_ad,
            "shift_normals": sum(r["normals_shift"] for r in block),
            "all_false_alarms_ref": sum(r["fa_all_ref"] for r in block),
            "all_false_alarms_adapted": sum(r["fa_all_adapted"] for r in block),
            "significant_gains": sum(r["significant_object"] and r["estimate"] > 0 for r in block),
            "significant_losses": losses,
            **{f"macro_auroc_{c}_{k}": float(np.mean([r[f"auroc_{c}_{k}"] for r in block]))
               for c in ("all", "regular", "shift") for k in ("ref", "adapted")},
            "macro_recall_ref": float(np.mean([r["recall_ref"] for r in block])),
            "macro_recall_adapted": float(np.mean([r["recall_adapted"] for r in block])),
            "works": bool(len(block) == len(CATEGORIES) and fa_ad <= 0.5 * fa_ref and losses == 0),
        })
    (out / "summary.json").write_text(json.dumps(decisions, indent=1), encoding="utf-8")
    print(json.dumps(decisions, indent=1))
    for r in rows:
        print(f"n={r['level']!s:<4}{r['category']:<12} AUROC {r['auroc_all_ref']:.3f} -> {r['auroc_all_adapted']:.3f} "
              f"(shift {r['auroc_shift_ref']:.3f} -> {r['auroc_shift_adapted']:.3f}; object "
              f"[{r['ci_low_object']:+.3f}, {r['ci_high_object']:+.3f}]{' *' if r['significant_object'] else ''})  "
              f"shift FA {r['fa_shift_ref']} -> {r['fa_shift_adapted']} / {r['normals_shift']}  "
              f"recall {r['recall_ref']:.2f} -> {r['recall_adapted']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    ap.add_argument("--levels", nargs="+", type=int, default=None, help="override the n levels (smoke tests)")
    ap.add_argument("--out", default="reports/studies/e16")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        run(args)
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
