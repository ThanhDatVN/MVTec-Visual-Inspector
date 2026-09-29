"""E03b: does a larger calibration pool repair the conformal threshold? (development only)

On `pcb1` the rank threshold from 136 validation normals admits 4-6% false alarms
against a 0.73% bound, because a few test normals score above every validation
normal. With n = 136 a 1% request can only use the single largest score (k = 1).

Cross-fitting enlarges the pool without touching test data: split the training
normals into K folds, fit on K-1, score the held-out fold, and pool the
out-of-fold scores (~770 on `pcb1`), so a 1% threshold uses k = 7. The
out-of-fold models see 80% of the training data, so their scores run slightly
high against the full model's — a conservative direction for a threshold.

Writes one JSON per category with the out-of-fold scores; the analysis compares
the realized test FPR of the full-model run (from the registry) at thresholds
from (a) validation only, (b) out-of-fold only, (c) both pooled.

    python scripts/crossfit_calibration.py --data-root <VisA> --categories pcb1 macaroni2 capsules \
        --reference-runs <run ids of the full-model runs> --out reports/studies/e03b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inspector.analysis import exceedance_p_value
from inspector.config import deep_merge, load_config
from inspector.data.core import DatasetIndex
from inspector.evaluate import SplitPredictions
from inspector.postproc.thresholds import conformal_rank
from inspector.runner import prepare

RECIPE = "configs/recipes/confirmation-v1-patchcore-640.yaml"


def folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    return [np.sort(part) for part in np.array_split(order, k)]


def subset(index: DatasetIndex, rows: np.ndarray) -> DatasetIndex:
    samples = list(index)
    return DatasetIndex([samples[i] for i in rows], root=index.root, category=index.category,
                        split_name=index.split_name, layout=index.layout)


def out_of_fold_scores(cfg: dict, method: str, category: str, data_root: str, k: int, seed: int) -> dict:
    _, _, indices = prepare(cfg, method=method, category=category, seed=seed, data_root=data_root)
    train = indices["train"]
    ids, scores = [], []
    for f, held_out in enumerate(folds(len(train), k, seed)):
        keep = np.setdiff1d(np.arange(len(train)), held_out)
        _, model, _ = prepare(cfg, method=method, category=category, seed=seed, data_root=data_root)
        model.fit(subset(train, keep))
        preds = SplitPredictions.from_model(model, subset(train, held_out))
        ids += preds.ids
        scores += preds.scores.tolist()
        print(f"  {category} fold {f + 1}/{k}: fit on {keep.size}, scored {held_out.size}")
    return {"category": category, "k_folds": k, "seed": seed, "ids": ids, "scores": scores}


def threshold_row(name: str, pool: np.ndarray, test: SplitPredictions, target: float) -> dict:
    n = pool.size
    k = conformal_rank(n, target) or 1
    thr = float(np.sort(pool)[::-1][k - 1])
    normals = test.labels == 0
    fp = int(np.sum(test.scores[normals] > thr))
    return {
        "source": name, "n": int(n), "k": k, "bound": k / (n + 1), "threshold": thr,
        "fp": fp, "test_normals": int(normals.sum()), "fpr": fp / int(normals.sum()),
        "recall": float(np.mean(test.scores[~normals] > thr)),
        "tail_p": exceedance_p_value(fp, int(normals.sum()), int(n), k),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--reference-runs", nargs="+", required=True,
                    help="registry run ids of the full-model runs, one per category, same order")
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--out", default="reports/studies/e03b")
    ap.add_argument("--data-config", default="configs/data/visa_pcb1.yaml")
    ap.add_argument("--recipe", default=RECIPE, help="model config fragment; 'none' for defaults")
    ap.add_argument("--method", default="patchcore")
    args = ap.parse_args()
    cfg = load_config(args.data_config)
    if args.recipe != "none":
        cfg = deep_merge(cfg, load_config(args.recipe))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for category, run_id in zip(args.categories, args.reference_runs):
        cache = out / f"{category}.oof.json"
        if cache.exists():
            oof = json.loads(cache.read_text(encoding="utf-8"))
        else:
            oof = out_of_fold_scores(cfg, args.method, category, args.data_root, args.folds, args.seed)
            cache.write_text(json.dumps(oof), encoding="utf-8")
        run_dir = Path(args.registry) / run_id
        val = SplitPredictions.load(run_dir, "validation")
        spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
        test = SplitPredictions.load(run_dir, spec["dataset"]["test_split"])
        oof_scores = np.asarray(oof["scores"])
        for name, pool in (("validation", val.scores), ("out-of-fold train", oof_scores),
                           ("pooled", np.concatenate([val.scores, oof_scores]))):
            rows.append({"category": category, "reference_run": run_id,
                         **threshold_row(name, pool, test, args.target)})

    (out / "e03b.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["| category | calibration pool | n | k | bound | FP / test normals | realized FPR | tail p | recall |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['category']}` | {r['source']} | {r['n']} | {r['k']} | {r['bound']:.2%} | "
                     f"{r['fp']} / {r['test_normals']} | {r['fpr']:.1%} | {r['tail_p']:.2g} | {r['recall']:.2f} |")
    (out / "e03b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
