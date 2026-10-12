"""R2 (docs/14): location-conditional map normalization, evaluated on stored maps (no refit).

For a category whose images share one geometry, the anomaly map of a normal image has structured
hot spots — edges, glare, texture seams — that set the image maximum as high as a defect does.
Standardizing each map location by the mean and spread of that location over the validation
normals removes them:

    z(x) = (m(x) - mu(x)) / max(sigma(x), floor),   image score = max_x z(x)

mu and sigma come from the validation normals; each validation image's own score is computed
leave-one-out, so the threshold set on them (the rank rule, k = 1 at these sizes) is not
optimistic. sigma is floored at the median sigma, so near-constant locations cannot explode.

Exploratory development study on MVTec AD 2 `test_public` (E14e), added after the E14 plan.

    python scripts/location_normalization_study.py --implementation af89b5297f4d --out reports/studies/e14
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from inspector.evaluate import SplitPredictions
from inspector.metrics.ranking import roc_auc


def zmax(maps: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((maps - mu) / sigma).reshape(len(maps), -1).max(axis=1)


def study(run_dir: Path, split: str) -> dict:
    val = SplitPredictions.load(run_dir, "validation")
    test = SplitPredictions.load(run_dir, split)
    v = np.stack([np.asarray(m, dtype=np.float64) for m in val.raw_maps])
    shapes = {np.asarray(m).shape for m in test.raw_maps} | {v.shape[1:]}
    if len(shapes) != 1:
        raise ValueError(f"maps differ in shape: {shapes}")
    t = np.stack([np.asarray(m, dtype=np.float64) for m in test.raw_maps])
    n = len(v)
    mu, sd = v.mean(axis=0), v.std(axis=0)
    floor = float(np.median(sd))
    test_scores = zmax(t, mu, np.maximum(sd, floor))
    loo = np.empty(n)
    for i in range(n):  # leave-one-out statistics for each validation image's own score
        rest = np.delete(v, i, axis=0)
        loo[i] = zmax(v[i:i + 1], rest.mean(axis=0), np.maximum(rest.std(axis=0), floor))[0]
    threshold = float(loo.max())  # k = 1: the rank rule at n < 99
    y = test.labels
    stored_thr = float(np.sort(val.scores)[-1])
    return {
        "auroc_stored": roc_auc(y, test.scores), "auroc_location_z": roc_auc(y, test_scores),
        "fpr_stored": float(np.mean(test.scores[y == 0] > stored_thr)),
        "fpr_location_z": float(np.mean(test_scores[y == 0] > threshold)),
        "recall_stored": float(np.mean(test.scores[y == 1] > stored_thr)),
        "recall_location_z": float(np.mean(test_scores[y == 1] > threshold)),
        "n_validation": n, "bound": 1.0 / (n + 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--implementation", required=True)
    ap.add_argument("--role", default="confirmation")
    ap.add_argument("--out", default="reports/studies/e14")
    args = ap.parse_args()
    rows = []
    for f in sorted(Path(args.registry).glob("*/spec.json")):
        spec = json.loads(f.read_text(encoding="utf-8"))
        if (spec["dataset"].get("layout") != "mvtec_ad2" or spec["model"]["name"] != "patchcore"
                or spec.get("role", "development") != args.role or spec["seed"] != 0
                or not spec["implementation_id"].startswith(args.implementation)):
            continue
        recipe = f"{spec['transform']['long_side']}px,{'+'.join(spec['model']['hparams']['layers'])}"
        res = study(f.parent, spec["dataset"]["test_split"])
        rows.append({"category": spec["dataset"]["category"], "recipe": recipe, **res})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    rows.sort(key=lambda r: (r["recipe"], r["category"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "location_normalization.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
