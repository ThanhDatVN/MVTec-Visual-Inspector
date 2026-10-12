"""MVTec AD 2 `test_public` by capture condition, and the E13 comparison with object clusters.

`test_public` shows every object several times: `regular` (the lighting of train and validation,
which hold nothing else) and changed captures, `overexposed`, `underexposed` and `shift_1`..`shift_k`
(file `<object>_<condition>.png`; k = 1 on `fruit_jelly`, 4 on `vial`, 3 elsewhere). The shifts are
pooled into one `shift` family here. Two consequences, both checked here on stored predictions:

1. Five sixths of the test normals are a shifted population by design. Per condition: image
   AUROC (that condition's anomalies against its normals), false alarms of the validation
   operating point among its normals, recall among its anomalies.
2. The six captures of one object are not independent samples. E13's image bootstrap resampled
   images; here the same paired comparison resamples *objects* (all captures of an object
   together, class-stratified), which is the sampling unit. Development analysis, written after
   E13; it does not change E13's frozen verdict, it measures how much the image bootstrap
   understated the uncertainty.
3. With `--guards DATA_ROOT`: which captures the input guards (`inspector.api.guards`, fitted on
   the training normals) refuse — the guards watch global exposure and focus.

    python scripts/ad2_condition_study.py --implementation af89b5297f4d --role confirmation \
        --compare 960px,layer2 640px,layer2+layer3 --out reports/studies/ad2-conditions
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from inspector.metrics.ranking import roc_auc
from inspector.postproc.thresholds import from_validation_fpr
from inspector.stats import holm_bonferroni

CONDITIONS = ("regular", "overexposed", "underexposed", "shift")
NAME = re.compile(r"^(\d+)_(.+)\.png$")
N_RESAMPLES = 10_000


def recipe_of(spec: dict) -> str:
    hp = spec["model"]["hparams"]
    parts = [f"{spec['transform']['long_side']}px", "+".join(hp.get("layers", []))]
    if spec["transform"].get("normalize", "imagenet") != "imagenet":
        parts.append(f"normalize={spec['transform']['normalize']}")
    if hp.get("bank_exposures"):
        parts.append(f"bank_exposures={hp['bank_exposures']}")
    return ",".join(parts)


def read_split(run_dir: Path, split: str) -> list[dict]:
    return json.loads((run_dir / f"{split}.predictions.json").read_text(encoding="utf-8"))


def load(registry: Path, implementations: list[str], role: str) -> dict:
    """(category, recipe) -> {seed: (val_scores, test_records)}."""
    runs: dict[tuple[str, str], dict[int, tuple]] = defaultdict(dict)
    for f in sorted(registry.glob("*/result.json")):
        if json.loads(f.read_text(encoding="utf-8")).get("status") != "completed":
            continue
        spec = json.loads(f.with_name("spec.json").read_text(encoding="utf-8"))
        if (spec["dataset"].get("layout") != "mvtec_ad2" or spec["model"]["name"] != "patchcore"
                or spec.get("role", "development") != role
                or not any(spec["implementation_id"].startswith(i) for i in implementations)):
            continue
        val = np.array([r["score"] for r in read_split(f.parent, "validation")])
        test = read_split(f.parent, spec["dataset"]["test_split"])
        runs[(spec["dataset"]["category"], recipe_of(spec))][int(spec["seed"])] = (val, test)
    return runs


def parse(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    labels, conditions, clusters, ids = [], [], [], []
    for r in records:
        name = r["id"].split("/")[-1]
        m = NAME.match(name)
        if not m:
            raise ValueError(f"unexpected AD 2 file name {name!r}")
        labels.append(r["label"])
        conditions.append("shift" if m.group(2).startswith("shift_") else m.group(2))
        clusters.append(f"{r['label']}:{m.group(1)}")
        ids.append(r["id"])
    return np.array(labels), np.array(conditions), np.array(clusters), ids


def by_condition(seeds: dict[int, tuple]) -> list[dict]:
    rows = []
    per_seed = []
    for _, (val, test) in sorted(seeds.items()):
        y, cond, _, _ = parse(test)
        s = np.array([r["score"] for r in test])
        thr = from_validation_fpr(val, target_fpr=0.01, policy="relax").value
        per_seed.append((y, cond, s, thr))
    for c in ("all", *CONDITIONS):
        auroc, fa, rec, n0, n1 = [], [], [], 0, 0
        for y, cond, s, thr in per_seed:
            keep = np.ones(y.size, bool) if c == "all" else cond == c
            yc, sc = y[keep], s[keep]
            n0, n1 = int(np.sum(yc == 0)), int(np.sum(yc == 1))
            auroc.append(roc_auc(yc, sc))
            fa.append(int(np.sum(sc[yc == 0] > thr)))
            rec.append(float(np.mean(sc[yc == 1] > thr)))
        rows.append({"condition": c, "normals": n0, "anomalies": n1,
                     "auroc": float(np.mean(auroc)), "auroc_sd": float(np.std(auroc, ddof=1)) if len(auroc) > 1 else 0.0,
                     "false_alarms_mean": float(np.mean(fa)), "fpr": float(np.mean(fa)) / max(n0, 1),
                     "recall": float(np.mean(rec)), "seeds": len(per_seed)})
    return rows


def cluster_paired(a: dict[int, tuple], b: dict[int, tuple], seed: int) -> dict:
    """Seed-averaged AUROC(a) - AUROC(b); bootstrap over objects and over images, class-stratified."""
    seeds = sorted(set(a) & set(b))
    y, _, clusters, ids = parse(a[seeds[0]][1])
    for s in seeds:
        if [r["id"] for r in a[s][1]] != ids or [r["id"] for r in b[s][1]] != ids:
            raise ValueError("arms were not scored on the same images")
    sa = [np.array([r["score"] for r in a[s][1]]) for s in seeds]
    sb = [np.array([r["score"] for r in b[s][1]]) for s in seeds]

    def diff(idx: np.ndarray) -> float:
        yy = y[idx]
        return float(np.mean([roc_auc(yy, x[idx]) for x in sa]) - np.mean([roc_auc(yy, x[idx]) for x in sb]))

    members = defaultdict(list)
    for i, c in enumerate(clusters):
        members[c].append(i)
    groups = {lab: [np.array(members[c]) for c in sorted(members) if c.startswith(f"{lab}:")] for lab in (0, 1)}
    rng = np.random.default_rng(seed)
    out = {"estimate": diff(np.arange(y.size)), "objects_normal": len(groups[0]),
           "objects_anomalous": len(groups[1]), "seeds": len(seeds)}
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for unit in ("image", "object"):
        draws = np.empty(N_RESAMPLES)
        for i in range(N_RESAMPLES):
            if unit == "image":
                idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
            else:
                idx = np.concatenate([groups[lab][j] for lab in (1, 0)
                                      for j in rng.integers(0, len(groups[lab]), len(groups[lab]))])
            draws[i] = diff(idx)
        tail = min(int(np.sum(draws <= 0)), int(np.sum(draws >= 0)))
        out[f"ci_low_{unit}"] = float(np.quantile(draws, 0.025))
        out[f"ci_high_{unit}"] = float(np.quantile(draws, 0.975))
        out[f"p_{unit}"] = float(min(1.0, 2 * (tail + 1) / (N_RESAMPLES + 1)))
    return out


def guards_by_condition(data_root: Path, categories: list[str]) -> list[dict]:
    """Share of test normals and anomalies each capture condition loses to the input guards."""
    from inspector.api.guards import InputGuards
    from inspector.data.transforms import load_image

    rows = []
    for category in categories:
        guards = InputGuards.fit(load_image(p) for p in sorted((data_root / category / "train" / "good").glob("*.png")))
        refused: dict[tuple[str, int], list[bool]] = defaultdict(list)
        for label, folder in ((0, "good"), (1, "bad")):
            for path in sorted((data_root / category / "test_public" / folder).glob("*.png")):
                m = NAME.match(path.name)
                cond = "shift" if m.group(2).startswith("shift_") else m.group(2)
                refused[(cond, label)].append(not guards.check(load_image(path))["ok"])
        for cond in CONDITIONS:
            rows.append({"category": category, "condition": cond,
                         "refused_normals": float(np.mean(refused[(cond, 0)])),
                         "refused_anomalies": float(np.mean(refused[(cond, 1)])),
                         "normals": len(refused[(cond, 0)]), "n_train": guards.n_reference})
        print(category, " ".join(f"{r['condition']}={r['refused_normals']:.0%}" for r in rows[-len(CONDITIONS):]),
              flush=True)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--implementation", nargs="+", required=True)
    ap.add_argument("--role", default="confirmation")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="recipes for the paired A - B comparison")
    ap.add_argument("--guards", metavar="DATA_ROOT", help="also report guard refusals by condition")
    ap.add_argument("--out", default="reports/studies/ad2-conditions")
    args = ap.parse_args()
    runs = load(Path(args.registry), args.implementation, args.role)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for (category, recipe), seeds in sorted(runs.items()):
        for r in by_condition(seeds):
            rows.append({"category": category, "recipe": recipe, **r})
    write_csv(out / "conditions.csv", rows)
    recipes = sorted({r["recipe"] for r in rows})
    for recipe in recipes:
        print(f"\n{recipe}: AUROC / false alarms per normal, by condition (mean over seeds)")
        print(f"{'category':<12}" + "".join(f"{c:>20}" for c in ("all", *CONDITIONS)))
        for category in sorted({r["category"] for r in rows}):
            cells = {r["condition"]: r for r in rows if r["recipe"] == recipe and r["category"] == category}
            print(f"{category:<12}" + "".join(
                f"{cells[c]['auroc']:>11.3f} {cells[c]['fpr']:>7.0%} " for c in ("all", *CONDITIONS)))
        for c in ("all", *CONDITIONS):
            block = [r for r in rows if r["recipe"] == recipe and r["condition"] == c]
            fa = sum(r["false_alarms_mean"] for r in block)
            n0 = sum(r["normals"] for r in block)
            print(f"  {c:<13} macro AUROC {np.mean([r['auroc'] for r in block]):.3f}  "
                  f"pooled false alarms {fa:.1f}/{n0} = {fa / n0:.1%}")

    if args.guards:
        write_csv(out / "guards_by_condition.csv",
                  guards_by_condition(Path(args.guards), sorted({c for c, _ in runs})))

    if args.compare:
        a_name, b_name = args.compare
        comp = []
        for ci, category in enumerate(sorted({c for c, _ in runs})):
            a, b = runs.get((category, a_name)), runs.get((category, b_name))
            if a and b:
                comp.append({"category": category, "a": a_name, "b": b_name, **cluster_paired(a, b, seed=ci)})
        for unit in ("image", "object"):
            for row, (adj, sig) in zip(comp, holm_bonferroni([r[f"p_{unit}"] for r in comp])):
                row[f"p_holm_{unit}"], row[f"significant_{unit}"] = adj, sig
        write_csv(out / "cluster_bootstrap.csv", comp)
        print(f"\n{a_name} - {b_name}: image AUROC, image vs object bootstrap (Holm over {len(comp)})")
        for r in comp:
            print(f"  {r['category']:<12} {r['estimate']:+.3f}  image [{r['ci_low_image']:+.3f}, {r['ci_high_image']:+.3f}] "
                  f"p_holm {r['p_holm_image']:.4f}{' *' if r['significant_image'] else '  '}  "
                  f"object [{r['ci_low_object']:+.3f}, {r['ci_high_object']:+.3f}] p_holm {r['p_holm_object']:.4f}"
                  f"{' *' if r['significant_object'] else ''}  ({r['objects_normal']}+{r['objects_anomalous']} objects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
