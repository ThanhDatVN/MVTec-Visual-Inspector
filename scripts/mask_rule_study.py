"""Mask rules for the MVTec AD 2 server, compared on `test_public` from stored maps (no refit).

For each run, the validation maps (normal images only) give the thresholds:

* `3sigma`  — mean + 3 SD of the validation pixels (what S1 submitted);
* `p95xG`   — the validation pixels' 95th percentile times a gain G (SuperADD's rule, which
              sets G = 1.421 for its own maps; the gain depends on the map type, so it is scanned);
* `qR`      — the validation pixel quantile at a normal-pixel rate R (1% down to 0.01%).

Each threshold is applied bare (`plain`) and with the shape post-processing of
`inspector.postproc.binarize.MaskRule` (`morph`). SegF1 and ClassF1 are computed as the server does
(pixels pooled over every image of a category, normal images included; an image is rejected when
its mask has any pixel), then averaged over categories. Development evidence: any choice made from
this table is a choice made on `test_public`.

    python scripts/mask_rule_study.py --data-root data/raw/mvtec_ad2 --runs <ids> --label v3-1280 --out reports/studies/mask-rules
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from inspector.evaluate import SplitPredictions
from inspector.metrics.aupro import PROAccumulator
from inspector.postproc.binarize import PLAIN, MaskRule, validation_thresholds

GAINS = tuple(round(1.0 + 0.05 * i, 2) for i in range(21))  # 1.00 .. 2.00
RATES = (1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4)
MORPH = MaskRule()
MARKS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


def thresholds(val: SplitPredictions, sigma: float, n_sigma: float) -> dict[str, float]:
    lo, hi = val.raw_range()
    return validation_thresholds(val.native_maps(sigma), lo, hi, n_sigma=n_sigma, gains=GAINS, rates=RATES)


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def study(run_dir: Path, data_root: str) -> list[dict]:
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    val = SplitPredictions.load(run_dir, "validation", root=data_root)
    test = SplitPredictions.load(run_dir, spec["dataset"]["test_split"], root=data_root)
    return evaluate_rules(val, test, spec["evaluation"], run_dir.name, spec["dataset"]["category"])


def evaluate_rules(val: SplitPredictions, test: SplitPredictions, ev: dict, name: str, category: str) -> list[dict]:
    """One row per (rule, subset): `all` images (the mixed-lighting proxy) and `regular` only (the
    `test_private` proxy: the private split holds the training lighting only).

    Each threshold/mask pair is also scored with `MaskRule.mark` at the factors in MARKS: an image
    whose mask is empty but whose (work-resolution) map exceeds c * tau anywhere gets a single
    work pixel at its maximum. The variants are derived from the unmarked mask, exactly as
    `MaskRule.apply` would produce them.
    """
    sigma = ev["smoothing_sigma"]
    taus = thresholds(val, sigma, ev["sigma_n"])
    base = {(t, kind): (taus[t], rule) for t in taus for kind, rule in (("plain", PLAIN), ("morph", MORPH))}
    keys = [(t, kind, c) for (t, kind) in base for c in (0.0, *MARKS)]
    subsets = ("all", "regular")
    pix = {(k, s): np.zeros(3, dtype=np.int64) for k in keys for s in subsets}
    img = {(k, s): np.zeros(3, dtype=np.int64) for k in keys for s in subsets}
    normals = dict.fromkeys(subsets, 0)
    rate = min(1.0, ev["max_negative_samples"] / max(1.0, float(sum(w * h for w, h in test.native_sizes))))
    pro = {sub: PROAccumulator(negative_rate=rate, seed=ev["sampling_seed"]) for sub in subsets}
    for native, mask, label, image_id in zip(test.native_maps(sigma), test.masks(), test.labels, test.ids):
        member = ("all", "regular") if image_id.endswith("_regular.png") else ("all",)
        for sub in member:
            normals[sub] += int(label == 0)
            pro[sub].add(native, mask)
        # the single-pixel mark of each mask kind, and the maximum it is compared with
        marks = {}
        for kind, rule in (("plain", PLAIN), ("morph", MORPH)):
            work = rule.work_map(native)
            one = np.zeros(work.shape, dtype=bool)
            one[np.unravel_index(int(np.argmax(work)), work.shape)] = True
            block = rule.to_native(one, native.shape)
            marks[kind] = (float(work.max()), int(np.count_nonzero(block & mask)), int(np.count_nonzero(block & ~mask)))
        for (t, kind), (tau, rule) in base.items():
            pred = rule.apply(native, tau)
            px = np.array((np.count_nonzero(pred & mask), np.count_nonzero(pred & ~mask),
                           np.count_nonzero(~pred & mask)), dtype=np.int64)
            empty = not pred.any()
            peak, btp, bfp = marks[kind]
            for c in (0.0, *MARKS):
                marked = c > 0 and empty and peak > c * tau
                p_c = px + np.array((btp, bfp, -btp), dtype=np.int64) if marked else px
                rej = (not empty) or marked
                im = (rej and label == 1, rej and label == 0, (not rej) and label == 1)
                for sub in member:
                    pix[((t, kind, c), sub)] += p_c
                    img[((t, kind, c), sub)] += im
    aupro = {sub: pro[sub].curves((0.05,), num_thresholds=ev["aupro_num_thresholds"])[0.05].au_pro
             for sub in subsets}
    rows = []
    for t, kind, c in keys:
        for sub in subsets:
            tp, fp, fn = (int(x) for x in img[((t, kind, c), sub)])
            rows.append({"run_id": name, "category": category, "subset": sub, "threshold": t,
                         "post": kind if c == 0 else f"{kind}+mark{c:g}", "tau": base[(t, kind)][0],
                         "segf1": f1(*(int(x) for x in pix[((t, kind, c), sub)])), "classf1": f1(tp, fp, fn),
                         "normals_rejected": fp / max(1, normals[sub]), "aupro_005": aupro[sub]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="reports/studies/mask-rules")
    args = ap.parse_args()
    rows = []
    for run_id in args.runs:
        got = study(Path(args.registry) / run_id, args.data_root)
        rows += [{"label": args.label, **r} for r in got]
        got = [r for r in got if r["subset"] == "all"]
        best = max(got, key=lambda r: r["segf1"])
        print(f"{got[0]['category']:<12} best {best['threshold']}/{best['post']} SegF1 {best['segf1']:.3f}  "
              f"3sigma/plain {next(r['segf1'] for r in got if r['threshold'] == '3sigma' and r['post'] == 'plain'):.3f}",
              flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"rules_{args.label}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = sorted({(r["threshold"], r["post"]) for r in rows})

    def macro(key, sub, metric):
        return float(np.mean([r[metric] for r in rows if (r["threshold"], r["post"]) == key and r["subset"] == sub]))

    print(f"\n{args.label}: macro over {len(args.runs)} categories (regular = test_private proxy, all = mixed proxy)")
    first = keys[0]
    print(f"  AU-PRO@0.05 regular {macro(first, 'regular', 'aupro_005'):.4f} all {macro(first, 'all', 'aupro_005'):.4f}")
    for key in sorted(keys, key=lambda k: -macro(k, "all", "segf1"))[:15]:
        print(f"  {key[0]:<9} {key[1]:<6} SegF1 regular {macro(key, 'regular', 'segf1'):.4f} "
              f"all {macro(key, 'all', 'segf1'):.4f}  ClassF1 regular {macro(key, 'regular', 'classf1'):.4f} "
              f"all {macro(key, 'all', 'classf1'):.4f}  normals rejected {macro(key, 'all', 'normals_rejected'):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
