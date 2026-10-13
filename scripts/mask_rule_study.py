"""Mask rules for the MVTec AD 2 server, compared on `test_public` from stored maps (no refit).

For each run, the validation maps (normal images only) give the thresholds:

* `3sigma`  — mean + 3 SD of the validation pixels (what S1 submitted);
* `p95xG`   — the validation pixels' 95th percentile times a gain G (SuperADD's rule, which
              sets G = 1.421 for its own maps; the gain depends on the map type, so it is scanned);
* `q0.1%`   — the validation pixel quantile at a 0.1% rate.

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
from inspector.metrics.pixel_level import PixelHistogram
from inspector.postproc.binarize import PLAIN, MaskRule
from inspector.postproc.thresholds import pixel_quantile_from_histogram, sigma_from_stats

GAINS = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.3)
MORPH = MaskRule()


def thresholds(val: SplitPredictions, sigma: float, n_sigma: float) -> dict[str, float]:
    total = total_sq = count = 0.0
    lo, hi = val.raw_range()
    hist = PixelHistogram(lo, hi, bins=1 << 16)
    for native in val.native_maps(sigma):
        flat = native.ravel()
        total += float(flat.sum())
        total_sq += float(np.square(flat).sum())
        count += flat.size
        hist.add(flat, np.zeros(flat.size, dtype=bool))
    mean = total / count
    std = float(np.sqrt(max(0.0, total_sq / count - mean**2)))
    p95 = pixel_quantile_from_histogram(hist.neg, hist.edges, pixel_fpr=0.05).value
    out = {"3sigma": sigma_from_stats(mean, std, n_sigma=n_sigma).value,
           "q0.1%": pixel_quantile_from_histogram(hist.neg, hist.edges, pixel_fpr=0.001).value}
    out.update({f"p95x{g:g}": p95 * g for g in GAINS})
    return out


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def study(run_dir: Path, data_root: str) -> list[dict]:
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    sigma = spec["evaluation"]["smoothing_sigma"]
    val = SplitPredictions.load(run_dir, "validation", root=data_root)
    test = SplitPredictions.load(run_dir, spec["dataset"]["test_split"], root=data_root)
    taus = thresholds(val, sigma, spec["evaluation"]["sigma_n"])
    rules = {(t, kind): (taus[t], rule) for t in taus for kind, rule in (("plain", PLAIN), ("morph", MORPH))}
    pix = {k: np.zeros(3, dtype=np.int64) for k in rules}
    img = {k: np.zeros(3, dtype=np.int64) for k in rules}
    for native, mask, label in zip(test.native_maps(sigma), test.masks(), test.labels):
        for key, (tau, rule) in rules.items():
            pred = rule.apply(native, tau)
            pix[key] += (np.count_nonzero(pred & mask), np.count_nonzero(pred & ~mask), np.count_nonzero(~pred & mask))
            rej = bool(pred.any())
            img[key] += (rej and label == 1, rej and label == 0, (not rej) and label == 1)
    rows = []
    for (t, kind), (tau, _) in rules.items():
        tp, fp, fn = (int(x) for x in img[(t, kind)])
        rows.append({"run_id": run_dir.name, "category": spec["dataset"]["category"], "threshold": t, "post": kind,
                     "tau": tau, "segf1": f1(*(int(x) for x in pix[(t, kind)])), "classf1": f1(tp, fp, fn),
                     "normals_rejected": fp / max(1, int(np.sum(test.labels == 0)))})
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
    print(f"\n{args.label}: macro over {len(args.runs)} categories")
    for t, kind in sorted(keys, key=lambda k: -np.mean([r["segf1"] for r in rows if (r["threshold"], r["post"]) == k])):
        sel = [r for r in rows if (r["threshold"], r["post"]) == (t, kind)]
        print(f"  {t:<9} {kind:<6} SegF1 {np.mean([r['segf1'] for r in sel]):.4f}  ClassF1 "
              f"{np.mean([r['classf1'] for r in sel]):.4f}  normals rejected {np.mean([r['normals_rejected'] for r in sel]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
