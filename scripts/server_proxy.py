"""The MVTec AD 2 server's metrics, recomputed on `test_public` from stored predictions.

The server reports, per split, the mean over the eight categories of AU-PRO@0.05, SegF1 and
ClassF1, where (AD 2 paper, Supplement B-B) an image is *rejected* when at least one pixel of its
submitted binary mask is set, and ClassF1 is the F1 of those rejections with the anomalous
images as the positive class. This script reproduces SegF1 and ClassF1 for a run on
`test_public`, so the one server result we have can be checked against its local proxy, and
mask rules can be compared before a submission is spent:

* `3sigma`    — pixels above the validation mean + 3 SD (what S1 submitted);
* `valq`      — pixels above the validation pixel quantile at a 0.1% rate;
* `*+gate`    — the same masks, emptied for images whose image score is not above the
                validation operating threshold (the rank rule, as in `evaluate.image_metrics`);
* `reject_all` — every image rejected: the trivial ClassF1 floor of the class balance.

Development evidence only (test_public). No figure is produced; nothing derived from AD 2 images
leaves the machine except these aggregate numbers.

    python scripts/server_proxy.py --data-root data/raw/mvtec_ad2 --runs <run ids> --out reports/studies/ad2-server
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from inspector.evaluate import SplitPredictions
from inspector.metrics.pixel_level import PixelHistogram
from inspector.postproc.thresholds import (
    from_validation_fpr,
    pixel_quantile_from_histogram,
    sigma_from_stats,
)

RULES = ("3sigma", "3sigma+gate", "valq", "valq+gate")


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def proxy(run_dir: Path, data_root: str) -> dict:
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["result"]
    ev = spec["evaluation"]
    sigma = ev["smoothing_sigma"]
    val = SplitPredictions.load(run_dir, "validation", root=data_root)
    test = SplitPredictions.load(run_dir, spec["dataset"]["test_split"], root=data_root)

    total = total_sq = count = 0.0
    lo, hi = val.raw_range()
    hist = PixelHistogram(lo, hi, bins=ev.get("pixel_auroc_bins", 1 << 16))
    for native in val.native_maps(sigma):
        flat = native.ravel()
        total += float(flat.sum())
        total_sq += float(np.square(flat).sum())
        count += flat.size
        hist.add(flat, np.zeros(flat.size, dtype=bool))
    mean = total / count
    tau = {"3sigma": sigma_from_stats(mean, float(np.sqrt(max(0.0, total_sq / count - mean**2))),
                                      n_sigma=ev["sigma_n"]).value,
           "valq": pixel_quantile_from_histogram(hist.neg, hist.edges, pixel_fpr=0.001).value}
    image_thr = from_validation_fpr(val.scores, target_fpr=ev["target_fpr"], policy=ev["threshold_policy"]).value

    pix = {r: np.zeros(3, dtype=np.int64) for r in RULES}  # tp, fp, fn
    img = {r: np.zeros(3, dtype=np.int64) for r in (*RULES, "reject_all")}
    for native, mask, label, score in zip(test.native_maps(sigma), test.masks(), test.labels, test.scores):
        passed = score > image_thr
        for rule in RULES:
            base = rule.split("+")[0]
            pred = native > tau[base]
            if rule.endswith("+gate") and not passed:
                pred = np.zeros_like(pred)
            pix[rule] += (np.count_nonzero(pred & mask), np.count_nonzero(pred & ~mask), np.count_nonzero(~pred & mask))
            rejected = bool(pred.any())
            img[rule] += (rejected and label == 1, rejected and label == 0, (not rejected) and label == 1)
        img["reject_all"] += (label == 1, label == 0, 0)
    row = {"run_id": run_dir.name, "category": spec["dataset"]["category"],
           "long_side": spec["transform"]["long_side"], "seed": spec["seed"],
           "aupro_005": result["aupro_005"], "image_auroc": result["image_auroc"],
           "segf1_3sigma_stored": result["segf1_3sigma"],
           "threshold_3sigma": tau["3sigma"], "threshold_valq": tau["valq"],
           "anomalous_share": float(np.mean(test.labels == 1))}
    for rule in RULES:
        row[f"segf1_{rule}"] = f1(*(int(x) for x in pix[rule]))
    for rule in (*RULES, "reject_all"):
        tp, fp, fn = (int(x) for x in img[rule])
        row[f"classf1_{rule}"] = f1(tp, fp, fn)
        row[f"normals_rejected_{rule}"] = fp / max(1, int(np.sum(test.labels == 0)))
    if abs(row["segf1_3sigma"] - row["segf1_3sigma_stored"]) > 1e-3:
        raise RuntimeError(f"{run_dir.name}: SegF1 at 3 sigma {row['segf1_3sigma']:.4f} does not reproduce "
                           f"the stored {row['segf1_3sigma_stored']:.4f}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--label", required=True, help="name of this group of runs, e.g. v1-640")
    ap.add_argument("--out", default="reports/studies/ad2-server")
    args = ap.parse_args()
    rows = []
    for run_id in args.runs:
        rows.append({"label": args.label, **proxy(Path(args.registry) / run_id, args.data_root)})
        r = rows[-1]
        print(f"{r['category']:<12} AU-PRO {r['aupro_005']:.3f}  SegF1 3s {r['segf1_3sigma']:.3f} valq {r['segf1_valq']:.3f} "
              f"(+gate {r['segf1_3sigma+gate']:.3f} / {r['segf1_valq+gate']:.3f})  ClassF1 3s {r['classf1_3sigma']:.3f} "
              f"(+gate {r['classf1_3sigma+gate']:.3f}) reject-all {r['classf1_reject_all']:.3f}  "
              f"normals rejected 3s {r['normals_rejected_3sigma']:.0%}", flush=True)
    keys = [k for k in rows[0] if k.startswith(("aupro", "segf1_", "classf1_", "normals_rejected", "image_auroc"))
            and k != "segf1_3sigma_stored"]
    print(f"\n{args.label} means over {len(rows)} categories:")
    for k in keys:
        print(f"  {k:<32} {np.mean([r[k] for r in rows]):.4f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"proxy_{args.label}.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
