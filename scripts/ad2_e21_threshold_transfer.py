"""Does a per-category mask rule chosen on `test_public` beat one global rule out of sample?

Choosing each category's best threshold on `test_public` adds up to +2.2 SegF1 points in sample
(E21), but in-sample gains on development data have shrunk on held-out data before (docs/14 D1).
This splits each category's objects in two (seeded, separately for normal and anomalous objects;
all captures of an object stay together), chooses on one half and scores on the other, both ways,
for `--splits` random splits:

* **global** — the one rule (threshold, mask variant, mark) with the best macro SegF1 on the
  choosing halves, applied to every category;
* **per-category** — each category's own best rule on its choosing half.

Input: the per-image counts written by `ad2_e21_fusion.py --focus --dump DIR` (or the mask study).
Scored as the server does: pixels pooled over the scoring half's images, normal images included.

    python scripts/ad2_e21_threshold_transfer.py --dump reports/studies/fusion/dump-<label> --mark 0.9
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

NUMBER = re.compile(r"(\d+)_")


def f1(tp: float, fp: float, fn: float) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def objects(ids: np.ndarray) -> np.ndarray:
    """`<cat>/test_public/<good|bad>/<NNN>_<condition>.png` -> `<good|bad>/<NNN>`."""
    out = []
    for i in ids:
        parts = str(i).split("/")
        number = NUMBER.match(parts[-1]).group(1)
        out.append(f"{parts[-2]}/{number}")
    return np.asarray(out)


def scores(counts: np.ndarray, labels: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SegF1 and ClassF1 of every rule over the images in `rows`."""
    c = counts[rows].sum(axis=0)  # (keys, 4)
    seg = np.array([f1(*k[:3]) for k in c])
    rej = counts[rows][:, :, 3].astype(bool)
    y = labels[rows][:, None] == 1
    tp, fp, fn = (rej & y).sum(0), (rej & ~y).sum(0), (~rej & y).sum(0)
    cls = np.array([f1(a, b, d) for a, b, d in zip(tp, fp, fn)])
    return seg, cls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--mark", type=float, default=0.9, help="restrict the choice to rules with this mark")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--reference", default="p95x1.25|morph|0.9", help="the frozen rule, scored for comparison")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = {}
    for f in sorted(Path(args.dump).glob("*.npz")):
        z = np.load(f)
        data[f.stem] = {"counts": z["counts"], "labels": z["labels"], "objects": objects(z["ids"]),
                        "keys": [str(k) for k in z["keys"]]}
    cats = sorted(data)
    keys = data[cats[0]]["keys"]
    if any(data[c]["keys"] != keys for c in cats):
        raise ValueError("categories were scored with different rule sets")
    allowed = np.array([k.endswith(f"|{args.mark:g}") for k in keys])
    ref = keys.index(args.reference) if args.reference in keys else None

    results = {"global": [], "per_category": [], "reference": [], "global_classf1": [], "per_category_classf1": []}
    chosen_global, chosen_cat = [], {c: [] for c in cats}
    for split in range(args.splits):
        rng = np.random.default_rng(split)
        halves = {}
        for c in cats:
            obj = data[c]["objects"]
            uniq = np.unique(obj)
            side = {}
            for cls in ("good", "bad"):
                members = [o for o in uniq if o.startswith(cls)]
                order = rng.permutation(len(members))
                for rank, i in enumerate(order):
                    side[members[i]] = rank % 2
            halves[c] = np.array([side[o] for o in obj])
        for choose in (0, 1):
            seg_choose = {c: scores(data[c]["counts"], data[c]["labels"], np.flatnonzero(halves[c] == choose))
                          for c in cats}
            seg_score = {c: scores(data[c]["counts"], data[c]["labels"], np.flatnonzero(halves[c] != choose))
                         for c in cats}
            macro = np.mean([seg_choose[c][0] for c in cats], axis=0)
            g = int(np.flatnonzero(allowed)[np.argmax(macro[allowed])])
            chosen_global.append(keys[g])
            results["global"].append(np.mean([seg_score[c][0][g] for c in cats]))
            results["global_classf1"].append(np.mean([seg_score[c][1][g] for c in cats]))
            per, per_cls = [], []
            for c in cats:
                k = int(np.flatnonzero(allowed)[np.argmax(seg_choose[c][0][allowed])])
                chosen_cat[c].append(keys[k])
                per.append(seg_score[c][0][k])
                per_cls.append(seg_score[c][1][k])
            results["per_category"].append(np.mean(per))
            results["per_category_classf1"].append(np.mean(per_cls))
            if ref is not None:
                results["reference"].append(np.mean([seg_score[c][0][ref] for c in cats]))
    summary = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in results.items() if v}
    diff = np.array(results["per_category"]) - np.array(results["global"])
    summary["per_category_minus_global"] = {"mean": float(diff.mean()), "sd": float(diff.std()),
                                            "share_positive": float(np.mean(diff > 0))}
    summary["global_choices"] = {k: chosen_global.count(k) for k in sorted(set(chosen_global))}
    summary["per_category_choices"] = {c: {k: v.count(k) for k in sorted(set(v))} for c, v in chosen_cat.items()}
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
