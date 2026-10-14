"""How sure is v6's AU-PRO gain over v5? Object bootstrap of the paired difference on test_public.

Builds both arms exactly as `scripts/hybrid_maps_check.py` does (v5's rule: the maximum of the
two P95-scaled component maps; v6: the two-input head on the same maps), collects every
defect region and the sampled negative pixels per image as `mask_rule_study.evaluate_rules`
does, and counts them on one fixed threshold grid per arm, so a resample's PRO curve is a sum of
per-image counts. Objects are resampled within each category, normal and defective objects
separately (all captures of one object go together, as in `ad2_condition_study.cluster_paired`);
the macro difference uses the same draw index in every category.

    python scripts/hybrid_bootstrap.py --features C:/Users/<you>/mvi_head_features --dino <H+ runs> --wrn <WRN runs>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from ad2_condition_study import NAME
from fusion_head_study import EV, category_heads, features2
from fusion_study import load, p95, quarter

from inspector.metrics.aupro import _region_scores, _trapz_to_limit

LIMIT = 0.05
SUBSETS = ("regular", "all")


def collect(maps, masks, rate: float):
    """Per image (streamed): its sorted sampled negatives and its sorted defect regions."""
    images = []
    for i, (score, mask) in enumerate(zip(maps, masks)):
        regions, neg = _region_scores(np.asarray(score, dtype=np.float64), np.asarray(mask).astype(bool))
        if rate < 1.0 and neg.size:
            neg = neg[np.random.default_rng([EV["sampling_seed"], i]).random(neg.size) < rate]
        images.append((np.sort(neg), regions))
    return images


def grid(images) -> np.ndarray:
    """Thresholds at negative quantiles for FPR 0 .. 0.1 (twice the limit, so resamples reach it)."""
    neg = np.sort(np.concatenate([n for n, _ in images]))
    fpr = np.concatenate((np.geomspace(1e-6, 1e-3, 200), np.linspace(1e-3, 2 * LIMIT, 1800)))
    t = neg[np.clip(neg.size - 1 - np.floor(fpr * neg.size).astype(int), 0, neg.size - 1)]
    return np.unique(np.concatenate((t, [neg[-1] + 1e-6])))[::-1]


def tabulate(images, thresholds):
    """(negatives above t per image, negatives per image, coverage per region, region -> image)."""
    above = np.stack([n.size - np.searchsorted(n, thresholds, side="left") for n, _ in images])
    n_neg = np.array([n.size for n, _ in images], dtype=np.float64)
    cover, owner = [], []
    for i, (_, regions) in enumerate(images):
        for r in regions:
            cover.append((r.size - np.searchsorted(r, thresholds, side="left")) / r.size)
            owner.append(i)
    return above.astype(np.float64), n_neg, np.array(cover), np.array(owner)


def au_pro(table, weight: np.ndarray) -> float:
    """AU-PRO@0.05 with image multiplicities `weight` (0 = left out)."""
    above, n_neg, cover, owner = table
    fpr = weight @ above / (weight @ n_neg)
    w_reg = weight[owner]
    pro = w_reg @ cover / w_reg.sum()
    return _trapz_to_limit(fpr, pro, LIMIT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dino", nargs="+", required=True)
    ap.add_argument("--wrn", nargs="+", required=True)
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--data-root", default="data/raw/mvtec_ad2")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/studies/fusion-head/hybrid_bootstrap.json")
    args = ap.parse_args()
    reg = Path(args.registry)

    def by_cat(ids):
        return {json.loads((reg / r / "spec.json").read_text(encoding="utf-8"))["dataset"]["category"]: r for r in ids}

    dino, wrn = by_cat(args.dino), by_cat(args.wrn)
    rng = np.random.default_rng(args.seed)
    per_cat, draws = {}, {sub: [] for sub in SUBSETS}
    for cat in sorted(dino):
        head = category_heads(Path(args.features) / cat, kinds=("lr2",),
                              groups_to_load=("head", "synthetic", "validation"))["heads"]["lr2"]
        (sd, vd, td), (sw, vw, tw) = load(reg / dino[cat], args.data_root), load(reg / wrn[cat], args.data_root)
        sig_d, sig_w = sd["evaluation"]["smoothing_sigma"], sw["evaluation"]["smoothing_sigma"]
        nd, nw = p95(vd, sig_d), p95(vw, sig_w)
        arms = {"v5": [], "v6": []}
        for md, mw in zip(td.native_maps(sig_d), tw.native_maps(sig_w)):
            qd, qw = quarter(md) / nd, quarter(mw) / nw
            arms["v5"].append(np.maximum(qd, qw))
            arms["v6"].append(head(features2(np.stack([qd, qw]), np.ones(2))).reshape(qd.shape))
        ids, labels = list(td.ids), np.asarray(td.labels)
        objects = [f"{lab}:{NAME.match(i.split('/')[-1]).group(1)}" for i, lab in zip(ids, labels)]
        rate = min(1.0, EV["max_negative_samples"] / max(1.0, float(sum(w * h for w, h in td.native_sizes))))
        tables, exact = {}, {}
        for arm, maps in arms.items():
            built = replace(td, raw_maps=maps, input_sizes=list(td.native_sizes),
                            scores=np.array([float(m.max()) for m in maps]))
            images = collect(built.native_maps(0.0), td.masks(), rate)
            for sub in SUBSETS:
                keep = [k for k, i in enumerate(ids) if sub == "all" or i.endswith("_regular.png")]
                subset = [images[k] for k in keep]
                tables[(arm, sub)] = (tabulate(subset, grid(subset)), keep)
                exact[(arm, sub)] = au_pro(tables[(arm, sub)][0], np.ones(len(keep)))
        result = {sub: {"v5": exact[("v5", sub)], "v6": exact[("v6", sub)],
                        "diff": exact[("v6", sub)] - exact[("v5", sub)]} for sub in SUBSETS}
        # one draw of objects per resample, shared by both arms and both subsets
        groups = {lab: sorted({o for o, y in zip(objects, labels) if y == lab}) for lab in (0, 1)}
        members = {o: [k for k, x in enumerate(objects) if x == o] for o in set(objects)}
        cat_draws = {sub: np.empty(args.resamples) for sub in SUBSETS}
        for b in range(args.resamples):
            mult = np.zeros(len(ids))
            for lab in (0, 1):
                for j in rng.integers(0, len(groups[lab]), len(groups[lab])):
                    mult[members[groups[lab][j]]] += 1
            for sub in SUBSETS:
                d = [au_pro(tables[(arm, sub)][0], mult[tables[(arm, sub)][1]]) for arm in ("v6", "v5")]
                cat_draws[sub][b] = d[0] - d[1]
        for sub in SUBSETS:
            lo, hi = np.percentile(cat_draws[sub], [2.5, 97.5])
            result[sub].update({"ci95": [float(lo), float(hi)], "p_v6_not_better": float(np.mean(cat_draws[sub] <= 0))})
            draws[sub].append(cat_draws[sub])
        result["objects"] = {"normal": len(groups[0]), "defective": len(groups[1])}
        per_cat[cat] = result
        print(f"{cat:<12} " + "  ".join(
            f"{sub} {result[sub]['v5']:.3f}->{result[sub]['v6']:.3f} "
            f"[{result[sub]['ci95'][0]:+.3f}, {result[sub]['ci95'][1]:+.3f}]" for sub in SUBSETS), flush=True)
    macro = {}
    for sub in SUBSETS:
        m = np.mean(np.stack(draws[sub]), axis=0)
        est = float(np.mean([per_cat[c][sub]["diff"] for c in per_cat]))
        lo, hi = np.percentile(m, [2.5, 97.5])
        macro[sub] = {"v5": float(np.mean([per_cat[c][sub]["v5"] for c in per_cat])),
                      "v6": float(np.mean([per_cat[c][sub]["v6"] for c in per_cat])),
                      "diff": est, "ci95": [float(lo), float(hi)], "p_v6_not_better": float(np.mean(m <= 0))}
    print(json.dumps(macro, indent=1))
    Path(args.out).write_text(json.dumps({"resamples": args.resamples, "seed": args.seed, "macro": macro,
                                          "categories": per_cat}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
