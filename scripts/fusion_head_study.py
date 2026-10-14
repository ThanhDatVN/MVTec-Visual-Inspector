"""E23, step 2 (CPU): a fusion head learned from synthetic defects, against v5's fixed rule.

Reads the per-image maps written by `head_features.py`. Every map is brought to a quarter of
native resolution: each DINOv3 block grid and WRN50's raw map are resized (WRN50 smoothed as the
evaluator does, sigma 4 in its input pixels; DINOv3 at the equivalent 1 quarter-pixel) and
divided by the 95th percentile of its own validation pixels.

* **rule** — v5's fixed fusion on these maps: max(mean of the four blocks / its P95, WRN50 / its P95);
* **head-lr** / **head-mlp** — per category, logistic regression / a 10-16-1 MLP on ten
  features per pixel (the five normalized maps and their 3 x 3 local maxima), trained on the
  held-out training images: synthetic-defect pixels positive, all other pixels negative. Its
  output probability is the fused map.

The maps are then thresholded and scored on `test_public` exactly as `mask_rule_study.py` does
(server-style, normal images included), at the P95 gains and at fixed probabilities. Nothing in
training touches a test image; the gain is chosen on `test_public`, as for v5 (development).

    python scripts/fusion_head_study.py --features C:/Users/<you>/mvi_head_features --out reports/studies/fusion-head
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mask_rule_study import evaluate_rules, gains_args

from inspector.evaluate import SplitPredictions
from inspector.postproc.binarize import MaskRule

EV = {"smoothing_sigma": 0.0, "sigma_n": 3.0, "max_negative_samples": 2_000_000, "sampling_seed": 0,
      "aupro_num_thresholds": 512}


@dataclass
class Record:
    channels: np.ndarray  # (5, Hq, Wq): four DINOv3 blocks, WRN50
    native: tuple[int, int]
    mask: np.ndarray | None
    label: int
    image_id: str
    mask_path: str | None


def quarter_channels(z) -> np.ndarray:
    import cv2

    from inspector.data.transforms import smooth_map, upsample_map

    w, h = (int(v) for v in z["native"])
    hq, wq = max(1, h // 4), max(1, w // 4)
    layers = [cv2.GaussianBlur(cv2.resize(m.astype(np.float32), (wq, hq), interpolation=cv2.INTER_LINEAR),
                               (0, 0), 1.0) for m in z["layers"]]
    wi, hi = (int(v) for v in z["wrn_input"])
    wrn = smooth_map(upsample_map(z["wrn"].astype(np.float32), (wi, hi)), 4.0)
    wrn = cv2.resize(wrn.astype(np.float32), (wq, hq), interpolation=cv2.INTER_AREA)
    return np.stack([*layers, wrn]).astype(np.float32)


REPO = Path(__file__).resolve().parents[1]


def resolve(path: str) -> str | None:
    """A stored mask path, absolute; relative ones were written relative to the repository root."""
    if not path:
        return None
    return path if Path(path).is_absolute() else str(REPO / path)


def load(group_dir: Path) -> list[Record]:
    out = []
    for f in sorted(group_dir.glob("*.npz")):
        z = np.load(f)
        out.append(Record(quarter_channels(z), tuple(int(v) for v in z["native"]),
                          z["mask"] if "mask" in z.files else None,
                          int(z["label"]) if "label" in z.files else 0,
                          str(z["id"]) if "id" in z.files else f.stem,
                          resolve(str(z["mask_path"])) if "mask_path" in z.files else None))
    return out


def p95(values: list[np.ndarray]) -> float:
    flat = np.concatenate([v.ravel() for v in values])
    return float(np.quantile(flat, 0.95))


def features(rec: Record, scales: np.ndarray) -> np.ndarray:
    """(Hq * Wq, 10): five normalized maps and their 3 x 3 local maxima."""
    import cv2

    norm = rec.channels / scales[:, None, None]
    local = np.stack([cv2.dilate(c, np.ones((3, 3), np.uint8)) for c in norm])
    return np.concatenate([norm, local]).reshape(10, -1).T


def two_channels(channels: np.ndarray) -> np.ndarray:
    """v5's two fused inputs: the DINOv3 block mean and WRN50 (what the v5 export kept)."""
    return np.stack([channels[:4].mean(0), channels[4]])


def features2(two: np.ndarray, scales2: np.ndarray) -> np.ndarray:
    """(Hq * Wq, 4): the two normalized maps and their 3 x 3 local maxima."""
    import cv2

    norm = two / scales2[:, None, None]
    local = np.stack([cv2.dilate(c, np.ones((3, 3), np.uint8)) for c in norm])
    return np.concatenate([norm, local]).reshape(4, -1).T


def full_bank_records(dino_dir: Path, wrn_dir: Path, split: str, data_root: str) -> list[Record]:
    """Two-channel quarter-resolution records from registry runs (full banks, as v5 was selected on),
    built like `quarter_channels`: DINOv3's token-grid map resized and blurred by 1 quarter pixel,
    WRN50's map smoothed in its input pixels and area-resized."""
    import cv2

    from inspector.data.transforms import smooth_map, upsample_map

    d = SplitPredictions.load(dino_dir, split, root=data_root)
    w = SplitPredictions.load(wrn_dir, split, root=data_root)
    if d.ids != w.ids:
        raise ValueError("runs were evaluated on different images")
    out = []
    for i in range(len(d.ids)):
        nw, nh = d.native_sizes[i]
        hq, wq = max(1, nh // 4), max(1, nw // 4)
        dq = cv2.GaussianBlur(cv2.resize(d.raw_maps[i].astype(np.float32), (wq, hq), interpolation=cv2.INTER_LINEAR),
                              (0, 0), 1.0)
        wr = smooth_map(upsample_map(w.raw_maps[i].astype(np.float32), w.input_sizes[i]), 4.0)
        wq_ = cv2.resize(wr.astype(np.float32), (wq, hq), interpolation=cv2.INTER_AREA)
        out.append(Record(np.stack([dq, wq_]), (nw, nh), None, int(d.labels[i]), d.ids[i], d.mask_paths[i]))
    return out


def train_head(pos: np.ndarray, neg: np.ndarray, kind: str, seed: int):
    import torch

    torch.manual_seed(seed)
    x = torch.as_tensor(np.concatenate([pos, neg]), dtype=torch.float32)
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    mu, sd = x.mean(0), x.std(0).clamp_min(1e-6)
    d = x.shape[1]
    net = torch.nn.Linear(d, 1) if kind == "lr" else torch.nn.Sequential(
        torch.nn.Linear(d, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    weight = torch.tensor(len(neg) / max(1, len(pos)))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    for _ in range(300):
        perm = torch.randperm(len(x))[:65536]
        opt.zero_grad()
        loss = loss_fn(net((x[perm] - mu) / sd).squeeze(1), y[perm])
        loss.backward()
        opt.step()

    def predict(feats: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(feats, dtype=torch.float32)
            return torch.sigmoid(net((t - mu) / sd).squeeze(1)).numpy()

    return predict


def category_heads(cat_dir: Path, *, seed: int = 0, negatives: int = 10, kinds=("lr", "mlp", "lr2"),
                   groups_to_load=("head", "synthetic", "validation", "test_public")) -> dict:
    """Train one category's heads from its stored features; shared by the study and the export.

    Scales are the 95th percentiles of the validation maps (per channel; for the two-input head,
    of the block mean and of WRN50). Training pixels: every synthetic-defect pixel positive; up to
    4,000 other pixels per image negative, then at most `negatives` per positive.
    """
    groups = {g: load(cat_dir / g) for g in groups_to_load}
    scales = np.array([p95([r.channels[c] for r in groups["validation"]]) for c in range(5)])
    scales2 = np.array([p95([r.channels[:4].mean(0) for r in groups["validation"]]), scales[4]])
    rng = np.random.default_rng(seed)
    pos, neg, pos2, neg2 = [], [], [], []
    for r in groups["synthetic"] + groups["head"]:
        f, f2 = features(r, scales), features2(two_channels(r.channels), scales2)
        m = r.mask.ravel() if r.mask is not None else np.zeros(len(f), dtype=bool)
        pick = rng.choice(np.flatnonzero(~m), size=min(int((~m).sum()), 4000), replace=False)
        pos.append(f[m])
        neg.append(f[pick])
        pos2.append(f2[m])
        neg2.append(f2[pick])
    pos, neg, pos2, neg2 = (np.concatenate(a) for a in (pos, neg, pos2, neg2))
    keep = rng.choice(len(neg), size=min(len(neg), negatives * max(1, len(pos))), replace=False)
    neg, neg2 = neg[keep], neg2[keep]
    heads = {k: train_head(pos2, neg2, "lr", seed) if k == "lr2" else train_head(pos, neg, k, seed) for k in kinds}
    return {"groups": groups, "scales": scales, "scales2": scales2, "heads": heads, "positives": len(pos)}


def as_split(name: str, records: list[Record], maps: list[np.ndarray]) -> SplitPredictions:
    return SplitPredictions(split=name, ids=[r.image_id for r in records],
                            labels=np.array([r.label for r in records]), defect_types=["" for _ in records],
                            scores=np.array([float(m.max()) for m in maps]), raw_maps=maps,
                            input_sizes=[r.native for r in records], native_sizes=[r.native for r in records],
                            mask_paths=[r.mask_path for r in records], mask_overrides=[])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="reports/studies/fusion-head")
    ap.add_argument("--negatives", type=int, default=10, help="negative pixels per positive pixel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply-dino", nargs="*", default=[], help="full-bank ViT-H+ registry runs (one per category)")
    ap.add_argument("--apply-wrn", nargs="*", default=[], help="full-bank WRN50 1280 px registry runs")
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--data-root", default="data/raw/mvtec_ad2")
    args = ap.parse_args()
    import json

    def by_category(ids):
        out = {}
        for rid in ids:
            spec = json.loads((Path(args.registry) / rid / "spec.json").read_text(encoding="utf-8"))
            out[spec["dataset"]["category"]] = Path(args.registry) / rid
        return out

    apply_dino, apply_wrn = by_category(args.apply_dino), by_category(args.apply_wrn)
    rows = []
    for cat_dir in sorted(d for d in Path(args.features).iterdir() if (d / "done.json").is_file()):
        cat = cat_dir.name
        fitted = category_heads(cat_dir, seed=args.seed, negatives=args.negatives)
        groups, scales, scales2, heads = fitted["groups"], fitted["scales"], fitted["scales2"], fitted["heads"]
        rule_scale_d, pos = scales2[0], np.zeros(fitted["positives"])
        methods = {}

        def rule(r: Record, s_d=rule_scale_d, s_w=scales[4]) -> np.ndarray:
            return np.maximum(r.channels[:4].mean(0) / s_d, r.channels[4] / s_w)

        methods["rule"] = rule
        for kind in ("lr", "mlp"):

            def apply(r: Record, head=heads[kind], scales=scales) -> np.ndarray:
                return head(features(r, scales)).reshape(r.channels.shape[1:])

            methods[f"head-{kind}"] = apply
        head2 = heads["lr2"]

        def apply2(r: Record, head2=head2, scales2=scales2) -> np.ndarray:
            return head2(features2(two_channels(r.channels), scales2)).reshape(r.channels.shape[1:])

        methods["head-lr2"] = apply2
        if cat in apply_dino and cat in apply_wrn:
            # the deployable test: the two-input head, trained on the seven-eighths features, applied
            # to the full-bank development maps v5 was selected on
            full = {sp: full_bank_records(apply_dino[cat], apply_wrn[cat], sp, args.data_root)
                    for sp in ("validation", "test_public")}
            s_full = np.array([p95([r.channels[0] for r in full["validation"]]),
                               p95([r.channels[1] for r in full["validation"]])])
            for name, fn in (("full-rule", lambda r, s=s_full: np.maximum(r.channels[0] / s[0], r.channels[1] / s[1])),
                             ("full-head-lr2", lambda r, s=s_full, h=head2: h(features2(r.channels, s)).reshape(
                                 r.channels.shape[1:]))):
                val = as_split("validation", full["validation"], [fn(r) for r in full["validation"]])
                test = as_split("test_public", full["test_public"], [fn(r) for r in full["test_public"]])
                got = evaluate_rules(val, test, EV, name, cat, posts={"morph": MaskRule()}, keep={"p95x1.3"})
                rows += [{"method": name, **r} for r in got]
                a = next(r for r in got if r["subset"] == "all")
                g = next(r for r in got if r["subset"] == "regular")
                print(f"{cat:<12} {name:<14} AU-PRO regular {g['aupro_005']:.3f} all {a['aupro_005']:.3f}", flush=True)
        for name, fn in methods.items():
            val = as_split("validation", groups["validation"], [fn(r) for r in groups["validation"]])
            test = as_split("test_public", groups["test_public"], [fn(r) for r in groups["test_public"]])
            extra = {"p0.5": 0.5, "p0.7": 0.7, "p0.9": 0.9} if name.startswith("head") else None
            got = evaluate_rules(val, test, EV, name, cat, **gains_args(), extra_taus=extra)
            rows += [{"method": name, **r} for r in got]
            best = max((r for r in got if r["subset"] == "all" and r["post"] == "morph+mark0.9"), key=lambda r: r["segf1"])
            print(f"{cat:<12} {name:<9} AU-PRO {best['aupro_005']:.3f}  best {best['threshold']} SegF1 {best['segf1']:.3f}  "
                  f"positives {len(pos)}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "head_rules.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for name in sorted({r["method"] for r in rows}):
        sel = [r for r in rows if r["method"] == name and r["post"] == "morph+mark0.9"]
        if not sel:  # the full-bank rows carry AU-PRO only
            sel = [r for r in rows if r["method"] == name and r["post"] == "morph"]
        cats = sorted({r["category"] for r in sel})
        thresholds = sorted({r["threshold"] for r in sel})

        def macro(t, sub, key, sel=sel, cats=cats):
            return float(np.mean([next(r[key] for r in sel if r["category"] == c and r["threshold"] == t
                                       and r["subset"] == sub) for c in cats]))

        t_best = max(thresholds, key=lambda t: macro(t, "all", "segf1"))
        print(f"{name:<9} macro over {len(cats)}: best {t_best} SegF1 regular {macro(t_best, 'regular', 'segf1'):.4f} "
              f"all {macro(t_best, 'all', 'segf1'):.4f}  ClassF1 {macro(t_best, 'all', 'classf1'):.4f}  AU-PRO regular "
              f"{macro(t_best, 'regular', 'aupro_005'):.4f} all {macro(t_best, 'all', 'aupro_005'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
