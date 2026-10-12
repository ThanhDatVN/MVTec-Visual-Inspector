"""R4 (docs/14): an image-level branch beside the patch score, on MVTec AD 2 `test_public`.

A patch-local nearest-neighbour score cannot see an anomaly made of normal-looking patches — a
missing or extra part, a wrong count or arrangement (`wallplugs`, `rice`). An image-level
descriptor can. Two global scorers on the same frozen WRN50-2 and the same 640 px input as
recipe v1, fitted on the training normals only:

* `gauss` — each of layer2..4 mean-pooled over space, a Gaussian per layer with Ledoit-Wolf
  shrinkage, the score is the summed Mahalanobis distance (Rippel et al., ICPR 2020);
* `knn` — the layers' pooled means, each L2-normalized and concatenated, mean distance to the
  5 nearest training descriptors.

Each is fused with the stored PatchCore image score of a reference run: both z-scored on the
validation normals, then summed (`sum`) or maxed (`max`). Image AUROC overall and by capture
condition, and the paired object bootstrap of each fusion against the patch score alone.
Exploratory development study on `test_public`; any adoption goes through a pre-registered
screen.

    python scripts/global_branch_study.py --data-root data/raw/mvtec_ad2 \
        --implementation af89b5297f4d --role confirmation --out reports/studies/global-branch
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ad2_condition_study import CONDITIONS, cluster_paired, parse, read_split

from inspector.data import discover_category
from inspector.data.transforms import ImageTransform, load_image
from inspector.metrics.ranking import roc_auc

LAYERS = ("layer2", "layer3", "layer4")


def pooled_features(paths: list[str], transform: ImageTransform, device: str, batch: int = 4) -> dict[str, np.ndarray]:
    import torch
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor

    model = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1").eval().to(device)
    fx = create_feature_extractor(model, {name: name for name in LAYERS})
    out: dict[str, list[np.ndarray]] = {name: [] for name in LAYERS}
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = np.stack([transform.normalize_image(transform.resize_image(load_image(p))) for p in paths[i:i + batch]])
            feats = fx(torch.as_tensor(x, device=device))
            for name in LAYERS:
                out[name].append(feats[name].mean(dim=(2, 3)).float().cpu().numpy())
    del fx, model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {name: np.concatenate(v).astype(np.float64) for name, v in out.items()}


def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and Ledoit-Wolf shrunk covariance of the rows of `x`."""
    mean = x.mean(axis=0)
    xc = x - mean
    n, p = xc.shape
    s = xc.T @ xc / n
    mu = np.trace(s) / p
    delta = (np.sum(s * s) - 2 * mu * np.trace(s) + mu * mu * p) / p
    row_sq = np.sum(xc * xc, axis=1)
    # sum_k ||x_k x_k^T - S||_F^2 = sum_k (||x_k||^4 - 2 x_k^T S x_k) + n ||S||_F^2
    xs = xc @ s
    beta = (np.sum(row_sq**2) - 2 * np.sum(xs * xc) + n * np.sum(s * s)) / (n * n * p)
    alpha = min(beta, delta) / delta if delta > 0 else 1.0
    return mean, alpha * mu * np.eye(p) + (1 - alpha) * s


class Gauss:
    def __init__(self, train: dict[str, np.ndarray]):
        self.params = {}
        for name, x in train.items():
            mean, cov = ledoit_wolf(x)
            self.params[name] = (mean, np.linalg.inv(cov))

    def score(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        total = 0.0
        for name, (mean, prec) in self.params.items():
            d = feats[name] - mean
            total = total + np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, prec, d), 0.0))
        return total


class KNN:
    def __init__(self, train: dict[str, np.ndarray], k: int = 5):
        self.k = k
        self.bank = self._embed(train)

    @staticmethod
    def _embed(feats: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([f / np.linalg.norm(f, axis=1, keepdims=True) for f in feats.values()], axis=1)

    def score(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        q = self._embed(feats)
        d = np.sqrt(np.maximum((q * q).sum(1)[:, None] - 2 * q @ self.bank.T + (self.bank * self.bank).sum(1)[None], 0))
        return np.sort(d, axis=1)[:, : self.k].mean(axis=1)


def zscore(val: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (x - val.mean()) / max(val.std(), 1e-12)


def reference_runs(registry: Path, implementation: str, role: str) -> dict:
    """category -> {seed: (val_scores, test_records)} of recipe v1."""
    out: dict[str, dict] = {}
    for f in sorted(registry.glob("*/result.json")):
        spec = json.loads(f.with_name("spec.json").read_text(encoding="utf-8"))
        hp = spec["model"]["hparams"]
        if (spec["dataset"].get("layout") != "mvtec_ad2" or spec["model"]["name"] != "patchcore"
                or spec.get("role", "development") != role or not spec["implementation_id"].startswith(implementation)
                or spec["transform"]["long_side"] != 640 or hp.get("layers") != ["layer2", "layer3"]
                or spec["transform"].get("normalize", "imagenet") != "imagenet"
                or hp.get("bank_exposures") or hp.get("bank_relight")):
            continue
        if json.loads(f.read_text(encoding="utf-8")).get("status") != "completed":
            continue
        val = np.array([r["score"] for r in read_split(f.parent, "validation")])
        out.setdefault(spec["dataset"]["category"], {})[int(spec["seed"])] = (
            val, read_split(f.parent, spec["dataset"]["test_split"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--implementation", default="af89b5297f4d")
    ap.add_argument("--role", default="confirmation")
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/studies/global-branch")
    args = ap.parse_args()
    root = Path(args.data_root)
    refs = reference_runs(Path(args.registry), args.implementation, args.role)
    transform = ImageTransform(long_side=640)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, comps = [], []
    for ci, category in enumerate(args.categories or sorted(refs)):
        idx = discover_category(root, category, layout="mvtec_ad2", splits=("train", "validation", "test_public"))
        paths = {s: [str(x.image_path) for x in idx[s]] for s in ("train", "validation", "test_public")}
        feats = {s: pooled_features(p, transform, args.device) for s, p in paths.items()}
        seeds = refs[category]
        test_ids = [r["id"] for r in next(iter(seeds.values()))[1]]
        rel = [str(Path(p).relative_to(root)).replace("\\", "/") for p in paths["test_public"]]
        order = [rel.index(i) for i in test_ids]  # align features with the stored predictions
        y, cond, _, _ = parse(next(iter(seeds.values()))[1])
        scorers = {"gauss": Gauss(feats["train"]), "knn": KNN(feats["train"])}
        arms: dict[str, dict[int, tuple]] = {"patch": seeds}
        for name, scorer in scorers.items():
            g_val, g_test = scorer.score(feats["validation"]), scorer.score(feats["test_public"])[order]
            arms[name] = {s: (g_val, [{**r, "score": float(v)} for r, v in zip(rec, g_test)])
                          for s, (_, rec) in seeds.items()}
            for fusion in ("sum", "max"):
                fused = {}
                for s, (p_val, rec) in seeds.items():
                    p_test = np.array([r["score"] for r in rec])
                    zv = np.stack([zscore(p_val, p_val), zscore(g_val, g_val)])
                    zt = np.stack([zscore(p_val, p_test), zscore(g_val, g_test)])
                    f = (np.sum if fusion == "sum" else np.max)
                    fused[s] = (f(zv, axis=0), [{**r, "score": float(v)} for r, v in zip(rec, f(zt, axis=0))])
                arms[f"patch+{name}:{fusion}"] = fused
        for arm, by_seed in arms.items():
            row = {"category": category, "arm": arm}
            for c in ("all", *CONDITIONS):
                vals = []
                for _, (_, rec) in sorted(by_seed.items()):
                    s = np.array([r["score"] for r in rec])
                    keep = np.ones(y.size, bool) if c == "all" else cond == c
                    vals.append(roc_auc(y[keep], s[keep]))
                row[f"auroc_{c}"] = float(np.mean(vals))
            rows.append(row)
            if arm != "patch":
                comps.append({"category": category, "arm": arm,
                              **cluster_paired(by_seed, arms["patch"], seed=1000 + ci)})
        print(category, "  ".join(f"{r['arm']}={r['auroc_all']:.3f}/{r['auroc_regular']:.3f}"
                                    for r in rows if r["category"] == category), flush=True)
    for name, data in (("global_branch.csv", rows), ("global_branch_vs_patch.csv", comps)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    for arm in sorted({r["arm"] for r in rows}):
        block = [r for r in rows if r["arm"] == arm]
        print(f"{arm:<18} macro AUROC all {np.mean([r['auroc_all'] for r in block]):.3f}  "
              + "  ".join(f"{c} {np.mean([r[f'auroc_{c}'] for r in block]):.3f}" for c in CONDITIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
