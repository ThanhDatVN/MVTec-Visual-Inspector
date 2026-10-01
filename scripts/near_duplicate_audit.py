"""A4: near-duplicate captures across splits (rule L6), with a calibrated threshold.

On a fixed rig every photograph of a part looks alike, so a fixed threshold either flags
everything or nothing. Each criterion is calibrated per category instead: for each training
image, the distance to its nearest *other* training image describes how close two distinct
captures in one split normally are, and a validation or test image closer to a training image
than the 1st percentile of that distribution is a near-duplicate candidate.

Two distances are reported. The 64-bit pHash of the original rule L6 turns out to be too
coarse on these rigs (distinct captures collide at distance 0), so the primary criterion is the
RMSE between 64x64 grey thumbnails; RMSE < 1 grey level is treated as pixel-identical.

    python scripts/near_duplicate_audit.py --data-root <VisA> --out reports/studies/a4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inspector.data import ensure_validation, load_category
from inspector.data.integrity import phash
from inspector.data.visa import summarize

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = a[:, None] ^ b[None, :]
    return POPCOUNT[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1)


def thumb(path: Path, size: int = 64) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("L").resize((size, size), Image.Resampling.BOX), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="reports/studies/a4")
    ap.add_argument("--percentile", type=float, default=1.0)
    args = ap.parse_args()

    rows = []
    for cat in summarize(args.data_root):
        idx = load_category(args.data_root, cat, layout="visa")
        idx, _ = ensure_validation(idx, val_fraction=0.15, seed=0)
        hashes = {name: np.array([phash(s.image_path) for s in split], dtype=np.uint64)
                  for name, split in idx.items()}
        thumbs = {name: np.stack([thumb(s.image_path).ravel() for s in split]) for name, split in idx.items()}

        def rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            d2 = (a * a).sum(1)[:, None] - 2 * a @ b.T + (b * b).sum(1)[None, :]
            return np.sqrt(np.clip(d2, 0, None) / a.shape[1])

        within_px = rmse(thumbs["train"], thumbs["train"])
        np.fill_diagonal(within_px, np.inf)
        nn_px = within_px.min(axis=1)
        px_threshold = float(np.percentile(nn_px, args.percentile))
        train = hashes["train"]
        within = hamming_matrix(train, train).astype(float)
        np.fill_diagonal(within, np.inf)
        nearest_within = within.min(axis=1)
        threshold = float(np.percentile(nearest_within, args.percentile))
        row = {"category": cat, "n_train": int(train.size),
               "train_nn_median": float(np.median(nearest_within)),
               "train_nn_p1": threshold, "train_exact_pairs": int((within == 0).sum() // 2),
               "train_px_nn_median": float(np.median(nn_px)), "train_px_nn_p1": px_threshold,
               "train_px_identical_pairs": int((within_px < 1.0).sum() // 2)}
        for other in ("validation", "test"):
            d = hamming_matrix(hashes[other], train)
            nn = d.min(axis=1)
            exact = []
            for i in np.flatnonzero(nn == 0):
                j = int(np.argmin(d[i]))
                a = thumb(list(idx[other])[i].image_path)
                b = thumb(list(idx["train"])[j].image_path)
                exact.append(float(np.sqrt(np.mean((a - b) ** 2))))
            row[f"{other}_nn_median"] = float(np.median(nn))
            row[f"{other}_below_threshold"] = int((nn < threshold).sum())
            row[f"{other}_n"] = int(nn.size)
            row[f"{other}_exact_hash"] = len(exact)
            row[f"{other}_exact_pixel_identical"] = int(sum(r < 1.0 for r in exact))
            px = rmse(thumbs[other], thumbs["train"]).min(axis=1)
            row[f"{other}_px_nn_median"] = float(np.median(px))
            row[f"{other}_px_below_p1"] = int((px < px_threshold).sum())
            row[f"{other}_px_identical"] = [list(idx[other])[i].rel_id(idx[other].root)
                                            for i in np.flatnonzero(px < 1.0)]
        rows.append(row)
        print(cat, {k: v for k, v in row.items() if k != "category"})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "near_duplicates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["| category | pHash: train nn median / p1 | pHash exact matches val / test | thumbnail RMSE: train nn median / p1 | val below p1 | test below p1 | pixel-identical val / test |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| `{r['category']}` | {r['train_nn_median']:.0f} / {r['train_nn_p1']:.0f} | "
            f"{r['validation_exact_hash']} / {r['test_exact_hash']} | "
            f"{r['train_px_nn_median']:.1f} / {r['train_px_nn_p1']:.1f} | "
            f"{r['validation_px_below_p1']}/{r['validation_n']} | {r['test_px_below_p1']}/{r['test_n']} | "
            f"{len(r['validation_px_identical'])} / {len(r['test_px_identical'])} |")
    (out / "near_duplicates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
