"""Render case-book panels locally from stored predictions (never committed).

Each panel: the image with the map's peak marked, the anomaly map on a colour scale fixed by
the run's operating threshold, and a crop around the peak. Figures contain dataset pixels, so
they are written under `artifacts/` (git-ignored) and must not be published without the
dataset's attribution.

    python scripts/case_book.py --data-root <VisA> --run 00b777fff4506da9 \
        --ids pcb1/Data/Images/Normal/0214.JPG pcb1/Data/Images/Normal/0694.JPG \
        --out artifacts/case-book/pcb1-debris.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inspector.data.transforms import load_image
from inspector.evaluate import SplitPredictions
from inspector.models.base import postprocess_map
from inspector.postproc.thresholds import conformal_rank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", required=True, help="registry run id")
    ap.add_argument("--ids", nargs="+", required=True, help="image ids as stored in the predictions")
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=300)
    args = ap.parse_args()

    run_dir = Path(args.registry) / args.run
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    sigma = float(spec["evaluation"]["smoothing_sigma"])
    val = SplitPredictions.load(run_dir, "validation")
    test = SplitPredictions.load(run_dir, spec["dataset"]["test_split"])
    k = conformal_rank(val.scores.size, float(spec["evaluation"]["target_fpr"])) or 1
    threshold = float(np.sort(val.scores)[::-1][k - 1])

    wanted = list(args.ids)
    index = {img_id.replace("\\", "/"): n for n, img_id in enumerate(test.ids)}
    missing = [w for w in wanted if w not in index]
    if missing:
        raise SystemExit(f"not in this run's test split: {missing}")

    fig, axes = plt.subplots(3, len(wanted), figsize=(4.5 * len(wanted), 12), squeeze=False)
    for col, img_id in enumerate(wanted):
        n = index[img_id]
        image = load_image(Path(args.data_root) / img_id)
        amap = postprocess_map(test.raw_maps[n], input_size=test.input_sizes[n],
                               native_size=test.native_sizes[n], sigma=sigma)
        y, x = np.unravel_index(int(np.argmax(amap)), amap.shape)
        kind = "defect" if test.labels[n] else "normal"
        verdict = "ALARM" if test.scores[n] > threshold else "pass"
        axes[0, col].imshow(image)
        axes[0, col].plot(x, y, "r+", ms=28, mew=3)
        axes[0, col].set_title(f"{Path(img_id).name} ({kind})\nscore {test.scores[n]:.3f} -> {verdict}", fontsize=9)
        axes[1, col].imshow(amap, cmap="inferno", vmin=0, vmax=1.5 * threshold)
        axes[1, col].set_title("map (colour 1.0 = 1.5 x threshold)", fontsize=8)
        half = args.crop // 2
        y0, x0 = max(0, y - half), max(0, x - half)
        axes[2, col].imshow(image[y0:y0 + args.crop, x0:x0 + args.crop])
        axes[2, col].set_title("crop at the peak", fontsize=8)
        for row in range(3):
            axes[row, col].axis("off")
    fig.suptitle(f"run {args.run} — {spec['dataset']['category']}, threshold {threshold:.3f} "
                 f"(k={k} of {val.scores.size} validation normals)", fontsize=10)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=80)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
