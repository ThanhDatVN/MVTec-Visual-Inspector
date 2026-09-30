"""E08b: do input guards catch the corruptions that break the operating point? (development)

Fits the exposure and focus guards on each category's training normals, writes them into the
category's artifact (`guards.json`), and measures, for every E08 corruption and severity, the
share of test images the guards refuse — normals and defects alike, with the same per-image
corruption seeds as E08. The corrupted images are never scored by a model; the guards are
image statistics.

    python scripts/guard_study.py --data-root <VisA>
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path

import numpy as np

from inspector.api.guards import InputGuards
from inspector.config import deep_merge, load_config
from inspector.data import ensure_validation, load_category
from inspector.data.transforms import load_image
from inspector.eda import measure_regions, summarize_regions
from inspector.robustness import MAIN_SUITE, SEVERITIES, apply_corruption

RECIPE = "configs/recipes/confirmation-v1-patchcore-640.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--categories", nargs="+", default=["pcb1", "macaroni2", "capsules"])
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="reports/studies/e08/guards.csv")
    args = ap.parse_args()
    cfg = deep_merge(load_config("configs/data/visa_pcb1.yaml"), load_config(RECIPE))
    data = cfg["data"]

    rows = []
    for cat in args.categories:
        idx = load_category(args.data_root, cat, layout=data["layout"])
        idx, _ = ensure_validation(idx, val_fraction=float(data["val_carve_fraction"]),
                                   seed=int(data["val_carve_seed"]))
        guards = InputGuards.fit(load_image(s.image_path) for s in idx["train"])
        art = Path(args.artifacts) / cat
        if art.is_dir():
            (art / "guards.json").write_text(json.dumps(guards.as_dict(), indent=2), encoding="utf-8")
        test = idx[data["test_split"]]
        d_med = summarize_regions(measure_regions(test, with_contrast=False))["diameter_px_median"]
        val_flags = np.mean([not guards.check(load_image(s.image_path))["ok"] for s in idx["validation"]])
        print(f"{cat}: guards from {guards.n_reference} training normals; clean validation refused "
              f"{val_flags:.1%} (bound {guards.false_flag_bound:.2%} per guard)")
        images = [(s, load_image(s.image_path)) for s in test]
        clean = np.mean([not guards.check(im)["ok"] for _, im in images])
        rows.append({"category": cat, "corruption": "clean", "severity": 0, "refused": clean})
        for corruption in MAIN_SUITE:
            for severity in SEVERITIES:
                refused = []
                for s, im in images:
                    seed = zlib.crc32(s.rel_id(test.root).encode())
                    out, _ = apply_corruption(im, None, corruption, severity,
                                              median_defect_diameter=d_med, seed=seed)
                    refused.append(not guards.check(out)["ok"])
                rows.append({"category": cat, "corruption": corruption, "severity": severity,
                             "refused": float(np.mean(refused))})
            print(f"  {cat} {corruption}: " + " ".join(f"{r['refused']:.0%}" for r in rows[-5:]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "corruption", "severity", "refused"])
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
