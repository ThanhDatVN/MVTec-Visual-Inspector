"""E08: robustness of the frozen recipe to optical and geometric change (development categories).

The model is fitted once on clean training normals and the operating threshold is set once on
clean validation normals; only the test split is corrupted, normals and defects alike, each
image with its own seed (a stable hash of its path) shared across severities. Re-thresholding
under corruption would answer a much easier question. Metrics come from the shared evaluator,
so clean and corrupted cells have the same negative population (review F09).

Severity is scaled by each category's measured median defect diameter (docs/05), so severity 3
means comparable physical change on every category.

    python scripts/robustness_study.py --data-root <VisA> --categories pcb1 macaroni2 capsules

Resumable: finished cells are read back from the CSV and skipped. The fitted model of each
category is saved to --artifacts for the demo service.
"""

from __future__ import annotations

import argparse
import csv
import time
import zlib
from pathlib import Path

from inspector.api.artifact import save_artifact
from inspector.config import deep_merge, load_config
from inspector.eda import measure_regions, summarize_regions
from inspector.evaluate import SplitPredictions, evaluate
from inspector.robustness import CORRUPTIONS, MAIN_SUITE, SEVERITIES, apply_corruption
from inspector.runner import eval_config_from_config, prepare

RECIPE = "configs/recipes/confirmation-v1-patchcore-640.yaml"
FIELDS = ["category", "corruption", "severity", "image_auroc", "aupro_005", "fpr_at_op1",
          "recall_at_op1", "effective_fpr", "fp", "tn", "tp", "fn", "seconds"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--categories", nargs="+", default=["pcb1", "macaroni2", "capsules"])
    ap.add_argument("--out", default="reports/studies/e08/robustness.csv")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corruptions", nargs="+", default=list(MAIN_SUITE), choices=sorted(CORRUPTIONS),
                    help="default: the main suite; `translate` is reported separately (docs/05)")
    ap.add_argument("--no-artifacts", action="store_true", help="do not (re)write artifacts")
    args = ap.parse_args()

    cfg = deep_merge(load_config("configs/data/visa_pcb1.yaml"), load_config(RECIPE))
    eval_cfg = eval_config_from_config(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str, int]] = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as fh:
            done = {(r["category"], r["corruption"], int(r["severity"])) for r in csv.DictReader(fh)}
    new_file = not out.exists()

    for cat in args.categories:
        cells = [(c, s) for c in args.corruptions for s in SEVERITIES if (cat, c, s) not in done]
        need_clean = (cat, "clean", 0) not in done
        if not cells and not need_clean:
            continue
        spec, model, idx = prepare(cfg, method="patchcore", category=cat, seed=args.seed,
                                   data_root=args.data_root)
        test_idx = idx[spec.dataset["test_split"]]
        d_med = summarize_regions(measure_regions(test_idx, with_contrast=False))["diameter_px_median"]
        t0 = time.perf_counter()
        model.fit(idx["train"])
        print(f"{cat}: fitted in {time.perf_counter() - t0:.0f} s; median defect diameter {d_med:.1f} px")
        val = SplitPredictions.from_model(model, idx["validation"])  # clean and frozen
        if not args.no_artifacts:
            save_artifact(model, val.scores, Path(args.artifacts) / cat, target_fpr=eval_cfg.target_fpr,
                          category=cat, extra={"run_spec_id": spec.run_id})

        def corruptor(corruption: str, severity: int, root=test_idx.root, diameter=d_med):
            def image_fn(image, mask, sample):
                seed = zlib.crc32(sample.rel_id(root).encode())
                return apply_corruption(image, mask, corruption, severity,
                                        median_defect_diameter=diameter, seed=seed)
            return image_fn

        plan = ([("clean", 0)] if need_clean else []) + cells
        for corruption, severity in plan:
            t0 = time.perf_counter()
            fn = None if corruption == "clean" else corruptor(corruption, severity)
            test = SplitPredictions.from_model(model, test_idx, image_fn=fn)
            m = evaluate(val, test, eval_cfg)
            row = {"category": cat, "corruption": corruption, "severity": severity,
                   **{k: m.get(k) for k in FIELDS[3:-1]}, "seconds": round(time.perf_counter() - t0, 1)}
            with open(out, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS)
                if new_file:
                    writer.writeheader()
                    new_file = False
                writer.writerow(row)
            print(f"  {cat:<10} {corruption:<16} s{severity}  AUROC={m['image_auroc']:.3f}  "
                  f"AU-PRO={m.get('aupro_005', float('nan')):.3f}  FPR={m['fpr_at_op1']:.3f}  "
                  f"recall={m['recall_at_op1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
