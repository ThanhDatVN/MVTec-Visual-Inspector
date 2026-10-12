"""Deployment benchmark: inference latency (p50/p95/p99) and peak memory, one process per measurement.

Protocol (docs/02 §1): batch size 1, 20 warm-up predictions, then 200 timed ones with
`torch.cuda.synchronize()` around each, through the service path (`Inspector.predict`: input
guards, backbone and k-NN, upsampling and smoothing to native size). Each configuration is fitted
and exported in its own process, and measured in a fresh one, so no measurement inherits memory
from another — the per-run VRAM peaks in the registry are not clean for that reason (fixed in
`FeatureExtractor`; see reports/K1). Peak VRAM is `max_memory_allocated` from the artifact load
to the last prediction (the registry's measure; DR-1 criterion 4), and `max_memory_reserved`,
the allocator's footprint, is reported beside it. The CUDA context itself is in neither.

    python scripts/bench_inference.py --data-root <VisA> --category pcb1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

RECIPE = ("configs/data/visa_pcb1.yaml", "configs/recipes/confirmation-v1-patchcore-640.yaml")
#: label -> (config patch, model overrides, variant name as scripts/kaggle_studies.py prints it)
CONFIGS = {
    "recipe": ({}, {}, ""),
    "layer2": ({}, {"layers": ["layer2"]}, "layers=['layer2']"),
    "960px": ({"preprocess": {"long_side": 960}}, {}, "long_side=960"),
    "layer2_960px": ({"preprocess": {"long_side": 960}}, {"layers": ["layer2"]}, "layers=['layer2'],long_side=960"),
    "timm_racm": ({}, {"weights": "timm:racm_in1k"}, "weights=timm:racm_in1k"),
}
WARMUP, TIMED, N_IMAGES = 20, 200, 40


def _cfg(label: str) -> dict:
    from inspector.config import deep_merge, load_config

    return deep_merge(deep_merge(load_config(RECIPE[0]), load_config(RECIPE[1])), CONFIGS[label][0])


def fit_worker(args) -> dict:
    """Fit on the category's training normals and export the artifact, as `inspector export`."""
    import torch

    from inspector.api.artifact import save_artifact
    from inspector.api.guards import InputGuards
    from inspector.data.transforms import load_image
    from inspector.evaluate import SplitPredictions
    from inspector.runner import eval_config_from_config, prepare
    from inspector.utils.memory import PeakRSS, PeakVRAM

    cfg = _cfg(args.label)
    spec, model, idx = prepare(cfg, method="patchcore", category=args.category, seed=0,
                               data_root=args.data_root, overrides=CONFIGS[args.label][1] or None)
    with PeakRSS() as rss, PeakVRAM() as vram:
        t0 = time.perf_counter()
        model.fit(idx["train"])
        fit_s = time.perf_counter() - t0
    val = SplitPredictions.from_model(model, idx["validation"])
    guards = InputGuards.fit(load_image(s.image_path) for s in idx["train"])
    save_artifact(model, val.scores, args.artifact, target_fpr=eval_config_from_config(cfg).target_fpr,
                  category=args.category, guards=guards,
                  train_ids=[s.rel_id(idx["train"].root) for s in idx["train"]],
                  extra={"run_spec_id": spec.run_id, "bench_label": args.label})
    return {"label": args.label, "fit_s": fit_s, "fit_peak_vram_mb": vram.peak_mb, "fit_peak_rss_mb": rss.peak_mb,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}


def bench_worker(args) -> dict:
    """Load the artifact in a fresh process and time the service path."""
    import numpy as np
    import torch

    from inspector.api.artifact import load_artifact
    from inspector.config import load_config
    from inspector.data import ensure_validation, load_category
    from inspector.data.transforms import load_image
    from inspector.utils.memory import PeakRSS

    cuda = args.device == "cuda"
    data = load_config(RECIPE[0])["data"]
    idx = load_category(args.data_root, args.category, layout=data["layout"])
    idx, _ = ensure_validation(idx, val_fraction=float(data["val_carve_fraction"]), seed=int(data["val_carve_seed"]))
    images = [load_image(s.image_path) for s in list(idx[data["test_split"]])[:N_IMAGES]]

    def sync():
        if cuda:
            torch.cuda.synchronize()

    with PeakRSS() as rss:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        insp = load_artifact(args.artifact, device=args.device)
        for i in range(WARMUP):
            insp.predict(images[i % len(images)])
        times = []
        for i in range(TIMED):
            sync()
            t0 = time.perf_counter()
            insp.predict(images[i % len(images)])
            sync()
            times.append(1000.0 * (time.perf_counter() - t0))
    ms = np.asarray(times)
    return {"label": args.label, "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
            "long_side": insp.meta["transform"]["long_side"], "precision": f"{insp.meta['hparams']['dtype']} features",
            "batch": 1, "n_timed": TIMED,
            "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "p99_ms": float(np.percentile(ms, 99)), "mean_ms": float(ms.mean()),
            "peak_vram_alloc_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if cuda else math.nan,
            "peak_vram_reserved_mb": float(torch.cuda.max_memory_reserved() / 1024**2) if cuda else math.nan,
            "peak_rss_mb": rss.peak_mb}


def _child(*extra: str) -> dict:
    out = subprocess.run([sys.executable, __file__, *extra], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"worker {extra} failed:\n{out.stderr[-3000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--category", default="pcb1")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--cpu-configs", nargs="*", default=["recipe"], choices=list(CONFIGS))
    ap.add_argument("--work", default="artifacts/bench")
    ap.add_argument("--out", default="reports/studies/bench")
    ap.add_argument("--worker", choices=["fit", "bench"])
    ap.add_argument("--label")
    ap.add_argument("--artifact")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.worker:  # one measurement, in its own process; the last stdout line is the result
        result = fit_worker(args) if args.worker == "fit" else bench_worker(args)
        print(json.dumps(result))
        return 0

    common = ["--data-root", args.data_root, "--category", args.category]
    fits, benches = [], []
    for label in args.configs:
        art = str(Path(args.work) / f"{args.category}_{label}")
        if not (Path(art) / "meta.json").is_file():
            fits.append(_child("--worker", "fit", "--label", label, "--artifact", art, *common))
            print("fitted ", fits[-1], flush=True)
        for device in ["cuda"] + (["cpu"] if label in args.cpu_configs else []):
            benches.append({**_child("--worker", "bench", "--label", label, "--artifact", art,
                                     "--device", device, *common),
                            "variant": CONFIGS[label][2], "category": args.category})
            print("measured", benches[-1], flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fit.csv", fits), ("latency.csv", benches)):
        if rows:
            with open(out / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
