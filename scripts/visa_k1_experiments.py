"""K1 and E8: the GPU experiments on the VisA development categories, through the shared runner.

These are the experiment definitions behind reports/07-kaggle-studies.md and
reports/08-gate-e-followup.md (first run in Kaggle T4 sessions). Every run goes through
`inspector.runner` into the registry, so the script resumes where it stopped and reuses
completed runs. Groups:

* `reference` — the frozen recipe (PatchCore, WRN50 layer2+3, 640 px, fixed 10k bank), 3 seeds
* `sweep`     — one-axis-at-a-time ablation screen around the recipe, seed 0, plus 960 / 1280 px
* `followup`  — Gate E follow-up: the screening winners at three seeds (pre-registered, E8)
* `cae`       — the convolutional autoencoder at 640 px, up to 150 epochs (Tier 1)
* `c2`        — our PatchCore against anomalib v2.3.0's at 320 px (E00 step C2, owner=lib)
* `mpdd`      — the transfer probe (ADR-12): the recipe and two floors on MPDD (`--mpdd-root`)

The robustness grid (E08) is `scripts/visa_e08_robustness.py`. The analysis of these runs is
`scripts/visa_k1_analysis.py`.

    python scripts/visa_k1_experiments.py --data-root data/raw/VisA_20220922 --groups reference sweep
    python scripts/visa_k1_experiments.py --data-root data/raw/VisA_20220922 --groups mpdd \
        --mpdd-root data/raw/MPDD
    python scripts/visa_k1_experiments.py ... --dry-run      # list the planned runs, run nothing

Host RAM, not VRAM, bounds the high resolutions: 960 and 1280 px need more than 16 GB.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

GROUPS = ("reference", "sweep", "followup", "cae", "c2", "mpdd")
CATEGORIES = ("pcb1", "macaroni2", "capsules")
MPDD_CATEGORIES = ("bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes")
SEEDS = (0, 1, 2)

# Section 4 of the original session: one axis at a time around the recipe, seed 0.
# At 640 px the candidate pool (10%) holds 190k-375k patches, so no bank below is capped (F08).
AXES = {
    "backbone": [{"backbone": b} for b in ("resnet18", "resnet50")],
    "layers": [{"layers": layers} for layers in (["layer2"], ["layer3"], ["layer2", "layer3", "layer4"])],
    "bank": [{"bank_ratio": None, "bank_size": b} for b in (2500, 5000, 20000, 40000)],
    "neighbours": [{"patch_neighbors": k, "patch_reduction": red} for k, red in ((3, "mean"), (9, "mean"), (3, "kth"))],
    "reweight": [{"reweight": False}],
    "projection": [{"projection_dim": d} for d in (256, 512)],
}
SWEEP_RESOLUTIONS = (960, 1280)
AE_VARIANTS = ({"loss": "l2"}, {"loss": "ssim"}, {"loss": "l2+ssim"})
AE_EPOCHS, AE_PATIENCE = 150, 20


def plan(args, repo: Path) -> list[tuple[str, dict, str, str, int, Path, str, dict]]:
    """(group, cfg, method, category, seed, data_root, role, overrides) for every planned run."""
    from inspector.config import deep_merge, load_config

    def cfg(*parts: str) -> dict:
        return load_config(repo / "configs" / Path(*parts))

    data_cfg = cfg("data", "visa_pcb1.yaml")
    recipe = deep_merge(data_cfg, cfg("recipes", "confirmation-v1-patchcore-640.yaml"))
    visa = Path(args.data_root)
    cats = args.categories
    runs = []
    if "reference" in args.groups:
        runs += [("reference", recipe, "patchcore", c, s, visa, "development", {}) for c in cats for s in SEEDS]
    if "sweep" in args.groups:
        runs += [("sweep", recipe, "patchcore", c, 0, visa, "development", v)
                 for variants in AXES.values() for v in variants for c in cats]
        for res in SWEEP_RESOLUTIONS:
            hi = deep_merge(recipe, {"preprocess": {"long_side": res}})
            runs += [("sweep", hi, "patchcore", c, 0, visa, "development", {}) for c in cats]
    if "followup" in args.groups:
        r960 = deep_merge(recipe, {"preprocess": {"long_side": 960}})
        followup = ((recipe, {"layers": ["layer2"]}), (r960, {}), (r960, {"layers": ["layer2"]}),
                    (recipe, {"weights": "timm:racm_in1k"}))
        runs += [("followup", c_, "patchcore", c, s, visa, "development", o)
                 for c_, o in followup for c in cats for s in SEEDS]
    if "cae" in args.groups:
        ae = deep_merge(deep_merge(data_cfg, cfg("models", "cae.yaml")), {"preprocess": {"long_side": 640}})
        runs += [("cae", ae, "cae", c, 0, visa, "development",
                  {"epochs": AE_EPOCHS, "early_stopping_patience": AE_PATIENCE, **v})
                 for v in AE_VARIANTS for c in cats]
    if "c2" in args.groups:
        c2 = deep_merge(data_cfg, {"preprocess": {"long_side": 320}})
        ours = deep_merge(deep_merge(c2, cfg("models", "patchcore.yaml")), {"preprocess": {"long_side": 320}})
        lib = deep_merge(c2, cfg("models", "anomalib_patchcore.yaml"))
        arms = ((ours, "patchcore", {}), (lib, "anomalib_patchcore", {}),
                (lib, "anomalib_patchcore", {"weights": "tv_in1k"}))
        runs += [("c2", a, m, c, s, visa, "development", o) for a, m, o in arms for c in cats for s in SEEDS]
    if "mpdd" in args.groups:
        if not args.mpdd_root:
            raise SystemExit("--groups mpdd needs --mpdd-root (python scripts/data_fetch_mpdd.py)")
        root = Path(args.mpdd_root)
        mpdd_cfg = cfg("data", "mpdd.yaml")
        m_recipe = deep_merge(mpdd_cfg, cfg("recipes", "confirmation-v1-patchcore-640.yaml"))
        floors = deep_merge(mpdd_cfg, {"preprocess": {"long_side": 320}})
        present = [c for c in MPDD_CATEGORIES if (root / c / "train" / "good").is_dir()]
        for c in present:
            runs += [("mpdd", m_recipe, "patchcore", c, s, root, "confirmation", {}) for s in SEEDS]
            runs += [("mpdd", floors, m, c, 0, root, "confirmation", {}) for m in ("pixel_pca", "histogram")]
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", required=True, help="VisA root (the folder holding split_csv/)")
    ap.add_argument("--groups", nargs="+", default=list(GROUPS[:-1]), choices=GROUPS)
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    ap.add_argument("--mpdd-root", default=None, help="MPDD root (holds <category>/train/good)")
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--dry-run", action="store_true", help="build every run spec and list it; run nothing")
    args = ap.parse_args()

    from inspector.runner import Registry, execute, prepare

    repo = Path(__file__).resolve().parents[1]
    registry = Registry(Path(args.registry))
    runs = plan(args, repo)
    print(f"planned runs: {len(runs)}", flush=True)
    t0, done, failed = time.time(), 0, 0
    for group, cfg, method, cat, seed, root, role, over in runs:
        spec, model, idx = prepare(cfg, method=method, category=cat, seed=seed, data_root=root, role=role,
                                   overrides=over or None)
        label = f"[{spec.run_id}] {group:<9} {method:<18} {cat:<13} s{seed} {over or ''}"
        if args.dry_run:
            print(("done    " if registry.completed(spec.run_id) else "planned ") + label)
            continue
        try:
            result, reused = execute(spec, model, idx, data_root=root, registry=registry)
            print(("reused  " if reused else "ran     ") + label + "  " + result.summary(), flush=True)
            done += 1
        except Exception as exc:  # recorded as a failed run by execute; keep going
            print(f"FAILED  {label}: {exc}", flush=True)
            failed += 1
        finally:
            del model
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    if not args.dry_run:
        print(f"completed {done}, failed {failed}, {(time.time() - t0) / 3600:.1f} h; "
              "regenerate the table with `inspector results`", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
