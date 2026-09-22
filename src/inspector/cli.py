"""Command line interface.

One principle (docs/06 §1): the laptop and Colab execute the *same* code path
with the same configs. A Colab notebook is a thin driver that installs this
package and calls these commands — never a place where model code lives.

    inspector fixtures --out data/synthetic
    inspector audit    --config configs/data/synth_strip.yaml
    inspector info

`fit`, `predict`, `evaluate` and `bench` are registered but not yet implemented;
they arrive with their phases (P3, P5, P9).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", required=True, help="path to a YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set model.k=3 (repeatable)",
    )
    parser.add_argument("--data-root", default=None, help="override data.root")


def _resolve(args) -> dict:
    from .config import load_config

    cfg = load_config(args.config, args.overrides)
    root = args.data_root or cfg.get("data", {}).get("root") or os.environ.get(
        "INSPECTOR_DATA_ROOT"
    )
    if root:
        cfg.setdefault("data", {})["root"] = str(root)
    return cfg


# --- commands ---------------------------------------------------------------


def cmd_info(args) -> int:
    from .config import config_hash
    from .utils.env import capture

    snapshot = capture()
    snapshot["inspector_version"] = __version__
    snapshot["empty_config_hash"] = config_hash({})
    print(json.dumps(snapshot, indent=2, default=str))
    return 0


def cmd_fixtures(args) -> int:
    from .fixtures import FixtureSpec, generate

    spec = FixtureSpec(
        n_train=args.n_train,
        n_validation=args.n_validation,
        n_test_good=args.n_test_good,
        n_test_per_defect=args.n_test_per_defect,
    )
    root = generate(args.out, spec=spec, seed=args.seed, overwrite=args.overwrite)
    images = sum(1 for _ in Path(root).rglob("*.png"))
    print(f"wrote {images} images to {root}")
    return 0


def cmd_audit(args) -> int:
    """Run the split-integrity audit (protocol §5) against real data.

    The same checks run in CI on synthetic fixtures; this is how they reach the
    real dataset, which CI can never see.
    """
    from .data import discover_category, ensure_validation
    from .data.integrity import audit

    cfg = _resolve(args)
    data: dict = cfg["data"]
    if not data.get("root"):
        print(
            "error: no data root. Pass --data-root, set data.root, or export "
            "INSPECTOR_DATA_ROOT.",
            file=sys.stderr,
        )
        return 2

    indices = discover_category(data["root"], data["category"], layout=data["layout"])
    indices, assignment = ensure_validation(
        indices,
        val_fraction=data.get("val_carve_fraction", 0.15),
        seed=data.get("val_carve_seed", 0),
    )

    print(f"dataset : {data['layout']} :: {data['category']}")
    print(f"root    : {data['root']}")
    for name, index in sorted(indices.items()):
        print(f"  {name:20s} n={len(index):5d}  normal={index.n_normal:5d}  "
              f"anomalous={index.n_anomalous:5d}  types={index.defect_types}")
    if assignment is not None:
        print(f"  validation carved from train: seed={assignment.seed} "
              f"digest={assignment.digest[:12]}")

    print()
    reports = audit(indices, near_duplicates=args.near_duplicates)
    for report in reports:
        print(report)

    failed = [r for r in reports if not r.ok]
    if args.write_manifests:
        out = Path(args.write_manifests)
        out.mkdir(parents=True, exist_ok=True)
        for name, index in indices.items():
            manifest = index.manifest()
            path = out / f"{data['category']}.{name}.json"
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"manifest -> {path}  sha256={str(manifest['manifest_sha256'])[:16]}")

    print(f"\n{len(reports) - len(failed)}/{len(reports)} checks passed")
    return 1 if failed else 0


def cmd_not_implemented(args) -> int:
    print(
        f"`inspector {args.command}` is not implemented yet; it lands with its "
        "phase (see docs/04-roadmap.md).",
        file=sys.stderr,
    )
    return 3


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspector",
        description="MVTec Visual Inspector - one-class industrial defect detection",
    )
    parser.add_argument("--version", action="version", version=f"inspector {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="print environment and provenance")
    p_info.set_defaults(func=cmd_info)

    p_fix = sub.add_parser("fixtures", help="generate the synthetic dataset")
    p_fix.add_argument("--out", "-o", default="data/synthetic")
    p_fix.add_argument("--seed", type=int, default=0)
    p_fix.add_argument("--n-train", type=int, default=12)
    p_fix.add_argument("--n-validation", type=int, default=5)
    p_fix.add_argument("--n-test-good", type=int, default=5)
    p_fix.add_argument("--n-test-per-defect", type=int, default=3)
    p_fix.add_argument("--overwrite", action="store_true")
    p_fix.set_defaults(func=cmd_fixtures)

    p_audit = sub.add_parser("audit", help="run the split-integrity audit (protocol section 5)")
    _add_config_args(p_audit)
    p_audit.add_argument("--near-duplicates", action="store_true", help="also run the pHash check (slow)")
    p_audit.add_argument("--write-manifests", default=None, metavar="DIR")
    p_audit.set_defaults(func=cmd_audit)

    for name, help_text in (
        ("fit", "fit a model on the train split (P3+)"),
        ("predict", "score a split with a fitted model (P3+)"),
        ("evaluate", "compute the metric table for a run (P3+)"),
        ("bench", "measure latency and memory (P9)"),
    ):
        p = sub.add_parser(name, help=help_text)
        _add_config_args(p)
        p.set_defaults(func=cmd_not_implemented)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
