"""Command line interface.

One principle (docs/06 §1): the laptop and Colab execute the *same* code path
with the same configs. A Colab notebook is a thin driver that installs this
package and calls these commands — never a place where model code lives.

    inspector fixtures --out data/synthetic
    inspector audit    --config configs/data/synth_strip.yaml
    inspector eda      --config configs/data/synth_strip.yaml
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
    parser.add_argument(
        "--config", "-c", required=True, action="append",
        help="YAML config; repeat to compose (later files override earlier ones)",
    )
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
    from .config import apply_overrides, deep_merge, load_config

    paths = args.config if isinstance(args.config, list) else [args.config]
    cfg: dict = {}
    for path in paths:
        cfg = deep_merge(cfg, load_config(path))
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)
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
    from .data import ensure_validation, load_category
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

    indices = load_category(
        data["root"], data["category"], layout=data["layout"], split_csv=data.get("split_csv")
    )
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


def cmd_eda(args) -> int:
    """Phase P1 exploratory analysis (docs/04, tasks 1.5-1.7).

    Answers the question that gates the whole study: what input resolution do
    these defects permit? Run it before fitting anything.
    """
    from .data import ensure_validation, load_category
    from .eda import analyse_category
    from .report import write_eda_report

    cfg = _resolve(args)
    data: dict = cfg["data"]
    if not data.get("root"):
        print("error: no data root. Pass --data-root or export INSPECTOR_DATA_ROOT.", file=sys.stderr)
        return 2

    categories = args.categories or [data["category"]]
    reports = []
    for category in categories:
        indices = load_category(
            data["root"], category, layout=data["layout"], split_csv=data.get("split_csv")
        )
        indices, _ = ensure_validation(
            indices,
            val_fraction=data.get("val_carve_fraction", 0.15),
            seed=data.get("val_carve_seed", 0),
        )
        print(f"analysing {category} ...", file=sys.stderr)
        reports.append(
            analyse_category(
                indices,
                test_split=data.get("test_split", "test_public"),
                candidates=tuple(args.candidates),
            )
        )

    md_path, json_path = write_eda_report(reports, args.out)
    print(f"report  -> {md_path}")
    print(f"json    -> {json_path}")
    for report in reports:
        rec = report.recommended
        print(
            f"  {report.category:14s} native={report.native_size[0]}x{report.native_size[1]} "
            f"regions={int(report.region_summary.get('n_regions', 0)):4d} "
            f"recommended={'TILING' if rec and rec.requires_tiling else (rec.long_side if rec else 'n/a')} "
            f"({rec.verdict if rec else '-'})"
        )
    return 0


def cmd_run(args) -> int:
    """Fit and evaluate models through the shared runner.

    Every (method, category, seed) becomes a fully resolved `RunSpec`; its id
    covers the effective hyperparameters, so `--methods`, `--categories` and
    `--seeds` are part of what gets hashed rather than edits to a config that
    was hashed beforehand (docs/13, F04/F05). Completed runs are reloaded.
    """
    from .models import TIER0_MODELS, TORCH_MODELS
    from .results import render_markdown, write_results
    from .runner import Registry, execute, prepare

    cfg = _resolve(args)
    data: dict = cfg["data"]
    if not data.get("root"):
        print("error: no data root. Pass --data-root or export INSPECTOR_DATA_ROOT.", file=sys.stderr)
        return 2

    known = sorted(TIER0_MODELS) + list(TORCH_MODELS)
    configured = cfg.get("model", {}).get("name")
    methods = args.methods or ([configured] if configured else list(TIER0_MODELS))
    unknown = [m for m in methods if m not in known]
    if unknown:
        print(f"error: unknown method(s) {unknown}; available: {known}", file=sys.stderr)
        return 2

    categories = args.categories or [data["category"]]
    seeds = args.seeds if args.seeds is not None else [cfg.get("run", {}).get("seed", 0)]
    registry = Registry(args.registry)

    results, failures = [], 0
    for category in categories:
        for method in methods:
            for seed in seeds:
                spec, model, indices = prepare(
                    cfg, method=method, category=category, seed=seed,
                    data_root=data["root"], role=args.role,
                )
                try:
                    result, reused = execute(
                        spec, model, indices, data_root=data["root"], registry=registry,
                        force=args.force, save_maps=not args.no_maps,
                    )
                except Exception as error:
                    failures += 1
                    print(f"  FAILED {method} {category} s{seed} [{spec.run_id}]: {error}", file=sys.stderr)
                    continue
                tag = "reused " if reused else "ran    "
                print(f"  {tag}[{spec.run_id}] " + result.summary(), file=sys.stderr)
                results.append(result)

    if args.results:
        path = write_results(registry.results(), args.results)
        print(f"\nregistry table -> {path}")
    print(render_markdown(results))
    if any(r.dirty for r in results):
        print(
            "NOTE: some runs were made from a dirty working tree; they are tagged dirty and "
            "may not enter the headline table (docs/06 section 2).",
            file=sys.stderr,
        )
    return 1 if failures else 0


def cmd_results(args) -> int:
    """Regenerate the results table from the registry — never by hand."""
    from .results import render_markdown, write_results
    from .runner import Registry

    registry = Registry(args.registry)
    results = registry.results()
    path = write_results(results, args.out)
    print(f"{len(results)} completed runs -> {path}")
    if args.markdown:
        print(render_markdown(results))
    return 0


def cmd_study(args) -> int:
    """E02 aggregation, E03 calibration and the exchangeability diagnostic,
    computed from stored predictions — no model is refitted."""
    from .analysis import render_studies, run_studies

    study = run_studies(
        args.registry,
        role=args.role,
        methods=args.methods,
        categories=args.categories,
        draws=args.draws,
        n_resamples=args.resamples,
        data_root=args.data_root,
        roi_methods=tuple(args.roi_methods),
        region_size=args.region_size,
        implementation=args.implementation,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "studies.json").write_text(json.dumps(study, indent=2, default=float), encoding="utf-8")
    (out / "studies.md").write_text(render_studies(study), encoding="utf-8")
    print(f"{len(study['groups'])} run groups -> {out / 'studies.md'}")
    return 0


def cmd_export(args) -> int:
    """Fit a model on one category and write a deployable artifact with its
    threshold frozen on that category's validation normals."""
    from .api.artifact import save_artifact
    from .evaluate import SplitPredictions
    from .models.patchcore import PatchCore
    from .runner import eval_config_from_config, prepare

    cfg = _resolve(args)
    root = cfg["data"]["root"]
    category = args.category or cfg["data"].get("category")
    spec, model, idx = prepare(cfg, method=cfg["model"]["name"], category=category,
                               seed=args.seed, data_root=root)
    if not isinstance(model, PatchCore):
        print("error: artifacts are implemented for PatchCore only", file=sys.stderr)
        return 2
    model.fit(idx["train"])
    val = SplitPredictions.from_model(model, idx["validation"])
    target = eval_config_from_config(cfg).target_fpr
    out = save_artifact(model, val.scores, args.out, target_fpr=target, category=category,
                        extra={"run_spec_id": spec.run_id})
    print(f"artifact -> {out} (threshold from {val.scores.size} validation normals)")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api.service import create_app

    uvicorn.run(create_app(args.artifact), host=args.host, port=args.port)
    return 0


def cmd_demo(args) -> int:
    from .app.demo import build_demo

    build_demo(args.artifact).launch(server_name=args.host, server_port=args.port)
    return 0


def cmd_fetch(args) -> int:
    """Download a dataset that does not require an account."""
    from .fetch import REGISTRY, fetch, verify

    if args.verify:
        status = verify(args.dataset, args.out)
        for name, ok in status.items():
            print(f"  {'OK  ' if ok else 'FAIL'} {name}")
        return 0 if all(status.values()) else 1

    spec = REGISTRY.get(args.dataset)
    if spec is None:
        print(f"error: unknown dataset {args.dataset!r}; available: {sorted(REGISTRY)}", file=sys.stderr)
        return 2

    last = [-1]

    def progress(done: int, total: int) -> None:
        pct = int(100 * done / total) if total else 0
        if pct != last[0] and pct % 5 == 0:
            last[0] = pct
            print(f"  {pct:3d}%  {done / 1e9:.2f} / {total / 1e9:.2f} GB", file=sys.stderr)

    manifest = fetch(
        args.dataset,
        args.out,
        extract=not args.no_extract,
        force=args.force,
        progress=progress,
    )
    print(f"\n{args.dataset}: {manifest.action}")
    print(f"extracted to : {manifest.extracted_to}")
    for name, digest in manifest.files.items():
        print(f"  {name:24s} sha256={digest[:16]}")
    print(f"\nlicense  : {manifest.license_note}")
    print(f"cite     : {manifest.citation}")
    return 0


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

    p_eda = sub.add_parser("eda", help="phase P1 exploratory analysis and resolution decision")
    _add_config_args(p_eda)
    p_eda.add_argument("--categories", nargs="+", default=None, help="override data.category")
    p_eda.add_argument("--candidates", nargs="+", type=int, default=[256, 320, 448, 512, 1024])
    p_eda.add_argument("--out", default="reports/eda")
    p_eda.set_defaults(func=cmd_eda)

    p_fetch = sub.add_parser("fetch", help="download a dataset that needs no account (e.g. visa)")
    p_fetch.add_argument("dataset", choices=["visa"])
    p_fetch.add_argument("--out", default="data/raw")
    p_fetch.add_argument("--no-extract", action="store_true")
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.add_argument("--verify", action="store_true", help="re-hash against the recorded manifest")
    p_fetch.set_defaults(func=cmd_fetch)

    p_run = sub.add_parser("run", help="fit and evaluate models through the shared runner")
    _add_config_args(p_run)
    p_run.add_argument("--methods", nargs="+", default=None,
                       help="model names; default = config model.name, else all of tier 0")
    p_run.add_argument("--categories", nargs="+", default=None)
    p_run.add_argument("--seeds", nargs="+", type=int, default=None)
    p_run.add_argument("--registry", default="reports/runs")
    p_run.add_argument("--results", default=None, help="also regenerate this CSV from the registry")
    p_run.add_argument("--role", default="development", choices=["development", "confirmation"])
    p_run.add_argument("--force", action="store_true", help="re-run even if completed")
    p_run.add_argument("--no-maps", action="store_true", help="do not store raw maps")
    p_run.set_defaults(func=cmd_run)

    p_res = sub.add_parser("results", help="regenerate the results table from the registry")
    p_res.add_argument("--registry", default="reports/runs")
    p_res.add_argument("--out", default="reports/results_registry.csv")
    p_res.add_argument("--markdown", action="store_true")
    p_res.set_defaults(func=cmd_results)

    p_study = sub.add_parser("study", help="E02/E03 studies on stored predictions (no refits)")
    p_study.add_argument("--registry", default="reports/runs")
    p_study.add_argument("--out", default="reports/studies")
    p_study.add_argument("--role", default="development", choices=["development", "confirmation"])
    p_study.add_argument("--methods", nargs="+", default=None)
    p_study.add_argument("--categories", nargs="+", default=None)
    p_study.add_argument("--draws", type=int, default=200, help="calibration subsets per size and seed")
    p_study.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples for E02")
    p_study.add_argument("--data-root", default=None,
                         help="dataset root; enables the model-free train-proximity diagnostic")
    p_study.add_argument("--roi-methods", nargs="*", default=[],
                         help="run the object-region study (E02b) for these methods; needs --data-root")
    p_study.add_argument("--implementation", default=None, metavar="PREFIX",
                         help="only runs whose implementation id starts with PREFIX")
    p_study.add_argument("--region-size", action="store_true",
                         help="localization by defect size (E04); needs --data-root")
    p_study.set_defaults(func=cmd_study)

    p_exp = sub.add_parser("export", help="fit one category and write a deployable artifact")
    _add_config_args(p_exp)
    p_exp.add_argument("--category", default=None)
    p_exp.add_argument("--seed", type=int, default=0)
    p_exp.add_argument("--out", required=True)
    p_exp.set_defaults(func=cmd_export)

    for name, func, port, help_text in (
        ("serve", cmd_serve, 8000, "HTTP inspection service over an artifact (FastAPI)"),
        ("demo", cmd_demo, 7860, "interactive demo over an artifact (Gradio)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--artifact", required=True)
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=port)
        p.set_defaults(func=func)

    for name, help_text in (
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
