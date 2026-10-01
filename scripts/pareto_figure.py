"""I3: accuracy against cost, from the registry (development categories).

One axis pair: macro image AUROC over pcb1 / macaroni2 / capsules against inference time per
image. Memory is not a third axis — it goes in the companion table, with peak host RSS, because a
dual encoding on one chart would be read wrong. Colours follow the validated reference palette
(fixed order); every point is also direct-labelled and marker-shaped, so identity is never colour
alone.

    python scripts/pareto_figure.py --out reports/figures
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

CATS = ("pcb1", "macaroni2", "capsules")
E04 = "e76083ff5305"   # resolution x bank policy, seed 0
E01 = "b76d4f8dd1b6"   # baseline ladder at 320 px, 3 seeds
SERIES = {  # fixed categorical order from the reference palette (validated: all checks pass)
    "PatchCore, bank 1% of patches": ("#2a78d6", "o"),
    "PatchCore, bank fixed 10k": ("#eb6834", "s"),
    "autoencoder (CAE)": ("#1baf7a", "^"),
    "linear floor (pixel PCA)": ("#eda100", "D"),
}


def load(registry: Path) -> list[dict]:
    rows = []
    for f in registry.glob("*/result.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("status") != "completed":
            continue
        spec = json.loads(f.with_name("spec.json").read_text(encoding="utf-8"))
        if spec.get("role") != "development" or "train_limit" in spec["dataset"]:
            continue
        rows.append({"spec": spec, "r": rec["result"], "impl": spec["implementation_id"][:12]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="reports/runs")
    ap.add_argument("--out", default="reports/figures")
    args = ap.parse_args()
    rows = load(Path(args.registry))

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for x in rows:
        r, spec = x["r"], x["spec"]
        if r["category"] not in CATS:
            continue
        name = spec["model"]["name"]
        if x["impl"] == E04 and name == "patchcore" and spec["seed"] == 0:
            fixed = spec["model"]["hparams"].get("bank_size") is not None
            series = "PatchCore, bank fixed 10k" if fixed else "PatchCore, bank 1% of patches"
            groups[(series, f"{spec['transform']['long_side']} px")].append(r)
        elif x["impl"] == E01 and name == "cae":
            groups[("autoencoder (CAE)", f"{spec['model']['hparams']['loss'].upper()}, 320 px")].append(r)
        elif x["impl"] == E01 and name == "pixel_pca":
            groups[("linear floor (pixel PCA)", "320 px")].append(r)

    points = []
    for (series, label), rs in groups.items():
        by_cat = defaultdict(list)
        for r in rs:
            by_cat[r["category"]].append(r)
        if set(by_cat) != set(CATS):
            continue
        auroc = statistics.mean(statistics.mean(r["image_auroc"] for r in v) for v in by_cat.values())
        ms = statistics.mean(1000 * r["predict_seconds"] / (r["n_validation"] + r["n_test"]) for r in rs)
        bank = statistics.mean(float(r["fit_extra"].get("bank_mb", 0) or 0) for r in rs)
        rss = max(float(r.get("peak_rss_fit_mb") or float("nan")) for r in rs)
        fit = statistics.mean(r["fit_seconds"] for r in rs)
        points.append({"series": series, "label": label, "auroc": auroc, "ms": ms,
                       "bank_mb": bank, "rss_mb": rss, "fit_s": fit})

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for series, (colour, marker) in SERIES.items():
        pts = sorted((p for p in points if p["series"] == series), key=lambda p: p["ms"])
        if not pts:
            continue
        # Only the resolution steps of one configuration form a sequence worth a line;
        # the CAE losses and the floor are separate points.
        joined = series.startswith("PatchCore")
        ax.plot([p["ms"] for p in pts], [p["auroc"] for p in pts], color=colour,
                lw=2 if joined else 0, marker=marker, ms=8, mec="white", mew=2, label=series)
        dy = 7 if "fixed" in series else -13  # keep the two PatchCore label rows apart
        for p in pts:
            ax.annotate(p["label"], (p["ms"], p["auroc"]), textcoords="offset points",
                        xytext=(6, dy if joined else -3), fontsize=8, color="#3d3d3a")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if v in (30, 50, 70, 150) else ""))
    ax.set_xlabel("inference time per image, ms (laptop RTX 3050; includes image decoding)", fontsize=9)
    ax.set_ylabel("macro image AUROC (3 development categories)", fontsize=9)
    ax.grid(True, color="#e6e5df", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Accuracy against cost — development categories, protocol v2", fontsize=10, loc="left")
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "pareto.png", dpi=150)

    lines = ["| series | configuration | macro image AUROC | ms / image | fit s | bank MB | peak RSS in fit MB |",
             "|---|---|---|---|---|---|---|"]
    for p in sorted(points, key=lambda p: (list(SERIES).index(p["series"]), p["ms"])):
        rss = "-" if p["rss_mb"] != p["rss_mb"] else f"{p['rss_mb']:.0f}"  # not recorded before F07
        bank = f"{p['bank_mb']:.1f}" if p["bank_mb"] else "-"
        lines.append(f"| {p['series']} | {p['label']} | {p['auroc']:.3f} | {p['ms']:.0f} | {p['fit_s']:.0f} | "
                     f"{bank} | {rss} |")
    (out / "pareto.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
