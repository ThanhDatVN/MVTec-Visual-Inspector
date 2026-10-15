"""Render the figures of `reports/00-final-report.md` from committed result files.

Every number comes from a CSV or JSON under `reports/` (or, where noted, from a table of a report
that cites its source). No dataset image, anomaly map or mask is read: the figures show metrics
only, so they carry no dataset imagery (see ATTRIBUTION.md).

    python scripts/report_figures.py            # writes reports/figures/fig*.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

R = Path("reports")
STUDIES = R / "studies"

# Reference palette (validated: first three categorical slots, all pairs, light surface).
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CONTEXT = "#b9b7ae"  # published systems: recessive, so the project's own marks carry the colour

CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def save(fig, out: Path, name: str, note: str) -> None:
    fig.text(0.01, -0.01, note, fontsize=7, color=MUTED, ha="left", va="top")
    fig.savefig(out / name, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", out / name)


def first_number(text) -> float:
    return float(str(text).split()[0])


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------

def server() -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    with open(STUDIES / "ad2-server" / "server_means.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = "private" if row["split"] == "test_private" else "mixed"
            out.setdefault(row["recipe"], {})[split] = {k: float(row[k]) for k in ("aupro_005", "segf1", "classf1")}
    return out


def published() -> pd.DataFrame:
    return pd.read_csv(STUDIES / "ad2-server" / "published_reference.csv")


def macro(path: Path, column: str, **where) -> float:
    d = pd.read_csv(path)
    for k, v in where.items():
        d = d[d[k] == v]
    assert d["category"].nunique() == 8, (path, where)
    return float(d.groupby("category")[column].mean().mean())


def development_path() -> list[dict]:
    """Macro AU-PRO@0.05 and SegF1 on `test_public`, all captures, seed 0, per AD 2 candidate."""
    s = STUDIES
    rows = [
        {"name": "v1", "label": "v1\nWRN50 640 px\n3\u03c3 masks",
         "aupro": macro(s / "ad2-server/proxy_v1-640.csv", "aupro_005"),
         "segf1": macro(s / "ad2-server/proxy_v1-640.csv", "segf1_3sigma")},
        {"name": "v3", "label": "v3\nWRN50 1280 px\n3\u03c3 masks",
         "aupro": macro(s / "ad2-server/proxy_v3-1280.csv", "aupro_005"),
         "segf1": macro(s / "ad2-server/proxy_v3-1280.csv", "segf1_3sigma")},
    ]
    v4 = {"subset": "all", "threshold": "p95x1.25", "post": "morph+mark0.9"}
    rows.append({"name": "v4", "label": "v4\n+ DINOv3-L tiles\nmask rule",
                 "aupro": macro(s / "fusion/rules_dinov3l-max-wrn1280.csv", "aupro_005", **v4),
                 "segf1": macro(s / "fusion/rules_dinov3l-max-wrn1280.csv", "segf1", **v4)})
    v5 = {"method": "full-rule", "subset": "all", "threshold": "p95x1.3", "post": "morph+mark0.9"}
    rows.append({"name": "v5", "label": "v5\nDINOv3 ViT-H+\n+ WRN50 1280",
                 "aupro": macro(s / "fusion-head/head_rules.csv", "aupro_005", **v5),
                 "segf1": macro(s / "fusion-head/head_rules.csv", "segf1", **v5)})
    hybrid = json.loads((s / "fusion-head/hybrid_check.json").read_text(encoding="utf-8"))
    rows.append({"name": "v6", "label": "v6\nlearned-head maps\nv5 masks",
                 "aupro": hybrid["macro"]["hybrid"]["all"], "segf1": rows[-1]["segf1"]})
    return rows


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------

def fig_server(out: Path) -> None:
    srv = server()
    pub = published()
    systems = [("v6 (this project)", srv["v6"]["private"]["segf1"], srv["v6"]["mixed"]["segf1"],
                srv["v6"]["private"]["aupro_005"], srv["v6"]["mixed"]["aupro_005"], BLUE),
               ("v1 (this project)", srv["v1"]["private"]["segf1"], srv["v1"]["mixed"]["segf1"],
                srv["v1"]["private"]["aupro_005"], srv["v1"]["mixed"]["aupro_005"], ORANGE)]
    short = {"PatchCore (ensemble": "PatchCore (AD 2 paper)", "RoBiS": "RoBiS", "SuperADD": "SuperADD",
             "ISVL": "ISVL", "SuperAD": "SuperAD", "EfficientAD": "EfficientAD (AD 2 paper)",
             "MSFlow": "MSFlow (AD 2 paper)"}
    for _, r in pub.iterrows():
        name = next((v for k, v in short.items() if str(r["method"]).startswith(k)), None)
        if name is None:
            continue
        systems.append((name, r["segf1_private"], r["segf1_mixed"], r["aupro_005_private"], r["aupro_005_mixed"],
                        CONTEXT))
    systems.sort(key=lambda t: t[1])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True)
    for ax, (title, i0, i1) in zip(axes, (("SegF1 (masks)", 1, 2), ("AU-PRO@0.05 (continuous maps)", 3, 4))):
        for y, t in enumerate(systems):
            a, b, color = t[i0], t[i1], t[5]
            if a is None or (isinstance(a, float) and np.isnan(a)):
                ax.text(1, y, "not reported", va="center", fontsize=7.5, color=MUTED)
                continue
            ax.plot([a, b], [y, y], color=color, lw=2, zorder=2, alpha=0.9)
            ax.scatter([a], [y], s=46, color=color, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.scatter([b], [y], s=46, facecolor=SURFACE, edgecolor=color, linewidth=1.8, zorder=3)
            if color != CONTEXT:
                ax.text(max(a, b) + 1.5, y, f"{a:.1f} / {b:.1f}", va="center", fontsize=8, color=INK)
        ax.set_title(title)
        ax.set_xlim(0, 80)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(systems)), [t[0] for t in systems])
    for lbl, t in zip(axes[0].get_yticklabels(), systems):
        if t[5] != CONTEXT:
            lbl.set_fontweight("semibold")
            lbl.set_color(INK)
    handles = [Line2D([], [], marker="o", ls="", color=INK2, markersize=7, label="test_private (training lighting)"),
               Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, markersize=7,
                      label="test_private_mixed (changed lighting)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.04), fontsize=8)
    fig.suptitle("MVTec AD 2 held-out splits, scored by MVTec's server (means over 8 categories)",
                 x=0.01, ha="left", y=1.10, fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, out, "fig1_server_comparison.png",
         "Published numbers as reported by their authors (reports/studies/ad2-server/published_reference.csv); "
         "the line joins a system's two splits.")


def fig_development(out: Path) -> None:
    rows = development_path()
    srv = server()
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for key, color, label, dy in (("aupro", BLUE, "AU-PRO@0.05, test_public", -8),
                                  ("segf1", ORANGE, "SegF1, test_public", 8)):
        y = [r[key] for r in rows]
        ax.plot(x, y, color=color, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=label, zorder=3)
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(-9, dy), textcoords="offset points", va="center",
                    ha="right", fontsize=8, color=INK)
        ax.annotate(f"{y[0]:.3f}", (x[0], y[0]), xytext=(-8, 0), textcoords="offset points", va="center",
                    ha="right", fontsize=8, color=INK)
    for name, i in (("v1", 0), ("v6", len(rows) - 1)):
        for metric, color in (("aupro_005", BLUE), ("segf1", ORANGE)):
            v = srv[name]["mixed"][metric] / 100
            ax.scatter([i + 0.22], [v], marker="D", s=36, facecolor=SURFACE, edgecolor=color, linewidth=1.6, zorder=4)
            ax.annotate(f"{v:.3f}", (i + 0.22, v), xytext=(7, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=INK2)
    ax.set_xticks(x, [r["label"] for r in rows], fontsize=8)
    ax.set_ylim(0, 0.75)
    ax.set_xlim(-0.6, len(rows) - 0.2)
    ax.set_ylabel("macro over 8 categories")
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="D", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, markersize=6))
    labels.append("server, test_private_mixed (v1 and v6 only)")
    ax.legend(handles, labels, loc="upper left", fontsize=8)
    ax.set_title("From v1 to v6 on MVTec AD 2: development estimate (test_public, all captures, seed 0)")
    fig.tight_layout()
    save(fig, out, "fig2_ad2_development_path.png",
         "Diamonds: the server's test_private_mixed score. v2 (selected on VisA) is omitted: E13 tested it on "
         "detection and it was not confirmed. v6 keeps v5's masks, so its SegF1 is v5's.")


def fig_head_gain(out: Path) -> None:
    b = json.loads((STUDIES / "fusion-head" / "hybrid_bootstrap.json").read_text(encoding="utf-8"))
    cats = sorted(CATEGORIES, key=lambda c: b["categories"][c]["all"]["diff"])
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for off, key, color, label in ((-0.14, "regular", BLUE, "regular captures (≈ test_private)"),
                                   (0.14, "all", ORANGE, "all captures (≈ test_private_mixed)")):
        for i, c in enumerate(cats):
            e = b["categories"][c][key]
            lo, hi = e["ci95"]
            ax.plot([lo * 100, hi * 100], [i + off] * 2, color=color, lw=2, alpha=0.55, zorder=2)
            ax.scatter([e["diff"] * 100], [i + off], s=42, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=label if i == 0 else None)
    m = b["macro"]
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    ax.set_yticks(range(len(cats)), cats)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("v6 \u2212 v5 AU-PRO@0.05, points (95% object-bootstrap interval)")
    ax.set_title(f"What the learned head adds to the maps, per category (macro +{m['regular']['diff'] * 100:.1f} / "
                 f"+{m['all']['diff'] * 100:.1f})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig3_head_gain_by_category.png",
         "test_public, seed 0; objects resampled within categories, 2,000 resamples "
         "(reports/studies/fusion-head/hybrid_bootstrap.json).")


def fig_proxy(out: Path) -> None:
    srv = server()
    # v1 proxy: E21 §4 table (scripts/server_proxy.py on v1's stored maps, regular ↔ private, all ↔ mixed).
    proxy = {"v1": {"aupro_005": (36.4, 35.1), "segf1": (23.2, 20.5), "classf1": (77.0, 77.7)}}
    hybrid = json.loads((STUDIES / "fusion-head" / "hybrid_check.json").read_text(encoding="utf-8"))["macro"]["hybrid"]
    rules = pd.read_csv(STUDIES / "fusion-head" / "head_rules.csv")
    rules = rules[(rules.method == "full-rule") & (rules.threshold == "p95x1.3") & (rules.post == "morph+mark0.9")]
    by = rules.groupby("subset")[["segf1", "classf1"]].mean() * 100
    proxy["v6"] = {"aupro_005": (hybrid["regular"] * 100, hybrid["all"] * 100),
                   "segf1": (by.loc["regular", "segf1"], by.loc["all", "segf1"]),
                   "classf1": (by.loc["regular", "classf1"], by.loc["all", "classf1"])}
    marker = {"aupro_005": "o", "segf1": "s", "classf1": "^"}
    name = {"aupro_005": "AU-PRO@0.05", "segf1": "SegF1", "classf1": "ClassF1"}
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([15, 90], [15, 90], color=AXIS, lw=1, zorder=1)
    ax.text(70, 72.5, "server = proxy", fontsize=7.5, color=MUTED, rotation=45, ha="center")
    for sub, color in (("v1", ORANGE), ("v6", BLUE)):
        for metric, (p_priv, p_mix) in proxy[sub].items():
            for p, split in ((p_priv, "private"), (p_mix, "mixed")):
                s = srv[sub][split][metric]
                ax.scatter([p], [s], marker=marker[metric], s=50, color=color, edgecolor=SURFACE, linewidth=1.2,
                           zorder=3)
            if sub == "v6" and metric == "segf1":
                ax.annotate("v6 SegF1: mask rule chosen on\ntest_public, 8\u20139 points optimistic",
                            (p_mix, srv[sub]["mixed"][metric]), xytext=(20, -55), textcoords="offset points",
                            fontsize=7.5, color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_xlim(15, 90)
    ax.set_ylim(15, 90)
    ax.set_aspect("equal")
    ax.set_xlabel("development proxy on test_public (regular ↔ private, all ↔ mixed)")
    ax.set_ylabel("MVTec server")
    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=lab)
               for c, lab in ((ORANGE, "v1 (S1)"), (BLUE, "v6 (S2)"))]
    handles += [Line2D([], [], marker=marker[m], ls="", color=INK2, markersize=6, label=name[m]) for m in marker]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_title("Did test_public predict the server?")
    fig.tight_layout()
    save(fig, out, "fig4_proxy_vs_server.png", "Each submission contributes two points per metric (two splits).")


def fig_datasets(out: Path) -> None:
    visa = json.loads((STUDIES / "confirmation-v1" / "confirmation.json").read_text(encoding="utf-8"))["descriptive"]
    ad2 = json.loads((STUDIES / "ad2-confirmation" / "confirmation.json").read_text(encoding="utf-8"))["descriptive"]
    mpdd = pd.read_csv(STUDIES / "kaggle" / "mpdd.csv")
    groups = [("VisA, 9 held-out\ncategories (E12)", {r["category"]: first_number(r["auroc_640"]) for r in visa}),
              ("MPDD, 6 categories\n(H1, transfer)", dict(zip(mpdd.category, mpdd.image_auroc_mean))),
              ("MVTec AD 2 test_public,\n8 categories (E13)", {r["category"]: first_number(r["auroc_v1"]) for r in ad2})]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    rng = np.random.default_rng(0)
    for i, (_label, vals) in enumerate(groups):
        v = np.array(list(vals.values()))
        jitter = rng.uniform(-0.12, 0.12, len(v))
        ax.scatter(i + jitter, v, s=40, color=BLUE, edgecolor=SURFACE, linewidth=1.3, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=INK, lw=2, zorder=4)
        ax.text(i + 0.3, v.mean(), f"macro {v.mean():.3f}", va="center", fontsize=8, color=INK)
        lo = min(vals, key=vals.get)
        ax.annotate(lo, (i + jitter[list(vals).index(lo)], vals[lo]), xytext=(6, -10), textcoords="offset points",
                    fontsize=7.5, color=INK2)
    ax.axhline(0.5, color=AXIS, lw=1, zorder=1)
    ax.text(2.45, 0.505, "chance", fontsize=7.5, color=MUTED, ha="right", va="bottom")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("image AUROC (per category, mean of 3 seeds)")
    ax.grid(axis="x", visible=False)
    ax.set_title("The frozen v1 recipe (PatchCore, 640 px) on data that shaped nothing")
    fig.tight_layout()
    save(fig, out, "fig5_heldout_by_dataset.png",
         "Each dot is one category; black bar = macro mean. Recipe frozen on 3 VisA development categories (ADR-11).")


def fig_robustness(out: Path) -> None:
    d = pd.read_csv(STUDIES / "e08" / "robustness.csv")
    clean = d[d.corruption == "clean"].set_index("category")
    names = {"gaussian_blur": "Gaussian blur", "defocus_blur": "defocus blur", "exposure_up": "over-exposure",
             "exposure_down": "under-exposure", "spatial_light": "illumination falloff",
             "resize_roundtrip": "resize round-trip", "noise_jpeg": "sensor noise + JPEG", "translate": "translation"}
    colors = {"pcb1": BLUE, "macaroni2": ORANGE, "capsules": AQUA}
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.8), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.flat, names.items()):
        sub = d[d.corruption == key]
        for cat, color in colors.items():
            s = sub[sub.category == cat].sort_values("severity")
            xs = [0, *s.severity.tolist()]
            ys = [clean.loc[cat, "fpr_at_op1"] * 100, *(s.fpr_at_op1 * 100).tolist()]
            ax.plot(xs, ys, color=color, marker="o", markersize=4, label=cat)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-4, 104)
        ax.set_xticks(range(6))
    for ax in axes[:, 0]:
        ax.set_ylabel("false alarms at the\nfrozen threshold, %")
    for ax in axes[1]:
        ax.set_xlabel("severity (0 = clean)")
    handles = [Line2D([], [], color=c, marker="o", markersize=5, label=k) for k, c in colors.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05), fontsize=8)
    fig.suptitle("E08: false alarms at the frozen threshold under eight corruptions (VisA development categories)",
                 x=0.01, ha="left", y=1.11, fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, out, "fig6_robustness_false_alarms.png",
         "Threshold frozen on clean validation normals; requested rate 1%. Over the same cells image AUROC moves far less "
         "(reports/05-robustness.md): the ranking outlives the operating point.")


def fig_data_efficiency(out: Path) -> None:
    d = pd.read_csv(STUDIES / "e09" / "data_efficiency.csv")
    colors = {"pcb1": BLUE, "macaroni2": ORANGE, "capsules": AQUA}
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for cat, color in colors.items():
        s = d[d.category == cat].groupby("n_train")["image_auroc"].agg(["mean", "std"]).reset_index()
        ax.plot(s.n_train, s["mean"], color=color, marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=cat)
        ax.fill_between(s.n_train, s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0), color=color,
                        alpha=0.12, linewidth=0)
        ax.annotate(cat, (s.n_train.iloc[-1], s["mean"].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 200, 500], ["10", "25", "50", "100", "200", "500"])
    ax.set_xlabel("normal images used to fit the memory bank (log scale; last point = full training set)")
    ax.set_ylabel("image AUROC (mean ± SD, 3 subset seeds)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("E09: how many normal images the recipe needs (640 px, fixed 10k bank)")
    fig.tight_layout()
    save(fig, out, "fig7_data_efficiency.png", "Calibration normals and test set fixed in every run.")


def fig_threshold(out: Path) -> None:
    d = pd.read_csv(STUDIES / "fusion" / "rules_gains-hplus-max-wrn1280.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, (metric, title) in zip(axes, (("segf1", "SegF1 against the threshold gain"),
                                          ("classf1", "ClassF1: the image mark decouples it"))):
        for subset, color, label in (("regular", BLUE, "regular captures"), ("all", ORANGE, "all captures")):
            for post, ls, suffix in (("morph+mark0.9", "-", ""), ("morph", ":", ", no mark")):
                if metric == "segf1" and post == "morph":
                    continue
                s = d[(d.subset == subset) & (d.post == post)].copy()
                s["gain"] = s.threshold.str.replace("p95x", "").astype(float)
                g = s.groupby("gain")[metric].mean()
                ax.plot(g.index, g.values, color=color, ls=ls, marker="o" if ls == "-" else None, markersize=4,
                        label=label + suffix)
        ax.axvline(1.3, color=INK2, lw=1, zorder=1)
        ax.text(1.31, 0.93, "chosen: P95 \u00d7 1.3", fontsize=7.5, color=INK2)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("threshold = gain \u00d7 95th percentile of validation (normal) pixels")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left" if metric == "segf1" else "lower right", fontsize=7.5)
    axes[0].set_ylabel("macro over 8 categories (test_public)")
    fig.suptitle("v5 / v6 masks: one global threshold rule, chosen on test_public (development)",
                 x=0.01, ha="left", y=1.03, fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, out, "fig8_threshold_gain.png",
         "Fused DINOv3 ViT-H+ / WRN50 maps; MaskRule with line closing. Per-category gains lost out of sample (E21b).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    for fig in (fig_server, fig_development, fig_head_gain, fig_proxy, fig_datasets, fig_robustness,
                fig_data_efficiency, fig_threshold):
        fig(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
