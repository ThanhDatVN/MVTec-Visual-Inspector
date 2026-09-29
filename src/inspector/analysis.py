"""Studies that run on stored predictions, without refitting (docs/13, E02/E03).

Both studies hold the fitted model fixed and change exactly one thing, which is
what makes their conclusions attributable:

* **E02 — aggregation.** The anomaly maps stay fixed; only the rule that turns
  a map into one image score changes. Localization metrics must therefore not
  move at all — that is the negative control, and the evaluator tests assert it.
* **E03 — calibration.** The scores stay fixed; only the calibration sample and
  the thresholding rule change. Repeated sub-sampling of the calibration pool
  shows how much of the operating point is the rule and how much is the draw.

Plus one diagnostic the first corrected run made necessary: a rank rule's
false-alarm bound assumes calibration and test normals are exchangeable. If
test normals systematically score higher than validation normals, the realized
FPR exceeds the bound for a reason no threshold rule can fix, and that is worth
measuring directly rather than inferring from one number.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

from .evaluate import SplitPredictions
from .metrics.ranking import average_precision, roc_auc
from .postproc.thresholds import conformal_rank
from .stats import binomial_upper_bound
from .utils.hashing import hash_object

# ---------------------------------------------------------------------------
# Aggregation rules (E02)
# ---------------------------------------------------------------------------

#: Domain of every rule: the model's **raw** map at its own resolution, before
#: smoothing or upsampling. Stated because the answer depends on it — a top-1%
#: rule over a 40x30 PatchCore grid averages 12 patches, over a 1.5 MP native
#: map it averages 15,000 pixels.
AGGREGATION_DOMAIN = "raw model-resolution map, before smoothing"


def _top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    flat = np.sort(values.ravel())[::-1]
    k = max(1, round(fraction * flat.size))
    return float(flat[:k].mean())


AGGREGATIONS: dict[str, Callable[[np.ndarray], float]] = {
    "max": lambda m: float(m.max()),
    "mean": lambda m: float(m.mean()),
    "top0.1%": lambda m: _top_fraction_mean(m, 0.001),
    "top1%": lambda m: _top_fraction_mean(m, 0.01),
    "top5%": lambda m: _top_fraction_mean(m, 0.05),
}


def aggregate(preds: SplitPredictions, rule: str) -> np.ndarray:
    """Image scores from stored raw maps under one rule; `stored` keeps the
    model's own image score (e.g. PatchCore's re-weighted maximum)."""
    if rule == "stored":
        return preds.scores.copy()
    if rule not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {rule!r}; known: stored, {sorted(AGGREGATIONS)}")
    if not preds.raw_maps:
        raise ValueError("predictions were stored without maps; aggregation needs them")
    fn = AGGREGATIONS[rule]
    return np.asarray([fn(m) for m in preds.raw_maps], dtype=np.float64)


# ---------------------------------------------------------------------------
# Operating points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatingPoint:
    rule: str
    requested_fpr: float
    effective_fpr: float
    target_met: bool
    threshold: float
    n_calibration: int
    test_fp: int
    test_normals: int
    test_tp: int
    test_anomalies: int

    @property
    def fpr(self) -> float:
        return self.test_fp / self.test_normals if self.test_normals else float("nan")

    @property
    def recall(self) -> float:
        return self.test_tp / self.test_anomalies if self.test_anomalies else float("nan")

    @property
    def fpr_upper95(self) -> float:
        """Clopper-Pearson upper bound for the realized test FPR."""
        return binomial_upper_bound(self.test_fp, self.test_normals)

    def as_dict(self) -> dict:
        return {**asdict(self), "fpr": self.fpr, "recall": self.recall, "fpr_upper95": self.fpr_upper95}


def rank_threshold(calibration: np.ndarray, target: float) -> tuple[float, float, bool]:
    """Conservative rank threshold with the `relax` policy; (value, effective, met)."""
    cal = np.sort(np.asarray(calibration, dtype=np.float64))[::-1]
    n = cal.size
    k = conformal_rank(n, target)
    met = k >= 1
    k = max(k, 1)
    return float(cal[k - 1]), k / (n + 1), met


def sigma_threshold(calibration: np.ndarray, n_sigma: float = 3.0) -> float:
    """Mean + n*std of image scores: the heuristic applied to image scores."""
    cal = np.asarray(calibration, dtype=np.float64)
    return float(cal.mean() + n_sigma * cal.std(ddof=1))


def mad_threshold(calibration: np.ndarray, n_sigma: float = 3.0) -> float:
    """Median + n * 1.4826 * MAD: a robust version of the sigma rule."""
    cal = np.asarray(calibration, dtype=np.float64)
    med = float(np.median(cal))
    return med + n_sigma * 1.4826 * float(np.median(np.abs(cal - med)))


def operating_point(
    rule: str,
    threshold: float,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    *,
    requested: float = float("nan"),
    effective: float = float("nan"),
    met: bool = False,
    n_calibration: int = 0,
) -> OperatingPoint:
    flagged = np.asarray(test_scores) > threshold
    labels = np.asarray(test_labels)
    return OperatingPoint(
        rule=rule,
        requested_fpr=requested,
        effective_fpr=effective,
        target_met=met,
        threshold=float(threshold),
        n_calibration=int(n_calibration),
        test_fp=int(np.sum(flagged & (labels == 0))),
        test_normals=int(np.sum(labels == 0)),
        test_tp=int(np.sum(flagged & (labels == 1))),
        test_anomalies=int(np.sum(labels == 1)),
    )


# ---------------------------------------------------------------------------
# The exchangeability diagnostic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalShift:
    """Do test normals look like calibration normals to the model?"""

    auroc: float  # P(test normal scores above a validation normal); 0.5 = exchangeable
    mannwhitney_p: float
    n_validation: int
    n_test_normals: int
    median_val: float
    median_test: float

    def as_dict(self) -> dict:
        return asdict(self)


def normal_shift(val_scores: np.ndarray, test: SplitPredictions) -> NormalShift:
    """AUROC between validation normals and test normals, and its two-sided test.

    Under exchangeability — the assumption the conservative threshold's bound
    rests on — this is 0.5. Above 0.5 means test normals score higher than the
    normals the threshold was calibrated on, so the realized FPR will exceed the
    bound no matter which rank rule is used.
    """
    val = np.asarray(val_scores, dtype=np.float64)
    test_normal = test.scores[test.labels == 0]
    labels = np.concatenate([np.zeros(val.size, int), np.ones(test_normal.size, int)])
    scores = np.concatenate([val, test_normal])
    p = float(sps.mannwhitneyu(test_normal, val, alternative="two-sided").pvalue)
    return NormalShift(
        auroc=roc_auc(labels, scores),
        mannwhitney_p=p,
        n_validation=int(val.size),
        n_test_normals=int(test_normal.size),
        median_val=float(np.median(val)),
        median_test=float(np.median(test_normal)),
    )


def calibration_source_study(
    val: SplitPredictions,
    test: SplitPredictions,
    *,
    targets: tuple[float, ...] = (0.02, 0.05, 0.10),
    draws: int = 500,
    seed: int = 0,
) -> list[dict]:
    """Does the *source* of the calibration normals explain an excess FPR?

    Each draw splits the test normals at random into halves A and B, and draws
    a validation subset V of the same size n. Both A and V set a rank-k
    threshold with the same n and k; both are evaluated on B.

    Threshold-from-A is the positive control: A and B are exchangeable by
    construction, so its expected FPR on B is exactly k/(n+1) whatever the
    model. If threshold-from-V overshoots while A hits the bound, the excess
    comes from validation normals not being exchangeable with test normals —
    not from the rank rule, and not from sampling noise.

    A diagnostic, not an operating procedure: calibrating on test normals
    spends test data and is never used for a reported operating point.
    """
    rng = np.random.default_rng(seed)
    normals = test.scores[test.labels == 0]
    anomalies = test.scores[test.labels == 1]
    n = normals.size // 2
    if n < 10 or val.scores.size < n:
        return []
    rows = []
    for target in targets:
        k = conformal_rank(n, target)
        if k < 1:
            continue
        fpr: dict[str, list[float]] = {"validation": [], "test half": []}
        rec: dict[str, list[float]] = {"validation": [], "test half": []}
        for _ in range(draws):
            perm = rng.permutation(normals.size)
            half_a, half_b = normals[perm[:n]], normals[perm[n:]]
            subset_v = rng.choice(val.scores, size=n, replace=False)
            for source, cal in (("validation", subset_v), ("test half", half_a)):
                thr = np.sort(cal)[::-1][k - 1]
                fpr[source].append(float(np.mean(half_b > thr)))
                rec[source].append(float(np.mean(anomalies > thr)) if anomalies.size else float("nan"))
        for source in fpr:
            f = np.asarray(fpr[source])
            rows.append({
                "source": source, "requested_fpr": target, "n_calibration": n, "rank_k": k,
                "bound": k / (n + 1), "fpr_mean": float(f.mean()),
                "fpr_mc_se": float(f.std(ddof=1) / np.sqrt(f.size)),
                "fpr_p05": float(np.quantile(f, 0.05)), "fpr_p95": float(np.quantile(f, 0.95)),
                "recall_mean": float(np.mean(rec[source])), "draws": draws,
            })
    return rows


def train_proximity(indices: dict, *, test_split: str, size: int = 64) -> dict[str, Any]:
    """How close is each calibration normal, and each test normal, to its
    nearest *training* image — in plain pixels, independent of any model?

    Validation carved from the training pool may share capture sessions and
    near-identical frames with the images the model was fitted on. A
    nearest-neighbour model then scores validation normals lower than fresh
    normals, and a threshold calibrated on them is too low. Thumbnails are
    grey, `size` x `size`, standardised per image; distance is Euclidean.
    `auroc` = P(a test normal is farther from train than a validation image).
    """
    from PIL import Image

    from .data.core import NORMAL

    def thumbs(samples) -> np.ndarray:
        out = []
        for s in samples:
            img = Image.open(s.image_path).convert("L").resize((size, size), Image.Resampling.BOX)
            a = np.asarray(img, dtype=np.float32).ravel()
            out.append((a - a.mean()) / (a.std() + 1e-6))
        return np.stack(out)

    train = thumbs(indices["train"])
    train_sq = (train * train).sum(axis=1)

    def nearest(samples) -> np.ndarray:
        x = thumbs(samples)
        d2 = (x * x).sum(axis=1)[:, None] - 2.0 * x @ train.T + train_sq[None, :]
        return np.sqrt(np.clip(d2.min(axis=1), 0.0, None))

    val_d = nearest(list(indices["validation"]))
    test_d = nearest([s for s in indices[test_split] if s.label == NORMAL])
    labels = np.r_[np.zeros(val_d.size, int), np.ones(test_d.size, int)]
    return {
        "n_validation": int(val_d.size),
        "n_test_normals": int(test_d.size),
        "median_val": float(np.median(val_d)),
        "median_test": float(np.median(test_d)),
        "auroc": roc_auc(labels, np.r_[val_d, test_d]),
        "mannwhitney_p": float(sps.mannwhitneyu(test_d, val_d, alternative="two-sided").pvalue),
        "thumbnail_size": size,
    }


def exceedance_p_value(fp: int, n_test: int, n_calibration: int, k: int) -> float:
    """Exact permutation p-value for `fp` test normals above the k-th largest
    of `n_calibration` calibration normals, if all normals are exchangeable.

    Under exchangeability every ordering of the pooled normals is equally
    likely, and at least `fp` test normals exceed the threshold iff the top
    `fp + k - 1` pooled scores contain at least `fp` test normals — a
    hypergeometric tail.

    A binomial test at the effective rate is the wrong reference here: the
    bound k/(n+1) is marginal over calibration draws, and conditional on one
    calibration pool the false-alarm rate is itself random. The binomial
    version ignores that and overstates the evidence — the random-score
    control showed it.
    """
    if n_test <= 0 or fp <= 0:
        return 1.0
    return float(sps.hypergeom.sf(fp - 1, n_calibration + n_test, n_test, fp + k - 1))


# ---------------------------------------------------------------------------
# E02 driver
# ---------------------------------------------------------------------------


def aggregation_study(
    runs: list[tuple[SplitPredictions, SplitPredictions]],
    *,
    rules: tuple[str, ...] = ("stored", "max", "top0.1%", "top1%", "top5%", "mean"),
    target_fpr: float = 0.01,
    reference: str = "stored",
    n_resamples: int = 1000,
    seed: int = 0,
    extra: dict[str, list[tuple[np.ndarray, np.ndarray]]] | None = None,
) -> dict[str, dict]:
    """Compare aggregation rules on the same maps, pooled over fitting seeds.

    `runs` holds (validation, test) predictions for several seeds of one
    method and category; the test images are identical across seeds. The
    bootstrap resamples *images* (stratified by class) and averages the paired
    AUROC difference over seeds in each resample, so its interval covers
    evaluation-sample uncertainty for the seed-averaged effect. Seed spread is
    reported separately.

    `extra` adds rules computed elsewhere — (validation, test) image scores
    per seed, in the same order as `runs` — such as region-restricted scores.
    """
    if not runs:
        raise ValueError("no runs")
    labels = runs[0][1].labels
    for _, test in runs:
        if not np.array_equal(test.labels, labels):
            raise ValueError("runs must share the same test images in the same order")

    scored = {
        rule: [(aggregate(val, rule), aggregate(test, rule)) for val, test in runs] for rule in rules
    }
    for rule, per_seed in (extra or {}).items():
        if len(per_seed) != len(runs):
            raise ValueError(f"extra rule {rule!r} has {len(per_seed)} seeds, runs have {len(runs)}")
        scored[rule] = [(np.asarray(v, dtype=np.float64), np.asarray(t, dtype=np.float64)) for v, t in per_seed]
    out: dict[str, dict] = {}
    for rule, per_seed in scored.items():
        aurocs = [roc_auc(labels, t) for _, t in per_seed]
        aps = [average_precision(labels, t) for _, t in per_seed]
        points = []
        for v, t in per_seed:
            thr, eff, met = rank_threshold(v, target_fpr)
            points.append(operating_point(rule, thr, t, labels, requested=target_fpr,
                                          effective=eff, met=met, n_calibration=v.size))
        k = conformal_rank(per_seed[0][0].size, target_fpr) or 1
        out[rule] = {
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else float("nan"),
            "ap_mean": float(np.mean(aps)),
            "recall_mean": float(np.mean([p.recall for p in points])),
            "fpr_mean": float(np.mean([p.fpr for p in points])),
            "exceed_p_median": float(np.median([
                exceedance_p_value(p.test_fp, p.test_normals, v.size, k)
                for p, (v, _) in zip(points, per_seed)
            ])),
            "n_seeds": len(per_seed),
        }

    # Paired, class-stratified bootstrap of the seed-averaged AUROC difference.
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    ref = scored[reference]
    for rule in scored:
        if rule == reference:
            continue
        cand = scored[rule]
        diffs = np.empty(n_resamples)
        for i in range(n_resamples):
            idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
            y = labels[idx]
            diffs[i] = np.mean([
                roc_auc(y, c[1][idx]) - roc_auc(y, r[1][idx]) for c, r in zip(cand, ref)
            ])
        estimate = out[rule]["auroc_mean"] - out[reference]["auroc_mean"]
        out[rule].update({
            "d_auroc_vs_ref": float(estimate),
            "d_auroc_ci_low": float(np.quantile(diffs, 0.025)),
            "d_auroc_ci_high": float(np.quantile(diffs, 0.975)),
        })
    return out


# ---------------------------------------------------------------------------
# E03 driver
# ---------------------------------------------------------------------------


def calibration_study(
    val: SplitPredictions,
    test: SplitPredictions,
    *,
    sizes: tuple[int, ...] = (20, 40, 80),
    targets: tuple[float, ...] = (0.01, 0.02, 0.05),
    draws: int = 200,
    seed: int = 0,
) -> list[dict]:
    """Operating-point variability as a function of calibration-set size.

    For each size, draws `draws` subsets of the calibration pool without
    replacement (the full pool is included once, as `n = all`), applies each
    rule, and records the realized test FPR and recall. The model and its
    scores never change; only the calibration sample and the rule do.
    """
    rng = np.random.default_rng(seed)
    pool = val.scores
    rows: list[dict] = []
    plan = [(n, draws) for n in sizes if n < pool.size] + [(pool.size, 1)]

    for n, n_draws in plan:
        for d in range(n_draws):
            cal = pool if n == pool.size else rng.choice(pool, size=n, replace=False)
            for target in targets:
                thr, eff, met = rank_threshold(cal, target)
                op = operating_point("rank", thr, test.scores, test.labels,
                                     requested=target, effective=eff, met=met, n_calibration=n)
                rows.append({**op.as_dict(), "draw": d, "size": n})
            for rule, thr in (("sigma3", sigma_threshold(cal)), ("mad3", mad_threshold(cal))):
                op = operating_point(rule, thr, test.scores, test.labels, n_calibration=n)
                rows.append({**op.as_dict(), "draw": d, "size": n})
    return rows


def summarize_calibration(rows: list[dict]) -> list[dict]:
    """Median and 5-95% range of realized FPR and recall per (rule, target, size)."""
    # Rules without a requested rate are keyed on None, never NaN: NaN hashes by
    # object identity, so a fresh NaN per row would give every draw its own group.
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["rule"], r["requested_fpr"] if r["rule"] == "rank" else None, r["size"])
        groups.setdefault(key, []).append(r)
    out = []
    for (rule, target, size), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0.0, kv[0][2])):
        fpr = np.array([i["fpr"] for i in items])
        rec = np.array([i["recall"] for i in items])
        out.append({
            "rule": rule,
            "requested_fpr": float("nan") if target is None else target,
            "size": size,
            "draws": len(items),
            "target_met_share": float(np.mean([i["target_met"] for i in items])),
            "effective_fpr": float(np.median([i["effective_fpr"] for i in items])),
            "fpr_median": float(np.median(fpr)),
            "fpr_p05": float(np.quantile(fpr, 0.05)),
            "fpr_p95": float(np.quantile(fpr, 0.95)),
            "recall_median": float(np.median(rec)),
            "recall_p05": float(np.quantile(rec, 0.05)),
            "recall_p95": float(np.quantile(rec, 0.95)),
        })
    return out


# ---------------------------------------------------------------------------
# Registry driver: every study over every group of seeds
# ---------------------------------------------------------------------------


@dataclass
class RunGroup:
    """Runs that differ only in their fitting seed, so they can be pooled."""

    key: str
    category: str
    method: str
    label: str
    test_split: str
    run_ids: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    dataset: dict[str, Any] = field(default_factory=dict)


def group_runs(registry_root: str | Path, *, role: str = "development") -> list[RunGroup]:
    """Group completed registry runs by everything except the seed.

    The implementation id is part of the key: runs produced by different code
    are never pooled, even when their configuration is identical.
    """
    root = Path(registry_root)
    groups: dict[str, RunGroup] = {}
    hparams: dict[str, dict] = {}
    for result_file in sorted(root.glob("*/result.json")):
        record = json.loads(result_file.read_text(encoding="utf-8"))
        spec_file = result_file.with_name("spec.json")
        if record.get("status") != "completed" or not spec_file.is_file():
            continue
        spec = json.loads(spec_file.read_text(encoding="utf-8"))
        if spec.get("role", "development") != role:
            continue
        # The seed appears twice: in the spec and in the model's hyperparameters.
        model_hp = {k: v for k, v in spec["model"].get("hparams", {}).items() if k != "seed"}
        unseeded = {k: v for k, v in spec.items() if k != "seed"}
        unseeded["model"] = {**spec["model"], "hparams": model_hp}
        key = hash_object(unseeded)[:12]
        category = spec["dataset"]["category"]
        method = spec["model"]["name"]
        if key not in groups:
            groups[key] = RunGroup(key, category, method, method, spec["dataset"]["test_split"],
                                   dataset=spec["dataset"])
            hparams[key] = {**model_hp, "long_side": spec["transform"].get("long_side")}
        groups[key].run_ids.append(record["run_id"])
        groups[key].seeds.append(int(spec["seed"]))

    # Label each group by the hyperparameters that distinguish it from other
    # groups of the same method and category — "cae loss=ssim", not a hash.
    for g in groups.values():
        siblings = [k for k, o in groups.items() if (o.method, o.category) == (g.method, g.category)]
        if len(siblings) > 1:
            names = sorted({n for k in siblings for n in hparams[k]})
            differing = [n for n in names if len({json.dumps(hparams[k].get(n), default=str) for k in siblings}) > 1]
            own = hparams[g.key]
            g.label = g.method + "".join(f" {n}={own.get(n)}" for n in differing)
    return sorted(groups.values(), key=lambda g: (g.category, g.method, g.label))


def run_studies(
    registry_root: str | Path,
    *,
    role: str = "development",
    methods: list[str] | None = None,
    categories: list[str] | None = None,
    target_fpr: float = 0.01,
    sizes: tuple[int, ...] = (20, 40, 80),
    targets: tuple[float, ...] = (0.01, 0.02, 0.05),
    draws: int = 200,
    n_resamples: int = 1000,
    seed: int = 0,
    data_root: str | Path | None = None,
    roi_methods: tuple[str, ...] = (),
) -> dict[str, Any]:
    """E02, E03 and the exchangeability diagnostics over a registry.

    With `data_root`, also measures each category's pixel-space proximity of
    validation and test normals to the training images (`train_proximity`),
    and runs the object-region study (E02b) for the methods in `roi_methods`.
    """
    root = Path(registry_root)
    out: dict[str, Any] = {
        "aggregation": [], "calibration": [], "normal_shift": [], "calibration_source": [],
        "train_proximity": [], "roi": [], "groups": [],
    }
    proximity_done: set[str] = set()
    for g in group_runs(root, role=role):
        if (methods and g.method not in methods) or (categories and g.category not in categories):
            continue
        if data_root is not None and g.category not in proximity_done:
            out["train_proximity"].append({"category": g.category, **_proximity_for(g, data_root)})
            proximity_done.add(g.category)
        runs = [
            (SplitPredictions.load(root / rid, "validation"), SplitPredictions.load(root / rid, g.test_split))
            for rid in g.run_ids
        ]
        out["groups"].append(asdict(g))
        base = {"category": g.category, "method": g.method, "label": g.label, "group": g.key}

        if all(val.raw_maps and test.raw_maps for val, test in runs):
            agg = aggregation_study(runs, target_fpr=target_fpr, n_resamples=n_resamples, seed=seed)
            out["aggregation"] += [{**base, "rule": rule, **vals} for rule, vals in agg.items()]

        cal_rows: list[dict] = []
        for i, (val, test) in enumerate(runs):
            cal_rows += calibration_study(val, test, sizes=sizes, targets=targets, draws=draws, seed=seed + i)
            shift = normal_shift(val.scores, test)
            thr, eff, met = rank_threshold(val.scores, target_fpr)
            op = operating_point("rank", thr, test.scores, test.labels, requested=target_fpr,
                                 effective=eff, met=met, n_calibration=val.scores.size)
            out["normal_shift"].append({
                **base, "run_id": g.run_ids[i], "seed": g.seeds[i], **shift.as_dict(),
                "effective_fpr": eff, "test_fp": op.test_fp, "test_normals": op.test_normals,
                "realized_fpr": op.fpr, "fpr_upper95": op.fpr_upper95, "recall": op.recall,
                "flagged_normals": [
                    i for i, s, y in zip(test.ids, test.scores, test.labels) if y == 0 and s > thr
                ],
                "exceedance_p": exceedance_p_value(
                    op.test_fp, op.test_normals, int(val.scores.size), conformal_rank(val.scores.size, target_fpr) or 1
                ),
            })
            out["calibration_source"] += [
                {**base, "seed": g.seeds[i], **row}
                for row in calibration_source_study(val, test, draws=draws, seed=seed + i)
            ]
        out["calibration"] += [{**base, **row} for row in summarize_calibration(cal_rows)]
        if data_root is not None and g.method in roi_methods:
            out["roi"].append(roi_study(g, root, data_root, target_fpr=target_fpr,
                                        n_resamples=n_resamples, seed=seed))
    out["settings"] = {
        "role": role, "target_fpr": target_fpr, "sizes": list(sizes), "targets": list(targets),
        "draws_per_seed": draws, "bootstrap_resamples": n_resamples, "seed": seed,
        "aggregation_domain": AGGREGATION_DOMAIN,
    }
    return out


def _indices_for(group: RunGroup, data_root: str | Path) -> dict:
    """Rebuild the group's exact splits.

    The split listing is re-hashed and compared with the one recorded in the
    run spec, so a diagnostic provably looks at the images the runs used.
    """
    from .data import ensure_validation, load_category
    from .runner import split_listing

    ds = group.dataset
    indices = load_category(data_root, group.category, layout=ds["layout"])
    indices, _ = ensure_validation(
        indices, val_fraction=float(ds["val_carve_fraction"]), seed=int(ds["val_carve_seed"])
    )
    if split_listing(indices) != ds["split_listing"]:
        raise ValueError(f"{group.category}: splits differ from the ones the runs used")
    return indices


def _proximity_for(group: RunGroup, data_root: str | Path) -> dict[str, Any]:
    return train_proximity(_indices_for(group, data_root), test_split=group.test_split)


#: Pre-declared before the ROI study was run; the study reports every margin
#: rather than selecting one (a sensitivity analysis, not a tuning sweep).
ROI_MARGINS: tuple[float, ...] = (0.01, 0.03, 0.06)
ROI_DEFAULT_MARGIN = 0.03


def roi_study(
    group: RunGroup,
    registry_root: str | Path,
    data_root: str | Path,
    *,
    margins: tuple[float, ...] = ROI_MARGINS,
    reducers: tuple[str, ...] = ("max", "top5%"),
    target_fpr: float = 0.01,
    n_resamples: int = 1000,
    seed: int = 0,
    max_train_images: int = 100,
) -> dict[str, Any]:
    """E02b: restrict the image score to the object region.

    The region comes from `postproc.roi`, fitted on training normals only. The
    maps are the stored ones; only the aggregation domain changes. Reported
    alongside: the assumption checks (border foreground share on training
    images, region size per split) and a *development-only* safety check — the
    share of annotated defect pixels that fall inside the region. That share
    is never used to choose a parameter.
    """
    from .data.transforms import load_image, load_mask
    from .postproc.roi import fit_background, mask_on_grid, object_mask

    root = Path(registry_root)
    indices = _indices_for(group, data_root)
    train = list(indices["train"])
    step = max(1, len(train) // max_train_images)
    background = fit_background(load_image(s.image_path) for s in train[::step])
    border_share = [background.border_foreground_share(load_image(s.image_path)) for s in train[::step]]
    train_frac = [float(object_mask(load_image(s.image_path), background).mean()) for s in train[::step * 4]]

    runs = [
        (SplitPredictions.load(root / rid, "validation"), SplitPredictions.load(root / rid, group.test_split))
        for rid in group.run_ids
    ]
    samples = {s.rel_id(indices["validation"].root): s for split in ("validation", group.test_split)
               for s in indices[split]}

    # One region per image and margin; the maps' grid is the same for all seeds.
    regions: dict[str, dict[float, np.ndarray]] = {}
    coverage: dict[float, list[float]] = {m: [] for m in margins}
    for image_id, sample in samples.items():
        image = load_image(sample.image_path)
        regions[image_id] = {m: object_mask(image, background, margin=m) for m in margins}
        if sample.mask_path is not None:
            import cv2

            defect = load_mask(sample.mask_path)
            for m in margins:
                region = regions[image_id][m]
                small = cv2.resize(defect.astype(np.uint8), (region.shape[1], region.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                if small.any():
                    coverage[m].append(float(region[small].mean()))

    def region_scores(preds: SplitPredictions, margin: float, reducer: str) -> np.ndarray:
        fn = AGGREGATIONS[reducer]
        out = []
        for image_id, raw in zip(preds.ids, preds.raw_maps):
            cells = mask_on_grid(regions[image_id][margin], raw.shape)
            out.append(fn(raw[cells]))
        return np.asarray(out, dtype=np.float64)

    extra = {
        f"roi{m:.0%}:{red}": [(region_scores(v, m, red), region_scores(t, m, red)) for v, t in runs]
        for m in margins for red in reducers
    }
    agg = aggregation_study(runs, rules=("stored", *reducers), target_fpr=target_fpr,
                            n_resamples=n_resamples, seed=seed, extra=extra)
    val0, test0 = runs[0]
    return {
        "category": group.category, "method": group.method, "label": group.label,
        "rules": agg,
        "checks": {
            "border_foreground_share_median": float(np.median(border_share)),
            "border_foreground_share_max": float(np.max(border_share)),
            "region_fraction_train_median": float(np.median(train_frac)),
            "region_fraction_by_split": {
                f"{m:.0%}": {
                    "validation": float(np.median([regions[i][m].mean() for i in val0.ids])),
                    "test": float(np.median([regions[i][m].mean() for i in test0.ids])),
                } for m in margins
            },
            "defect_pixels_inside_median": {
                f"{m:.0%}": float(np.median(c)) if c else float("nan") for m, c in coverage.items()
            },
            "defect_images_fully_inside_share": {
                f"{m:.0%}": float(np.mean(np.asarray(c) >= 0.999)) if c else float("nan")
                for m, c in coverage.items()
            },
            "defects_measured": len(coverage[margins[0]]),
        },
    }


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return format(value, spec)


def render_studies(study: dict[str, Any]) -> str:
    """The generated study tables. Interpretation belongs in a hand-written
    report that cites these; this file only states what was computed."""
    s = study["settings"]
    lines = [
        "# E02 / E03 — studies on stored predictions",
        "",
        "Generated by `inspector study`. Do not edit by hand. Role: "
        f"`{s['role']}`. No model was refitted: every number is computed from the run registry's "
        "stored scores and raw maps.",
        "",
        "## E02 — image-score aggregation",
        "",
        f"Maps fixed; only the map-to-score rule changes. Domain: {s['aggregation_domain']}. "
        f"Operating point: conservative rank rule at a {s['target_fpr']:.0%} request on the same "
        "rule's validation scores. `dAUROC` is the seed-averaged paired difference against the "
        f"model's stored score, with a class-stratified image bootstrap 95% interval "
        f"({s['bootstrap_resamples']} resamples); intervals cover test-sample uncertainty, not "
        "seed spread.",
        "",
        "| category | model | rule | seeds | I-AUROC | ± seed sd | AP | recall@OP | FPR@OP | dAUROC [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in study["aggregation"]:
        ci = ("-" if "d_auroc_vs_ref" not in r else
              f"{r['d_auroc_vs_ref']:+.3f} [{r['d_auroc_ci_low']:+.3f}, {r['d_auroc_ci_high']:+.3f}]")
        lines.append(
            f"| {r['category']} | {r['label']} | {r['rule']} | {r['n_seeds']} | {_fmt(r['auroc_mean'])} | "
            f"{_fmt(r['auroc_std'])} | {_fmt(r['ap_mean'])} | {_fmt(r['recall_mean'])} | "
            f"{_fmt(r['fpr_mean'])} | {ci} |"
        )
    lines += [
        "",
        "## Exchangeability of validation and test normals",
        "",
        "`shift AUROC` = P(a test normal scores above a validation normal); 0.5 under "
        "exchangeability — a test of the *bulk*. `exceed p` is the exact permutation p-value "
        "of at least this many test normals above the validation threshold if validation and "
        "test normals were exchangeable — a test of the *upper tail*, which is what the "
        "threshold depends on. Seeds share their images, so their p-values are not independent. "
        "`FPR 95% UB` is the Clopper-Pearson upper bound of the realized rate.",
        "",
        "| category | model | seed | n val | n test normals | shift AUROC | MW p | eff. bound | FP | realized FPR | FPR 95% UB | exceed p | recall |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in study["normal_shift"]:
        lines.append(
            f"| {r['category']} | {r['label']} | {r['seed']} | {r['n_validation']} | {r['n_test_normals']} | "
            f"{_fmt(r['auroc'])} | {_fmt(r['mannwhitney_p'], '.2g')} | {_fmt(r['effective_fpr'], '.2%')} | "
            f"{r['test_fp']} | {_fmt(r['realized_fpr'], '.1%')} | {_fmt(r['fpr_upper95'], '.1%')} | "
            f"{_fmt(r['exceedance_p'], '.2g')} | {_fmt(r['recall'])} |"
        )
    lines += [
        "",
        "## Calibration source: validation vs exchangeable test normals",
        "",
        "Per draw, the test normals are split into halves A and B, and a validation subset V of "
        "the same size is drawn; A and V each set a rank-k threshold (same n, same k) that is "
        "evaluated on B. A is the positive control: its expected FPR is exactly the bound. Means "
        "are over draws, then over seeds. `MC se` is the Monte Carlo error of the mean, not a "
        "sampling interval. A diagnostic only: no reported operating point uses test normals.",
        "",
        "| category | model | request | n | k | bound | FPR from validation | FPR from test half | MC se | recall (val / test half) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    cs_groups: dict[tuple, dict[str, list[dict]]] = {}
    for r in study.get("calibration_source", []):
        key = (r["category"], r["label"], r["requested_fpr"])
        cs_groups.setdefault(key, {}).setdefault(r["source"], []).append(r)
    for (category, label, target), by_source in cs_groups.items():
        v, t = by_source.get("validation", []), by_source.get("test half", [])
        if not v or not t:
            continue
        se = max(max(x["fpr_mc_se"] for x in v), max(x["fpr_mc_se"] for x in t))
        lines.append(
            f"| {category} | {label} | {target:.0%} | {v[0]['n_calibration']} | {v[0]['rank_k']} | "
            f"{v[0]['bound']:.2%} | {np.mean([x['fpr_mean'] for x in v]):.2%} | "
            f"{np.mean([x['fpr_mean'] for x in t]):.2%} | {se:.2%} | "
            f"{np.mean([x['recall_mean'] for x in v]):.3f} / {np.mean([x['recall_mean'] for x in t]):.3f} |"
        )
    if study.get("train_proximity"):
        lines += [
            "",
            "## Pixel-space proximity to the training images",
            "",
            "Distance from each image to its nearest training image (grey thumbnails, standardised "
            "per image; model-free). `AUROC` = P(a test normal is farther from train than a "
            "validation image); 0.5 means validation normals are as novel as test normals.",
            "",
            "| category | n val | n test normals | median dist (val) | median dist (test) | AUROC | MW p |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in study["train_proximity"]:
            lines.append(
                f"| {r['category']} | {r['n_validation']} | {r['n_test_normals']} | "
                f"{r['median_val']:.2f} | {r['median_test']:.2f} | {r['auroc']:.3f} | "
                f"{r['mannwhitney_p']:.2g} |"
            )
    if study.get("roi"):
        lines += [
            "",
            "## E02b — object-region aggregation domain",
            "",
            "The image score is computed only over map cells inside an object region estimated "
            "from training normals (`inspector.postproc.roi`); maps are unchanged. Margins were "
            f"declared before the run ({', '.join(f'{m:.0%}' for m in ROI_MARGINS)}; default "
            f"{ROI_DEFAULT_MARGIN:.0%}) and all are reported. `exceed p` is the median over seeds of "
            "the exact tail-exchangeability p-value at the operating point. `defects inside` is a "
            "development-only safety check (median share of annotated defect pixels inside the "
            "region); it chose nothing.",
            "",
            "| category | model | rule | I-AUROC | recall@OP | FPR@OP | exceed p | dAUROC vs stored [95% CI] |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for block in study["roi"]:
            for rule, r in block["rules"].items():
                ci = ("-" if "d_auroc_vs_ref" not in r else
                      f"{r['d_auroc_vs_ref']:+.3f} [{r['d_auroc_ci_low']:+.3f}, {r['d_auroc_ci_high']:+.3f}]")
                lines.append(
                    f"| {block['category']} | {block['label']} | {rule} | {_fmt(r['auroc_mean'])} | "
                    f"{_fmt(r['recall_mean'])} | {_fmt(r['fpr_mean'])} | {_fmt(r['exceed_p_median'], '.2g')} | {ci} |"
                )
        lines += [
            "",
            "| category | border fg share (median / max) | region share, train | "
            + " | ".join(f"defects inside @{m:.0%} (median / fully)" for m in ROI_MARGINS) + " |",
            "|---|---|---|" + "---|" * len(ROI_MARGINS),
        ]
        for block in study["roi"]:
            c = block["checks"]
            inside = " | ".join(
                f"{_fmt(c['defect_pixels_inside_median'][f'{m:.0%}'], '.1%')} / "
                f"{_fmt(c['defect_images_fully_inside_share'][f'{m:.0%}'], '.0%')}"
                for m in ROI_MARGINS
            )
            lines.append(
                f"| {block['category']} | {c['border_foreground_share_median']:.3f} / "
                f"{c['border_foreground_share_max']:.3f} | {c['region_fraction_train_median']:.2f} | {inside} |"
            )
    lines += [
        "",
        "## E03 — calibration rule and calibration-set size",
        "",
        f"Scores fixed; the calibration sample and the rule change. {s['draws_per_seed']} draws "
        "without replacement per size and seed, pooled over seeds; the full pool appears once per "
        "seed. `rank` is the conservative rule (ADR-9, `relax` policy); `sigma3` and `mad3` are "
        "mean + 3 sd and median + 3 * 1.4826 MAD of the calibration image scores. Ranges are the "
        "5th-95th percentiles over draws.",
        "",
        "| category | model | rule | request | n cal | draws | target met | eff. bound | FPR median [p5, p95] | recall median [p5, p95] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in study["calibration"]:
        lines.append(
            f"| {r['category']} | {r['label']} | {r['rule']} | {_fmt(r['requested_fpr'], '.0%')} | "
            f"{r['size']} | {r['draws']} | {_fmt(r['target_met_share'], '.0%')} | "
            f"{_fmt(r['effective_fpr'], '.2%')} | "
            f"{_fmt(r['fpr_median'], '.1%')} [{_fmt(r['fpr_p05'], '.1%')}, {_fmt(r['fpr_p95'], '.1%')}] | "
            f"{_fmt(r['recall_median'])} [{_fmt(r['recall_p05'])}, {_fmt(r['recall_p95'])}] |"
        )
    return "\n".join(lines) + "\n"
