# P2 — Tier 0 Floors and Gate G2

Phase P2 of [docs/04](../docs/04-roadmap.md). Tier 0 exists to settle two questions before any
respectable method runs: **is the evaluation code correct**, and **is any category trivially
separable**. Both were answered, and a third finding appeared that changes the protocol.

> **These numbers are from synthetic fixtures, not MVTec data.** No Tier 0 number here is a
> result about anomaly detection. The fixtures exist so every code path is executable and
> assertable without the dataset (which is CC BY-NC-SA and cannot be committed). Real numbers
> arrive when the AD 2 download completes.

---

## 1. Gate G2 — metric validation

| Control | Expected | Measured | Status |
|---------|----------|----------|--------|
| `random`, pixel AUROC | 0.500 | 0.500 ± 0.08 over 8 seeds | pass |
| `random`, AU-PRO@0.05 | 0.025 (= L/2) | 0.025 ± 0.035 | pass |
| `random`, AU-PRO@0.30 | 0.150 | 0.151 (analytic suite, 30 images) | pass |
| perfect predictor, AU-PRO | 1.000 | 1.000 exactly | pass |
| inverted predictor, AU-PRO | 0.000 | 0.000 exactly | pass |
| histogram pixel AUROC vs `sklearn` | equal | within 1e-4 | pass |
| hand-computed SegF1 / IoU | 0.500 / 0.333 | exact | pass |

The random control converging to `L/2` for both integration limits is the load-bearing check: it
constrains the FPR axis, the PRO averaging, and the integration together. A metric bug that
survives it would have to be conspiratorial.

**Gate G2 passes** for the analytic half. The remaining half — agreement with an external
reference implementation on real data, and reproduction of a published PatchCore number — is
still open and belongs to Gate G5.

## 2. Are the categories trivially separable?

Not on the fixtures: no global-statistic model (`mean_intensity`, `histogram`) exceeds 0.94 image
AUROC on any category, and both sit near or below chance on two of three. This check must be
repeated on the real categories, where it matters much more — a category separable by exposure
alone would invalidate any claim built on it.

## 3. The finding that changed the protocol

Implementing `OP-FPR1` exposed that **its 1% false-alarm target is not statistically achievable
on any of the three study categories.**

A threshold set as the k-th largest of `n` validation scores is exceeded by a fresh normal sample
with probability `k/(n+1)`, for any continuous score distribution. The most extreme choice,
`k = 1`, floors the achievable rate at `1/(n+1)`:

| category | validation `n` | minimum achievable FPR | protocol asked for |
|----------|----------------|------------------------|--------------------|
| `sheet_metal` | 19 | **5.0%** | 1% |
| `fruit_jelly` | 37 | **2.6%** | 1% |
| `walnuts` | 48 | **2.0%** | 1% |

Nothing errors when you ask for this. `numpy.percentile` interpolates between the top two order
statistics and returns a number; simulation puts the realized rate of that "p99" threshold at
about 5% for `n = 19`. The report would have claimed 1% while the system delivered five times
that, and the gap would have been invisible because the realized FPR looks like an ordinary
distribution-shift effect.

Resolved in [ADR-7](../docs/07-risks-and-decisions.md#adr-7--op-fpr1-targets-the-achievable-rate-not-1):
the operating point is now set per category at its achievable floor, computed from the
distribution-free order statistic, with the expected rate recorded in the threshold's provenance.
`strict=True` refuses an unachievable target rather than clamping it silently.

**Consequence for the model card:** buying a lower false-alarm rate on `sheet_metal` requires more
normal validation images. That is an actionable statement, and it is only available because the
limit was computed rather than assumed.

## 4. Two observations to carry into Tier 1

**Localization and detection are separate abilities.** `pixel_pca` reaches 0.985 pixel AUROC and
0.846 AU-PRO@0.05 on `synth_strip` while its image AUROC is 0.53 — barely chance. Its anomaly maps
are good; its *aggregation* of a map into one scalar (the mean residual) is not, because a small
defect barely moves a mean taken over the whole image. Tier 1 should ablate the aggregation
(mean vs max vs top-k mean) separately from the model, and the same question will return for
PatchCore, whose published AD 2 profile shows the mirror-image failure: competitive AU-PRO
(28.8%) with a collapsed SegF1 (3.7%).

**AU-PRO partially rewards detection even with no localization.** `mean_intensity` emits a
*constant* map yet scores AU-PRO@0.05 of 0.061, above the 0.025 no-information baseline. The
mechanism: with a constant map, lowering the threshold switches whole images on at once, so a
model that merely ranks images well collects whole regions at low FPR. On an 11-image fixture the
value is then decided by how many regions happen to sit in the top-scoring image — pure
discretization. On a 150-image split the effect is far smaller but not zero, which is a reason to
read AU-PRO alongside a per-image metric rather than alone.

## 5. What is not yet established

- No agreement check against an external reference implementation on real data (Gate G5).
- No PatchCore reproduction (Gate G5).
- AUPIMO is specified in [docs/02](../docs/02-metrics-and-baselines.md) but not yet implemented.
- The single-pass memory budget used by the resolution decision remains an assumption from
  docs/06 §3, to be measured at Gate G5.
- Everything above is fixture-scale. The fixtures were built to exercise code, and their
  11-image test splits make every metric noisy by construction.

---

Data: MVTec AD / MVTec AD 2 (c) MVTec Software GmbH, CC BY-NC-SA 4.0. Non-commercial use only.
See [ATTRIBUTION.md](../ATTRIBUTION.md). No dataset images are redistributed in this repository.
