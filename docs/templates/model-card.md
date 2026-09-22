# Model Card — MVTec Visual Inspector

*Template. Complete at Phase P10. Structure follows Mitchell et al., "Model Cards for Model
Reporting" (FAT* 2019). Every `<...>` must be filled or explicitly marked "not applicable".*

---

## Model details

| Field | Value |
|-------|-------|
| Name / version | `<inspector-{category}-{method}-v{semver}>` |
| Date | `<>` |
| Type | Unsupervised one-class anomaly detection and localization |
| Architecture | `<backbone, layers, memory-bank size, projection dim>` |
| Training regime | Normal images only; `<fit procedure>` |
| Input | RGB, `<resolution>`, `<resize policy>`, ImageNet normalization |
| Output | Image anomaly score; H×W anomaly map at native resolution; binary decision; binary mask |
| Artifact size | `<MB>` |
| Frameworks | PyTorch `<v>`, torchvision `<v>`, OpenCV `<v>` |
| Code | `<repo URL, commit SHA>` |
| License (code) | MIT |
| License (weights / memory bank) | Derived from MVTec AD 2 — CC BY-NC-SA 4.0, non-commercial only |

## Intended use

**Intended:** research and educational demonstration of cold-start visual anomaly detection on the
MVTec AD 2 categories `<...>`, under the fixed capture conditions of that dataset.

**Intended users:** researchers and engineers evaluating one-class inspection methods.

**Out of scope — do not use for:**
- **Any commercial purpose.** Precluded by the dataset license (CC BY-NC-SA 4.0, NonCommercial).
- Safety-critical inspection, or any setting where a missed defect causes harm.
- Categories, parts, lighting rigs, or cameras other than those it was fitted on. This model has a
  memory of *these* normal parts under *these* conditions; it does not transfer.
- Any decision without human review at the operating point stated below.

## Operating point

| Field | Value |
|-------|-------|
| Threshold `tau` | `<>` |
| **Threshold source** | `validation/good`, 99th percentile (`OP-FPR1`) — *never* test-derived |
| Pixel threshold `tau_pix` | `<>` (`OP-3SIGMA`: validation mean + 3·std) |
| Target FPR | 1% |
| **Realized FPR on test normals** | `<>` ← the calibration error; report it even when it is unflattering |
| Recall at this threshold | `<>` |

## Performance

*All numbers on `test_public` unless stated. `test_public` functioned as a development set — see
[ADR-3](../07-risks-and-decisions.md). Not directly comparable to published `TEST_priv` numbers.*

| Category | Image AUROC | AU-PRO@0.05 | AU-PRO@0.30 | Pixel AUROC | AUPIMO (med) | SegF1 | FPR@OP1 | Recall@OP1 |
|----------|-------------|-------------|-------------|-------------|--------------|-------|---------|------------|
| `<cat1>` | | | | | | | | |
| `<cat2>` | | | | | | | | |
| `<cat3>` | | | | | | | | |
| **Mean** | | | | | | | | |

Seeds: `<n>`, reported as mean ± std. External anchors: [02](../02-metrics-and-baselines.md) §2.

### Performance across conditions

| Condition | Image AUROC | AU-PRO@0.05 | Realized FPR@OP1 |
|-----------|-------------|-------------|------------------|
| Clean | | | |
| Unseen lighting (native AD 2 split) | | | |
| Synthetic blur, severity 3 | | | |
| Synthetic exposure ±1 stop | | | |
| Resize round-trip 0.5× | | | |

Disaggregated performance by **defect type** (not just per category) goes here — a mean hides the
fact that one defect class is missed entirely, which is precisely what a user needs to know.

### Systems

| Device | Precision | Resolution | p50 | p95 | p99 | Peak VRAM | Peak RSS |
|--------|-----------|-----------|-----|-----|-----|-----------|----------|
| RTX 3050 Laptop (4 GB) | | | | | | | |
| Colab T4 | | | | | | | |
| CPU | | | | | | | |

## Training data

MVTec AD 2, categories `<...>`, `train/good` split only: `<n>` normal images per category.
No defect labels were used at fit time. Validation (`validation/good`, `<n>` images) was used for
threshold selection and normalization statistics only.

## Evaluation data

`test_public`: `<n>` images per category, with pixel-precise ground truth.
Evaluation budget consumed: `<n>` / 40 per category (see `reports/test_set_budget.md`).

## Ethical considerations

Industrial parts; no people, no personal data, no demographic attributes. The relevant harm is
economic and physical: a missed defect ships a faulty part. The model is therefore specified for
human-in-the-loop use, and the operating point deliberately favours `<recall | precision>` because
`<justification>`.

## Limitations

*Write this section as the strongest critic of the work would write it. If it is short, it is wrong.*

- Fitted per category; no cross-category generalization is claimed or tested.
- Assumes a fixed capture rig. `<State the measured sensitivity to translation/rotation.>`
- `<The realized-FPR drift under lighting shift, stated in numbers.>`
- Reconstruction and memory-based methods `<state the specific defect types that were missed>`.
- `test_public` was used for development; reported numbers carry an optimism bias of unknown size.
- `<Any Tier 4 comparator run at default hyperparameters — its number is not that method's ceiling.>`
- `<Compute constraints: which configurations could not be evaluated, and what that leaves unknown.>`
- `<Every finding from reports/negative-results.md that bounds the claims made here.>`

## Attribution

*(Reproduce the full block from [08-licensing-and-attribution.md](../08-licensing-and-attribution.md) §4.)*
