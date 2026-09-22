"""Split integrity and leakage detection (protocol §5, rules L1-L7).

These functions back the tests in `tests/data/`. They are library code rather
than test-only helpers because the same checks run in the CLI (`inspector
audit`) against the real dataset, where they cannot live in a CI-only path.

Design note: every check returns a *report* rather than raising. A leakage
report that names the offending files is actionable; an AssertionError that says
`False is not True` is not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .core import ANOMALOUS, DatasetIndex


@dataclass
class LeakReport:
    """Result of a pairwise split-overlap check."""

    rule: str
    ok: bool
    detail: str = ""
    offenders: list[tuple[str, str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        head = f"[{self.rule}] {'PASS' if self.ok else 'FAIL'}"
        if self.ok:
            return f"{head} {self.detail}".rstrip()
        listing = "\n".join(f"    {a}  <->  {b}" for a, b in self.offenders[:20])
        more = (
            f"\n    ... and {len(self.offenders) - 20} more" if len(self.offenders) > 20 else ""
        )
        return f"{head} {self.detail}\n{listing}{more}"


def check_splits_disjoint(indices: Mapping[str, DatasetIndex]) -> LeakReport:
    """L1: no image content appears in two splits.

    Compares SHA-256 of file *bytes*. Filename comparison would miss the real
    failure mode, which is the same capture copied under a different name.
    """
    hashes: dict[str, dict[str, str]] = {
        name: index.content_hashes() for name, index in indices.items()
    }
    by_hash: dict[str, list[tuple[str, str]]] = {}
    for split, mapping in hashes.items():
        for rel_id, digest in mapping.items():
            by_hash.setdefault(digest, []).append((split, rel_id))

    offenders: list[tuple[str, str]] = []
    for entries in by_hash.values():
        splits = {split for split, _ in entries}
        if len(splits) > 1:
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    (s1, a), (s2, b) = entries[i], entries[j]
                    if s1 != s2:
                        offenders.append((f"{s1}/{a}", f"{s2}/{b}"))

    n = sum(len(m) for m in hashes.values())
    return LeakReport(
        rule="L1-splits-disjoint",
        ok=not offenders,
        detail=f"{n} images across {len(hashes)} splits, {len(offenders)} cross-split duplicates",
        offenders=offenders,
    )


def check_duplicates_within(index: DatasetIndex) -> LeakReport:
    """Exact duplicates inside one split.

    Not a leak by itself, but duplicated normals inflate a memory bank with
    redundant entries and duplicated test images double-count in every metric,
    so it is worth knowing about.
    """
    by_hash: dict[str, list[str]] = {}
    for rel_id, digest in index.content_hashes().items():
        by_hash.setdefault(digest, []).append(rel_id)
    offenders = [
        (ids[0], other) for ids in by_hash.values() if len(ids) > 1 for other in ids[1:]
    ]
    return LeakReport(
        rule="dup-within-split",
        ok=not offenders,
        detail=f"split={index.split_name}, {len(index)} images, {len(offenders)} duplicate pairs",
        offenders=offenders,
    )


# ---------------------------------------------------------------------------
# Near-duplicate detection (L6)
# ---------------------------------------------------------------------------


def phash(path: str | Path, hash_size: int = 8) -> int:
    """Perceptual hash (DCT-based) as a 64-bit int.

    Implemented here rather than pulled in as a dependency: it is 15 lines, and
    an opaque third-party pHash whose preprocessing we cannot see would make the
    calibration step in rule L6 meaningless.
    """
    size = hash_size * 4
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)

    # 2-D DCT-II via the orthonormal basis; scipy is available but keeping this
    # explicit makes the low-frequency crop below obvious.
    basis = np.cos(np.pi * (2 * np.arange(size)[:, None] + 1) * np.arange(size)[None, :] / (2 * size))
    dct = basis.T @ pixels @ basis
    low = dct[:hash_size, :hash_size].flatten()
    median = np.median(low[1:])  # drop DC, which only encodes mean brightness
    bits = low > median

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def check_near_duplicates(
    indices: Mapping[str, DatasetIndex],
    *,
    max_distance: int = 4,
    pairs: tuple[str, str] = ("train", "test_public"),
) -> LeakReport:
    """L6: near-identical captures across two splits.

    MVTec categories are shot on a fixed rig, so images are genuinely similar by
    design and this will flag pairs at a loose threshold. Calibrate
    `max_distance` on within-train pairs first (see `calibrate_phash_threshold`)
    and record the calibration; the goal is to catch duplicate *captures*, not
    to discover that all photographs of the same part look alike.
    """
    left, right = pairs
    if left not in indices or right not in indices:
        return LeakReport(
            rule="L6-near-duplicates",
            ok=True,
            detail=f"skipped: need both {left!r} and {right!r}",
        )

    lh = [(s.rel_id(indices[left].root), phash(s.image_path)) for s in indices[left]]
    rh = [(s.rel_id(indices[right].root), phash(s.image_path)) for s in indices[right]]

    offenders = [
        (f"{left}/{a}", f"{right}/{b}")
        for a, ha in lh
        for b, hb in rh
        if hamming(ha, hb) <= max_distance
    ]
    return LeakReport(
        rule="L6-near-duplicates",
        ok=not offenders,
        detail=(
            f"{len(lh)}x{len(rh)} comparisons at hamming<={max_distance}, "
            f"{len(offenders)} near-duplicate pairs"
        ),
        offenders=offenders,
    )


def calibrate_phash_threshold(index: DatasetIndex, *, percentile: float = 1.0) -> dict[str, float]:
    """Within-split pHash distance distribution, used to pick `max_distance`.

    Returns the distance percentiles so the chosen threshold can be justified in
    writing rather than guessed.
    """
    hashes = [phash(s.image_path) for s in index]
    distances = [
        hamming(hashes[i], hashes[j])
        for i in range(len(hashes))
        for j in range(i + 1, len(hashes))
    ]
    if not distances:
        return {"n_pairs": 0}
    arr = np.asarray(distances, dtype=float)
    return {
        "n_pairs": float(len(arr)),
        "min": float(arr.min()),
        f"p{percentile:g}": float(np.percentile(arr, percentile)),
        "p5": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


def check_masks_present(index: DatasetIndex) -> LeakReport:
    """Every anomalous sample in a labelled split has a mask."""
    missing = [
        (s.rel_id(index.root), "no mask")
        for s in index
        if s.label == ANOMALOUS and s.mask_path is None
    ]
    return LeakReport(
        rule="masks-present",
        ok=not missing,
        detail=f"split={index.split_name}, {index.n_anomalous} anomalous samples",
        offenders=missing,
    )


def check_mask_binary(index: DatasetIndex, *, limit: int | None = None) -> LeakReport:
    """Masks contain exactly two values.

    A mask that has been bilinearly resized somewhere upstream will have
    intermediate values, and every pixel metric computed against it is then
    subtly wrong in a way no other test would reveal.
    """
    offenders: list[tuple[str, str]] = []
    checked = 0
    for sample in index:
        if sample.mask_path is None:
            continue
        if limit is not None and checked >= limit:
            break
        values = np.unique(np.asarray(Image.open(sample.mask_path).convert("L")))
        checked += 1
        # A label mask holds at most two values. More than two means it was
        # interpolated somewhere upstream, and every pixel metric computed
        # against it is then subtly wrong with no other symptom.
        if len(values) > 2:
            offenders.append((sample.rel_id(index.root), f"values={values[:6].tolist()}..."))
    return LeakReport(
        rule="mask-binary",
        ok=not offenders,
        detail=f"{checked} masks checked",
        offenders=offenders,
    )


def check_private_split_unlabeled(index: DatasetIndex) -> LeakReport:
    """L7: private splits expose no ground truth."""
    offenders = [
        (s.rel_id(index.root), str(s.mask_path)) for s in index if s.mask_path is not None
    ]
    return LeakReport(
        rule="L7-private-unlabeled",
        ok=not offenders,
        detail=f"split={index.split_name}, {len(index)} images",
        offenders=offenders,
    )


def audit(indices: Mapping[str, DatasetIndex], *, near_duplicates: bool = False) -> list[LeakReport]:
    """Run the full structural audit. Used by `inspector audit` and by tests."""
    reports = [check_splits_disjoint(indices)]
    for name, index in indices.items():
        reports.append(check_duplicates_within(index))
        if index.layout.masks_available(name):
            reports.append(check_masks_present(index))
        if name in index.layout.unlabeled_splits:
            reports.append(check_private_split_unlabeled(index))
    if near_duplicates:
        reports.append(check_near_duplicates(indices))
    return reports
