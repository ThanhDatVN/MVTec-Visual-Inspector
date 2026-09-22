"""VisA (Visual Anomaly) dataset support.

VisA is the second most-used industrial anomaly-detection benchmark after MVTec
AD, and unlike MVTec it is:

* **downloadable without registration** — AWS Open Data, one 1.8 GB tar;
* **CC BY 4.0** rather than CC BY-NC-SA, so attribution is the only obligation.

That combination makes it the dataset this project can actually run on today,
with MVTec AD 2 remaining the headline benchmark once its download completes.

Its on-disk layout does not match MVTec's, so it gets its own reader rather than
a `DatasetLayout` entry::

    VisA_20220922/
      <category>/
        Data/Images/Normal/0000.JPG
        Data/Images/Anomaly/000.JPG
        Data/Masks/Anomaly/000.png
        image_anno.csv
      split_csv/1cls.csv

The **official `1cls.csv` split is authoritative** and is what makes results
comparable with published work. Inventing a split would silently produce numbers
that cannot be compared with anything, so a missing CSV is an error here rather
than a cue to fall back on directory scanning.

Note that VisA, like classic MVTec AD, ships **no validation split**: `train` is
normal-only and `test` holds both classes. `ensure_validation` carves one with a
recorded seed (protocol §3.4).

Attribution (CC BY 4.0, required):
    Zou, Y., Jeong, J., Pemula, L., Zhang, D., Dabeer, O.
    "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
    and Segmentation." ECCV 2022. arXiv:2207.14315
    Accessed from https://registry.opendata.aws/visa
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .core import ANOMALOUS, NORMAL, DatasetIndex, Sample
from .layout import DatasetLayout

#: The 12 VisA categories, grouped by the structural challenge the paper assigns
#: them. Group membership is what makes a three-category selection defensible:
#: picking three from one group would test one difficulty three times.
VISA_GROUPS: dict[str, tuple[str, ...]] = {
    "complex_structure": ("pcb1", "pcb2", "pcb3", "pcb4"),
    "multiple_instances": ("capsules", "candle", "macaroni1", "macaroni2"),
    "single_instance": ("cashew", "chewinggum", "fryum", "pipe_fryum"),
}

VISA_CATEGORIES: tuple[str, ...] = tuple(
    sorted(c for group in VISA_GROUPS.values() for c in group)
)

VISA = DatasetLayout(
    name="visa",
    splits=("train", "test"),
    fit_splits=("train",),
    val_split=None,  # carved from train; see splits.py
    test_splits=("test",),
    unlabeled_splits=(),
    mask_root="Data/Masks",  # unused: mask paths come from the split CSV
)

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
SPLIT_CSV_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"


@dataclass(frozen=True)
class VisaRow:
    """One line of `1cls.csv`."""

    object: str
    split: str
    label: str
    image: str
    mask: str

    @property
    def is_normal(self) -> bool:
        return self.label == "normal"


def read_split_csv(path: str | Path) -> list[VisaRow]:
    """Read the official one-class split."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"VisA split CSV not found at {path}. It defines the official one-class "
            f"train/test partition; download it from {SPLIT_CSV_URL}. Scanning directories "
            "instead would invent a split, and results on an invented split are not "
            "comparable with any published number."
        )
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"object", "split", "label", "image", "mask"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"VisA split CSV is missing columns: {sorted(missing)}")
        return [
            VisaRow(
                object=row["object"].strip(),
                split=row["split"].strip(),
                label=row["label"].strip(),
                image=row["image"].strip(),
                mask=row["mask"].strip(),
            )
            for row in reader
        ]


def discover_visa(
    root: str | Path,
    category: str,
    *,
    split_csv: str | Path | None = None,
    require_masks: bool = True,
    verify_exists: bool = True,
) -> dict[str, DatasetIndex]:
    """Index one VisA category from the official split.

    Args:
        root: the extracted VisA root (the directory holding the category folders).
        category: one of `VISA_CATEGORIES`.
        split_csv: path to `1cls.csv`; defaults to `<root>/split_csv/1cls.csv`.
        require_masks: raise if an anomalous image has no mask. On by default —
            an anomalous image silently treated as having an empty ground truth
            depresses every pixel metric with no other symptom.
        verify_exists: check that each listed file is on disk. Worth the stat
            calls: a partial extraction otherwise surfaces as a confusing error
            deep inside a training loop.
    """
    root = Path(root)
    if category not in VISA_CATEGORIES:
        raise ValueError(f"unknown VisA category {category!r}; known: {VISA_CATEGORIES}")

    csv_path = Path(split_csv) if split_csv else root / "split_csv" / "1cls.csv"
    rows = [r for r in read_split_csv(csv_path) if r.object == category]
    if not rows:
        raise ValueError(f"split CSV contains no rows for category {category!r}")

    by_split: dict[str, list[Sample]] = {"train": [], "test": []}
    for row in rows:
        if row.split not in by_split:
            raise ValueError(f"unexpected split {row.split!r} in VisA CSV")

        image_path = root / row.image
        if verify_exists and not image_path.is_file():
            raise FileNotFoundError(
                f"VisA image listed in the split CSV is missing on disk: {image_path}. "
                "The extraction is incomplete or `root` points at the wrong directory."
            )

        mask_path = None
        if not row.is_normal:
            if row.mask:
                mask_path = root / row.mask
                if verify_exists and not mask_path.is_file():
                    raise FileNotFoundError(f"VisA mask missing on disk: {mask_path}")
            elif require_masks:
                raise FileNotFoundError(
                    f"no ground-truth mask listed for anomalous image {row.image}"
                )

        by_split[row.split].append(
            Sample(
                image_path=image_path,
                label=NORMAL if row.is_normal else ANOMALOUS,
                # The one-class CSV carries no per-defect-type label; `anomaly`
                # is the honest name for what it does carry. Per-type labels
                # live in each category's image_anno.csv and are loaded
                # separately by `read_defect_types`.
                defect_type="good" if row.is_normal else "anomaly",
                split=row.split,
                category=category,
                mask_path=mask_path,
            )
        )

    indices = {}
    for split, samples in by_split.items():
        if not samples:
            continue
        samples.sort(key=lambda s: s.image_path.as_posix())
        indices[split] = DatasetIndex(
            samples, root=root, category=category, split_name=split, layout=VISA
        )

    if "train" in indices and indices["train"].n_anomalous:
        raise ValueError("VisA train split should be normal-only; the CSV disagrees")
    return indices


def read_defect_types(root: str | Path, category: str) -> dict[str, str]:
    """Map image path -> defect type from a category's `image_anno.csv`.

    The one-class split CSV collapses every defect to `anomaly`. Per-type labels
    are what turn a single recall number into a failure profile — "which defect
    class does this model miss entirely" is the question a process engineer
    actually asks — so they are read separately when available.
    """
    path = Path(root) / category / "image_anno.csv"
    if not path.is_file():
        return {}

    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            keys = {k.lower(): k for k in row}
            image_key = keys.get("image") or keys.get("image_name") or keys.get("filename")
            type_key = keys.get("label") or keys.get("type") or keys.get("anomaly_type")
            if image_key and type_key:
                mapping[row[image_key].strip()] = row[type_key].strip()
    return mapping


def summarize(root: str | Path, *, split_csv: str | Path | None = None) -> dict[str, dict]:
    """Per-category counts, for the dataset audit report."""
    root = Path(root)
    csv_path = Path(split_csv) if split_csv else root / "split_csv" / "1cls.csv"
    rows = read_split_csv(csv_path)

    per_category: dict[str, Counter] = {}
    for row in rows:
        counter = per_category.setdefault(row.object, Counter())
        counter[f"{row.split}_{row.label}"] += 1
        if row.mask:
            counter["with_mask"] += 1

    group_of = {c: g for g, cats in VISA_GROUPS.items() for c in cats}
    return {
        category: {
            "group": group_of.get(category, "unknown"),
            "train_normal": counter["train_normal"],
            "test_normal": counter["test_normal"],
            "test_anomaly": counter["test_anomaly"],
            "with_mask": counter["with_mask"],
        }
        for category, counter in sorted(per_category.items())
    }
