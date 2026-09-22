"""Dataset indexing — torch-free.

This is the layer the leakage tests (protocol §5, rules L1-L7) run against. It
is deliberately free of torch and of any image decoding so that CI can verify
split integrity on any machine, and so that a split bug surfaces as a failing
unit test rather than as an inexplicably good AUROC in week nine.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ..utils.hashing import hash_file, hash_object
from .layout import IMAGE_EXTENSIONS, NORMAL_DIR, DatasetLayout, get_layout

NORMAL = 0
ANOMALOUS = 1


@dataclass(frozen=True)
class Sample:
    """One image, its label, and where its mask is (if any)."""

    image_path: Path
    label: int
    defect_type: str
    split: str
    category: str
    mask_path: Path | None = None

    @property
    def is_normal(self) -> bool:
        return self.label == NORMAL

    @property
    def stem(self) -> str:
        return self.image_path.stem

    def rel_id(self, root: Path) -> str:
        """Stable identifier for reporting: path relative to the dataset root,
        POSIX-normalised so it matches across Windows and Linux runs."""
        try:
            return self.image_path.relative_to(root).as_posix()
        except ValueError:
            return self.image_path.as_posix()


class DatasetIndex:
    """An ordered, immutable collection of `Sample`s with split-level provenance.

    `split_name` records which protocol split these samples came from. Fit code
    asserts on it (rule L2), which is why it is carried on the index rather than
    inferred from paths at the point of use.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        root: Path,
        category: str,
        split_name: str,
        layout: DatasetLayout,
    ) -> None:
        self._samples = tuple(samples)
        self.root = Path(root)
        self.category = category
        self.split_name = split_name
        self.layout = layout

    # -- sequence protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, i: int) -> Sample:
        return self._samples[i]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __repr__(self) -> str:
        return (
            f"DatasetIndex(category={self.category!r}, split={self.split_name!r}, "
            f"n={len(self)}, normal={self.n_normal}, anomalous={self.n_anomalous})"
        )

    # -- views -------------------------------------------------------------
    @property
    def samples(self) -> tuple[Sample, ...]:
        return self._samples

    @property
    def n_normal(self) -> int:
        return sum(s.is_normal for s in self._samples)

    @property
    def n_anomalous(self) -> int:
        return len(self) - self.n_normal

    @property
    def labels(self) -> list[int]:
        return [s.label for s in self._samples]

    @property
    def defect_types(self) -> list[str]:
        return sorted({s.defect_type for s in self._samples})

    def filter(self, *, label: int | None = None, defect_type: str | None = None) -> DatasetIndex:
        picked = [
            s
            for s in self._samples
            if (label is None or s.label == label)
            and (defect_type is None or s.defect_type == defect_type)
        ]
        return self._with(picked)

    def subset(self, indices: Sequence[int]) -> DatasetIndex:
        return self._with([self._samples[i] for i in indices])

    def rename_split(self, split_name: str) -> DatasetIndex:
        """Relabel the protocol split (used when carving validation from train).

        The per-sample `split` field is rewritten too, so a sample can never
        claim one split while its index claims another.
        """
        renamed = [replace(s, split=split_name) for s in self._samples]
        return DatasetIndex(
            renamed,
            root=self.root,
            category=self.category,
            split_name=split_name,
            layout=self.layout,
        )

    def _with(self, samples: Sequence[Sample]) -> DatasetIndex:
        return DatasetIndex(
            samples,
            root=self.root,
            category=self.category,
            split_name=self.split_name,
            layout=self.layout,
        )

    # -- integrity ---------------------------------------------------------
    def content_hashes(self) -> dict[str, str]:
        """SHA-256 of each image's *bytes*, keyed by relative id.

        Bytes, not filenames: two splits containing the same photograph under
        different names is exactly the leak rule L1 exists to catch.
        """
        return {s.rel_id(self.root): hash_file(s.image_path) for s in self._samples}

    def manifest(self) -> dict[str, object]:
        entries = [
            {
                "id": s.rel_id(self.root),
                "sha256": hash_file(s.image_path),
                "label": s.label,
                "defect_type": s.defect_type,
                "mask": s.mask_path.relative_to(self.root).as_posix() if s.mask_path else None,
            }
            for s in sorted(self._samples, key=lambda x: x.rel_id(self.root))
        ]
        body = {
            "dataset": self.layout.name,
            "category": self.category,
            "split": self.split_name,
            "n_samples": len(entries),
            "n_normal": self.n_normal,
            "n_anomalous": self.n_anomalous,
            "entries": entries,
        }
        return {**body, "manifest_sha256": hash_object(body)}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _iter_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name)


def _find_mask(
    mask_dir: Path, image_path: Path, suffix: str
) -> Path | None:
    """Locate the mask for an image.

    MVTec writes ``<stem>_mask.png``, but the extension is not guaranteed to
    match the image's, so try the suffixed stem across known extensions and then
    the bare stem. Returning None is meaningful — an anomalous image with no
    mask is a dataset problem the caller must decide about, not something to
    paper over with an empty mask.
    """
    if not mask_dir.is_dir():
        return None
    for stem in (f"{image_path.stem}{suffix}", image_path.stem):
        for ext in (image_path.suffix, *IMAGE_EXTENSIONS):
            candidate = mask_dir / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
    return None


def discover_split(
    root: str | Path,
    category: str,
    split: str,
    *,
    layout: DatasetLayout | str = "mvtec_ad2",
    require_masks: bool = True,
) -> DatasetIndex:
    """Index one split of one category.

    Args:
        root: dataset root containing category folders.
        category: category name.
        split: protocol split name, must be declared by the layout.
        layout: a `DatasetLayout` or its registered name.
        require_masks: raise if a labelled anomalous image has no mask. Left on
            by default: a silently mask-less anomalous image would be scored as
            if it had an empty ground truth, quietly depressing every pixel
            metric.
    """
    layout = get_layout(layout) if isinstance(layout, str) else layout
    root = Path(root)
    if split not in layout.splits:
        raise ValueError(
            f"{split!r} is not a split of layout {layout.name!r}; known: {layout.splits}"
        )

    split_dir = root / category / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_dir}")

    mask_root = root / category / layout.mask_root.format(split=split)
    samples: list[Sample] = []

    # An unlabeled split (AD 2's private sets) holds images directly, with no
    # class subfolders and no ground truth.
    direct = _iter_images(split_dir)
    if split in layout.unlabeled_splits or (direct and not any(p.is_dir() for p in split_dir.iterdir())):
        samples = [
            Sample(
                image_path=p,
                label=NORMAL,  # placeholder; never read for unlabeled splits
                defect_type="unknown",
                split=split,
                category=category,
            )
            for p in direct
        ]
        return DatasetIndex(
            samples, root=root, category=category, split_name=split, layout=layout
        )

    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        class_name = class_dir.name
        if class_name == "ground_truth":
            continue
        is_normal = class_name == NORMAL_DIR
        label = NORMAL if is_normal else ANOMALOUS
        for image_path in _iter_images(class_dir):
            mask_path = None
            if not is_normal and layout.masks_available(split):
                mask_path = _find_mask(mask_root / class_name, image_path, layout.mask_suffix)
                if mask_path is None and require_masks:
                    raise FileNotFoundError(
                        f"no ground-truth mask for anomalous image {image_path}; "
                        f"looked in {mask_root / class_name}"
                    )
            samples.append(
                Sample(
                    image_path=image_path,
                    label=label,
                    defect_type=class_name,
                    split=split,
                    category=category,
                    mask_path=mask_path,
                )
            )

    return DatasetIndex(samples, root=root, category=category, split_name=split, layout=layout)


def discover_category(
    root: str | Path,
    category: str,
    *,
    layout: DatasetLayout | str = "mvtec_ad2",
    splits: Sequence[str] | None = None,
    require_masks: bool = True,
) -> dict[str, DatasetIndex]:
    """Index every split that exists on disk for a category.

    Missing splits are skipped rather than raising: AD 2's private splits are
    only present for users who downloaded them, and classic AD has no
    validation split at all.
    """
    layout = get_layout(layout) if isinstance(layout, str) else layout
    wanted = splits if splits is not None else layout.splits
    out: dict[str, DatasetIndex] = {}
    for split in wanted:
        if not (Path(root) / category / split).is_dir():
            continue
        out[split] = discover_split(
            root, category, split, layout=layout, require_masks=require_masks
        )
    return out
