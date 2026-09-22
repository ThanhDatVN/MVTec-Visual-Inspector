"""On-disk layout descriptions for the datasets we read.

The three layouts we must support differ in ways that are easy to get subtly
wrong, so they are declared as data rather than buried in `if` statements:

MVTec AD 2 — one `bad` folder, ground truth *inside* the test split::

    <category>/train/good/*.png
    <category>/validation/good/*.png
    <category>/test_public/good/*.png
    <category>/test_public/bad/*.png
    <category>/test_public/ground_truth/bad/*_mask.png
    <category>/test_private/*            (no ground truth)
    <category>/test_private_mixed/*      (no ground truth)

MVTec AD classic — per-defect-type folders, ground truth at *category* root,
and no validation split at all (we carve one from train)::

    <category>/train/good/*.png
    <category>/test/good/*.png
    <category>/test/<defect_type>/*.png
    <category>/ground_truth/<defect_type>/*_mask.png

Synthetic fixtures — AD 2 paths with per-defect-type folders, which exercises
the multi-class branch that AD 2 itself does not.

Rather than hard-code a defect-folder name, the reader discovers subfolders:
`good` is normal, anything else is an anomaly class. That is what lets one
loader read all three, and it degrades gracefully if MVTec adds a folder.
"""

from __future__ import annotations

from dataclasses import dataclass

NORMAL_DIR = "good"
MASK_SUFFIX = "_mask"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class DatasetLayout:
    """Where splits and masks live for one dataset family."""

    name: str
    splits: tuple[str, ...]
    fit_splits: tuple[str, ...]
    val_split: str | None
    test_splits: tuple[str, ...]
    unlabeled_splits: tuple[str, ...]
    #: mask root relative to the category dir; "{split}" is substituted.
    mask_root: str
    mask_suffix: str = MASK_SUFFIX

    def masks_available(self, split: str) -> bool:
        return split in self.test_splits and split not in self.unlabeled_splits


MVTEC_AD2 = DatasetLayout(
    name="mvtec_ad2",
    splits=("train", "validation", "test_public", "test_private", "test_private_mixed"),
    fit_splits=("train",),
    val_split="validation",
    test_splits=("test_public", "test_private", "test_private_mixed"),
    unlabeled_splits=("test_private", "test_private_mixed"),
    mask_root="{split}/ground_truth",
)

MVTEC_AD = DatasetLayout(
    name="mvtec_ad",
    splits=("train", "test"),
    fit_splits=("train",),
    val_split=None,  # carved from train with a fixed seed; see splits.py
    test_splits=("test",),
    unlabeled_splits=(),
    mask_root="ground_truth",
)

SYNTHETIC = DatasetLayout(
    name="synthetic",
    splits=("train", "validation", "test_public"),
    fit_splits=("train",),
    val_split="validation",
    test_splits=("test_public",),
    unlabeled_splits=(),
    mask_root="{split}/ground_truth",
)

LAYOUTS: dict[str, DatasetLayout] = {
    layout.name: layout for layout in (MVTEC_AD2, MVTEC_AD, SYNTHETIC)
}


def get_layout(name: str) -> DatasetLayout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise KeyError(f"unknown dataset layout {name!r}; known: {sorted(LAYOUTS)}") from None
