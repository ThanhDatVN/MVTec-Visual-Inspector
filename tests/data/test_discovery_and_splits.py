"""Dataset discovery across the three on-disk layouts, and validation carving."""

from __future__ import annotations

import pytest
from PIL import Image

from inspector.data import (
    ANOMALOUS,
    MVTEC_AD,
    MVTEC_AD2,
    NORMAL,
    SYNTHETIC,
    carve_validation,
    discover_category,
    discover_split,
    ensure_validation,
    get_layout,
)


def test_discovers_every_split(strip_indices, fixture_spec):
    assert set(strip_indices) == {"train", "validation", "test_public"}
    assert len(strip_indices["train"]) == fixture_spec.n_train
    assert len(strip_indices["validation"]) == fixture_spec.n_validation


def test_labels_and_defect_types(strip_indices, fixture_spec):
    test = strip_indices["test_public"]
    assert test.n_normal == fixture_spec.n_test_good
    assert test.n_anomalous == 2 * fixture_spec.n_test_per_defect
    assert test.defect_types == ["good", "pit", "scratch"]
    assert all(s.label == NORMAL for s in test.filter(defect_type="good"))
    assert all(s.label == ANOMALOUS for s in test.filter(defect_type="pit"))


def test_anomalous_samples_are_paired_with_their_mask(strip_indices):
    for sample in strip_indices["test_public"].filter(label=ANOMALOUS):
        assert sample.mask_path is not None
        assert sample.mask_path.is_file()
        assert sample.mask_path.stem.startswith(sample.image_path.stem)
        # Mask and image must agree in size, or every pixel metric is misaligned.
        assert Image.open(sample.mask_path).size == Image.open(sample.image_path).size


def test_train_and_validation_are_normal_only(strip_indices):
    for split in ("train", "validation"):
        assert strip_indices[split].n_anomalous == 0


def test_ordering_is_deterministic(synthetic_root):
    first = discover_split(synthetic_root, "synth_grain", "train", layout=SYNTHETIC)
    second = discover_split(synthetic_root, "synth_grain", "train", layout=SYNTHETIC)
    assert [s.rel_id(first.root) for s in first] == [s.rel_id(second.root) for s in second]


def test_rel_id_is_posix_on_every_platform(strip_indices):
    for sample in strip_indices["train"]:
        rel = sample.rel_id(strip_indices["train"].root)
        assert "\\" not in rel
        assert rel.startswith("synth_strip/train/good/")


def test_manifest_hash_is_content_addressed(strip_indices):
    manifest = strip_indices["train"].manifest()
    again = strip_indices["train"].manifest()
    assert manifest["manifest_sha256"] == again["manifest_sha256"]
    assert manifest["n_samples"] == len(strip_indices["train"])
    assert all(len(entry["sha256"]) == 64 for entry in manifest["entries"])


def test_rejects_unknown_split_for_layout(synthetic_root):
    with pytest.raises(ValueError, match="not a split of layout"):
        discover_split(synthetic_root, "synth_strip", "test_private", layout=SYNTHETIC)


def test_rejects_missing_category(synthetic_root):
    with pytest.raises(FileNotFoundError):
        discover_split(synthetic_root, "no_such_category", "train", layout=SYNTHETIC)


def test_discover_category_skips_absent_splits(synthetic_root):
    """Classic AD has no validation split and private AD 2 splits are optional,
    so absence must not be an error."""
    found = discover_category(synthetic_root, "synth_strip", layout=SYNTHETIC)
    assert "test_private" not in found


# --- layouts ----------------------------------------------------------------


def test_layouts_declare_the_conventions_that_actually_differ():
    # AD 2 keeps ground truth inside the test split; classic AD at category root.
    assert MVTEC_AD2.mask_root == "{split}/ground_truth"
    assert MVTEC_AD.mask_root == "ground_truth"
    # AD 2 ships a validation split; classic AD does not.
    assert MVTEC_AD2.val_split == "validation"
    assert MVTEC_AD.val_split is None


def test_get_layout_rejects_an_unknown_name():
    with pytest.raises(KeyError, match="unknown dataset layout"):
        get_layout("mvtec_ad3")


# --- validation carving -----------------------------------------------------


def test_carve_is_deterministic_for_a_seed(strip_indices):
    a = carve_validation(strip_indices["train"], seed=0, min_val=3)
    b = carve_validation(strip_indices["train"], seed=0, min_val=3)
    assert a[2].digest == b[2].digest
    assert a[2].val_ids == b[2].val_ids


def test_carve_differs_across_seeds(strip_indices):
    a = carve_validation(strip_indices["train"], seed=0, min_val=3)
    b = carve_validation(strip_indices["train"], seed=1, min_val=3)
    assert a[2].val_ids != b[2].val_ids


def test_carve_partitions_without_overlap(strip_indices):
    train, val, assignment = carve_validation(strip_indices["train"], seed=0, min_val=3)
    assert len(train) + len(val) == len(strip_indices["train"])
    assert not set(assignment.train_ids) & set(assignment.val_ids)
    assert val.split_name == "validation"
    assert {s.split for s in val} == {"validation"}


def test_carve_refuses_a_split_containing_anomalies(strip_indices):
    """Validation must be normal-only: carving from a split with defects would
    hand threshold selection labelled anomalies and void the protocol."""
    with pytest.raises(ValueError, match="normal-only"):
        carve_validation(strip_indices["test_public"], seed=0)


def test_carve_enforces_a_minimum_validation_size(strip_indices):
    """Below a handful of images a percentile threshold is not estimable and
    OP-FPR1 is meaningless."""
    _, val, _ = carve_validation(strip_indices["train"], val_fraction=0.01, seed=0, min_val=4)
    assert len(val) >= 4


def test_ensure_validation_leaves_an_official_split_untouched(strip_indices):
    out, assignment = ensure_validation(strip_indices)
    assert assignment is None, "AD 2 ships a validation split; carving one would be wrong"
    assert out["validation"] is strip_indices["validation"]


def test_ensure_validation_carves_when_absent(strip_indices):
    without = {k: v for k, v in strip_indices.items() if k != "validation"}
    out, assignment = ensure_validation(without, seed=0)
    assert assignment is not None
    assert len(out["validation"]) >= 4
    assert out["train"].n_anomalous == 0
