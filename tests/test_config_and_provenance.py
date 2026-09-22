# ruff: noqa: NPY002  -- these tests deliberately exercise the legacy global RNG,
# which is exactly what seed_everything must make reproducible.
"""Config composition, hashing, and run provenance.

The config *is* the experiment (docs/06 §2), so its hash is the primary
experiment key and must be stable, order-independent, and sensitive to every
value that changes a result.
"""

from __future__ import annotations

import pytest
import yaml

from inspector.config import (
    ConfigError,
    apply_overrides,
    config_hash,
    deep_merge,
    flatten,
    load_config,
    parse_override,
)
from inspector.utils.env import capture, git_info
from inspector.utils.hashing import hash_object
from inspector.utils.seed import make_rng, seed_everything


def write(tmp_path, name, data):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- composition ------------------------------------------------------------


def test_deep_merge_overrides_nested_values():
    base = {"model": {"backbone": "wrn50", "layers": ["l2", "l3"]}, "seed": 0}
    merged = deep_merge(base, {"model": {"backbone": "resnet50"}})
    assert merged["model"]["backbone"] == "resnet50"
    assert merged["model"]["layers"] == ["l2", "l3"]
    assert merged["seed"] == 0


def test_lists_are_replaced_not_concatenated():
    """Concatenating on merge would make it impossible for a child config to
    *remove* an entry — a trap that surfaces the first time you drop one
    augmentation from a list."""
    merged = deep_merge({"augs": ["flip", "jitter"]}, {"augs": ["flip"]})
    assert merged["augs"] == ["flip"]


def test_merge_does_not_mutate_its_inputs():
    base = {"model": {"k": 1}}
    deep_merge(base, {"model": {"k": 9}})
    assert base["model"]["k"] == 1


def test_base_composition(tmp_path):
    write(tmp_path, "base.yaml", {"seed": 0, "model": {"backbone": "wrn50", "k": 1}})
    child = write(tmp_path, "child.yaml", {"_base_": "base.yaml", "model": {"k": 9}})

    cfg = load_config(child)
    assert cfg["seed"] == 0
    assert cfg["model"] == {"backbone": "wrn50", "k": 9}
    assert "_base_" not in cfg


def test_multiple_bases_apply_left_to_right(tmp_path):
    write(tmp_path, "a.yaml", {"x": 1, "y": 1})
    write(tmp_path, "b.yaml", {"y": 2})
    child = write(tmp_path, "c.yaml", {"_base_": ["a.yaml", "b.yaml"]})
    assert load_config(child) == {"x": 1, "y": 2}


def test_circular_base_is_rejected(tmp_path):
    write(tmp_path, "a.yaml", {"_base_": "b.yaml"})
    write(tmp_path, "b.yaml", {"_base_": "a.yaml"})
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


# --- overrides --------------------------------------------------------------


def test_override_values_get_natural_types():
    assert parse_override("a.b=3") == (["a", "b"], 3)
    assert parse_override("a=true") == (["a"], True)
    assert parse_override("a=0.1") == (["a"], 0.1)
    assert parse_override("a=null") == (["a"], None)
    assert parse_override("a=wrn50") == (["a"], "wrn50")


def test_override_applies_to_a_nested_key():
    cfg = {"model": {"k": 1}}
    assert apply_overrides(cfg, ["model.k=9"])["model"]["k"] == 9


def test_override_of_an_unknown_key_is_an_error():
    """A typo must fail loudly. Silently accepting `patchore.k=3` would produce
    a run that ignores the flag you thought you set, while still hashing to a
    distinct config — the worst of both worlds."""
    with pytest.raises(ConfigError, match="does not exist"):
        apply_overrides({"model": {"k": 1}}, ["patchore.k=3"])
    with pytest.raises(ConfigError, match="does not exist"):
        apply_overrides({"model": {"k": 1}}, ["model.kk=3"])


def test_malformed_override_is_rejected():
    with pytest.raises(ConfigError, match="key=value"):
        parse_override("model.k")


# --- hashing ----------------------------------------------------------------


def test_hash_is_independent_of_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"m": {"x": 1, "y": 2}}) == config_hash({"m": {"y": 2, "x": 1}})


def test_hash_changes_when_any_value_changes():
    base = {"model": {"backbone": "wrn50", "coreset": 0.01}}
    changed = {"model": {"backbone": "wrn50", "coreset": 0.1}}
    assert config_hash(base) != config_hash(changed)


def test_hash_distinguishes_types():
    """`1` and `"1"` must not collide: one is a coreset count, the other a
    string that would take a different code path."""
    assert hash_object({"k": 1}) != hash_object({"k": "1"})


def test_flatten_produces_mlflow_ready_params():
    flat = flatten({"model": {"backbone": "wrn50", "layers": ["l2", "l3"]}, "seed": 0})
    assert flat["model.backbone"] == "wrn50"
    assert flat["seed"] == 0
    assert flat["model.layers"] == '["l2","l3"]'


# --- provenance -------------------------------------------------------------


def test_git_info_reports_sha_and_dirty_flag():
    info = git_info()
    assert set(info) == {"git_sha", "git_branch", "dirty"}
    assert isinstance(info["dirty"], bool)


def test_unknown_git_state_counts_as_dirty(monkeypatch):
    """'We could not tell' must never be recorded as 'it was clean', or a dirty
    run could slip into the headline table."""
    monkeypatch.setattr("inspector.utils.env._git", lambda *a: None)
    assert git_info()["dirty"] is True


def test_capture_includes_everything_a_rerun_needs():
    snapshot = capture()
    assert {"git_sha", "dirty", "libraries", "device"} <= set(snapshot)
    assert "python" in snapshot["libraries"]
    assert "numpy" in snapshot["libraries"]


# --- seeding ----------------------------------------------------------------


def test_seed_everything_is_reproducible():
    import numpy as np

    seed_everything(0)
    a = np.random.rand(5)
    seed_everything(0)
    assert np.allclose(a, np.random.rand(5))


def test_local_rng_does_not_touch_global_state():
    """Library code that mutates the global RNG makes a caller's results depend
    on call order — a reproducibility bug that only surfaces weeks later."""
    import numpy as np

    seed_everything(0)
    before = np.random.rand(3)

    seed_everything(0)
    make_rng(999).random(100)
    after = np.random.rand(3)

    assert np.allclose(before, after)
