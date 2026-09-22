"""Configuration loading, composition and hashing.

The config *is* the experiment (docs/06 §2). A run is
``inspector fit --config configs/x.yaml --seed 0``, and the resolved config's
SHA-256 is the primary experiment key.

Composition is deliberately tiny — a ``_base_`` key plus deep-merge — rather
than a framework. Hydra/OmegaConf would add a heavy dependency and an
interpolation language whose evaluation order becomes another thing that has to
be reproduced.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .utils.hashing import canonical_json, hash_object

BASE_KEY = "_base_"


class ConfigError(ValueError):
    """Raised for a malformed config, a missing base, or a bad override."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``; ``override`` wins.

    Lists are replaced wholesale, not concatenated. Concatenating a list on
    merge means you cannot ever *remove* an element in a child config, which is
    a trap that surfaces the first time you try to drop one augmentation.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config must be a mapping at top level: {path}")
    return data


def _resolve(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ConfigError(f"circular _base_ reference: {chain}")

    raw = _read_yaml(path)
    bases = raw.pop(BASE_KEY, [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list):
        raise ConfigError(f"{BASE_KEY} must be a string or list of strings in {path}")

    merged: dict[str, Any] = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        merged = deep_merge(merged, _resolve(base_path, (*seen, path)))
    return deep_merge(merged, raw)


def parse_override(item: str) -> tuple[list[str], Any]:
    """Parse a ``a.b.c=value`` CLI override. Values go through the YAML scalar
    parser so ``true``/``3``/``0.1``/``null`` acquire their natural types."""
    if "=" not in item:
        raise ConfigError(f"override must be key=value, got: {item!r}")
    key, _, raw = item.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"override has an empty key: {item!r}")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse override value in {item!r}: {exc}") from exc
    return key.split("."), value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b=value`` overrides.

    Creating a *new* key is an error: it is almost always a typo, and silently
    accepting ``patchore.k=3`` would produce a run that ignores the flag you
    thought you set while still hashing to a distinct config.
    """
    out = copy.deepcopy(cfg)
    for item in overrides:
        parts, value = parse_override(item)
        node: Any = out
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"override path does not exist in config: {'.'.join(parts)}")
            node = node[part]
        leaf = parts[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise ConfigError(f"override path does not exist in config: {'.'.join(parts)}")
        node[leaf] = value
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load, compose and override a config, returning the fully resolved mapping.

    The resolved mapping — not the composition — is what gets stored with the
    run, because that is what can actually be re-executed later.
    """
    cfg = _resolve(Path(path))
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Primary experiment key: SHA-256 of the canonical resolved config."""
    return hash_object(cfg)


def flatten(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten to ``a.b.c -> value`` for MLflow params, which are flat."""
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        elif isinstance(value, list):
            flat[name] = canonical_json(value)
        else:
            flat[name] = value
    return flat


def dump(cfg: dict[str, Any], path: str | Path) -> None:
    """Write the resolved config next to the run artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)
