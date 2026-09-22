"""Split construction.

MVTec AD 2 ships an official normal-only `validation` split, so we use it
verbatim. Classic MVTec AD ships none, so one is carved from `train` with a
fixed, recorded seed (protocol §3.4).

The carve is deterministic and content-addressed: the assignment depends on the
sorted order of file ids plus the seed, never on directory iteration order,
which varies across filesystems and would otherwise make the "fixed seed"
promise false on a different machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.hashing import hash_object
from .core import DatasetIndex


@dataclass(frozen=True)
class SplitAssignment:
    """A recorded train/validation partition, reproducible from its own fields."""

    seed: int
    val_fraction: float
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]

    @property
    def digest(self) -> str:
        return hash_object(
            {
                "seed": self.seed,
                "val_fraction": self.val_fraction,
                "train_ids": list(self.train_ids),
                "val_ids": list(self.val_ids),
            }
        )


def carve_validation(
    train_index: DatasetIndex,
    *,
    val_fraction: float = 0.15,
    seed: int = 0,
    min_val: int = 4,
) -> tuple[DatasetIndex, DatasetIndex, SplitAssignment]:
    """Split a normal-only train index into (train, validation).

    Args:
        train_index: the index to split. Must contain only normal samples —
            carving a validation set that contains defects would hand the
            threshold-selection step labelled anomalies, defeating the entire
            normal-only protocol.
        val_fraction: fraction routed to validation.
        seed: recorded in the returned assignment.
        min_val: floor on validation size; below roughly this many images a
            percentile threshold is not estimable and `OP-FPR1` is meaningless.

    Returns:
        (train, validation, assignment)
    """
    if train_index.n_anomalous:
        raise ValueError(
            f"cannot carve validation from a split containing {train_index.n_anomalous} "
            "anomalous samples; validation must be normal-only (protocol §3.4)"
        )
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    n = len(train_index)
    n_val = max(min_val, round(n * val_fraction))
    if n_val >= n:
        raise ValueError(
            f"validation size {n_val} would consume the whole train split of {n} images"
        )

    # Sort by stable id so the permutation is filesystem-independent.
    order = sorted(range(n), key=lambda i: train_index[i].rel_id(train_index.root))
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(order)
    val_pos = sorted(permuted[:n_val].tolist())
    train_pos = sorted(permuted[n_val:].tolist())

    val_index = train_index.subset(val_pos).rename_split("validation")
    new_train = train_index.subset(train_pos)

    assignment = SplitAssignment(
        seed=seed,
        val_fraction=val_fraction,
        train_ids=tuple(train_index[i].rel_id(train_index.root) for i in train_pos),
        val_ids=tuple(train_index[i].rel_id(train_index.root) for i in val_pos),
    )
    return new_train, val_index, assignment


def ensure_validation(
    indices: dict[str, DatasetIndex],
    *,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[dict[str, DatasetIndex], SplitAssignment | None]:
    """Return indices guaranteed to have a normal-only `validation` entry.

    If the dataset already provides one (AD 2), it is returned untouched and the
    assignment is None — recording that no carve happened matters as much as
    recording one that did.
    """
    if "validation" in indices:
        return indices, None
    if "train" not in indices:
        raise KeyError("cannot create a validation split without a train split")

    train, val, assignment = carve_validation(
        indices["train"], val_fraction=val_fraction, seed=seed
    )
    out = dict(indices)
    out["train"], out["validation"] = train, val
    return out, assignment
