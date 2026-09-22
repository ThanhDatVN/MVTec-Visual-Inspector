"""Deterministic seeding.

Protocol §4.2 requires seeds {0, 1, 2} for every stochastic method and full
determinism for reported runs. `seed_everything` covers stdlib, numpy and (if
installed) torch. Torch is imported lazily so this module stays usable in the
torch-free CI path.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed every RNG this project touches.

    Args:
        seed: the seed.
        deterministic: if True, also force cuDNN into deterministic mode. This
            costs throughput, so sweeps may set it False — but any run that
            enters the headline table must have it True.

    Returns:
        The seed, so callers can log exactly what was used.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Seeding the *legacy* global RNG is deliberate, not an oversight: third-party
    # code (sklearn, older torch paths) still draws from it, so leaving it unseeded
    # would leave part of the run unreproducible. Our own code uses `make_rng`.
    np.random.seed(seed)  # noqa: NPY002

    try:
        import torch
    except Exception:  # ImportError, or OSError from a broken/blocked install
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    return seed


def make_rng(seed: int) -> np.random.Generator:
    """A local numpy Generator.

    Prefer this over the global numpy RNG inside library code: a function that
    mutates global RNG state makes a caller's results depend on call order,
    which is a reproducibility bug that only shows up weeks later.
    """
    return np.random.default_rng(seed)
