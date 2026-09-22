"""Content hashing.

Every hash in this project is SHA-256 of *bytes*, never of a filename. Split
integrity (protocol rule L1) depends on this: two files with different names and
identical content are the same image, and a leakage check keyed on names would
miss it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # 1 MiB


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, streamed so multi-megapixel images do not
    have to be held in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace.

    Used so that two configs that differ only in key ordering hash identically.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_object(obj: Any) -> str:
    """SHA-256 of any JSON-serializable object, order-independent."""
    return hash_bytes(canonical_json(obj).encode("utf-8"))


def short(digest: str, n: int = 12) -> str:
    """Abbreviate a digest for log lines and run names."""
    return digest[:n]
