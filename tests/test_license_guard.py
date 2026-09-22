"""License guard (docs/08 §2).

MVTec AD / AD 2 are CC BY-NC-SA 4.0 and must never enter this repository. The
usual failure is not "I committed the dataset"; it is "I committed one example
image in week 2, removed it in week 3, and pushed the history in week 12". So
this checks the *history*, not just the working tree.

Run both as a test and as the CI `license-guard` job.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Anything that could carry dataset pixels.
FORBIDDEN_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif",
    ".ppm", ".pgm", ".npy", ".npz", ".mp4", ".avi",
}

#: Figures we author ourselves are allowed back in.
ALLOWED_PREFIXES = ("docs/", "reports/figures/")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.skip(f"git unavailable or not a repository: {result.stderr.strip()}")
    return result.stdout


def _offenders(paths: list[str]) -> list[str]:
    out = []
    for path in paths:
        path = path.strip()
        if not path:
            continue
        if Path(path).suffix.lower() not in FORBIDDEN_SUFFIXES:
            continue
        if path.startswith(ALLOWED_PREFIXES):
            continue
        out.append(path)
    return out


def test_working_tree_contains_no_dataset_images():
    tracked = _git("ls-files").splitlines()
    offenders = _offenders(tracked)
    assert not offenders, (
        "image files are tracked in the working tree; MVTec data is CC BY-NC-SA "
        "and must not be committed:\n  " + "\n  ".join(offenders)
    )


def test_git_history_never_contained_dataset_images():
    """A file deleted in a later commit is still published once the history is
    pushed, so the tip being clean is not sufficient."""
    added = _git("log", "--all", "--diff-filter=A", "--name-only", "--pretty=format:")
    offenders = sorted(set(_offenders(added.splitlines())))
    assert not offenders, (
        "image files appear in git history and would be published with it:\n  "
        + "\n  ".join(offenders)
    )


def test_no_dataset_directory_is_tracked():
    tracked = [p for p in _git("ls-files").splitlines() if p.startswith("data/")]
    disallowed = [p for p in tracked if not p.startswith("data/manifests/")]
    assert not disallowed, (
        "only manifests may be tracked under data/:\n  " + "\n  ".join(disallowed)
    )


def test_gitignore_declares_the_dataset_exclusions():
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/**", "*.png", "*.npy"):
        assert pattern in text, f".gitignore is missing {pattern!r}"


def test_attribution_file_is_present_and_complete():
    text = (REPO / "ATTRIBUTION.md").read_text(encoding="utf-8")
    for required in (
        "CC BY-NC-SA 4.0",
        "MVTec",
        "10.1007/s11263-026-02743-0",  # AD 2
        "10.1109/CVPR.2019.00982",  # AD classic
        "non-commercial",
    ):
        assert required in text, f"ATTRIBUTION.md is missing {required!r}"


def test_license_scopes_itself_to_code():
    """The MIT license must not appear to cover the dataset or artifacts
    derived from it."""
    text = (REPO / "LICENSE").read_text(encoding="utf-8")
    assert "SOURCE CODE" in text
    assert "CC BY-NC-SA 4.0" in text
