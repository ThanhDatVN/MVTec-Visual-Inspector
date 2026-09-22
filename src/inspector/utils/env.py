"""Run provenance capture.

Protocol §4.2: every run records git SHA + dirty flag, library versions, device
and CUDA version. Engineering rule: a dirty git tree marks the run `dirty=True`,
and dirty runs may not enter the headline table.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info() -> dict[str, Any]:
    """Current commit, branch, and whether the working tree has uncommitted changes."""
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_sha": sha,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # status is None when git failed; treat that as dirty rather than clean,
        # because "we could not tell" must never be recorded as "it was clean".
        "dirty": True if status is None else bool(status),
    }


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "sklearn", "skimage", "PIL", "torch", "torchvision", "cv2"):
        try:
            mod = __import__(name)
        except ImportError:
            continue
        except Exception as exc:
            # A package can be installed and still fail to load — a mismatched
            # CUDA runtime, or Windows Application Control blocking an unsigned
            # DLL, both raise OSError from `import torch`. Provenance capture
            # must record that and carry on, never abort the run it is
            # describing.
            versions[name] = f"present-but-unloadable: {type(exc).__name__}"
            continue
        versions[name] = str(getattr(mod, "__version__", "unknown"))
    return versions


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    try:
        import torch
    except Exception:  # ImportError, or OSError from a broken/blocked install
        info["cuda_available"] = False
        return info

    info["cuda_available"] = torch.cuda.is_available()
    info["torch_cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_total_mb"] = round(props.total_memory / 1024**2)
        info["cudnn_version"] = torch.backends.cudnn.version()
    return info


def capture() -> dict[str, Any]:
    """Full provenance record, logged as MLflow params/tags on every run."""
    return {**git_info(), "libraries": library_versions(), "device": device_info()}
