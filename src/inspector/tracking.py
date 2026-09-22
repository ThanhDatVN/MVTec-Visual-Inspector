"""MLflow run tracking (docs/06 §4).

Two rules are enforced here rather than left to discipline:

1. **Every run records its provenance** — git SHA, dirty flag, config hash,
   dataset manifest hash, library versions, device.
2. **A dirty working tree tags the run `dirty=True`.** Dirty runs may not enter
   the headline table. This single rule prevents more irreproducibility than
   anything else in the project, and it only works if it is automatic.

MLflow is optional: with tracking disabled (or the package absent) the run still
executes and still returns its metrics, so the test suite and a quick laptop
experiment never depend on a tracking server being up.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import config_hash, flatten
from .utils.env import capture
from .utils.hashing import short


class RunTracker:
    """A thin wrapper over an MLflow run. Also usable as a no-op."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        enabled: bool = True,
        uri: str | None = None,
        experiment: str = "mvtec-visual-inspector",
        run_name: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.config_hash = config_hash(cfg)
        self.provenance = capture()
        self.enabled = enabled
        self._mlflow = None
        self._active = None
        self.metrics: dict[str, Any] = {}

        if not enabled:
            return
        try:
            import mlflow
        except ImportError:
            self.enabled = False
            return

        self._mlflow = mlflow
        mlflow.set_tracking_uri(uri or f"file:{Path.cwd() / 'mlruns'}")
        mlflow.set_experiment(experiment)
        self.run_name = run_name or self._default_name()

    def _default_name(self) -> str:
        exp = self.cfg.get("experiment", {})
        data = self.cfg.get("data", {})
        parts = [
            str(exp.get("tier") or "T?"),
            str(self.cfg.get("model", {}).get("name") or exp.get("name") or "run"),
            str(data.get("category") or "?"),
            f"s{self.cfg.get('run', {}).get('seed', 0)}",
            short(self.config_hash, 8),
        ]
        return "-".join(parts)

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> RunTracker:
        if self.enabled and self._mlflow is not None:
            self._active = self._mlflow.start_run(run_name=self.run_name)
            self._log_startup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._active is not None and self._mlflow is not None:
            if exc_type is not None:
                self._mlflow.set_tag("failed", True)
                self._mlflow.set_tag("error", f"{exc_type.__name__}: {exc}")
            self._mlflow.end_run()
            self._active = None

    def _log_startup(self) -> None:
        assert self._mlflow is not None
        params = flatten(self.cfg)
        # MLflow caps param values; long ones go to an artifact instead.
        self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        self._mlflow.set_tags(
            {
                "config_hash": self.config_hash,
                "git_sha": self.provenance.get("git_sha") or "unknown",
                "git_branch": self.provenance.get("git_branch") or "unknown",
                "dirty": self.provenance.get("dirty", True),
                "tier": self.cfg.get("experiment", {}).get("tier") or "unknown",
                "owner": self.cfg.get("experiment", {}).get("owner", "own"),
                "category": self.cfg.get("data", {}).get("category") or "unknown",
            }
        )
        self.log_dict(self.cfg, "config.resolved.yaml.json")
        self.log_dict(self.provenance, "provenance.json")

    # -- logging -----------------------------------------------------------
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if _is_number(v)}
        self.metrics.update(clean)
        if self.enabled and self._mlflow is not None and clean:
            self._mlflow.log_metrics(clean, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.set_tags(tags)

    def log_dict(self, payload: dict[str, Any], name: str) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.log_text(json.dumps(payload, indent=2, default=str), name)

    def log_artifact(self, path: str | Path, subdir: str | None = None) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.log_artifact(str(path), artifact_path=subdir)

    def log_dataset_manifest(self, manifests: dict[str, dict]) -> None:
        """Record the split manifests so a result can be tied to exact bytes."""
        digests = {name: m["manifest_sha256"] for name, m in manifests.items()}
        self.set_tags({f"manifest.{name}": d for name, d in digests.items()})
        self.log_dict(digests, "dataset_manifests.json")

    @property
    def reportable(self) -> bool:
        """False when the tree was dirty: such a run may not enter the headline
        table (docs/06 §2)."""
        return not self.provenance.get("dirty", True)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@contextmanager
def track(cfg: dict[str, Any], **kwargs):
    """`with track(cfg) as run:` — the standard entry point for an experiment."""
    tracker = RunTracker(cfg, **kwargs)
    with tracker:
        yield tracker
