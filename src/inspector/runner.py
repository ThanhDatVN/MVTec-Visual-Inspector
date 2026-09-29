"""The experiment runner — one source of truth for CLI and notebooks.

Before this module, the notebook appended pipeline results with empty
`config_hash` and `git_sha`, resumed by human-readable keys that omitted some
effective parameters, and the CLI hashed the config *before* `--methods`,
`--categories` and `--seeds` changed what actually ran (docs/13, F04/F05). A
reported row could not be traced to the configuration that produced it.

Now every run is described by a `RunSpec` that is **fully resolved**: the
model's effective hyperparameters including defaults, the input transform, the
evaluation configuration, the seed, the split listing, and the protocol version.
Its hash, combined with a hash of the package source (`implementation_id`), is
the `run_id`. Changing any of those changes the id; a restarted session reloads
completed runs by id instead of recomputing or silently reusing them.

Registry layout (`reports/runs/<run_id>/`)::

    spec.json         the resolved specification and the identity inputs
    result.json       status, every metric, provenance, timings — or the failure
    <split>.predictions.json   per-image ids, labels, scores (small, committed)
    <split>.maps.npz           raw maps, float16 (large, gitignored)

Failed and out-of-memory runs are kept, with their reason: a registry that only
remembers successes misrepresents what the method can do.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

from .data import ensure_validation, load_category
from .data.core import DatasetIndex
from .data.transforms import ImageTransform
from .evaluate import EvalConfig, SplitPredictions
from .models import build_model
from .models.base import AnomalyModel
from .pipeline import ExperimentResult, run_with_predictions
from .utils.env import capture, git_info
from .utils.hashing import canonical_json, hash_object
from .utils.seed import seed_everything

#: Bump when the protocol changes in a way that makes old results incomparable.
#: v2: conservative strict thresholds (F01), input-pixel smoothing, corrected
#: PatchCore scoring (F02/F03), streaming evaluation including normals (F09).
PROTOCOL_VERSION = "2"

PACKAGE_ROOT = Path(__file__).resolve().parent


#: Modules that read, analyse or present stored results but cannot change the
#: numbers a run produces. Editing them must not invalidate hours of completed
#: runs. Everything else — data, features, models, metrics, post-processing,
#: evaluation, and this runner — is hashed. When unsure, a module stays in.
PRESENTATION_MODULES: tuple[str, ...] = (
    "analysis.py", "cli.py", "report.py", "results.py", "stats.py", "tracking.py",
    "api/", "app/", "viz/",
)


def affects_results(rel_path: str) -> bool:
    """Whether a package-relative source path is part of the implementation id."""
    return not any(
        rel_path.startswith(m) if m.endswith("/") else rel_path == m for m in PRESENTATION_MODULES
    )


def source_hash(root: Path) -> str:
    """SHA-256 over the result-producing Python sources under `root`.

    Line endings are normalized first. Hashing raw bytes made the id depend on
    the checkout rather than the code: a Windows worktree (CRLF) and a working
    tree with LF-written files produced different ids for identical sources.
    """
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if not affects_results(rel):
            continue
        h.update(rel.encode())
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


@cache
def implementation_id() -> str:
    """The package's `source_hash`, in sorted path order.

    Changes only when code that can move a number changes — unlike a git SHA, a
    documentation commit or a new analysis module does not invalidate every
    completed run, while any edit to data, model or evaluation code does.
    """
    return source_hash(PACKAGE_ROOT)


def split_listing(indices: dict[str, DatasetIndex]) -> dict[str, str]:
    """Hash of each split's image ids: cheap, and it pins the split assignment.

    File *contents* are verified by the audit (rule L1) and by `fetch --verify`,
    and the content manifest is recorded as provenance; putting the content
    hash in the identity would mean hashing ~1.6 GB before every resume check.
    """
    return {
        name: hash_object(sorted(s.rel_id(idx.root) for s in idx))[:16]
        for name, idx in sorted(indices.items())
    }


@dataclass(frozen=True)
class RunSpec:
    """Everything that determines a run's result."""

    dataset: dict[str, Any]
    model: dict[str, Any]
    transform: dict[str, Any]
    evaluation: dict[str, Any]
    seed: int
    role: str = "development"
    protocol_version: str = PROTOCOL_VERSION

    def identity(self) -> dict[str, Any]:
        return {**asdict(self), "implementation_id": implementation_id()}

    @property
    def run_id(self) -> str:
        return hash_object(self.identity())[:16]

    @property
    def config_hash(self) -> str:
        """Hash of the spec alone, without the implementation — two runs with
        the same config hash and different run ids differ only in code."""
        return hash_object(asdict(self))[:16]


# ---------------------------------------------------------------------------
# Building a run from a configuration
# ---------------------------------------------------------------------------


def transform_from_config(cfg: dict[str, Any]) -> ImageTransform:
    pre = cfg.get("preprocess", {})
    return ImageTransform(
        mode=pre.get("resize_mode", "aspect_preserving"),
        long_side=pre.get("long_side"),
        short_side=pre.get("short_side"),
        interpolation_down=pre.get("interpolation_down", "area"),
        interpolation_up=pre.get("interpolation_up", "linear"),
        normalize=pre.get("normalize", "imagenet"),
    )


def eval_config_from_config(cfg: dict[str, Any]) -> EvalConfig:
    post = cfg.get("postproc", {})
    thr = cfg.get("thresholds", {})
    met = cfg.get("metrics", {})
    return EvalConfig(
        smoothing_sigma=float(post.get("gaussian_sigma", 4.0)),
        target_fpr=float(thr.get("target_fpr", 0.01)),
        threshold_policy=str(thr.get("policy", "relax")),
        sigma_n=float(thr.get("op_sigma_n", 3.0)),
        aupro_limits=tuple(float(x) for x in met.get("aupro_limits", (0.05, 0.30))),
        aupro_num_thresholds=int(met.get("aupro_num_thresholds", 512)),
        max_negative_samples=int(met.get("max_negative_samples", 2_000_000)),
        sampling_seed=int(met.get("sampling_seed", 0)),
        report_oracle=bool(thr.get("report_oracle", True)),
    )


def model_kwargs(cfg: dict[str, Any], method: str) -> dict[str, Any]:
    """Constructor arguments for `method` from the config's `model:` block.

    The block applies only when it names this method; any other method runs at
    its defaults, and those defaults are captured by `hparams()` either way.
    """
    block = dict(cfg.get("model") or {})
    if block.get("name") != method:
        return {}
    block.pop("name", None)
    if "layers" in block and isinstance(block["layers"], list):
        block["layers"] = tuple(block["layers"])
    return block


def prepare(
    cfg: dict[str, Any],
    *,
    method: str,
    category: str,
    seed: int,
    data_root: str | Path,
    role: str = "development",
    overrides: dict[str, Any] | None = None,
) -> tuple[RunSpec, AnomalyModel, dict[str, DatasetIndex]]:
    """Resolve a run: build the model, index the data, and freeze the spec."""
    data = cfg.get("data", {})
    transform = transform_from_config(cfg)
    kwargs = {**model_kwargs(cfg, method), **(overrides or {})}
    model = build_model(method, transform, seed=seed, **kwargs)

    indices = load_category(
        data_root, category, layout=data.get("layout", "mvtec_ad2"), split_csv=data.get("split_csv")
    )
    indices, _ = ensure_validation(
        indices,
        val_fraction=float(data.get("val_carve_fraction", 0.15)),
        seed=int(data.get("val_carve_seed", 0)),
    )

    spec = RunSpec(
        dataset={
            "layout": data.get("layout", "mvtec_ad2"),
            "category": category,
            "test_split": data.get("test_split", "test_public"),
            "val_carve_fraction": float(data.get("val_carve_fraction", 0.15)),
            "val_carve_seed": int(data.get("val_carve_seed", 0)),
            "split_listing": split_listing(indices),
        },
        model={"name": method, "class": type(model).__name__, "hparams": model.hparams()},
        transform=transform.as_dict(),
        evaluation=eval_config_from_config(cfg).as_dict(),
        seed=seed,
        role=role,
    )
    return spec, model, indices


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _result_from_dict(data: dict[str, Any]) -> ExperimentResult:
    names = {f.name for f in fields(ExperimentResult)}
    return ExperimentResult(**{k: v for k, v in data.items() if k in names})


class Registry:
    """Filesystem registry of runs, keyed by `run_id`."""

    def __init__(self, root: str | Path = "reports/runs") -> None:
        self.root = Path(root)

    def path(self, run_id: str) -> Path:
        return self.root / run_id

    def load_result(self, run_id: str) -> dict[str, Any] | None:
        f = self.path(run_id) / "result.json"
        if not f.is_file():
            return None
        return json.loads(f.read_text(encoding="utf-8"))

    def completed(self, run_id: str) -> ExperimentResult | None:
        """A completed result whose prediction artifacts are present."""
        data = self.load_result(run_id)
        if not data or data.get("status") != "completed":
            return None
        directory = self.path(run_id)
        if not any(directory.glob("*.predictions.json")):
            return None  # result without its predictions is not a completed run
        return _result_from_dict(data["result"])

    def write(
        self,
        spec: RunSpec,
        result: ExperimentResult,
        predictions: list[SplitPredictions],
        *,
        data_root: str | Path,
        provenance: dict[str, Any],
        save_maps: bool = True,
    ) -> Path:
        directory = self.path(spec.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(directory / "spec.json", json.dumps(spec.identity(), indent=2, default=str))
        for preds in predictions:
            preds.save(directory, with_maps=save_maps, relative_to=data_root)
        record = {
            "run_id": spec.run_id,
            "status": "completed",
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prediction_digests": {p.split: p.digest() for p in predictions},
            "provenance": provenance,
            "result": asdict(result),
        }
        _atomic_write(directory / "result.json", json.dumps(record, indent=2, default=str))
        return directory

    def write_failure(self, spec: RunSpec, error: BaseException, provenance: dict[str, Any]) -> Path:
        directory = self.path(spec.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(directory / "spec.json", json.dumps(spec.identity(), indent=2, default=str))
        record = {
            "run_id": spec.run_id,
            "status": "failed",
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "failure_reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exception(type(error), error, error.__traceback__)[-6:],
            "provenance": provenance,
        }
        _atomic_write(directory / "result.json", json.dumps(record, indent=2, default=str))
        return directory

    def results(self, *, status: str = "completed", **match: Any) -> list[ExperimentResult]:
        """All registry results, optionally filtered on result fields."""
        out = []
        for f in sorted(self.root.glob("*/result.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("status") != status or "result" not in data:
                continue
            result = _result_from_dict(data["result"])
            if all(getattr(result, k, None) == v for k, v in match.items()):
                out.append(result)
        return out

    def load_predictions(
        self, run_id: str, split: str, *, root: str | Path | None = None
    ) -> SplitPredictions:
        return SplitPredictions.load(self.path(run_id), split, root=root)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute(
    spec: RunSpec,
    model: AnomalyModel,
    indices: dict[str, DatasetIndex],
    *,
    data_root: str | Path,
    registry: Registry,
    force: bool = False,
    save_maps: bool = True,
) -> tuple[ExperimentResult, bool]:
    """Run once, or reload the completed run. Returns (result, reused)."""
    if not force:
        previous = registry.completed(spec.run_id)
        if previous is not None:
            return previous, True

    info = git_info()
    provenance = {
        "git_sha": info["git_sha"],
        "git_branch": info["git_branch"],
        "dirty": info["dirty"],
        "implementation_id": implementation_id(),
        "environment": capture()["libraries"],
        "device": capture()["device"],
        "dataset_manifests": {
            name: str(idx.manifest()["manifest_sha256"]) for name, idx in sorted(indices.items())
        },
    }

    seed_everything(spec.seed)
    try:
        result, val_preds, test_preds = run_with_predictions(
            model,
            indices,
            seed=spec.seed,
            test_split=spec.dataset["test_split"],
            eval_config=EvalConfig(**{
                **spec.evaluation,
                "aupro_limits": tuple(spec.evaluation["aupro_limits"]),
            }),
        )
    except Exception as error:  # recorded, then re-raised: a failure is a result too
        registry.write_failure(spec, error, provenance)
        raise

    result.run_id = spec.run_id
    result.config_hash = spec.config_hash
    result.implementation_id = implementation_id()[:16]
    result.git_sha = (info["git_sha"] or "unknown")[:12]
    result.dirty = bool(info["dirty"])
    result.notes = f"role={spec.role}"
    registry.write(
        spec, result, [val_preds, test_preds],
        data_root=data_root, provenance=provenance, save_maps=save_maps,
    )
    return result, False


def spec_summary(spec: RunSpec) -> str:
    return canonical_json({"model": spec.model["name"], "category": spec.dataset["category"],
                           "seed": spec.seed, "run_id": spec.run_id})
