"""The shared runner: run identity, resume, failure records (docs/13, F04/F05)."""

from __future__ import annotations

import json

import pytest

from inspector.config import load_config
from inspector.runner import Registry, execute, implementation_id, prepare

CONFIG = "configs/data/synth_strip.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


def make(cfg, synthetic_root, **kwargs):
    defaults = {"method": "pixel_pca", "category": "synth_strip", "seed": 0}
    return prepare(cfg, data_root=synthetic_root, **{**defaults, **kwargs})


def test_run_id_is_stable(cfg, synthetic_root):
    a, _, _ = make(cfg, synthetic_root)
    b, _, _ = make(cfg, synthetic_root)
    assert a.run_id == b.run_id


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 1},
        {"method": "histogram"},
        {"overrides": {"n_components": 8}},
        {"overrides": {"work_size": 48}},
    ],
)
def test_meaningful_changes_change_the_run_id(cfg, synthetic_root, change):
    """Including a model *default* that nobody wrote in the config: changing a
    default must not let an old completed run be reused."""
    base, _, _ = make(cfg, synthetic_root)
    other, _, _ = make(cfg, synthetic_root, **change)
    assert base.run_id != other.run_id


def test_evaluation_settings_are_part_of_the_identity(synthetic_root):
    from inspector.config import deep_merge

    base_cfg = load_config(CONFIG)
    changed = deep_merge(base_cfg, {"postproc": {"gaussian_sigma": 0.0}})
    a, _, _ = make(base_cfg, synthetic_root)
    b, _, _ = make(changed, synthetic_root)
    assert a.run_id != b.run_id, "smoothing changed but the run id did not"


def test_spec_captures_effective_hyperparameters(cfg, synthetic_root):
    spec, model, _ = make(cfg, synthetic_root)
    assert spec.model["hparams"] == model.hparams()
    assert spec.model["hparams"]["n_components"] == 16  # a constructor default


def test_implementation_id_is_a_source_hash():
    assert len(implementation_id()) == 64


@pytest.mark.parametrize(
    ("rel_path", "hashed"),
    [
        ("models/patchcore.py", True),
        ("metrics/aupro.py", True),
        ("evaluate.py", True),
        ("runner.py", True),
        ("data/visa.py", True),
        ("analysis.py", False),
        ("results.py", False),
        ("viz/heatmap.py", False),
    ],
)
def test_only_result_producing_code_is_in_the_implementation_id(rel_path, hashed):
    """An analysis or table-formatting edit must not force hours of reruns;
    a model, metric or data edit must."""
    from inspector.runner import affects_results

    assert affects_results(rel_path) is hashed


def test_execute_records_complete_provenance(cfg, synthetic_root, tmp_path):
    registry = Registry(tmp_path / "runs")
    spec, model, indices = make(cfg, synthetic_root)
    result, reused = execute(spec, model, indices, data_root=synthetic_root, registry=registry)

    assert not reused
    assert result.run_id == spec.run_id
    assert result.config_hash and result.implementation_id and result.git_sha
    record = json.loads((tmp_path / "runs" / spec.run_id / "result.json").read_text())
    assert record["status"] == "completed"
    assert set(record["provenance"]["dataset_manifests"]) == {"train", "validation", "test_public"}
    assert (tmp_path / "runs" / spec.run_id / "test_public.predictions.json").is_file()


def test_resume_reloads_the_completed_run(cfg, synthetic_root, tmp_path):
    registry = Registry(tmp_path / "runs")
    spec, model, indices = make(cfg, synthetic_root)
    first, _ = execute(spec, model, indices, data_root=synthetic_root, registry=registry)

    spec2, model2, indices2 = make(cfg, synthetic_root)
    again, reused = execute(spec2, model2, indices2, data_root=synthetic_root, registry=registry)
    assert reused
    assert again.image_auroc == first.image_auroc
    assert not model2.fitted, "a reused run must not refit"


def test_registry_table_includes_resumed_runs(cfg, synthetic_root, tmp_path):
    """The old notebook helper returned None for skipped runs, so a resumed
    session's summary silently lost everything completed earlier."""
    registry = Registry(tmp_path / "runs")
    for seed in (0, 1):
        spec, model, indices = make(cfg, synthetic_root, method="random", seed=seed)
        execute(spec, model, indices, data_root=synthetic_root, registry=registry)
    assert len(registry.results(method="random")) == 2


def test_failures_are_recorded(cfg, synthetic_root, tmp_path):
    registry = Registry(tmp_path / "runs")
    spec, model, indices = make(cfg, synthetic_root)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated out of memory")

    model._fit = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        execute(spec, model, indices, data_root=synthetic_root, registry=registry)
    record = json.loads((tmp_path / "runs" / spec.run_id / "result.json").read_text())
    assert record["status"] == "failed"
    assert "simulated out of memory" in record["failure_reason"]
    assert registry.completed(spec.run_id) is None


def test_stored_predictions_reproduce_the_metrics(cfg, synthetic_root, tmp_path):
    from inspector.evaluate import EvalConfig, evaluate

    registry = Registry(tmp_path / "runs")
    spec, model, indices = make(cfg, synthetic_root)
    result, _ = execute(spec, model, indices, data_root=synthetic_root, registry=registry)

    val = registry.load_predictions(spec.run_id, "validation", root=synthetic_root)
    test = registry.load_predictions(spec.run_id, "test_public", root=synthetic_root)
    ev = spec.evaluation
    again = evaluate(val, test, EvalConfig(**{**ev, "aupro_limits": tuple(ev["aupro_limits"])}))
    assert again["image_auroc"] == pytest.approx(result.image_auroc)
    assert again["aupro_005"] == pytest.approx(result.aupro_005, abs=5e-3)  # float16 maps


def test_cli_accepts_every_model_family(synthetic_root, tmp_path):
    """F05: `--methods patchcore` used to fail at validation before fitting."""
    from inspector.cli import build_parser

    parser = build_parser()
    for method in ("patchcore", "cae", "pixel_pca"):
        args = parser.parse_args(["run", "-c", CONFIG, "--methods", method,
                                  "--data-root", str(synthetic_root)])
        assert args.methods == [method]


def test_cli_run_populates_and_reuses_the_registry(synthetic_root, tmp_path):
    from inspector.cli import main

    argv = ["run", "-c", CONFIG, "--methods", "mean_intensity", "--seeds", "0",
            "--data-root", str(synthetic_root), "--registry", str(tmp_path / "runs"),
            "--results", str(tmp_path / "table.csv")]
    assert main(argv) == 0
    assert main(argv) == 0  # second call reuses
    assert len(list((tmp_path / "runs").glob("*/result.json"))) == 1
    assert (tmp_path / "table.csv").read_text().count("\n") == 2  # header + one row


def test_model_fragment_composes_over_a_data_config():
    """Model configs are fragments. When they inherited base.yaml, composing
    one after a data config silently reset the layout to mvtec_ad2 and the
    category to None."""
    from inspector.cli import _resolve, build_parser

    args = build_parser().parse_args([
        "run", "-c", "configs/data/visa_pcb1.yaml", "-c", "configs/models/patchcore.yaml",
        "--data-root", "unused",
    ])
    cfg = _resolve(args)
    assert cfg["data"]["layout"] == "visa"
    assert cfg["data"]["category"] == "pcb1"
    assert cfg["model"]["name"] == "patchcore"
    assert cfg["preprocess"]["long_side"] == 320
