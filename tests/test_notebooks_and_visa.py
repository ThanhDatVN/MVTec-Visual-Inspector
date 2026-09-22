"""Notebooks and the VisA reader.

The notebook tests exist because a notebook is the easiest artifact in a project
to break silently: nothing imports it, nothing runs it in CI, and a stale cell
looks identical to a working one until someone opens it.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((REPO / "notebooks").glob("*.ipynb"))


# --- notebooks --------------------------------------------------------------


def test_notebooks_exist():
    assert NOTEBOOKS, "no notebooks found; run scripts/build_notebooks.py"
    names = {p.name for p in NOTEBOOKS}
    assert "01_laptop_data_and_eda.ipynb" in names
    assert "02_kaggle_experiments.ipynb" in names


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json_and_schema(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook has no cells"
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_compiles(path):
    """A syntax error in cell 14 is only discovered by whoever runs cell 14,
    which on Kaggle is 40 minutes into a session."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            compile(source, f"{path.name}:cell{i}", "exec")
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{path.name} cell {i} line {exc.lineno}: {exc.msg}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_carry_no_outputs(path):
    """Committed outputs bloat diffs and, worse, make a notebook look like it
    ran when the code beneath has since changed."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [], f"cell {i} has stored outputs"
            assert cell.get("execution_count") is None, f"cell {i} has an execution count"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_are_thin_drivers(path):
    """docs/06 §1: notebooks install the package and call its API. Model code in
    a cell cannot be tested, reviewed or re-run, so it must not accumulate."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        assert "class " not in source, f"cell {i} defines a class; move it into src/inspector"
        assert "nn.Module" not in source, f"cell {i} defines a network; move it into src/inspector"
        assert "torch.optim" not in source, f"cell {i} trains in the notebook; use the pipeline"


def test_notebooks_are_regenerable():
    """The builder is the source of truth; the committed notebooks must match
    what it currently emits."""
    before = {p: p.read_text(encoding="utf-8") for p in NOTEBOOKS}
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_notebooks.py")],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for path, original in before.items():
        assert path.read_text(encoding="utf-8") == original, (
            f"{path.name} differs from what scripts/build_notebooks.py produces; "
            "edit the builder, not the notebook"
        )


def test_notebooks_carry_attribution():
    """CC BY 4.0 obligates attribution wherever the data is used."""
    for path in NOTEBOOKS:
        text = path.read_text(encoding="utf-8")
        assert "VisA" in text
        assert "CC BY 4.0" in text
        assert "ATTRIBUTION.md" in text


# --- VisA reader ------------------------------------------------------------


def write_visa_tree(root: Path, category: str = "pcb1", n_train: int = 6) -> Path:
    """A miniature VisA tree with the real layout and a real-format split CSV."""
    from PIL import Image

    rows = [["object", "split", "label", "image", "mask"]]

    normal_dir = root / category / "Data" / "Images" / "Normal"
    anomaly_dir = root / category / "Data" / "Images" / "Anomaly"
    mask_dir = root / category / "Data" / "Masks" / "Anomaly"
    for directory in (normal_dir, anomaly_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for i in range(n_train):
        name = f"{i:04d}.JPG"
        Image.new("RGB", (32, 32), (90, 90, 90)).save(normal_dir / name)
        rel = f"{category}/Data/Images/Normal/{name}"
        rows.append([category, "train", "normal", rel, ""])

    for i in range(2):
        name = f"t{i:03d}.JPG"
        Image.new("RGB", (32, 32), (92, 92, 92)).save(normal_dir / name)
        rows.append([category, "test", "normal", f"{category}/Data/Images/Normal/{name}", ""])

    for i in range(3):
        name = f"{i:03d}.JPG"
        Image.new("RGB", (32, 32), (60, 60, 60)).save(anomaly_dir / name)
        mask = Image.new("L", (32, 32), 0)
        mask.paste(255, (8, 8, 14, 14))
        mask.save(mask_dir / f"{i:03d}.png")
        rows.append([
            category,
            "test",
            "anomaly",
            f"{category}/Data/Images/Anomaly/{name}",
            f"{category}/Data/Masks/Anomaly/{i:03d}.png",
        ])

    csv_path = root / "split_csv" / "1cls.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return root


@pytest.fixture
def visa_tree(tmp_path):
    return write_visa_tree(tmp_path / "VisA")


def test_reads_the_official_split(visa_tree):
    from inspector.data import load_category

    indices = load_category(visa_tree, "pcb1", layout="visa")
    assert set(indices) == {"train", "test"}
    assert indices["train"].n_normal == 6
    assert indices["train"].n_anomalous == 0
    assert indices["test"].n_normal == 2
    assert indices["test"].n_anomalous == 3


def test_train_split_is_normal_only(visa_tree):
    """The one-class protocol depends on it, so it is asserted rather than
    assumed from the CSV's contents."""
    from inspector.data import load_category

    assert load_category(visa_tree, "pcb1", layout="visa")["train"].n_anomalous == 0


def test_masks_are_paired_with_anomalous_images(visa_tree):
    from inspector.data import ANOMALOUS, load_category

    for sample in load_category(visa_tree, "pcb1", layout="visa")["test"].filter(label=ANOMALOUS):
        assert sample.mask_path is not None
        assert sample.mask_path.is_file()


def test_missing_split_csv_is_an_error_not_a_fallback(tmp_path):
    """Scanning directories instead would invent a split, and a result on an
    invented split is comparable with nothing."""
    from inspector.data.visa import discover_visa

    (tmp_path / "pcb1" / "Data" / "Images" / "Normal").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="split CSV not found"):
        discover_visa(tmp_path, "pcb1")


def test_missing_image_on_disk_is_reported(visa_tree):
    from inspector.data.visa import discover_visa

    next((visa_tree / "pcb1" / "Data" / "Images" / "Normal").glob("*.JPG")).unlink()
    with pytest.raises(FileNotFoundError, match="missing on disk"):
        discover_visa(visa_tree, "pcb1")


def test_unknown_category_is_rejected(visa_tree):
    from inspector.data.visa import discover_visa

    with pytest.raises(ValueError, match="unknown VisA category"):
        discover_visa(visa_tree, "not_a_category")


def test_summary_counts_match_the_csv(visa_tree):
    from inspector.data.visa import summarize

    summary = summarize(visa_tree)["pcb1"]
    assert summary["train_normal"] == 6
    assert summary["test_anomaly"] == 3
    assert summary["group"] == "complex_structure"


def test_groups_partition_the_categories():
    from inspector.data.visa import VISA_CATEGORIES, VISA_GROUPS

    flat = [c for group in VISA_GROUPS.values() for c in group]
    assert len(flat) == len(set(flat)) == len(VISA_CATEGORIES) == 12


def test_validation_is_carved_because_visa_ships_none(visa_tree):
    from inspector.data import ensure_validation, load_category

    indices, assignment = ensure_validation(
        load_category(visa_tree, "pcb1", layout="visa"), val_fraction=0.3, seed=0
    )
    assert assignment is not None, "VisA has no validation split; one must be carved"
    assert indices["validation"].n_anomalous == 0
    assert len(indices["train"]) + len(indices["validation"]) == 6
