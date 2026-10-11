"""Dataset acquisition: layout of the extracted tree, pinned digests, resumability.

The archives are built here and served over `file://`, so nothing touches the network.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from inspector import fetch as fetch_mod
from inspector.fetch import EXTRACTED_MARKER, RemoteDataset, fetch

CSV = b"object,split,label,image,mask\npcb1,train,normal,pcb1/Data/Images/Normal/0000.JPG,\n"


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _register(monkeypatch, tmp_path, members, *, pin=True, csv_digest=None):
    src = tmp_path / "src"
    src.mkdir()
    archive = _tar(src / "DS.tar", members)
    (src / "1cls.csv").write_bytes(CSV)
    spec = RemoteDataset(
        key="ds", url=archive.as_uri(), filename="DS.tar",
        sha256=_sha(archive) if pin else "",
        extra_files=((
            (src / "1cls.csv").as_uri(), "ds_1cls.csv",
            csv_digest if csv_digest is not None else hashlib.sha256(CSV).hexdigest(),
        ),),
        extract_dirname="DS_ROOT",
        split_file="ds_1cls.csv",
    )
    monkeypatch.setitem(fetch_mod.REGISTRY, "ds", spec)
    return spec


def test_top_level_members_are_extracted_inside_the_dataset_root(monkeypatch, tmp_path):
    # VisA's tar holds `pcb1/...` at the top level, with no `VisA_20220922/` folder.
    _register(monkeypatch, tmp_path, {"pcb1/Data/Images/Normal/0000.JPG": b"img"})
    out = tmp_path / "data"
    m = fetch("ds", out, manifest_dir=tmp_path / "manifests")
    root = Path(m.extracted_to)
    assert root == out / "DS_ROOT"
    assert (root / "pcb1/Data/Images/Normal/0000.JPG").read_bytes() == b"img"
    assert (root / "split_csv" / "1cls.csv").read_bytes() == CSV
    assert not (out / "pcb1").exists()  # nothing lands beside the root
    assert (root / EXTRACTED_MARKER).is_file()


def test_wrapped_archives_are_not_nested_twice(monkeypatch, tmp_path):
    _register(monkeypatch, tmp_path, {"DS_ROOT/pcb1/a.JPG": b"img"})
    m = fetch("ds", tmp_path / "data", manifest_dir=tmp_path / "manifests")
    assert (Path(m.extracted_to) / "pcb1" / "a.JPG").is_file()
    assert not (Path(m.extracted_to) / "DS_ROOT").exists()


def test_manifest_goes_where_it_is_told(monkeypatch, tmp_path):
    _register(monkeypatch, tmp_path, {"pcb1/a.JPG": b"img"})
    monkeypatch.chdir(tmp_path)  # the default would write ./data/manifests here
    fetch("ds", tmp_path / "data", manifest_dir=tmp_path / "elsewhere")
    assert json.loads((tmp_path / "elsewhere" / "ds.fetch.json").read_text())["dataset"] == "ds"
    assert not (tmp_path / "data" / "manifests").exists()


def test_digest_mismatch_is_refused(monkeypatch, tmp_path):
    _register(monkeypatch, tmp_path, {"pcb1/a.JPG": b"img"}, csv_digest="0" * 64)
    with pytest.raises(ValueError, match="does not match the pinned"):
        fetch("ds", tmp_path / "data", manifest_dir=tmp_path / "manifests")


def test_a_half_extracted_root_is_extracted_again(monkeypatch, tmp_path):
    _register(monkeypatch, tmp_path, {"pcb1/a.JPG": b"img", "pcb1/b.JPG": b"img2"})
    root = tmp_path / "data" / "DS_ROOT"
    (root / "pcb1").mkdir(parents=True)
    (root / "pcb1" / "a.JPG").write_bytes(b"img")  # a killed extraction, no marker
    fetch("ds", tmp_path / "data", manifest_dir=tmp_path / "manifests")
    assert (root / "pcb1" / "b.JPG").read_bytes() == b"img2"


def test_a_completed_extraction_is_not_repeated(monkeypatch, tmp_path):
    _register(monkeypatch, tmp_path, {"pcb1/a.JPG": b"img"})
    out = tmp_path / "data"
    fetch("ds", out, manifest_dir=tmp_path / "manifests")
    (out / "DS_ROOT" / "pcb1" / "a.JPG").unlink()
    fetch("ds", out, manifest_dir=tmp_path / "manifests")  # marker matches: no re-extract
    assert not (out / "DS_ROOT" / "pcb1" / "a.JPG").exists()


def test_unsafe_members_are_refused(monkeypatch, tmp_path):
    _register(monkeypatch, tmp_path, {"../escape.txt": b"x"}, pin=False)
    with pytest.raises(ValueError, match="unsafe"):
        fetch("ds", tmp_path / "data", manifest_dir=tmp_path / "manifests")


def test_visa_is_pinned():
    visa = fetch_mod.REGISTRY["visa"]
    assert len(visa.sha256) == 64
    assert all(len(digest) == 64 for _, _, digest in visa.extra_files)


def test_an_unchanged_fetch_leaves_the_manifest_untouched(monkeypatch, tmp_path):
    _register(monkeypatch, tmp_path, {"pcb1/a.JPG": b"img"})
    manifests = tmp_path / "manifests"
    fetch("ds", tmp_path / "data", manifest_dir=manifests)
    first = (manifests / "ds.fetch.json").read_bytes()
    fetch("ds", tmp_path / "data", manifest_dir=manifests)  # "already present", new timestamp
    assert (manifests / "ds.fetch.json").read_bytes() == first
