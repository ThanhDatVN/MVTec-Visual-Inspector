"""Dataset acquisition.

Only datasets that can be fetched without an account live here. MVTec AD and
AD 2 require registration and are therefore a manual step, documented in
[docs/10-datasets.md](../../docs/10-datasets.md) rather than automated behind a
command that would only fail.

Every fetch records a SHA-256 manifest of what it downloaded. A silently updated
or partially written archive is otherwise indistinguishable from a good one
until a model trained on it behaves strangely weeks later.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .utils.hashing import hash_file


@dataclass(frozen=True)
class RemoteDataset:
    key: str
    url: str
    filename: str
    extra_files: tuple[tuple[str, str], ...] = ()
    #: Directory the archive expands into, relative to the download dir.
    extract_dirname: str = ""
    license_note: str = ""
    citation: str = ""


VISA = RemoteDataset(
    key="visa",
    url="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
    filename="VisA_20220922.tar",
    extra_files=(
        (
            "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv",
            "visa_1cls.csv",
        ),
    ),
    extract_dirname="VisA_20220922",
    license_note=(
        "CC BY 4.0 per the AWS Open Data registry and the source paper. Some third-party "
        "documentation states CC BY-NC-SA 4.0; until that is settled this project behaves as if "
        "the stricter reading applied and redistributes no images."
    ),
    citation=(
        "Zou, Y., Jeong, J., Pemula, L., Zhang, D., Dabeer, O. 'SPot-the-Difference "
        "Self-Supervised Pre-training for Anomaly Detection and Segmentation.' ECCV 2022. "
        "arXiv:2207.14315. Accessed from https://registry.opendata.aws/visa"
    ),
)

@dataclass(frozen=True)
class FetchResult:
    """What a fetch did, and the hashes needed to prove it later."""

    dataset: str
    url: str
    fetched_utc: str
    action: str
    files: dict[str, str]
    extracted_to: str | None
    license_note: str
    citation: str


REGISTRY: dict[str, RemoteDataset] = {VISA.key: VISA}


def _download(url: str, dest: Path, *, chunk: int = 1 << 20, progress=None) -> Path:
    """Stream a URL to `dest`, writing to a temporary file first.

    The temp-then-rename is what makes an interrupted download detectable: a
    partial file never occupies the final name, so a later run cannot mistake it
    for a complete one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        with open(tmp, "wb") as fh:
            while block := response.read(chunk):
                fh.write(block)
                written += len(block)
                if progress:
                    progress(written, total)

    if total and written != total:
        tmp.unlink(missing_ok=True)
        raise OSError(f"truncated download: got {written} of {total} bytes from {url}")

    tmp.replace(dest)
    return dest


def fetch(
    key: str,
    out_dir: str | Path = "data/raw",
    *,
    extract: bool = True,
    force: bool = False,
    progress=None,
) -> FetchResult:
    """Download (and optionally extract) a dataset, recording its manifest."""
    try:
        spec = REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown dataset {key!r}; available: {sorted(REGISTRY)}") from None

    out_dir = Path(out_dir)
    archive = out_dir / spec.filename

    if archive.is_file() and not force:
        action = "already present"
    else:
        _download(spec.url, archive, progress=progress)
        action = "downloaded"

    files = {spec.filename: hash_file(archive)}
    for url, name in spec.extra_files:
        path = out_dir / name
        if not path.is_file() or force:
            _download(url, path)
        files[name] = hash_file(path)

    extracted: Path | None = None
    if extract:
        extracted = _extract(archive, out_dir, spec)
        # The split CSV lives in the repo, not the tar, so place it where the
        # loader expects to find it.
        split_target = extracted / "split_csv" / "1cls.csv"
        if not split_target.is_file() and (out_dir / "visa_1cls.csv").is_file():
            split_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out_dir / "visa_1cls.csv", split_target)

    manifest = FetchResult(
        dataset=spec.key,
        url=spec.url,
        fetched_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        action=action,
        files=files,
        extracted_to=str(extracted) if extracted else None,
        license_note=spec.license_note,
        citation=spec.citation,
    )

    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{spec.key}.fetch.json").write_text(
        json.dumps(asdict(manifest), indent=2), encoding="utf-8"
    )
    return manifest


def _extract(archive: Path, out_dir: Path, spec: RemoteDataset) -> Path:
    target = out_dir / spec.extract_dirname if spec.extract_dirname else out_dir
    if target.is_dir() and any(target.iterdir()):
        return target

    with tarfile.open(archive) as tar:
        _safe_extract(tar, out_dir)
    return target


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract, refusing members that would escape `dest`.

    A tar entry named `../../etc/x` writes outside the destination. This archive
    comes from AWS Open Data and is not expected to contain one, but "not
    expected to" is not a reason to extract a downloaded archive unchecked.
    """
    dest = dest.resolve()
    for member in tar.getmembers():
        resolved = (dest / member.name).resolve()
        if not resolved.is_relative_to(dest):
            raise ValueError(f"refusing unsafe tar member escaping the target: {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"refusing link member in archive: {member.name!r}")
    tar.extractall(dest)


def verify(key: str, out_dir: str | Path = "data/raw") -> dict[str, bool]:
    """Re-hash the downloaded files against the recorded manifest.

    Run after any sync to Drive or Kaggle: sync corruption is silent, and
    discovering it a week later costs a week.
    """
    manifest_path = Path("data/manifests") / f"{key}.fetch.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no fetch manifest at {manifest_path}; run `inspector fetch` first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    return {
        name: (out_dir / name).is_file() and hash_file(out_dir / name) == digest
        for name, digest in manifest["files"].items()
    }
