"""Dataset acquisition.

Only datasets that can be fetched without an account live here. MVTec AD and
AD 2 require registration and are therefore a manual step, documented in
[docs/10-datasets.md](../../docs/10-datasets.md) rather than automated behind a
command that would only fail.

Every fetch checks the files against pinned SHA-256 digests and records a
manifest of what it downloaded. A silently updated or partially written archive
is otherwise indistinguishable from a good one until a model trained on it
behaves strangely weeks later — and the split CSV comes from a repository's
default branch, so an upstream edit would silently change the protocol.
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
    #: SHA-256 of the archive the committed results were produced from ("" = not pinned).
    sha256: str = ""
    #: (url, local name, sha256) for files fetched alongside the archive.
    extra_files: tuple[tuple[str, str, str], ...] = ()
    #: Directory the dataset lives in, relative to the download dir. Archives that
    #: already carry this top-level folder are extracted as they are; archives whose
    #: members sit at the top level (VisA's do) are extracted into it.
    extract_dirname: str = ""
    #: Local name of the extra file that is the official split; it is copied to
    #: `<root>/split_csv/1cls.csv`, where the loader looks for it.
    split_file: str = ""
    license_note: str = ""
    citation: str = ""


VISA = RemoteDataset(
    key="visa",
    url="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
    filename="VisA_20220922.tar",
    sha256="2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362",
    extra_files=(
        (
            "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv",
            "visa_1cls.csv",
            "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995",
        ),
    ),
    extract_dirname="VisA_20220922",
    split_file="visa_1cls.csv",
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

#: Written into the dataset directory once extraction has finished; holds the
#: archive's digest. Without it a half-extracted directory looks complete.
EXTRACTED_MARKER = ".extracted"


def _check_digest(path: Path, expected: str) -> str:
    digest = hash_file(path)
    if expected and digest != expected:
        raise ValueError(
            f"{path.name}: SHA-256 {digest} does not match the pinned {expected}. The file is "
            "corrupt, or upstream changed it; the committed results were produced from the "
            "pinned version. Delete the file to re-download, and do not mix the two."
        )
    return digest


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
    manifest_dir: str | Path = "data/manifests",
) -> FetchResult:
    """Download (and optionally extract) a dataset, verify it, and record its manifest.

    `manifest_dir` is relative to the working directory. Point it outside a git
    checkout that must stay clean (a Kaggle clone): an untracked manifest there
    would mark every later run as dirty.
    """
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

    files = {spec.filename: _check_digest(archive, spec.sha256)}
    for url, name, digest in spec.extra_files:
        path = out_dir / name
        if not path.is_file() or force:
            _download(url, path)
        files[name] = _check_digest(path, digest)

    extracted: Path | None = None
    if extract:
        extracted = _extract(archive, out_dir, spec, digest=files[spec.filename])
        # The split CSV ships separately from the archive; place the verified copy
        # where the loader looks for it, every time, so a stale split is never read.
        if spec.split_file:
            split_target = extracted / "split_csv" / "1cls.csv"
            split_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out_dir / spec.split_file, split_target)

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

    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{spec.key}.fetch.json"
    # Rewritten only when what was fetched changed: a timestamp-only rewrite of a
    # committed manifest would leave the tree dirty and tag every later run dirty.
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = None
    stable = ("dataset", "url", "files", "extracted_to")
    if previous is None or any(previous.get(k) != getattr(manifest, k) for k in stable):
        path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return manifest


def _extract(archive: Path, out_dir: Path, spec: RemoteDataset, *, digest: str) -> Path:
    """Extract into `out_dir / spec.extract_dirname`, once.

    Whether the archive carries that folder itself is read from its members, not
    assumed: VisA's tar holds `pcb1/...` at the top level, and extracting it into
    `out_dir` put the images beside the dataset root instead of inside it.
    """
    target = out_dir / spec.extract_dirname if spec.extract_dirname else out_dir
    marker = target / EXTRACTED_MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
        return target

    with tarfile.open(archive) as tar:
        prefix = spec.extract_dirname.strip("/")
        names = tar.getnames()
        wrapped = bool(prefix) and all(n == prefix or n.startswith(prefix + "/") for n in names)
        dest = out_dir if (wrapped or not prefix) else target
        dest.mkdir(parents=True, exist_ok=True)
        _safe_extract(tar, dest)
    target.mkdir(parents=True, exist_ok=True)
    marker.write_text(digest + "\n", encoding="utf-8")
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
    if hasattr(tarfile, "data_filter"):  # Python >= 3.12, and security backports
        tar.extractall(dest, filter="data")
    else:  # pragma: no cover - older interpreters
        tar.extractall(dest)


def verify(
    key: str, out_dir: str | Path = "data/raw", manifest_dir: str | Path = "data/manifests"
) -> dict[str, bool]:
    """Re-hash the downloaded files against the recorded manifest.

    Run after any sync to Drive or Kaggle: sync corruption is silent, and
    discovering it a week later costs a week.
    """
    manifest_path = Path(manifest_dir) / f"{key}.fetch.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no fetch manifest at {manifest_path}; run `inspector fetch` first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    return {
        name: (out_dir / name).is_file() and hash_file(out_dir / name) == digest
        for name, digest in manifest["files"].items()
    }
