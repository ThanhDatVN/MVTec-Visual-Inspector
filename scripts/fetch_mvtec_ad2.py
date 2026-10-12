"""Fetch MVTec AD 2 one category at a time, verify it, and keep a SHA-256 manifest.

MVTec AD 2 (Heckler-Kram et al., 2025; CC BY-NC-SA 4.0) is offered as one archive per category
on MVTec's download page, served directly from mydrive.ch. Each archive is downloaded to a
temporary name, checked against its Content-Length, hashed, extracted with the members checked
(no absolute paths, no links, nothing outside the target), and deleted once extracted, so the
peak disk use is the extracted dataset plus one archive. A marker per category makes a re-run
skip what is already complete; a killed extraction is redone.

    python -I scripts/fetch_mvtec_ad2.py                 # all eight -> data/raw/mvtec_ad2
    python -I scripts/fetch_mvtec_ad2.py --categories vial can
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

PAGE = "https://www.mvtec.com/company/research/datasets/mvtec-ad-2/downloads"
CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
UA = {"User-Agent": "Mozilla/5.0"}
MARKER = ".fetched"


def archive_links() -> dict[str, str]:
    """Category -> archive URL, read from the official download page."""
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(urllib.request.Request(PAGE, headers=UA), timeout=60) as r:
                html = r.read().decode("utf-8", "replace")
            break
        except OSError:
            if attempt == 5:
                raise
            time.sleep(5 * attempt)
    # per-category archives only: the whole-dataset and code-utils archives have digits or capitals
    return {m.group(2): m.group(1)
            for m in re.finditer(r'(https://www\.mydrive\.ch/shares/[^"\s]+?/([a-z_]+)\.tar\.gz)', html)}


def _get(url: str, start: int):
    headers = dict(UA, **({"Range": f"bytes={start}-"} if start else {}))
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def download(url: str, dest: Path, attempts: int = 8) -> tuple[int, str]:
    """Download to `<dest>.part`, resuming with HTTP ranges after a dropped connection."""
    tmp = dest.with_name(dest.name + ".part")
    total = 0
    for attempt in range(1, attempts + 1):
        start = tmp.stat().st_size if tmp.is_file() else 0
        try:
            with _get(url, start) as r:
                if start and r.status != 206:  # the server ignored the range: start over
                    start = 0
                if r.status == 206:
                    total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                else:
                    total = int(r.headers.get("Content-Length") or 0)
                with open(tmp, "ab" if start else "wb") as fh:
                    written = start
                    while block := r.read(1 << 22):
                        fh.write(block)
                        written += len(block)
                        if total and written % (1 << 30) < (1 << 22):
                            print(f"    {written / 1e9:.1f} / {total / 1e9:.1f} GB", flush=True)
            break
        except (OSError, TimeoutError) as exc:  # URLError, resets and timeouts are OSErrors
            print(f"    attempt {attempt} failed at {tmp.stat().st_size if tmp.is_file() else 0} bytes: {exc}",
                  flush=True)
            if attempt == attempts:
                raise
            time.sleep(min(60, 5 * attempt))
    size = tmp.stat().st_size
    if total and size != total:
        raise OSError(f"truncated download: {size} of {total} bytes from {url}")
    digest = sha256_of(tmp)
    tmp.replace(dest)
    return size, digest


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def safe_extract(archive: Path, dest: Path) -> list[str]:
    dest = dest.resolve()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for m in members:
            if m.issym() or m.islnk():
                raise ValueError(f"refusing link member {m.name!r}")
            if not (dest / m.name).resolve().is_relative_to(dest):
                raise ValueError(f"refusing member escaping the target: {m.name!r}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:  # pragma: no cover
            tar.extractall(dest)
    return sorted({m.name.split("/")[0] for m in members})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/raw/mvtec_ad2")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    ap.add_argument("--manifest", default="data/manifests/mvtec_ad2.fetch.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
        "source": PAGE, "licence": "CC BY-NC-SA 4.0", "archives": {}}
    links = archive_links()
    missing = [c for c in args.categories if c not in links]
    if missing:
        print(f"error: no download link found for {missing} on {PAGE}", file=sys.stderr)
        return 1

    for cat in args.categories:
        marker = out / cat / MARKER
        if marker.is_file():
            print(f"{cat}: already fetched")
            continue
        archive = out / f"{cat}.tar.gz"
        if archive.is_file():  # renamed from .part only once complete and size-checked
            size, digest = archive.stat().st_size, sha256_of(archive)
            print(f"{cat}: reusing the downloaded archive", flush=True)
        else:
            print(f"{cat}: downloading {links[cat]}", flush=True)
            size, digest = download(links[cat], archive)
        if (out / cat).exists():  # a killed extraction: start the category again
            shutil.rmtree(out / cat)
        tops = safe_extract(archive, out)
        # each archive holds its category folder plus the dataset's licence.txt and readme.txt
        if cat not in tops or any((out / t).is_dir() for t in tops if t != cat):
            print(f"error: {cat}.tar.gz holds {tops}, expected the {cat!r} folder", file=sys.stderr)
            return 1
        archive.unlink()
        n_files = sum(1 for p in (out / cat).rglob("*") if p.is_file())
        marker.write_text(digest + "\n", encoding="utf-8")
        manifest["archives"][cat] = {"url": links[cat], "bytes": size, "sha256": digest, "files": n_files}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
        print(f"{cat}: {size / 1e9:.2f} GB, {n_files} files, sha256 {digest[:16]}...", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
