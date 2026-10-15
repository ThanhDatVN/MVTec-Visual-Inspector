"""Fetch MPDD from the authors' shared folder, verify it, and optionally zip it for Kaggle.

MPDD (Jezek et al., ICUMT 2021; CC BY-NC-SA 4.0) is distributed as a SharePoint folder linked
from https://github.com/stepanje/MPDD. Following the share link gives a guest session; the
SharePoint REST API then lists the folder tree and serves each file. Every file is checked
against the listed size, and a SHA-256 manifest is written. Re-running skips files already
present with the right size.

    python -I scripts/data_fetch_mpdd.py                                   # -> data/raw/MPDD
    python -I scripts/data_fetch_mpdd.py --zip data/kaggle_upload/mpdd.zip # + an archive to upload

The archive holds `MPDD/<category>/...` plus a README with the citation and licence. Upload it
as a **private** Kaggle dataset: the transfer probe (ADR-12) is evaluation only, and this
repository publishes no MPDD image or derived figure.
"""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import sys
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHARE = ("https://vutbr-my.sharepoint.com/:f:/g/personal/xjezek16_vutbr_cz/"
         "EhHS_ufVigxDo3MC6Lweau0BVMuoCmhMZj6ddamiQ7-FnA?e=oHKCxI")
SITE = "https://vutbr-my.sharepoint.com/personal/xjezek16_vutbr_cz"
ROOT = "/personal/xjezek16_vutbr_cz/Documents/MPDD"
CATEGORIES = ("bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes")

README = """MPDD - Metal Parts Defect Detection dataset

Source:  https://github.com/stepanje/MPDD (authors' shared folder)
Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
         https://creativecommons.org/licenses/by-nc-sa/4.0/

Stepan Jezek, Martin Jonak, Radim Burget, Pavel Dvorak, Milos Skotak. "Deep learning-based
defect detection of metal parts: evaluating current methods in complex conditions."
ICUMT 2021, pp. 66-71. DOI: 10.1109/ICUMT54235.2021.9631567

Unmodified copy, repackaged as one archive for private use in a Kaggle notebook.
"""

_jar = http.cookiejar.CookieJar()
_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_jar))
_opener.addheaders = [("User-Agent", "Mozilla/5.0")]


def _quote(path: str) -> str:
    return urllib.parse.quote(path.replace("'", "''"))


def _api(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json;odata=nometadata"})
    with _opener.open(req, timeout=60) as r:
        return json.load(r)


def list_files(folder: str = ROOT) -> list[tuple[str, int]]:
    """(server-relative URL, size in bytes) for every file under `folder`."""
    d = _api(f"{SITE}/_api/web/GetFolderByServerRelativeUrl('{_quote(folder)}')"
             "?$expand=Folders,Files")
    files = [(f["ServerRelativeUrl"], int(f["Length"])) for f in d.get("Files", [])]
    for sub in d.get("Folders", []):
        if sub["Name"] != "Forms":
            files += list_files(sub["ServerRelativeUrl"])
    return files


def _fetch(item: tuple[str, int], dest_root: Path) -> tuple[str, int, str]:
    url_path, size = item
    rel = url_path[len(ROOT) + 1:]
    dest = dest_root / rel
    if dest.is_file() and dest.stat().st_size == size:
        return rel, size, hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{SITE}/_api/web/GetFileByServerRelativeUrl('{_quote(url_path)}')/$value"
    error: Exception | None = None
    for _ in range(4):
        try:
            with _opener.open(url, timeout=120) as r:
                data = r.read()
            if len(data) != size:
                raise OSError(f"got {len(data)} bytes, listing says {size}")
            dest.write_bytes(data)
            return rel, size, hashlib.sha256(data).hexdigest()
        except Exception as exc:  # transient network errors: retry
            error = exc
    raise RuntimeError(f"{rel}: {error}")


def write_zip(dest_root: Path, rels: list[str], archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    # PNG is already compressed; storing keeps the archive fast to build and to extract.
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("MPDD/README.txt", README)
        for rel in sorted(rels):
            zf.write(dest_root / rel, f"MPDD/{rel}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/raw", help="MPDD/ is created inside")
    ap.add_argument("--manifest", default="data/manifests/mpdd.sha256.json")
    ap.add_argument("--zip", default=None, help="also write an archive for upload")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    dest_root = Path(args.out) / "MPDD"
    _opener.open(SHARE, timeout=60).read()  # guest session cookie
    files = list_files()
    print(f"listing: {len(files)} files, {sum(s for _, s in files) / 1e9:.2f} GB", flush=True)
    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        for k, row in enumerate(pool.map(lambda it: _fetch(it, dest_root), files), 1):
            rows.append(row)
            if k % 250 == 0:
                print(f"  {k}/{len(files)}", flush=True)

    missing = [c for c in CATEGORIES if not (dest_root / c / "train" / "good").is_dir()]
    if missing:
        print(f"error: categories without train/good: {missing}", file=sys.stderr)
        return 1
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "source": SHARE, "licence": "CC BY-NC-SA 4.0",
        "files": {r: {"bytes": s, "sha256": h} for r, s, h in sorted(rows)},
    }, indent=1), encoding="utf-8")
    print(f"verified: {len(rows)} files by size; manifest {manifest}")

    if args.zip:
        write_zip(dest_root, [r for r, _, _ in rows], Path(args.zip))
        print(f"archive : {args.zip} ({Path(args.zip).stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
