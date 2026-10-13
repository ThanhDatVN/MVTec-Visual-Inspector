"""ADR-14 H4: candidate v3 against v1 on MVTec AD 2's private splits, from the server's numbers.

Committed with ADR-14, before either submission's results exist. The owner transcribes the
server's per-category results into `reports/studies/ad2-server/server_results.csv`:

    recipe,split,category,aupro_005,segf1
    v1,test_private,can,0.123,0.045
    ...

(`recipe` in {v1, v3}; `split` in {test_private, test_private_mixed}; AU-PRO@0.05 and SegF1 as
fractions, not percent). H4 holds if v3's AU-PRO@0.05 on `test_private` exceeds v1's on at least
7 of the 8 categories (one-sided exact sign test, p = 0.035). Everything else — SegF1, the
`test_private_mixed` split, the private-vs-mixed drop of each recipe — is descriptive. If the
server reports only means, H4 cannot be evaluated and the means are reported as received.

    python scripts/ad2_server_analysis.py --results reports/studies/ad2-server/server_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
H4_MIN = 7


def sign_test_p(successes: int, n: int) -> float:
    return sum(math.comb(n, k) for k in range(successes, n + 1)) / 2**n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="reports/studies/ad2-server/server_results.csv")
    args = ap.parse_args()
    table: dict[tuple[str, str, str], dict[str, float]] = {}
    with open(args.results, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            table[(r["recipe"], r["split"], r["category"])] = {
                k: float(r[k]) for k in ("aupro_005", "segf1") if r.get(k) not in (None, "")}
    out: dict[str, object] = {}
    complete = all(("v1", "test_private", c) in table and ("v3", "test_private", c) in table for c in CATEGORIES)
    if complete:
        wins = [c for c in CATEGORIES
                if table[("v3", "test_private", c)]["aupro_005"] > table[("v1", "test_private", c)]["aupro_005"]]
        out["H4"] = {"v3_higher_on": wins, "count": len(wins), "rule": f">= {H4_MIN} of 8",
                     "sign_test_p": sign_test_p(len(wins), 8), "holds": len(wins) >= H4_MIN}
    else:
        out["H4"] = "not evaluable: per-category test_private numbers for both recipes are missing"
    for recipe in ("v1", "v3"):
        for split in ("test_private", "test_private_mixed"):
            rows = [table[(recipe, split, c)] for c in CATEGORIES if (recipe, split, c) in table]
            if rows:
                out[f"{recipe}/{split}"] = {k: sum(r[k] for r in rows if k in r) / len(rows)
                                            for k in ("aupro_005", "segf1")} | {"categories": len(rows)}
    for recipe in ("v1", "v3"):  # the lighting drop on held-out data, per category
        drops = {c: table[(recipe, "test_private", c)]["aupro_005"] - table[(recipe, "test_private_mixed", c)]["aupro_005"]
                 for c in CATEGORIES
                 if (recipe, "test_private", c) in table and (recipe, "test_private_mixed", c) in table}
        if drops:
            out[f"{recipe}/aupro_drop_private_to_mixed"] = drops
    Path(args.results).with_name("server_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
