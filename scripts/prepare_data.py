#!/usr/bin/env python3
"""Build coords.npy and polars.npy — polars use only observed AoA (no extrapolation)."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
from foilform.paths import DATA_PROCESSED, RAW, ensure_dirs  # noqa: E402

CSV_CUR = RAW / "COMPILED AIRFOIL DATA.csv"
CSV_BAK = RAW / "COMPILED AIRFOIL DATA.csv.bak"


def polars_from_bak(filenames: list[str]) -> tuple[np.ndarray, np.ndarray]:
    by_fn = defaultdict(dict)
    aoa_seen = set()
    with open(CSV_BAK, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["Filename"]
            aoa = int(float(row["AoA"]))
            aoa_seen.add(aoa)
            by_fn[fn][aoa] = (float(row["Cl"]), float(row["Cd"]))
    return _fill_grid(by_fn, aoa_seen, filenames)


def polars_from_compiled_json(filenames: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Use AoA, Cl, Cd JSON lists in COMPILED AIRFOIL DATA.csv (no extrapolation)."""
    by_fn = {}
    aoa_seen = set()
    with open(CSV_CUR, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["Filename"]
            aos = json.loads(row["AoA"])
            cls = json.loads(row["Cl"])
            cds = json.loads(row["Cd"])
            d = {}
            for k in range(len(aos)):
                if cls[k] is None or cds[k] is None:
                    continue
                try:
                    a = int(round(float(aos[k])))
                except (TypeError, ValueError):
                    continue
                d[a] = (float(cls[k]), float(cds[k]))
                aoa_seen.add(a)
            by_fn[fn] = d
    return _fill_grid(by_fn, aoa_seen, filenames)


def _fill_grid(by_fn: dict, aoa_seen: set, filenames: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = filenames
    n = len(rows)
    aoa_grid = np.array(sorted(aoa_seen), dtype=np.float64)
    n_aoa = len(aoa_grid)
    pol = np.full((n, n_aoa, 3), np.nan, dtype=np.float64)
    for i, fn in enumerate(rows):
        d = by_fn[fn]
        for j, tgt in enumerate(aoa_grid):
            key = int(tgt)
            pol[i, j, 0] = tgt
            if key in d:
                pol[i, j, 1] = d[key][0]
                pol[i, j, 2] = d[key][1]
    return pol, aoa_grid


def main():
    ensure_dirs()
    with open(CSV_CUR, newline="", encoding="utf-8") as f:
        filenames = [row["Filename"] for row in csv.DictReader(f)]
    if CSV_BAK.exists():
        polars, aoa_grid = polars_from_bak(filenames)
        source = "bak"
    else:
        polars, aoa_grid = polars_from_compiled_json(filenames)
        source = "compiled JSON (no .bak)"
    coords = []
    with open(CSV_CUR, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = json.loads(row["Geometry"])
            xy = np.asarray(g, dtype=np.float64)
            assert xy.shape == (501, 2), xy.shape
            coords.append(xy)
    coords = np.stack(coords, axis=0)
    np.save(DATA_PROCESSED / "coords.npy", coords)
    np.save(DATA_PROCESSED / "polars.npy", polars)
    np.save(DATA_PROCESSED / "aoa_grid.npy", aoa_grid)
    n_valid = int(np.isfinite(polars[:, :, 1]).sum())
    print("source:", source)
    print("coords", coords.shape, "polars", polars.shape, "aoa_grid", aoa_grid)
    print("valid polar entries (non-NaN):", n_valid)


if __name__ == "__main__":
    main()
