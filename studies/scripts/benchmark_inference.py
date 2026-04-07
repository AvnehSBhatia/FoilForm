#!/usr/bin/env python3
"""CPU timing: decode_append vs NeuralFoil xxxlarge; optional xfoil check."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

_STUDIES = Path(__file__).resolve().parent.parent
_REPO = _STUDIES.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_STUDIES / "src"))

from foilform.checkpoint_utils import load_geom_transformer  # noqa: E402
from foilform.paths import DATA_PROCESSED, FIGURES  # noqa: E402


def build_targets(polars: np.ndarray):
    n, nc, _ = polars.shape
    counts = [int(np.isfinite(polars[i, :, 1]).sum()) for i in range(n)]
    ms = max(counts)
    y = np.zeros((n, ms, 3), dtype=np.float32)
    for i in range(n):
        rows = []
        for j in range(nc):
            if np.isfinite(polars[i, j, 1]):
                rows.append((float(polars[i, j, 1]), float(polars[i, j, 2]), float(polars[i, j, 0])))
        if rows:
            arr = np.asarray(rows, dtype=np.float32)
            y[i, : arr.shape[0], :] = arr
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geom-checkpoint", type=str, required=True)
    parser.add_argument("--airfoil-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--Re", type=float, default=1e5)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cpu")
    geom_ckpt = Path(args.geom_checkpoint)
    if not geom_ckpt.is_file():
        geom_ckpt = _REPO / args.geom_checkpoint
    model = load_geom_transformer(geom_ckpt, device)

    polars = np.load(DATA_PROCESSED / "polars.npy").astype(np.float32)
    geom_emb = np.load(DATA_PROCESSED / "geom_embeddings.npy").astype(np.float32)
    coords = np.load(DATA_PROCESSED / "coords.npy").astype(np.float64)
    y = build_targets(polars)
    i = args.airfoil_index
    g = torch.from_numpy(geom_emb[i : i + 1]).float()
    yt = torch.from_numpy(y[i : i + 1]).float()
    s_steps = y.shape[1]

    for _ in range(args.warmup):
        _ = model.decode_append(g, s_steps, aoa_ground_truth=yt[:, :, 2])

    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        _ = model.decode_append(g, s_steps, aoa_ground_truth=yt[:, :, 2])
        times.append(time.perf_counter() - t0)
    ms_dec = float(np.median(times) * 1000.0)
    n_params = model.count_parameters()

    # NeuralFoil same airfoil / AoA schedule
    import warnings

    warnings.filterwarnings("ignore")
    import neuralfoil as nf

    js = [j for j in range(polars.shape[1]) if np.isfinite(polars[i, j, 1])]
    aoa_valid = np.array([polars[i, j, 0] for j in js], dtype=np.float64)
    for _ in range(args.warmup):
        _ = nf.get_aero_from_coordinates(
            coordinates=coords[i], alpha=aoa_valid, Re=args.Re, model_size="xxxlarge"
        )
    times_nf = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        _ = nf.get_aero_from_coordinates(
            coordinates=coords[i], alpha=aoa_valid, Re=args.Re, model_size="xxxlarge"
        )
        times_nf.append(time.perf_counter() - t0)
    ms_nf = float(np.median(times_nf) * 1000.0)

    xfoil_status = "skipped"
    if shutil.which("xfoil"):
        xfoil_status = "binary_present_not_benchmarked"

    out = {
        "decode_append_median_ms_cpu": ms_dec,
        "neuralfoil_xxxlarge_median_ms_cpu": ms_nf,
        "n_params_transformer": n_params,
        "airfoil_index": i,
        "xfoil": xfoil_status,
    }
    print(json.dumps(out, indent=2))

    FIGURES.mkdir(parents=True, exist_ok=True)
    outp = Path(args.output_json) if args.output_json else FIGURES / "benchmark_inference.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
