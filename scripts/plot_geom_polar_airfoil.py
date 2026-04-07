#!/usr/bin/env python3
"""Plot Cl & Cd vs AoA (°) from -8 to 8 for one random airfoil: GT vs model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
from foilform.paths import DATA_PROCESSED, FIGURES, RUNS, ensure_dirs  # noqa: E402
from foilform.geom_polar_transformer import GeomPolarTransformer  # noqa: E402


def find_latest_best_checkpoint() -> Path | None:
    if not RUNS.is_dir():
        return None
    candidates = list(RUNS.glob("*/best_geom_polar_transformer.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_targets(polars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, n_cols, _ = polars.shape
    counts = [int(np.isfinite(polars[i, :, 1]).sum()) for i in range(n)]
    max_steps = max(counts)
    y = np.zeros((n, max_steps, 3), dtype=np.float32)
    m = np.zeros((n, max_steps), dtype=np.float32)
    for i in range(n):
        rows = []
        for j in range(n_cols):
            if np.isfinite(polars[i, j, 1]):
                aoa = float(polars[i, j, 0])
                cl = float(polars[i, j, 1])
                cd = float(polars[i, j, 2])
                rows.append((cl, cd, aoa))
        if rows:
            arr = np.asarray(rows, dtype=np.float32)
            s = arr.shape[0]
            y[i, :s, :] = arr
            m[i, :s] = 1.0
    return y, m


def resolve_device(prefer: str) -> torch.device:
    p = prefer.lower()
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--airfoil", type=int, default=-1, help="Global airfoil index, or -1 for random.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="PNG path (default: figures/geom_polar_airfoil_<idx>.png).",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dirs()
    polars_path = DATA_PROCESSED / "polars.npy"
    geom_path = DATA_PROCESSED / "geom_embeddings.npy"
    if not polars_path.is_file() or not geom_path.is_file():
        raise FileNotFoundError("Need data/processed/polars.npy and geom_embeddings.npy")

    polars = np.load(polars_path).astype(np.float32)
    geom = np.load(geom_path).astype(np.float32)
    y, m = build_targets(polars)
    n = geom.shape[0]

    rng = np.random.default_rng(args.seed)
    if args.airfoil < 0:
        idx = int(rng.integers(0, n))
    else:
        idx = int(args.airfoil) % n

    device = resolve_device(args.device)
    model = GeomPolarTransformer().to(device)
    ckpt: Path | None = None
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_file():
            ckpt = _REPO / args.checkpoint
    else:
        ckpt = find_latest_best_checkpoint()
    if ckpt is not None and ckpt.is_file():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print("Warning: no checkpoint found — random weights.")

    model.eval()
    g = torch.from_numpy(geom[idx : idx + 1]).to(device)
    yt = torch.from_numpy(y[idx : idx + 1]).to(device)

    pred = model.decode_append(
        geom_context=g,
        target_steps=y.shape[1],
        teacher_tuples=None,
        aoa_ground_truth=yt[:, :, 2],
    )

    mask = m[idx] > 0.5
    aoa = y[idx, :, 2]
    gt_cl = y[idx, :, 0]
    gt_cd = y[idx, :, 1]
    p_cl = pred[0, :, 0].cpu().numpy()
    p_cd = pred[0, :, 1].cpu().numpy()

    order = np.argsort(aoa[mask])
    aoa_p = aoa[mask][order]
    gt_cl_p = gt_cl[mask][order]
    gt_cd_p = gt_cd[mask][order]
    p_cl_p = p_cl[mask][order]
    p_cd_p = p_cd[mask][order]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax0.plot(aoa_p, gt_cl_p, "o-", color="0.2", label="Ground truth", markersize=5)
    ax0.plot(aoa_p, p_cl_p, "s--", color="tab:blue", label="Pred (Cl, Cd + gt AoA chain)", markersize=4)
    ax0.set_ylabel("Cl")
    ax0.legend(loc="best", fontsize=9)
    ax0.grid(True, alpha=0.3)
    ax0.set_title(f"Airfoil index {idx}  |  device={device}")

    ax1.plot(aoa_p, gt_cd_p, "o-", color="0.2", label="Ground truth", markersize=5)
    ax1.plot(aoa_p, p_cd_p, "s--", color="tab:blue", label="Pred (Cl, Cd + gt AoA chain)", markersize=4)
    ax1.set_ylabel("Cd")
    ax1.set_xlabel("AoA (°)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)
    ax1.set_xlim(-8.5, 8.5)

    fig.tight_layout()
    out = Path(args.out) if args.out else FIGURES / f"geom_polar_airfoil_{idx}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
