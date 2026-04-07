#!/usr/bin/env python3
"""Histogram / heatmap of ΔCl, ΔCd from correction MLP on val airfoils."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_STUDIES = Path(__file__).resolve().parent.parent
_REPO = _STUDIES.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_STUDIES / "src"))

from foilform.checkpoint_utils import load_geom_transformer  # noqa: E402
from foilform.paths import DATA_PROCESSED, FIGURES  # noqa: E402
from foilform.polar_correction_mlp import N_SLOTS, POLAR_DIM, PolarCorrectionMLP  # noqa: E402


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


def col_indices_all(polars: np.ndarray):
    return [[j for j in range(polars.shape[1]) if np.isfinite(polars[i, j, 1])] for i in range(polars.shape[0])]


def pred_to_polar34_batch(pred: np.ndarray, col_indices: list[list[int]]) -> np.ndarray:
    B = pred.shape[0]
    out = np.zeros((B, POLAR_DIM), dtype=np.float32)
    for b in range(B):
        js = col_indices[b]
        for t, j in enumerate(js):
            if j >= N_SLOTS or t >= pred.shape[1]:
                continue
            out[b, j] = pred[b, t, 0]
            out[b, N_SLOTS + j] = pred[b, t, 1]
    return out


def resolve_device(p: str) -> torch.device:
    p = p.lower()
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geom-checkpoint", type=str, required=True)
    parser.add_argument("--corr-checkpoint", type=str, required=True)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = resolve_device(args.device)
    polars = np.load(DATA_PROCESSED / "polars.npy").astype(np.float32)
    geom_emb = np.load(DATA_PROCESSED / "geom_embeddings.npy").astype(np.float32)
    coords = np.load(DATA_PROCESSED / "coords.npy").astype(np.float32)
    y = build_targets(polars)
    all_js = col_indices_all(polars)
    n = geom_emb.shape[0]
    s_steps = y.shape[1]

    perm = np.random.default_rng(args.seed).permutation(n)
    n_train = max(1, min(int(round(args.train_frac * n)), n - 1))
    val_idx = np.sort(perm[n_train:])

    geom_ckpt = Path(args.geom_checkpoint)
    if not geom_ckpt.is_file():
        geom_ckpt = _REPO / args.geom_checkpoint
    corr_ckpt = Path(args.corr_checkpoint)
    if not corr_ckpt.is_file():
        corr_ckpt = _REPO / args.corr_checkpoint

    t_model = load_geom_transformer(geom_ckpt, device)
    corr = PolarCorrectionMLP().to(device)
    corr.load_state_dict(torch.load(corr_ckpt, map_location=device, weights_only=False)["model"], strict=True)
    corr.eval()

    d_cl, d_cd = [], []
    with torch.no_grad():
        for start in range(0, len(val_idx), args.batch_size):
            end = min(start + args.batch_size, len(val_idx))
            idx_batch = val_idx[start:end]
            g = torch.from_numpy(geom_emb[idx_batch]).to(device)
            yt = torch.from_numpy(y[idx_batch]).to(device)
            pred_t = t_model.decode_append(g, s_steps, aoa_ground_truth=yt[:, :, 2])
            base34 = pred_to_polar34_batch(pred_t.cpu().numpy(), [all_js[i] for i in idx_batch])
            gy = torch.from_numpy(coords[idx_batch].astype(np.float32)).to(device)
            b34_t = torch.from_numpy(base34).to(device)
            out = corr.predict(gy, b34_t).cpu().numpy()
            for bi, i_global in enumerate(idx_batch):
                for t_idx, j in enumerate(all_js[i_global]):
                    if j >= N_SLOTS:
                        continue
                    bcl = base34[bi, j]
                    bcd = base34[bi, N_SLOTS + j]
                    d_cl.append(out[bi, j] - bcl)
                    d_cd.append(out[bi, N_SLOTS + j] - bcd)

    d_cl = np.asarray(d_cl, dtype=np.float64)
    d_cd = np.asarray(d_cd, dtype=np.float64)

    FIGURES.mkdir(parents=True, exist_ok=True)
    out_dir = FIGURES / "corrector_delta"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "delta_cl_cd.npz", delta_cl=d_cl, delta_cd=d_cd)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    axs[0].hist(d_cl, bins=60, color="steelblue", alpha=0.85)
    axs[0].set_title("ΔCl correction")
    axs[1].hist(d_cd, bins=60, color="darkorange", alpha=0.85)
    axs[1].set_title("ΔCd correction")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_histograms.png", dpi=160)
    plt.close()
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
