#!/usr/bin/env python3
"""Top/bottom K airfoils by per-airfoil error; plot outlines from coords.npy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_STUDIES = Path(__file__).resolve().parent.parent
_REPO = _STUDIES.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_STUDIES / "src"))

from foilform.checkpoint_utils import load_geom_transformer  # noqa: E402
from foilform.paths import DATA_PROCESSED, FIGURES  # noqa: E402
from foilform.polar_correction_mlp import N_SLOTS, POLAR_DIM  # noqa: E402

import torch  # noqa: E402


def build_targets(polars: np.ndarray):
    n, nc, _ = polars.shape
    counts = [int(np.isfinite(polars[i, :, 1]).sum()) for i in range(n)]
    ms = max(counts)
    y = np.zeros((n, ms, 3), dtype=np.float32)
    m = np.zeros((n, ms), dtype=np.float32)
    for i in range(n):
        rows = []
        for j in range(nc):
            if np.isfinite(polars[i, j, 1]):
                rows.append((float(polars[i, j, 1]), float(polars[i, j, 2]), float(polars[i, j, 0])))
        if rows:
            arr = np.asarray(rows, dtype=np.float32)
            y[i, : arr.shape[0], :] = arr
            m[i, : arr.shape[0]] = 1.0
    return y, m


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


def geom_stats(x: np.ndarray, yc: np.ndarray) -> dict[str, float]:
    """Simple thickness/camber proxies from coords (chord ~ x)."""
    t = float(np.max(yc) - np.min(yc))
    cam = float(np.trapezoid(yc, x) / (x.max() - x.min() + 1e-8))
    return {"thickness_span": t, "camber_proxy": cam}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geom-checkpoint", type=str, required=True)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = resolve_device(args.device)
    polars = np.load(DATA_PROCESSED / "polars.npy").astype(np.float32)
    geom_emb = np.load(DATA_PROCESSED / "geom_embeddings.npy").astype(np.float32)
    coords = np.load(DATA_PROCESSED / "coords.npy").astype(np.float32)
    y, _ = build_targets(polars)
    all_js = col_indices_all(polars)
    n = geom_emb.shape[0]
    s_steps = y.shape[1]

    perm = np.random.default_rng(args.seed).permutation(n)
    n_train = max(1, min(int(round(args.train_frac * n)), n - 1))
    val_idx = np.sort(perm[n_train:])

    geom_ckpt = Path(args.geom_checkpoint)
    if not geom_ckpt.is_file():
        geom_ckpt = _REPO / args.geom_checkpoint
    model = load_geom_transformer(geom_ckpt, device)

    err = np.zeros(len(val_idx), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(val_idx), args.batch_size):
            end = min(start + args.batch_size, len(val_idx))
            idx_batch = val_idx[start:end]
            g = torch.from_numpy(geom_emb[idx_batch]).to(device)
            yt = torch.from_numpy(y[idx_batch]).to(device)
            pred_t = model.decode_append(g, s_steps, aoa_ground_truth=yt[:, :, 2])
            pred_np = pred_t.cpu().numpy()
            batch_js = [all_js[i] for i in idx_batch]
            base34 = pred_to_polar34_batch(pred_np, batch_js)
            for bi, i_global in enumerate(idx_batch):
                js = all_js[i_global]
                e = []
                for t_idx, j in enumerate(js):
                    if j >= N_SLOTS:
                        continue
                    e.append(abs(base34[bi, j] - y[i_global, t_idx, 0]))
                    e.append(abs(base34[bi, N_SLOTS + j] - y[i_global, t_idx, 1]))
                err[start + bi] = float(np.mean(e)) if e else np.nan

    order = np.argsort(-err)
    worst = val_idx[order[: args.top_k]]
    best = val_idx[order[-args.top_k :]]

    out_dir = FIGURES / "worst_best_airfoils"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"worst": [], "best": []}
    for tag, indices in [("worst", worst), ("best", best)]:
        for rank, i in enumerate(indices):
            x = coords[i, :, 0]
            yc = coords[i, :, 1]
            st = geom_stats(x, yc)
            meta[tag].append({"global_index": int(i), "rank": rank, **st})

    with (out_dir / "indices.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(4, 5, figsize=(14, 10))
        axs = axs.ravel()
        for ax, i in zip(axs, worst):
            ax.plot(coords[i, :, 0], coords[i, :, 1], "k-", lw=0.8)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"worst #{int(i)}", fontsize=8)
        fig.suptitle("Highest-error val airfoils (transformer)")
        fig.tight_layout()
        fig.savefig(out_dir / "worst_geometries.png", dpi=140)
        plt.close()

        fig, axs = plt.subplots(4, 5, figsize=(14, 10))
        axs = axs.ravel()
        for ax, i in zip(axs, best):
            ax.plot(coords[i, :, 0], coords[i, :, 1], "k-", lw=0.8)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"best #{int(i)}", fontsize=8)
        fig.suptitle("Lowest-error val airfoils (transformer)")
        fig.tight_layout()
        fig.savefig(out_dir / "best_geometries.png", dpi=140)
        plt.close()
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
