#!/usr/bin/env python3
"""Per-airfoil MAE Cl/Cd on val: histograms + boxplots (transformer vs corrected)."""

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
from foilform.polar_correction_mlp import N_SLOTS, POLAR_DIM, PolarCorrectionMLP  # noqa: E402

import torch  # noqa: E402


def resolve_device(prefer: str) -> torch.device:
    p = prefer.lower()
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geom-checkpoint", type=str, required=True)
    parser.add_argument("--corr-checkpoint", type=str, default="")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
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
    t_model = load_geom_transformer(geom_ckpt, device)
    corr = None
    if args.corr_checkpoint:
        ck = Path(args.corr_checkpoint)
        if not ck.is_file():
            ck = _REPO / args.corr_checkpoint
        if ck.is_file():
            corr = PolarCorrectionMLP().to(device)
            corr.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"], strict=True)
            corr.eval()

    mae_cl_t = np.zeros(len(val_idx), dtype=np.float64)
    mae_cd_t = np.zeros(len(val_idx), dtype=np.float64)
    mae_cl_c = np.zeros(len(val_idx), dtype=np.float64)
    mae_cd_c = np.zeros(len(val_idx), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(val_idx), args.batch_size):
            end = min(start + args.batch_size, len(val_idx))
            idx_batch = val_idx[start:end]
            g = torch.from_numpy(geom_emb[idx_batch]).to(device)
            yt = torch.from_numpy(y[idx_batch]).to(device)
            pred_t = t_model.decode_append(g, s_steps, aoa_ground_truth=yt[:, :, 2])
            pred_np = pred_t.detach().cpu().numpy()
            batch_js = [all_js[i] for i in idx_batch]
            base34 = pred_to_polar34_batch(pred_np, batch_js)
            if corr is not None:
                gy = torch.from_numpy(coords[idx_batch].astype(np.float32)).to(device)
                b34_t = torch.from_numpy(base34).to(device)
                corr34 = corr.predict(gy, b34_t).detach().cpu().numpy()
            else:
                corr34 = base34

            for bi, i_global in enumerate(idx_batch):
                js = all_js[i_global]
                ec_l = []
                ec_d = []
                for t_idx, j in enumerate(js):
                    if j >= N_SLOTS:
                        continue
                    gt_cl = y[i_global, t_idx, 0]
                    gt_cd = y[i_global, t_idx, 1]
                    pb = base34[bi, j] - gt_cl
                    qb = base34[bi, N_SLOTS + j] - gt_cd
                    ec_l.append(abs(float(pb)))
                    ec_d.append(abs(float(qb)))
                mae_cl_t[start + bi] = np.mean(ec_l) if ec_l else np.nan
                mae_cd_t[start + bi] = np.mean(ec_d) if ec_d else np.nan
                ec_l2 = []
                ec_d2 = []
                for t_idx, j in enumerate(js):
                    if j >= N_SLOTS:
                        continue
                    gt_cl = y[i_global, t_idx, 0]
                    gt_cd = y[i_global, t_idx, 1]
                    pc = corr34[bi, j] - gt_cl
                    qc = corr34[bi, N_SLOTS + j] - gt_cd
                    ec_l2.append(abs(float(pc)))
                    ec_d2.append(abs(float(qc)))
                mae_cl_c[start + bi] = np.mean(ec_l2) if ec_l2 else np.nan
                mae_cd_c[start + bi] = np.mean(ec_d2) if ec_d2 else np.nan

    FIGURES.mkdir(parents=True, exist_ok=True)
    out_dir = FIGURES / "val_distribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "per_airfoil_mae.npz",
        val_idx=val_idx,
        mae_cl_transformer=mae_cl_t,
        mae_cd_transformer=mae_cd_t,
        mae_cl_corrected=mae_cl_c,
        mae_cd_corrected=mae_cd_c,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].hist(mae_cl_t, bins=40, color="steelblue", alpha=0.7)
        axs[0, 0].set_title("MAE Cl — transformer")
        axs[0, 1].hist(mae_cd_t, bins=40, color="steelblue", alpha=0.7)
        axs[0, 1].set_title("MAE Cd — transformer")
        axs[1, 0].hist(mae_cl_c, bins=40, color="darkorange", alpha=0.7)
        axs[1, 0].set_title("MAE Cl — corrected")
        axs[1, 1].hist(mae_cd_c, bins=40, color="darkorange", alpha=0.7)
        axs[1, 1].set_title("MAE Cd — corrected")
        fig.tight_layout()
        fig.savefig(out_dir / "histograms_mae.png", dpi=160)
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 4))
        bp = ax.boxplot(
            [mae_cl_t, mae_cd_t, mae_cl_c, mae_cd_c],
            labels=["Cl tfm", "Cd tfm", "Cl corr", "Cd corr"],
        )
        ax.set_ylabel("per-airfoil MAE")
        ax.set_title("Val set distribution")
        fig.tight_layout()
        fig.savefig(out_dir / "boxplot_mae.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"Plotting skipped: {e}")

    summary = {
        "n_val": int(len(val_idx)),
        "median_mae_cl_tfm": float(np.nanmedian(mae_cl_t)),
        "median_mae_cd_tfm": float(np.nanmedian(mae_cd_t)),
        "median_mae_cl_corr": float(np.nanmedian(mae_cl_c)),
        "median_mae_cd_corr": float(np.nanmedian(mae_cd_c)),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
