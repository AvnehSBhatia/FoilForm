#!/usr/bin/env python3
"""Val MAE Cl, Cd, L/D + timing; studies checkpoints; optional NeuralFoil."""

from __future__ import annotations

import argparse
import json
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
from foilform.checkpoints import resolve_geom_polar_transformer, resolve_polar_correction  # noqa: E402
from foilform.manifest import append_manifest  # noqa: E402
from foilform.paths import DATA_PROCESSED  # noqa: E402
from foilform.polar_correction_mlp import (  # noqa: E402
    N_SLOTS,
    POLAR_DIM,
    PolarCorrectionMLP,
)


def build_targets(polars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def col_indices_all(polars: np.ndarray) -> list[list[int]]:
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


def resolve_device(prefer: str) -> torch.device:
    p = prefer.lower()
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Studies eval: MAE + speed")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--Re", type=float, default=1e5)
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_nf", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--geom-checkpoint",
        type=str,
        default="",
        help="Path to best_geom_polar_transformer.pt (default: models/geom_polar_transformer.pt or latest runs/).",
    )
    parser.add_argument(
        "--corr-checkpoint",
        type=str,
        default="",
        help="Path to best_polar_correction.pt (optional; default: models/polar_correction.pt or latest runs/).",
    )
    parser.add_argument("--output-json", type=str, default="", help="Write metrics JSON here.")
    parser.add_argument("--experiment-id", type=str, default="eval")
    args = parser.parse_args()

    polars = np.load(DATA_PROCESSED / "polars.npy").astype(np.float32)
    geom_emb = np.load(DATA_PROCESSED / "geom_embeddings.npy").astype(np.float32)
    coords = np.load(DATA_PROCESSED / "coords.npy").astype(np.float32)
    y, mask_unused = build_targets(polars)
    all_js = col_indices_all(polars)
    n = geom_emb.shape[0]
    s_steps = y.shape[1]

    perm = np.random.default_rng(args.seed).permutation(n)
    n_train = max(1, min(int(round(args.train_frac * n)), n - 1))
    val_idx = np.sort(perm[n_train:])
    n_val = len(val_idx)
    print(f"Val airfoils: {n_val}  |  batch_size: {args.batch_size}")

    device = resolve_device(args.device)

    if args.geom_checkpoint:
        geom_ckpt = Path(args.geom_checkpoint)
        if not geom_ckpt.is_file():
            geom_ckpt = _REPO / args.geom_checkpoint
    else:
        geom_ckpt = resolve_geom_polar_transformer()
    if geom_ckpt is None or not geom_ckpt.is_file():
        raise FileNotFoundError(
            "No geom checkpoint: add models/geom_polar_transformer.pt or pass --geom-checkpoint."
        )
    t_model = load_geom_transformer(geom_ckpt, device)
    if args.compile:
        t_model = torch.compile(t_model)
    n_params = t_model.count_parameters()
    print(f"Transformer: {geom_ckpt} ({n_params} params)")

    corr_model = None
    corr_ckpt: Path | None = None
    if args.corr_checkpoint:
        corr_ckpt = Path(args.corr_checkpoint)
        if not corr_ckpt.is_file():
            corr_ckpt = _REPO / args.corr_checkpoint
    else:
        corr_ckpt = resolve_polar_correction()
    if corr_ckpt is not None and corr_ckpt.is_file():
        corr_model = PolarCorrectionMLP().to(device)
        corr_model.load_state_dict(
            torch.load(corr_ckpt, map_location=device, weights_only=False)["model"], strict=True
        )
        corr_model.eval()
        print(f"Correction MLP: {corr_ckpt}")
    else:
        print("Correction MLP: skipped (no checkpoint)")

    tfm_base34 = np.zeros((n_val, POLAR_DIM), dtype=np.float32)
    corr_pred34 = np.zeros((n_val, POLAR_DIM), dtype=np.float32)
    gt34 = np.zeros((n_val, POLAR_DIM), dtype=np.float32)
    mask34 = np.zeros((n_val, POLAR_DIM), dtype=np.float32)

    device_sync(device)
    t0_total = time.perf_counter()

    with torch.no_grad():
        for start in range(0, n_val, args.batch_size):
            end = min(start + args.batch_size, n_val)
            idx_batch = val_idx[start:end]
            bs = end - start

            g = torch.from_numpy(geom_emb[idx_batch]).to(device)
            yt = torch.from_numpy(y[idx_batch]).to(device)
            pred_t = t_model.decode_append(g, s_steps, aoa_ground_truth=yt[:, :, 2])
            pred_np = pred_t.cpu().numpy()

            batch_js = [all_js[i] for i in idx_batch]
            base34_batch = pred_to_polar34_batch(pred_np, batch_js)
            tfm_base34[start:end] = base34_batch

            if corr_model is not None:
                gy = torch.from_numpy(coords[idx_batch].astype(np.float32)).to(device)
                b34_t = torch.from_numpy(base34_batch).to(device)
                corr_out = corr_model.predict(gy, b34_t).cpu().numpy()
                corr_pred34[start:end] = corr_out

            for bi in range(bs):
                i_global = idx_batch[bi]
                for t_idx, j in enumerate(all_js[i_global]):
                    if j >= N_SLOTS:
                        continue
                    gt34[start + bi, j] = y[i_global, t_idx, 0]
                    gt34[start + bi, N_SLOTS + j] = y[i_global, t_idx, 1]
                    mask34[start + bi, j] = 1.0
                    mask34[start + bi, N_SLOTS + j] = 1.0

    device_sync(device)
    t_pipeline = time.perf_counter() - t0_total

    m_cl = mask34[:, :N_SLOTS]
    m_cd = mask34[:, N_SLOTS:]
    gt_cl, gt_cd = gt34[:, :N_SLOTS], gt34[:, N_SLOTS:]
    tfm_cl, tfm_cd = tfm_base34[:, :N_SLOTS], tfm_base34[:, N_SLOTS:]
    corr_cl, corr_cd = corr_pred34[:, :N_SLOTS], corr_pred34[:, N_SLOTS:]

    def masked_mae(pred: np.ndarray, gt: np.ndarray, m: np.ndarray) -> float:
        valid = m > 0.5
        if valid.sum() == 0:
            return float("nan")
        return float(np.abs(pred[valid] - gt[valid]).mean())

    def masked_mae_ld(p_cl: np.ndarray, p_cd: np.ndarray, g_cl: np.ndarray, g_cd: np.ndarray, m: np.ndarray) -> float:
        valid = (m[:, :N_SLOTS] > 0.5) & (np.abs(g_cd) > 1e-6)
        if valid.sum() == 0:
            return float("nan")
        gt_ld = g_cl[valid] / g_cd[valid]
        cd_safe = np.where(np.abs(p_cd[valid]) > 1e-8, p_cd[valid], 1e-8)
        return float(np.abs(p_cl[valid] / cd_safe - gt_ld).mean())

    mae_tfm_cl = masked_mae(tfm_cl, gt_cl, m_cl)
    mae_tfm_cd = masked_mae(tfm_cd, gt_cd, m_cd)
    mae_tfm_ld = masked_mae_ld(tfm_cl, tfm_cd, gt_cl, gt_cd, mask34)

    mae_corr_cl = mae_corr_cd = mae_corr_ld = float("nan")
    if corr_model is not None:
        mae_corr_cl = masked_mae(corr_cl, gt_cl, m_cl)
        mae_corr_cd = masked_mae(corr_cd, gt_cd, m_cd)
        mae_corr_ld = masked_mae_ld(corr_cl, corr_cd, gt_cl, gt_cd, mask34)

    mae_nf_cl = mae_nf_cd = mae_nf_ld = float("nan")
    t_nf = float("nan")
    if not args.skip_nf:
        import warnings

        warnings.filterwarnings("ignore")
        import neuralfoil as nf

        nf_cl_all, nf_cd_all = np.zeros_like(gt_cl), np.zeros_like(gt_cd)
        n_skip = 0
        t0_nf = time.perf_counter()
        for vi in range(n_val):
            i = val_idx[vi]
            js = all_js[i]
            n_v = len(js)
            if n_v == 0:
                continue
            aoa_valid = np.array([y[i, t_idx, 2] for t_idx in range(n_v)], dtype=np.float64)
            aero = nf.get_aero_from_coordinates(
                coordinates=coords[i], alpha=aoa_valid, Re=args.Re, model_size="xxxlarge"
            )
            cl_nf = np.asarray(aero["CL"], dtype=np.float32)
            cd_nf = np.asarray(aero["CD"], dtype=np.float32)
            if np.any(np.isnan(cl_nf)) or np.any(np.isnan(cd_nf)):
                n_skip += 1
                continue
            for t_idx, j in enumerate(js):
                if j >= N_SLOTS:
                    continue
                nf_cl_all[vi, j] = cl_nf[t_idx]
                nf_cd_all[vi, j] = cd_nf[t_idx]
        t_nf = time.perf_counter() - t0_nf

        mae_nf_cl = masked_mae(nf_cl_all, gt_cl, m_cl)
        mae_nf_cd = masked_mae(nf_cd_all, gt_cd, m_cd)
        mae_nf_ld = masked_mae_ld(nf_cl_all, nf_cd_all, gt_cl, gt_cd, mask34)
        if n_skip:
            print(f"NeuralFoil: skipped {n_skip} airfoils with NaN")

    print(f"\n{'':25s} {'Transformer':>14s} {'Corrected MLP':>14s} {'NF xxxlarge':>14s}")
    print("-" * 70)
    print(f"{'MAE Cl':25s} {mae_tfm_cl:14.6f} {mae_corr_cl:14.6f} {mae_nf_cl:14.6f}")
    print(f"{'MAE Cd':25s} {mae_tfm_cd:14.6f} {mae_corr_cd:14.6f} {mae_nf_cd:14.6f}")
    print(f"{'MAE L/D':25s} {mae_tfm_ld:14.6f} {mae_corr_ld:14.6f} {mae_nf_ld:14.6f}")
    print()

    ms_per = t_pipeline / n_val * 1000
    print(f"Transformer pipeline ({n_val} airfoils, B={args.batch_size}):")
    print(f"  Total: {t_pipeline:.2f}s  |  {ms_per:.2f} ms/airfoil")
    if not args.skip_nf and np.isfinite(t_nf):
        nf_ms = t_nf / n_val * 1000
        print(f"NeuralFoil xxxlarge: Total {t_nf:.2f}s  |  {nf_ms:.2f} ms/airfoil")

    out = {
        "experiment_id": args.experiment_id,
        "geom_checkpoint": str(geom_ckpt),
        "corr_checkpoint": str(corr_ckpt) if corr_ckpt and corr_ckpt.is_file() else None,
        "n_val": n_val,
        "n_params_transformer": n_params,
        "mae_transformer": {"cl": mae_tfm_cl, "cd": mae_tfm_cd, "ld": mae_tfm_ld},
        "mae_corrected": {"cl": mae_corr_cl, "cd": mae_corr_cd, "ld": mae_corr_ld},
        "mae_neuralfoil": {"cl": mae_nf_cl, "cd": mae_nf_cd, "ld": mae_nf_ld},
        "inference_ms_per_airfoil": ms_per,
        "wall_time_sec": t_pipeline,
    }
    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {outp}")

    append_manifest({"kind": "eval_metrics", **out})


if __name__ == "__main__":
    main()
