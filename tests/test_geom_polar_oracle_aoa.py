#!/usr/bin/env python3
"""
Sanity check: model predicts Cl, Cd only; the autoregressive chain uses
``[pred_Cl, pred_Cd, ground_truth_AoA]`` (see ``decode_append(..., aoa_ground_truth=...)``).

Usage (from repo root):
  python tests/test_geom_polar_oracle_aoa.py
  python tests/test_geom_polar_oracle_aoa.py --checkpoint models/geom_polar_transformer.pt
  python tests/test_geom_polar_oracle_aoa.py --random_init
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
from foilform.checkpoints import resolve_geom_polar_transformer  # noqa: E402
from foilform.geom_polar_transformer import GeomPolarTransformer  # noqa: E402
from foilform.paths import DATA_PROCESSED  # noqa: E402


def airfoil_train_val_mask(n_airfoils: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_airfoils)
    n_train = int(round(train_frac * n_airfoils))
    n_train = max(1, min(n_train, n_airfoils - 1))
    train = np.zeros(n_airfoils, dtype=bool)
    train[perm[:n_train]] = True
    return train, ~train


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


@torch.no_grad()
def masked_channel_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    w = mask.unsqueeze(-1)
    err = torch.abs(pred - target[..., :2]) * w
    num = err.sum(dim=(0, 1))
    den = w.sum(dim=(0, 1)).clamp_min(1.0)
    return num / den


def device_sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to checkpoint. If empty, uses models/geom_polar_transformer.pt or newest runs/*/best_geom_polar_transformer.pt.",
    )
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Ignore checkpoints and use random weights (sanity only).",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, choices=("val", "train"), default="val")
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="If >0, only run this many batches (smoke test).",
    )
    parser.add_argument(
        "--timing_warmup",
        type=int,
        default=2,
        help="Decode passes on the first batch before timing (GPU/MPS). Use 0 to skip.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else "mps"
        if args.device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )

    geom = np.load(DATA_PROCESSED / "geom_embeddings.npy").astype(np.float32)
    polars = np.load(DATA_PROCESSED / "polars.npy").astype(np.float32)
    y, m = build_targets(polars)
    n = geom.shape[0]
    tr, va = airfoil_train_val_mask(n, args.train_frac, args.seed)
    idx = tr if args.split == "train" else va
    geom_s = geom[idx]
    y_s = y[idx]
    m_s = m[idx]

    model = GeomPolarTransformer().to(device)
    ckpt_path: Path | None = None
    if args.random_init:
        print("Using random weights (--random_init).")
    elif args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_file():
            ckpt_path = _REPO / args.checkpoint
    else:
        ckpt_path = resolve_geom_polar_transformer()

    if ckpt_path is not None and ckpt_path.is_file() and not args.random_init:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        print(f"Loaded checkpoint: {ckpt_path}")
    elif not args.random_init and args.checkpoint and ckpt_path is not None and not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    elif not args.random_init and ckpt_path is None:
        print("No models/geom_polar_transformer.pt or runs/*/best_geom_polar_transformer.pt — using random weights.")

    model.eval()
    n_samples = geom_s.shape[0]
    s_steps = y_s.shape[1]

    ch_sum = torch.zeros(2, device=device)
    n_batches = 0
    n_airfoils_used = 0
    total_decode_steps = 0

    # First batch for warmup (same shape as real batches).
    w_end = min(args.batch_size, n_samples)
    g0 = torch.from_numpy(geom_s[:w_end]).to(device)
    yt0 = torch.from_numpy(y_s[:w_end]).to(device)
    for _ in range(max(0, args.timing_warmup)):
        device_sync(device)
        _ = model.decode_append(
            geom_context=g0,
            target_steps=s_steps,
            teacher_tuples=None,
            aoa_ground_truth=yt0[:, :, 2],
        )
        device_sync(device)

    device_sync(device)
    t0 = time.perf_counter()
    for start in range(0, n_samples, args.batch_size):
        end = min(start + args.batch_size, n_samples)
        g = torch.from_numpy(geom_s[start:end]).to(device)
        yt = torch.from_numpy(y_s[start:end]).to(device)
        mk = torch.from_numpy(m_s[start:end]).to(device)
        bsz = end - start
        pred = model.decode_append(
            geom_context=g,
            target_steps=s_steps,
            teacher_tuples=None,
            aoa_ground_truth=yt[:, :, 2],
        )
        ch_sum += masked_channel_mae(pred, yt, mk)
        n_batches += 1
        n_airfoils_used += bsz
        total_decode_steps += bsz * s_steps
        if args.max_batches > 0 and n_batches >= args.max_batches:
            break
    device_sync(device)
    elapsed = time.perf_counter() - t0

    ch = (ch_sum / max(1, n_batches)).cpu().numpy()
    sec_per_airfoil = elapsed / max(1, n_airfoils_used)
    sec_per_ar_step = elapsed / max(1, total_decode_steps)
    print(
        f"split={args.split}  n={n_samples}  steps={s_steps}  device={device}  "
        f"batches_used={n_batches}"
        + (f" (capped by --max_batches={args.max_batches})" if args.max_batches > 0 else ""),
    )
    print(
        f"timing: wall={elapsed:.4f}s  airfoils_timed={n_airfoils_used}  "
        f"decode_steps={total_decode_steps}  "
        f"sec/airfoil={sec_per_airfoil:.6f}  sec/autoreg_step(mean)={sec_per_ar_step:.6f}",
    )
    print("Masked MAE [Cl, Cd] (chain uses pred Cl/Cd + ground-truth AoA):")
    print(f"  Cl={ch[0]:.6f}  Cd={ch[1]:.6f}")


if __name__ == "__main__":
    main()
