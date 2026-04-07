#!/usr/bin/env python3
"""Train PolarCorrectionMLP (studies); --geom-checkpoint required."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

_STUDIES = Path(__file__).resolve().parent.parent
_REPO = _STUDIES.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_STUDIES / "src"))

from foilform.checkpoint_utils import load_geom_transformer  # noqa: E402
from foilform.manifest import append_manifest  # noqa: E402
from foilform.paths import DATA_PROCESSED, RUNS, ensure_dirs  # noqa: E402
from foilform.polar_correction_mlp import (  # noqa: E402
    EXPECTED_PARAMETER_COUNT,
    GEOM_STATIONS,
    N_SLOTS,
    POLAR_DIM,
    PolarCorrectionMLP,
    split_cl_cd,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def airfoil_train_val_mask(n_airfoils: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_airfoils)
    n_train = int(round(train_frac * n_airfoils))
    n_train = max(1, min(n_train, n_airfoils - 1))
    train = np.zeros(n_airfoils, dtype=bool)
    train[perm[:n_train]] = True
    return train, ~train


def build_geom_xy(coords: np.ndarray) -> np.ndarray:
    if coords.shape[1:] != (GEOM_STATIONS, 2):
        raise ValueError(f"Expected coords (N, {GEOM_STATIONS}, 2), got {coords.shape}")
    return coords.astype(np.float32)


def build_polar34_and_mask(polars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, k, _ = polars.shape
    if k > N_SLOTS:
        raise ValueError(f"Need at most {N_SLOTS} AoA columns, got {k}")
    y = np.zeros((n, POLAR_DIM), dtype=np.float32)
    mask = np.zeros((n, POLAR_DIM), dtype=np.float32)
    for i in range(n):
        for j in range(k):
            if np.isfinite(polars[i, j, 1]):
                y[i, j] = float(polars[i, j, 1])
                y[i, N_SLOTS + j] = float(polars[i, j, 2])
                mask[i, j] = 1.0
                mask[i, N_SLOTS + j] = 1.0
    return y, mask


def build_targets_seq(polars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, n_cols, _ = polars.shape
    counts = [int(np.isfinite(polars[i, :, 1]).sum()) for i in range(n)]
    max_steps = max(counts)
    y = np.zeros((n, max_steps, 3), dtype=np.float32)
    m = np.zeros((n, max_steps), dtype=np.float32)
    for i in range(n):
        rows = []
        for j in range(n_cols):
            if np.isfinite(polars[i, j, 1]):
                rows.append((float(polars[i, j, 1]), float(polars[i, j, 2]), float(polars[i, j, 0])))
        if rows:
            arr = np.asarray(rows, dtype=np.float32)
            y[i, : arr.shape[0], :] = arr
            m[i, : arr.shape[0]] = 1.0
    return y, m


def column_indices_for_airfoils(polars: np.ndarray) -> list[list[int]]:
    out = []
    for i in range(polars.shape[0]):
        js = [j for j in range(polars.shape[1]) if np.isfinite(polars[i, j, 1])]
        out.append(js)
    return out


@torch.no_grad()
def generate_transformer_base(
    geom_emb: np.ndarray,
    y_seq: np.ndarray,
    col_indices: list[list[int]],
    ckpt_path: Path,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model = load_geom_transformer(ckpt_path, device)
    s_steps = y_seq.shape[1]
    n = geom_emb.shape[0]
    base34 = np.zeros((n, POLAR_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="gen_base", unit="batch"):
            end = min(start + batch_size, n)
            g = torch.from_numpy(geom_emb[start:end]).to(device)
            yt = torch.from_numpy(y_seq[start:end]).to(device)
            pred = model.decode_append(
                geom_context=g,
                target_steps=s_steps,
                teacher_tuples=None,
                aoa_ground_truth=yt[:, :, 2],
            )
            pred_np = pred.cpu().numpy()
            for b_idx in range(end - start):
                i = start + b_idx
                js = col_indices[i]
                for t, j in enumerate(js):
                    if j >= N_SLOTS or t >= s_steps:
                        continue
                    base34[i, j] = float(pred_np[b_idx, t, 0])
                    base34[i, N_SLOTS + j] = float(pred_np[b_idx, t, 1])
    return base34


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask
    err2 = (pred - target) ** 2 * w
    den = w.sum().clamp_min(1.0)
    return err2.sum() / den


class PolarCorrDataset(Dataset):
    def __init__(self, geom: np.ndarray, target: np.ndarray, mask: np.ndarray, base: np.ndarray) -> None:
        self.geom = torch.from_numpy(geom).float()
        self.target = torch.from_numpy(target).float()
        self.mask = torch.from_numpy(mask).float()
        self.base = torch.from_numpy(base).float()

    def __len__(self) -> int:
        return int(self.geom.shape[0])

    def __getitem__(self, idx: int):
        return self.geom[idx], self.target[idx], self.mask[idx], self.base[idx]


@torch.no_grad()
def evaluate(model: PolarCorrectionMLP, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    sum_mse = sum_mae_cl = sum_mae_cd = 0.0
    n_batches = 0
    for geom, tgt, m, base in loader:
        geom, tgt, m, base = geom.to(device), tgt.to(device), m.to(device), base.to(device)
        pred = model.predict(geom, base)
        sum_mse += float(masked_mse(pred, tgt, m).item())
        p_cl, p_cd = split_cl_cd(pred)
        t_cl, t_cd = split_cl_cd(tgt)
        m_cl, m_cd = m[:, :N_SLOTS], m[:, N_SLOTS:]
        sum_mae_cl += float((torch.abs(p_cl - t_cl) * m_cl).sum() / m_cl.sum().clamp_min(1.0))
        sum_mae_cd += float((torch.abs(p_cd - t_cd) * m_cd).sum() / m_cd.sum().clamp_min(1.0))
        n_batches += 1
    if n_batches == 0:
        return float("nan"), float("nan"), float("nan")
    inv = 1.0 / n_batches
    return sum_mse * inv, sum_mae_cl * inv, sum_mae_cd * inv


def resolve_device(prefer: str) -> torch.device:
    p = prefer.lower()
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--geom-checkpoint",
        type=str,
        required=True,
        help="Path to best_geom_polar_transformer.pt (studies run).",
    )
    parser.add_argument("--experiment-id", type=str, default="")
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="If set, run directory is polar_corr_<run_name>.",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="If polar_corr_<run-name>/best_polar_correction.pt exists, exit 0 without training.",
    )
    args = parser.parse_args()

    geom_ckpt = Path(args.geom_checkpoint)
    if not geom_ckpt.is_file():
        geom_ckpt = _REPO / args.geom_checkpoint
    if not geom_ckpt.is_file():
        raise FileNotFoundError(f"geom checkpoint not found: {args.geom_checkpoint}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    ensure_dirs()

    if args.run_name and args.skip_if_exists:
        run_dir_skip = RUNS / f"polar_corr_{args.run_name}"
        ckpt_skip = run_dir_skip / "best_polar_correction.pt"
        if ckpt_skip.is_file():
            print(f"Skipping (--skip-if-exists): {ckpt_skip}")
            sys.exit(0)

    coords_path = DATA_PROCESSED / "coords.npy"
    polars_path = DATA_PROCESSED / "polars.npy"
    geom_emb_path = DATA_PROCESSED / "geom_embeddings.npy"
    for p in (coords_path, polars_path, geom_emb_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")

    coords = np.load(coords_path).astype(np.float32)
    polars = np.load(polars_path).astype(np.float32)
    geom_emb = np.load(geom_emb_path).astype(np.float32)
    n = coords.shape[0]
    geom_xy = build_geom_xy(coords)
    y34, m34 = build_polar34_and_mask(polars)
    y_seq, m_seq = build_targets_seq(polars)
    col_indices = column_indices_for_airfoils(polars)

    print(f"Generating base predictions from transformer: {geom_ckpt}")
    base34 = generate_transformer_base(geom_emb, y_seq, col_indices, geom_ckpt, device)
    residual_mag = float(np.abs(y34 - base34)[m34 > 0.5].mean())
    print(f"Mean |GT - transformer_base| over valid slots: {residual_mag:.6f}")

    tr_mask, va_mask = airfoil_train_val_mask(n, args.train_frac, args.seed)
    train_ds = PolarCorrDataset(geom_xy[tr_mask], y34[tr_mask], m34[tr_mask], base34[tr_mask])
    val_ds = PolarCorrDataset(geom_xy[va_mask], y34[va_mask], m34[va_mask], base34[va_mask])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = PolarCorrectionMLP().to(device)
    total_n = model.count_parameters()
    if total_n != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_PARAMETER_COUNT} params, got {total_n}")
    print(f"PolarCorrectionMLP: {total_n} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    RUNS.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        run_dir = RUNS / f"polar_corr_{args.run_name}"
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_id = datetime.now().strftime("polar_corr_%Y%m%d_%H%M%S")
        run_dir = RUNS / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
    ckpt = run_dir / "best_polar_correction.pt"

    exp_id = args.experiment_id or f"polarcorr_{geom_ckpt.parent.name}"

    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(
            {**vars(args), "total_params": total_n, "geom_checkpoint": str(geom_ckpt), "experiment_id": exp_id},
            f,
            indent=2,
        )

    best_val = float("inf")
    best_epoch = -1
    history: list[dict] = []

    for ep in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for geom_b, tgt_b, mask_b, base_b in train_loader:
            geom_b, tgt_b, mask_b, base_b = (
                geom_b.to(device),
                tgt_b.to(device),
                mask_b.to(device),
                base_b.to(device),
            )
            opt.zero_grad(set_to_none=True)
            pred = model.predict(geom_b, base_b)
            loss = masked_mse(pred, tgt_b, mask_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()

        tr_mse = float(np.mean(losses)) if losses else float("nan")
        va_mse, va_mae_cl, va_mae_cd = evaluate(model, val_loader, device)
        row = {
            "epoch": ep,
            "train_mse": tr_mse,
            "val_mse": va_mse,
            "val_mae_cl": va_mae_cl,
            "val_mae_cd": va_mae_cd,
        }
        history.append(row)
        print(
            f"epoch {ep}/{args.epochs}  train_mse={tr_mse:.6f}  val_mse={va_mse:.6f}  "
            f"val_mae_cl={va_mae_cl:.6f}  val_mae_cd={va_mae_cd:.6f}",
        )

        if np.isfinite(va_mse) and va_mse < best_val:
            best_val = va_mse
            best_epoch = ep
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "best_val_mse": best_val,
                    "args": vars(args),
                    "geom_checkpoint": str(geom_ckpt),
                },
                ckpt,
            )

        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    wall_s = time.perf_counter() - t0
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_mse": best_val,
                "best_epoch": best_epoch,
                "checkpoint": str(ckpt),
                "wall_time_sec": wall_s,
                "total_params_corr": total_n,
                "experiment_id": exp_id,
            },
            f,
            indent=2,
        )

    print(f"Best val MSE: {best_val:.6f} | checkpoint: {ckpt}")

    append_manifest(
        {
            "kind": "train_polar_correction_mlp",
            "experiment_id": exp_id,
            "run_dir": str(run_dir),
            "geom_checkpoint": str(geom_ckpt),
            "total_params_corr": total_n,
            "best_epoch": best_epoch,
            "wall_time_sec": wall_s,
            "best_val_mse": best_val,
            "args": vars(args),
        }
    )


if __name__ == "__main__":
    main()
