#!/usr/bin/env python3
"""Train GeomPolarTransformer (studies): block_type, n_layers, dropout, AoA order."""

from __future__ import annotations

import argparse
import contextlib
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

from foilform.geom_polar_transformer import (  # noqa: E402
    BLOCK_TYPES,
    GeomPolarTransformer,
)
from foilform.manifest import append_manifest  # noqa: E402
from foilform.paths import DATA_PROCESSED, RUNS, ensure_dirs  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(prefer: str) -> torch.device:
    p = prefer.lower()
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def airfoil_train_val_mask(n_airfoils: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_airfoils)
    n_train = int(round(train_frac * n_airfoils))
    n_train = max(1, min(n_train, n_airfoils - 1))
    train = np.zeros(n_airfoils, dtype=bool)
    train[perm[:n_train]] = True
    return train, ~train


def build_targets(polars: np.ndarray, *, reverse_aoa_order: bool = False) -> tuple[np.ndarray, np.ndarray]:
    n, n_cols, _ = polars.shape
    counts = [int(np.isfinite(polars[i, :, 1]).sum()) for i in range(n)]
    max_steps = max(counts)
    y = np.zeros((n, max_steps, 3), dtype=np.float32)
    m = np.zeros((n, max_steps), dtype=np.float32)
    for i in tqdm(range(n), desc="build_targets", unit="airfoil"):
        rows = []
        for j in range(n_cols):
            if np.isfinite(polars[i, j, 1]):
                aoa = float(polars[i, j, 0])
                cl = float(polars[i, j, 1])
                cd = float(polars[i, j, 2])
                rows.append((cl, cd, aoa))
        if rows:
            arr = np.asarray(rows, dtype=np.float32)
            if reverse_aoa_order:
                idx = np.argsort(-arr[:, 2])
                arr = arr[idx]
            s = arr.shape[0]
            y[i, :s, :] = arr
            m[i, :s] = 1.0
    return y, m


class AirfoilSeqDataset(Dataset):
    def __init__(self, geom: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        self.geom = torch.from_numpy(geom).float()
        self.y = torch.from_numpy(y).float()
        self.mask = torch.from_numpy(mask).float()

    def __len__(self) -> int:
        return int(self.geom.shape[0])

    def __getitem__(self, idx: int):
        return self.geom[idx], self.y[idx], self.mask[idx]


def compute_channel_scales(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    y2 = y[..., :2]
    w = mask[..., None].astype(np.float64)
    denom = np.maximum(w.sum(axis=(0, 1)), 1.0)
    mean_abs = (np.abs(y2).astype(np.float64) * w).sum(axis=(0, 1)) / denom
    return np.maximum(mean_abs, 1e-6).astype(np.float32)


def masked_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    channel_scales: torch.Tensor,
) -> torch.Tensor:
    w = mask.unsqueeze(-1)
    err = pred - target[..., :2]
    err_bal2 = (err / channel_scales.view(1, 1, 2)).pow(2)
    num = (err_bal2 * w).sum()
    den = (w.sum() * pred.shape[-1]).clamp_min(1.0)
    return num / den


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


def autocast_cm(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def save_history_csv(path: Path, rows: list[dict]) -> None:
    header = [
        "epoch",
        "train_bal_mse",
        "val_bal_mse",
        "lr",
        "train_mae_cl",
        "train_mae_cd",
        "val_mae_cl",
        "val_mae_cd",
        "skipped_nonfinite",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            vals = [
                r["epoch"],
                r["train_bal_mse"],
                r["val_bal_mse"],
                r["lr"],
                r["train_mae_cl"],
                r["train_mae_cd"],
                r["val_mae_cl"],
                r["val_mae_cd"],
                r["skipped_nonfinite"],
            ]
            f.write(",".join(str(v) for v in vals) + "\n")


def save_plots(run_dir: Path, history: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting skipped: {e}")
        return

    ep = np.array([r["epoch"] for r in history], dtype=np.int32)
    tr_mse = np.array([r["train_bal_mse"] for r in history], dtype=np.float64)
    va_mse = np.array([r["val_bal_mse"] for r in history], dtype=np.float64)
    lr = np.array([r["lr"] for r in history], dtype=np.float64)
    skip = np.array([r["skipped_nonfinite"] for r in history], dtype=np.float64)
    tr_cl = np.array([r["train_mae_cl"] for r in history], dtype=np.float64)
    tr_cd = np.array([r["train_mae_cd"] for r in history], dtype=np.float64)
    va_cl = np.array([r["val_mae_cl"] for r in history], dtype=np.float64)
    va_cd = np.array([r["val_mae_cd"] for r in history], dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(ep, tr_mse, label="train_bal_mse")
    ax.plot(ep, va_mse, label="val_bal_mse")
    ax.set_xlabel("epoch")
    ax.set_ylabel("balanced MSE")
    ax.set_title("Balanced Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "curve_balanced_mse.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(ep, tr_cl, label="train")
    axs[0].plot(ep, va_cl, label="val")
    axs[0].set_title("MAE Cl")
    axs[1].plot(ep, tr_cd, label="train")
    axs[1].plot(ep, va_cd, label="val")
    axs[1].set_title("MAE Cd")
    for ax in axs:
        ax.set_xlabel("epoch")
        ax.set_ylabel("raw MAE")
        ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "curve_raw_mae_channels.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.5))
    ax1.plot(ep, lr, color="#2563eb", label="lr")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("learning rate", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(ep, skip, color="#dc2626", label="skipped_nonfinite")
    ax2.set_ylabel("skipped non-finite batches", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax1.set_title("LR and Non-finite Batch Skips")
    fig.tight_layout()
    fig.savefig(run_dir / "curve_lr_and_skips.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def evaluate(
    model: GeomPolarTransformer,
    loader: DataLoader,
    device: torch.device,
    channel_scales: torch.Tensor,
) -> tuple[float, np.ndarray]:
    model.eval()
    losses = []
    ch_sum = torch.zeros(2, device=device)
    n_batches = 0
    for geom, y, mask in tqdm(loader, desc="val", leave=False, unit="batch"):
        geom = geom.to(device)
        y = y.to(device)
        mask = mask.to(device)
        with autocast_cm(device):
            pred = model.decode_append(
                geom_context=geom,
                target_steps=y.shape[1],
                teacher_tuples=None,
                aoa_ground_truth=y[:, :, 2],
            )
        if not torch.isfinite(pred).all():
            continue
        pf = pred.float()
        yf = y.float()
        losses.append(masked_weighted_mse(pf, yf, mask, channel_scales).item())
        ch_sum += masked_channel_mae(pf, yf, mask)
        n_batches += 1
    if not losses:
        return float("nan"), np.array([np.nan, np.nan], dtype=np.float32)
    ch = (ch_sum / max(1, n_batches)).detach().cpu().numpy().astype(np.float32)
    return float(np.mean(losses)), ch


def main() -> None:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--train_frac",
        "--train-frac",
        type=float,
        default=0.6,
        dest="train_frac",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--early_stop_warmup", type=int, default=10)
    parser.add_argument(
        "--block-type",
        type=str,
        default="pairwise",
        help=f"One of {BLOCK_TYPES}",
    )
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--reverse-aoa-order",
        action="store_true",
        help="Sort polar rows by descending AoA before packing.",
    )
    parser.add_argument("--experiment-id", type=str, default="", help="Tag for manifest (default: auto).")
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="If set, run directory is geom_polar_<run_name> (must not exist).",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="If geom_polar_<run-name>/best_geom_polar_transformer.pt exists, exit 0 without training.",
    )
    args = parser.parse_args()

    bt = args.block_type.lower().strip()
    if bt not in BLOCK_TYPES:
        raise ValueError(f"--block-type must be one of {BLOCK_TYPES}, got {args.block_type!r}")

    set_seed(args.seed)
    device = resolve_device(args.device)

    ensure_dirs()
    if args.run_name and args.skip_if_exists:
        run_dir_skip = RUNS / f"geom_polar_{args.run_name}"
        ckpt_skip = run_dir_skip / "best_geom_polar_transformer.pt"
        if ckpt_skip.is_file():
            print(f"Skipping (--skip-if-exists): {ckpt_skip}")
            sys.exit(0)

    geom_path = DATA_PROCESSED / "geom_embeddings.npy"
    polars_path = DATA_PROCESSED / "polars.npy"
    if not geom_path.is_file():
        raise FileNotFoundError("geom_embeddings.npy not found.")
    if not polars_path.is_file():
        raise FileNotFoundError("polars.npy not found.")

    geom = np.load(geom_path).astype(np.float32)
    polars = np.load(polars_path).astype(np.float32)
    if geom.ndim != 3 or geom.shape[-1] != 8:
        raise ValueError(f"Expected geom_embeddings shape (N, L, 8), got {geom.shape}")
    n_airfoils = geom.shape[0]

    y, m = build_targets(polars, reverse_aoa_order=args.reverse_aoa_order)
    n_steps = y.shape[1]
    tr_mask, va_mask = airfoil_train_val_mask(n_airfoils, args.train_frac, args.seed)
    print(
        f"Split train/val: {int(tr_mask.sum())}/{int(va_mask.sum())} airfoils "
        f"(train_frac={args.train_frac:.2f}); target_steps={n_steps}; block_type={bt} "
        f"n_layers={args.n_layers} dropout={args.dropout} reverse_aoa={args.reverse_aoa_order}",
    )

    train_ds = AirfoilSeqDataset(geom[tr_mask], y[tr_mask], m[tr_mask])
    val_ds = AirfoilSeqDataset(geom[va_mask], y[va_mask], m[va_mask])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    channel_scales_np = compute_channel_scales(y[tr_mask], m[tr_mask])
    channel_scales = torch.from_numpy(channel_scales_np).to(device)
    channel_weights = 1.0 / channel_scales_np
    print(
        "Balanced MAE scales [Cl, Cd]="
        f"{channel_scales_np.tolist()} | weights={channel_weights.tolist()}"
    )

    model = GeomPolarTransformer(
        n_layers=args.n_layers,
        dropout=args.dropout,
        block_type=bt,
    ).to(device)
    n_params = model.count_parameters()
    print(f"Parameters: {n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    runs_dir = RUNS
    runs_dir.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        run_dir = runs_dir / f"geom_polar_{args.run_name}"
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_id = datetime.now().strftime("geom_polar_%Y%m%d_%H%M%S")
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

    exp_id = args.experiment_id or f"{bt}_nl{args.n_layers}_dr{args.dropout}_tf{args.train_frac}_seed{args.seed}"
    if args.reverse_aoa_order:
        exp_id += "_revaoa"

    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "n_params": n_params, "experiment_id": exp_id}, f, indent=2)

    best_val = float("inf")
    ckpt = run_dir / "best_geom_polar_transformer.pt"
    history: list[dict] = []
    no_improve_checks = 0
    best_epoch = -1

    print(f"Run directory: {run_dir}")

    epoch_pbar = tqdm(
        range(1, args.epochs + 1),
        desc="epoch",
        unit="ep",
        dynamic_ncols=True,
    )
    for ep in epoch_pbar:
        model.train()
        train_losses = []
        train_ch_sum = torch.zeros(2, device=device)
        train_n_batches = 0
        skipped_nonfinite = 0
        batch_pbar = tqdm(
            train_loader,
            desc=f"train ep {ep}/{args.epochs}",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )
        for geom_b, y_b, m_b in batch_pbar:
            geom_b = geom_b.to(device)
            y_b = y_b.to(device)
            m_b = m_b.to(device)

            with autocast_cm(device):
                pred = model.decode_append(
                    geom_context=geom_b,
                    target_steps=n_steps,
                    teacher_tuples=None,
                    aoa_ground_truth=y_b[:, :, 2],
                )
            if not torch.isfinite(pred).all():
                skipped_nonfinite += 1
                batch_pbar.set_postfix(skip=skipped_nonfinite)
                continue
            loss = masked_weighted_mse(pred.float(), y_b.float(), m_b, channel_scales)
            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                batch_pbar.set_postfix(skip=skipped_nonfinite)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                batch_pbar.set_postfix(skip=skipped_nonfinite)
                continue
            opt.step()
            train_losses.append(loss.item())
            with torch.no_grad():
                train_ch_sum += masked_channel_mae(pred.float(), y_b.float(), m_b)
                train_n_batches += 1
            batch_pbar.set_postfix(
                loss=f"{float(np.mean(train_losses)):.5f}",
                skip=skipped_nonfinite,
            )

        sched.step()
        train_mae = float(np.mean(train_losses)) if train_losses else float("nan")
        if train_n_batches > 0:
            train_ch = (train_ch_sum / train_n_batches).detach().cpu().numpy()
        else:
            train_ch = np.array([np.nan, np.nan], dtype=np.float32)
        do_val = (ep % max(1, int(args.val_every)) == 0) or (ep == 1) or (ep == args.epochs)
        if do_val:
            val_mae, val_ch = evaluate(model, val_loader, device, channel_scales)
        else:
            val_mae = float("nan")
            val_ch = np.array([np.nan, np.nan], dtype=np.float32)
        lr = float(opt.param_groups[0]["lr"])
        if do_val:
            epoch_pbar.set_postfix(
                tr_mse=f"{train_mae:.5f}",
                val_mse=f"{val_mae:.5f}",
                lr=f"{lr:.1e}",
                tr_cl=f"{train_ch[0]:.4f}",
                tr_cd=f"{train_ch[1]:.4f}",
                va_cl=f"{val_ch[0]:.4f}",
                va_cd=f"{val_ch[1]:.4f}",
                skip=skipped_nonfinite,
            )
        else:
            epoch_pbar.set_postfix(
                tr_mse=f"{train_mae:.5f}",
                val_mse="—",
                lr=f"{lr:.1e}",
                tr_cl=f"{train_ch[0]:.4f}",
                tr_cd=f"{train_ch[1]:.4f}",
                va_cl="—",
                va_cd="—",
                skip=skipped_nonfinite,
            )

        row = {
            "epoch": int(ep),
            "train_bal_mse": float(train_mae),
            "val_bal_mse": float(val_mae),
            "lr": float(lr),
            "train_mae_cl": float(train_ch[0]),
            "train_mae_cd": float(train_ch[1]),
            "val_mae_cl": float(val_ch[0]),
            "val_mae_cd": float(val_ch[1]),
            "skipped_nonfinite": int(skipped_nonfinite),
        }
        history.append(row)

        improved = False
        if do_val and np.isfinite(val_mae) and (best_val - val_mae) > args.min_delta:
            improved = True
            best_val = val_mae
            best_epoch = ep
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "best_val_mse": best_val,
                    "args": vars(args),
                    "channel_scales": channel_scales_np.tolist(),
                    "n_params": n_params,
                    "block_type": bt,
                    "n_layers": args.n_layers,
                },
                ckpt,
            )

        if do_val:
            if improved:
                no_improve_checks = 0
            else:
                no_improve_checks += 1

        save_history_csv(run_dir / "history.csv", history)
        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if (
            ep >= args.early_stop_warmup
            and do_val
            and no_improve_checks >= args.patience
        ):
            tqdm.write(
                "Early stopping triggered: "
                f"no val improvement > {args.min_delta} for {no_improve_checks} checks."
            )
            break

    save_plots(run_dir, history)
    wall_s = time.perf_counter() - t0
    summary = {
        "best_val_mse": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(history[-1]["epoch"]) if history else 0,
        "checkpoint": str(ckpt),
        "history_csv": str(run_dir / "history.csv"),
        "history_json": str(run_dir / "history.json"),
        "plot_balanced_mse": str(run_dir / "curve_balanced_mse.png"),
        "plot_raw_mae_channels": str(run_dir / "curve_raw_mae_channels.png"),
        "plot_lr_skips": str(run_dir / "curve_lr_and_skips.png"),
        "n_params": n_params,
        "wall_time_sec": wall_s,
        "experiment_id": exp_id,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best val MSE: {best_val:.6f} (epoch {best_epoch}) | checkpoint: {ckpt}")
    print(f"Saved history/plots: {run_dir}")

    val_at_best = next((r for r in history if r["epoch"] == best_epoch), None)
    append_manifest(
        {
            "kind": "train_geom_polar_transformer",
            "experiment_id": exp_id,
            "run_dir": str(run_dir),
            "n_params": n_params,
            "best_epoch": best_epoch,
            "epochs_ran": summary["epochs_ran"],
            "wall_time_sec": wall_s,
            "val_mae_cl": float(val_at_best["val_mae_cl"]) if val_at_best else None,
            "val_mae_cd": float(val_at_best["val_mae_cd"]) if val_at_best else None,
            "best_val_mse": best_val,
            "args": vars(args),
        }
    )


if __name__ == "__main__":
    main()
