#!/usr/bin/env python3
"""Train geometry-triplet and aero tokenizer autoencoders.

Produces (under repo root):
  artifacts/geom_tokenizer.pt
  artifacts/aero_tokenizer.pt
  data/processed/geom_embeddings.npy  (N, 167, 8)
  data/processed/aero_embeddings.npy  (N, 9, 8) NaN-padded

Train/val is by airfoil id (default train_frac=0.6) so triplets and polars do not leak.

Embedding plots:  python visualize_embeddings.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
from foilform.paths import ARTIFACTS, DATA_PROCESSED, ensure_dirs  # noqa: E402
from foilform.tokenizer_model import AeroDecoder, AeroEncoder, TripletDecoder, TripletEncoder  # noqa: E402

N_PATCHES = 167
PATCH_PTS = 3


# ── Data preparation ───────────────────────────────────────────────────────────

def extract_triplets(coords: np.ndarray) -> np.ndarray:
    """coords (N, 501, 2) → triplets (N, 167, 6)."""
    N = coords.shape[0]
    out = np.zeros((N, N_PATCHES, 6), dtype=np.float32)
    for k in range(N_PATCHES):
        i0 = k * PATCH_PTS
        out[:, k, 0:2] = coords[:, i0, :]
        out[:, k, 2:4] = coords[:, i0 + 1, :]
        out[:, k, 4:6] = coords[:, i0 + 2, :]
    return out


def extract_aero(polars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """polars (N, 9, 3) → flat valid observations (M, 3), indices (M, 2) for (airfoil, aoa_col)."""
    rows = []
    idxs = []
    for i in range(polars.shape[0]):
        for j in range(polars.shape[1]):
            if np.isfinite(polars[i, j, 1]):
                rows.append(polars[i, j])
                idxs.append((i, j))
    return np.array(rows, dtype=np.float32), np.array(idxs, dtype=np.int64)


# ── Training helpers ───────────────────────────────────────────────────────────

@torch.no_grad()
def _recon_mse_batches(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    data: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    """Returns (recon_mse, mean ||z||²) over *data* (no noise, eval mode)."""
    encoder.eval()
    decoder.eval()
    tot = 0.0
    tot_z2 = 0.0
    n = 0
    for i in range(0, data.shape[0], batch_size):
        batch = data[i : i + batch_size].to(device)
        z = encoder(batch)
        recon = decoder(z)
        tot += F.mse_loss(recon, batch, reduction="sum").item()
        tot_z2 += z.pow(2).sum().item()
        n += batch.size(0)
    denom = max(n, 1)
    return tot / denom, tot_z2 / denom


def train_autoencoder(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    data: torch.Tensor,
    *,
    device: torch.device,
    val_data: torch.Tensor | None = None,
    epochs: int = 300,
    batch_size: int = 4096,
    lr: float = 1e-3,
    embed_l2: float = 0.01,
    noise_std: float = 0.0,
    tag: str = "model",
) -> None:
    """Denoising autoencoder training loop with embedding L2 regularisation."""
    encoder.to(device).train()
    decoder.to(device).train()
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    total_steps = epochs * max(1, len(data) // batch_size)
    warmup = min(500, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    val_bs = min(batch_size, 8192)
    step = 0
    for ep in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        ep_loss = 0.0
        n = 0
        for (batch,) in dl:
            batch = batch.to(device)
            if noise_std > 0:
                batch_noisy = batch + torch.randn_like(batch) * noise_std
            else:
                batch_noisy = batch

            z = encoder(batch_noisy)
            recon = decoder(z)
            loss_recon = F.mse_loss(recon, batch)
            loss_reg = embed_l2 * z.pow(2).mean()
            loss = loss_recon + loss_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1

            ep_loss += loss_recon.item() * batch.size(0)
            n += batch.size(0)

        mse = ep_loss / max(n, 1)
        if ep % 50 == 0 or ep == 1 or ep == epochs:
            if val_data is not None and val_data.shape[0] > 0:
                v_mse, _ = _recon_mse_batches(
                    encoder, decoder, val_data, device, val_bs,
                )
                print(
                    f"  [{tag}] epoch {ep:3d}/{epochs}  train_recon_mse={mse:.8f}  "
                    f"val_recon_mse={v_mse:.8f}",
                )
            else:
                print(f"  [{tag}] epoch {ep:3d}/{epochs}  train_recon_mse={mse:.8f}")

    encoder.eval()
    decoder.eval()


@torch.no_grad()
def encode_all(
    encoder: torch.nn.Module,
    data: torch.Tensor,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    encoder.eval()
    parts = []
    for i in range(0, data.shape[0], batch_size):
        x = data[i : i + batch_size].to(device)
        parts.append(encoder(x).cpu().numpy())
    return np.concatenate(parts, axis=0)


# ── Main ───────────────────────────────────────────────────────────────────────

def airfoil_train_val_mask(n_airfoils: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratify by airfoil id so no contour leaks between train and val."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_airfoils)
    n_train = int(round(train_frac * n_airfoils))
    n_train = max(1, min(n_train, n_airfoils - 1))
    train_ids = np.zeros(n_airfoils, dtype=bool)
    train_ids[perm[:n_train]] = True
    val_ids = ~train_ids
    return train_ids, val_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--train_frac", type=float, default=0.6, help="Fraction of airfoils in train (rest = val).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device(args.device)
    coords = np.load(DATA_PROCESSED / "coords.npy")   # (2946, 501, 2)
    polars = np.load(DATA_PROCESSED / "polars.npy")    # (2946, 9, 3)
    N, n_aoa = polars.shape[0], polars.shape[1]

    train_airfoil, val_airfoil = airfoil_train_val_mask(N, args.train_frac, args.seed)
    n_tr = int(train_airfoil.sum())
    n_va = int(val_airfoil.sum())
    print(f"Airfoil split: train={n_tr}  val={n_va}  (train_frac={args.train_frac})")

    # ── Extract raw data ───────────────────────────────────────────────────────
    triplets = extract_triplets(coords)                          # (N, 167, 6)
    aero_flat, aero_idx = extract_aero(polars)                   # (M, 3), (M, 2)
    print(f"Geometry triplets: {N} airfoils × {N_PATCHES} = {N * N_PATCHES:,}")
    print(f"Aero observations: {aero_flat.shape[0]:,}")

    trip_flat = torch.from_numpy(triplets.reshape(-1, 6))        # (N*167, 6)
    airfoil_per_trip = np.repeat(np.arange(N), N_PATCHES)
    trip_train_mask = train_airfoil[airfoil_per_trip]
    trip_val_mask = val_airfoil[airfoil_per_trip]
    trip_train = trip_flat[trip_train_mask]
    trip_val = trip_flat[trip_val_mask]

    aero_t = torch.from_numpy(aero_flat)
    aero_airfoil = aero_idx[:, 0]
    aero_train_mask = train_airfoil[aero_airfoil]
    aero_train = aero_t[aero_train_mask]
    aero_val = aero_t[~aero_train_mask]

    print(
        f"Triplet rows: train={trip_train.shape[0]:,}  val={trip_val.shape[0]:,} | "
        f"Aero rows: train={aero_train.shape[0]:,}  val={aero_val.shape[0]:,}",
    )

    # ── Train geometry autoencoder ─────────────────────────────────────────────
    print("\n=== Geometry tokenizer ===")
    geom_enc = TripletEncoder()
    geom_dec = TripletDecoder()
    train_autoencoder(
        geom_enc, geom_dec, trip_train,
        val_data=trip_val,
        device=device, epochs=args.epochs, batch_size=args.batch_size,
        embed_l2=0.01, noise_std=0.005, tag="geom",
    )
    torch.save(
        {"encoder": geom_enc.state_dict(), "decoder": geom_dec.state_dict()},
        ARTIFACTS / "geom_tokenizer.pt",
    )

    # ── Train aero autoencoder ─────────────────────────────────────────────────
    print("\n=== Aero tokenizer ===")
    aero_enc = AeroEncoder()
    aero_dec = AeroDecoder()
    train_autoencoder(
        aero_enc, aero_dec, aero_train,
        val_data=aero_val,
        device=device, epochs=args.epochs, batch_size=min(2048, args.batch_size),
        embed_l2=0.01, noise_std=0.01, tag="aero",
    )
    torch.save(
        {"encoder": aero_enc.state_dict(), "decoder": aero_dec.state_dict()},
        ARTIFACTS / "aero_tokenizer.pt",
    )

    # ── Encode everything & save ───────────────────────────────────────────────
    print("\nEncoding all data…")
    geom_emb_flat = encode_all(geom_enc, trip_flat, device)
    geom_emb = geom_emb_flat.reshape(N, N_PATCHES, -1)           # (N, 167, 8)

    aero_emb_full = np.full((N, n_aoa, 8), np.nan, dtype=np.float32)
    aero_emb_flat = encode_all(aero_enc, aero_t, device)
    for k, (ai, aj) in enumerate(aero_idx):
        aero_emb_full[ai, aj] = aero_emb_flat[k]

    np.save(DATA_PROCESSED / "geom_embeddings.npy", geom_emb)
    np.save(DATA_PROCESSED / "aero_embeddings.npy", aero_emb_full)
    print(f"Saved geom_embeddings.npy  {geom_emb.shape}")
    print(f"Saved aero_embeddings.npy  {aero_emb_full.shape}")

    # ── Reconstruction quality (train vs val — no leakage) ────────────────────
    bs_eval = min(8192, args.batch_size)
    g_tr_mse, _ = _recon_mse_batches(geom_enc, geom_dec, trip_train, device, bs_eval)
    g_va_mse, _ = _recon_mse_batches(geom_enc, geom_dec, trip_val, device, bs_eval)
    a_tr_mse, _ = _recon_mse_batches(aero_enc, aero_dec, aero_train, device, bs_eval)
    a_va_mse, _ = _recon_mse_batches(aero_enc, aero_dec, aero_val, device, bs_eval)

    with torch.no_grad():
        geom_recon = geom_dec(torch.from_numpy(geom_emb_flat).to(device)).cpu().numpy()
        geom_mae = float(np.mean(np.abs(geom_recon - trip_flat.numpy())))

        aero_recon = aero_dec(torch.from_numpy(aero_emb_flat).to(device)).cpu().numpy()
        aero_mae = float(np.mean(np.abs(aero_recon - aero_flat)))

    print(f"\nFinal reconstruction (MSE):")
    print(f"  Geometry  train={g_tr_mse:.2e}  val={g_va_mse:.2e}  | all MAE={geom_mae:.2e}")
    print(f"  Aero      train={a_tr_mse:.2e}  val={a_va_mse:.2e}  | all MAE={aero_mae:.2e}")

    # ── Embedding statistics ───────────────────────────────────────────────────
    print(f"\nGeom embedding  mean={geom_emb_flat.mean():.4f}  std={geom_emb_flat.std():.4f}  "
          f"per-dim std=[{', '.join(f'{s:.3f}' for s in geom_emb_flat.std(axis=0))}]")
    print(f"Aero embedding  mean={aero_emb_flat.mean():.4f}  std={aero_emb_flat.std():.4f}  "
          f"per-dim std=[{', '.join(f'{s:.3f}' for s in aero_emb_flat.std(axis=0))}]")

    print("\nPlots:  python scripts/visualize_embeddings.py")


if __name__ == "__main__":
    main()
