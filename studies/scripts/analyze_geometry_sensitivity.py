#!/usr/bin/env python3
"""Perturb one airfoil coords, re-encode triplets, plot Cl/Cd vs AoA."""

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
from foilform.paths import ARTIFACTS, DATA_PROCESSED, FIGURES  # noqa: E402
from foilform.tokenizer_model import TripletEncoder  # noqa: E402

N_PATCHES = 167
PATCH_PTS = 3


def extract_triplets(coords: np.ndarray) -> np.ndarray:
    N = coords.shape[0]
    out = np.zeros((N, N_PATCHES, 6), dtype=np.float32)
    for k in range(N_PATCHES):
        i0 = k * PATCH_PTS
        out[:, k, 0:2] = coords[:, i0, :]
        out[:, k, 2:4] = coords[:, i0 + 1, :]
        out[:, k, 4:6] = coords[:, i0 + 2, :]
    return out


def encode_geom(enc: TripletEncoder, triplets: np.ndarray, device: torch.device) -> np.ndarray:
    flat = torch.from_numpy(triplets.reshape(-1, 6)).to(device)
    with torch.no_grad():
        z = enc(flat).cpu().numpy()
    return z.reshape(triplets.shape[0], N_PATCHES, -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geom-checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="", help="Path to geom_tokenizer.pt")
    parser.add_argument("--airfoil-index", type=int, default=0)
    parser.add_argument("--camber-pct", type=float, default=0.005)
    parser.add_argument("--thickness-pct", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    tok_path = Path(args.tokenizer) if args.tokenizer else ARTIFACTS / "geom_tokenizer.pt"
    if not tok_path.is_file():
        print(f"Missing tokenizer {tok_path}; skip geometry sensitivity (train tokenizers first).")
        return

    device = torch.device(args.device)
    sd = torch.load(tok_path, map_location=device, weights_only=False)
    enc = TripletEncoder().to(device)
    enc.load_state_dict(sd["encoder"], strict=True)
    enc.eval()

    coords = np.load(DATA_PROCESSED / "coords.npy").astype(np.float32)
    polars = np.load(DATA_PROCESSED / "polars.npy").astype(np.float32)
    i = args.airfoil_index
    c0 = coords[i : i + 1].copy()
    x = c0[0, :, 0]
    y = c0[0, :, 1].copy()
    # thickness: scale y about 0
    y_th = y * (1.0 + args.thickness_pct)
    # camber: add bump along chord
    xc = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_cb = y + args.camber_pct * np.sin(np.pi * xc)

    def pack(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        a = np.stack([cx, cy], axis=-1).astype(np.float32)[None, ...]
        trip = extract_triplets(a)
        return encode_geom(enc, trip, device)

    g_base = pack(c0[0, :, 0], c0[0, :, 1])
    g_th = pack(c0[0, :, 0], y_th)
    g_cb = pack(c0[0, :, 0], y_cb)

    geom_ckpt = Path(args.geom_checkpoint)
    if not geom_ckpt.is_file():
        geom_ckpt = _REPO / args.geom_checkpoint
    model = load_geom_transformer(geom_ckpt, device)

    ncols = polars.shape[1]
    rows = []
    for j in range(ncols):
        if np.isfinite(polars[i, j, 1]):
            rows.append((float(polars[i, j, 1]), float(polars[i, j, 2]), float(polars[i, j, 0])))
    arr = np.asarray(rows, dtype=np.float32)
    s = arr.shape[0]
    y_seq = np.zeros((1, s, 3), dtype=np.float32)
    y_seq[0, :s, :] = arr
    yt = torch.from_numpy(y_seq).to(device)

    def run(g: np.ndarray):
        g_t = torch.from_numpy(g).float().to(device)
        with torch.no_grad():
            pred = model.decode_append(g_t, s, aoa_ground_truth=yt[:, :, 2])
        return pred.cpu().numpy()[0]

    p0 = run(g_base)
    p_th = run(g_th)
    p_cb = run(g_cb)
    aoa = arr[:, 2]

    FIGURES.mkdir(parents=True, exist_ok=True)
    out_dir = FIGURES / "geometry_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        axs[0].plot(aoa, arr[:, 0], "k-", label="GT Cl")
        axs[0].plot(aoa, p0[:, 0], "b--", label="base")
        axs[0].plot(aoa, p_th[:, 0], "r:", label=f"th+{args.thickness_pct}")
        axs[0].plot(aoa, p_cb[:, 0], "g:", label=f"cam+{args.camber_pct}")
        axs[0].set_xlabel("AoA")
        axs[0].set_ylabel("Cl")
        axs[0].legend(fontsize=8)
        axs[1].plot(aoa, arr[:, 1], "k-", label="GT Cd")
        axs[1].plot(aoa, p0[:, 1], "b--", label="base")
        axs[1].plot(aoa, p_th[:, 1], "r:", label="thick")
        axs[1].plot(aoa, p_cb[:, 1], "g:", label="camber")
        axs[1].set_xlabel("AoA")
        axs[1].set_ylabel("Cd")
        fig.tight_layout()
        fig.savefig(out_dir / "sensitivity_cl_cd.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"Wrote {out_dir / 'sensitivity_cl_cd.png'}")


if __name__ == "__main__":
    main()
