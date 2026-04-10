#!/usr/bin/env python3
"""Plot Cl & Cd vs AoA: ground truth, transformer pred, correction MLP, and NeuralFoil xxxlarge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
from foilform.checkpoints import resolve_geom_polar_transformer, resolve_polar_correction  # noqa: E402
from foilform.geom_polar_transformer import GeomPolarTransformer, N_LAYERS  # noqa: E402
from foilform.paths import DATA_PROCESSED, FIGURES, ensure_dirs  # noqa: E402
from foilform.polar_correction_mlp import GEOM_STATIONS, N_SLOTS, POLAR_DIM, PolarCorrectionMLP  # noqa: E402


def infer_transformer_n_layers(state_dict: dict) -> int | None:
    mx = -1
    for k in state_dict:
        if not k.startswith("blocks."):
            continue
        parts = k.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            mx = max(mx, int(parts[1]))
    return (mx + 1) if mx >= 0 else None


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


def column_indices_for_airfoil(polars_i: np.ndarray) -> list[int]:
    """Column j in polars order for each packed step t (matches ``build_targets``)."""
    js = []
    for j in range(polars_i.shape[0]):
        if np.isfinite(polars_i[j, 1]):
            js.append(j)
    return js


def pred_to_polar34(pred_cl_cd: np.ndarray, js: list[int]) -> np.ndarray:
    """Map (S,2) Cl/Cd in packed order to fixed (34,) layout [Cl×17, Cd×17]."""
    out = np.zeros(POLAR_DIM, dtype=np.float32)
    for t, j in enumerate(js):
        if j >= N_SLOTS:
            continue
        out[j] = float(pred_cl_cd[t, 0])
        out[N_SLOTS + j] = float(pred_cl_cd[t, 1])
    return out


def polar34_to_packed(cl_cd_34: np.ndarray, js: list[int], n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack (34,) to (n_steps,) Cl and Cd in packed step order."""
    cl = np.zeros(n_steps, dtype=np.float32)
    cd = np.zeros(n_steps, dtype=np.float32)
    for t, j in enumerate(js):
        if t >= n_steps or j >= N_SLOTS:
            break
        cl[t] = float(cl_cd_34[j])
        cd[t] = float(cl_cd_34[N_SLOTS + j])
    return cl, cd


def build_geom_xy(coords: np.ndarray) -> np.ndarray:
    if coords.shape[1:] != (GEOM_STATIONS, 2):
        raise ValueError(f"Expected coords (N, {GEOM_STATIONS}, 2), got {coords.shape}")
    return coords.astype(np.float32)


# UIUC / AeroSandbox names for ``--nf_coords`` (canonical repanel to GEOM_STATIONS).
NF_CANONICAL_UIUC = (
    "naca0012",
    "s1223",
)


def canonical_aerosandbox_coords(uiuc_name: str) -> np.ndarray:
    """NACA / UIUC name (e.g. ``naca0012``) repanelled to ``GEOM_STATIONS`` points for NeuralFoil.

    Uses AeroSandbox cosine spacing (``n_points_per_side`` such that ``2*n-1 == GEOM_STATIONS``).
    """
    import aerosandbox as asb

    nps = (GEOM_STATIONS + 1) // 2
    af = asb.Airfoil(uiuc_name).repanel(n_points_per_side=nps)
    # float64: NeuralFoil / Kulfan fit can fail (SVD) with float32 on some airfoils (e.g. S1223).
    xy = np.asarray(af.coordinates, dtype=np.float64)
    if xy.shape != (GEOM_STATIONS, 2):
        raise ValueError(f"Expected ({GEOM_STATIONS}, 2) from repanel, got {xy.shape}")
    return xy


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
    parser.add_argument(
        "--airfoil",
        type=int,
        default=-1,
        help="Global airfoil index (0..N-1). If -1, randomly pick from the val split.",
    )
    parser.add_argument("--geom_checkpoint", type=str, default="", help="Transformer checkpoint path.")
    parser.add_argument(
        "--corr_checkpoint",
        type=str,
        default="",
        help="Polar correction MLP checkpoint (best_polar_correction.pt).",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--Re", type=float, default=1e5, help="Reynolds number for NeuralFoil (default: 1e5).")
    parser.add_argument(
        "--nf_coords",
        type=str,
        choices=("dataset",) + NF_CANONICAL_UIUC,
        default="dataset",
        help=(
            "NeuralFoil geometry: ``dataset`` (same as GT), or canonical AeroSandbox airfoil "
            f"({', '.join(NF_CANONICAL_UIUC)}), repanelled to {GEOM_STATIONS} points."
        ),
    )
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG (default: figures/polar_corr_airfoil_<idx>.png).",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dirs()
    polars_path = DATA_PROCESSED / "polars.npy"
    geom_emb_path = DATA_PROCESSED / "geom_embeddings.npy"
    coords_path = DATA_PROCESSED / "coords.npy"
    for p in (polars_path, geom_emb_path, coords_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")

    polars = np.load(polars_path).astype(np.float32)
    geom = np.load(geom_emb_path).astype(np.float32)
    coords = np.load(coords_path).astype(np.float32)
    y, m = build_targets(polars)
    n = geom.shape[0]

    if args.airfoil >= 0:
        idx = int(args.airfoil) % n
    else:
        rng = np.random.default_rng(seed=None)
        tr_mask = np.zeros(n, dtype=bool)
        perm = np.random.default_rng(args.seed).permutation(n)
        n_train = max(1, min(int(round(args.train_frac * n)), n - 1))
        tr_mask[perm[:n_train]] = True
        val_indices = np.where(~tr_mask)[0]
        idx = int(rng.choice(val_indices))
        print(f"Randomly selected val airfoil index: {idx}")

    device = resolve_device(args.device)

    geom_ckpt: Path | None = None
    if args.geom_checkpoint:
        geom_ckpt = Path(args.geom_checkpoint)
        if not geom_ckpt.is_file():
            geom_ckpt = _REPO / args.geom_checkpoint
    else:
        geom_ckpt = resolve_geom_polar_transformer()
    t_model: GeomPolarTransformer | None = None
    if geom_ckpt is not None and geom_ckpt.is_file():
        state = torch.load(geom_ckpt, map_location=device, weights_only=False)
        sd = state["model"]
        n_l = infer_transformer_n_layers(sd)
        if n_l is None:
            n_l = N_LAYERS
        t_model = GeomPolarTransformer(n_layers=n_l).to(device)
        t_model.load_state_dict(sd, strict=True)
        print(f"Loaded transformer: {geom_ckpt} (n_layers={n_l})")
    else:
        t_model = GeomPolarTransformer().to(device)
        print("Warning: no geom transformer checkpoint — random weights.")

    corr_ckpt: Path | None = None
    if args.corr_checkpoint:
        corr_ckpt = Path(args.corr_checkpoint)
        if not corr_ckpt.is_file():
            corr_ckpt = _REPO / args.corr_checkpoint
    else:
        corr_ckpt = resolve_polar_correction()
    corr_model = PolarCorrectionMLP().to(device)
    if corr_ckpt is not None and corr_ckpt.is_file():
        state = torch.load(corr_ckpt, map_location=device, weights_only=False)
        corr_model.load_state_dict(state["model"], strict=True)
        print(f"Loaded polar correction: {corr_ckpt}")
    else:
        print("Warning: no polar correction checkpoint — random weights.")

    t_model.eval()
    corr_model.eval()

    g = torch.from_numpy(geom[idx : idx + 1]).to(device)
    yt = torch.from_numpy(y[idx : idx + 1]).to(device)
    pred_t = t_model.decode_append(
        geom_context=g,
        target_steps=y.shape[1],
        teacher_tuples=None,
        aoa_ground_truth=yt[:, :, 2],
    )
    pred_cl_cd = pred_t[0].cpu().numpy()

    js = column_indices_for_airfoil(polars[idx])
    n_valid = int(m[idx].sum())
    if len(js) != n_valid:
        raise RuntimeError(f"Column count mismatch: js={len(js)} vs mask sum={n_valid}")

    base34 = pred_to_polar34(pred_cl_cd[:n_valid], js)
    gy = torch.from_numpy(build_geom_xy(coords[idx : idx + 1])).float().to(device)
    b34 = torch.from_numpy(base34.reshape(1, -1)).float().to(device)
    corrected34 = corr_model.predict(gy, b34).cpu().numpy().reshape(-1)

    import neuralfoil as nf

    _nf_labels = {
        "naca0012": "AeroSandbox NACA0012 (canonical)",
        "s1223": "AeroSandbox S1223 (canonical)",
    }
    if args.nf_coords in NF_CANONICAL_UIUC:
        airfoil_coords = canonical_aerosandbox_coords(args.nf_coords)
        nf_geom_note = f"NF geom: {_nf_labels[args.nf_coords]}"
        print(
            f"NeuralFoil: canonical {args.nf_coords} coords "
            "(GT / transformer / corrected still use dataset airfoil geometry)."
        )
    else:
        airfoil_coords = coords[idx]  # (501, 2)
        nf_geom_note = "NF geom: dataset coords"
    mask = m[idx] > 0.5
    aoa = y[idx, :, 2]
    gt_cl = y[idx, :, 0]
    gt_cd = y[idx, :, 1]
    p_cl = pred_cl_cd[:, 0]
    p_cd = pred_cl_cd[:, 1]
    c_cl, c_cd = polar34_to_packed(corrected34, js, y.shape[1])

    aoa_valid = aoa[mask]
    nf_aero = nf.get_aero_from_coordinates(
        coordinates=np.asarray(airfoil_coords, dtype=np.float64),
        alpha=aoa_valid.astype(np.float64),
        Re=args.Re,
        model_size="xxxlarge",
    )
    nf_cl = np.asarray(nf_aero["CL"], dtype=np.float32)
    nf_cd = np.asarray(nf_aero["CD"], dtype=np.float32)

    order = np.argsort(aoa[mask])
    aoa_p = aoa[mask][order]
    gt_cl_p = gt_cl[mask][order]
    gt_cd_p = gt_cd[mask][order]
    p_cl_p = p_cl[mask][order]
    p_cd_p = p_cd[mask][order]
    c_cl_p = c_cl[mask][order]
    c_cd_p = c_cd[mask][order]
    nf_cl_p = nf_cl[order]
    nf_cd_p = nf_cd[order]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax0.plot(aoa_p, gt_cl_p, "o-", color="0.2", label="Ground truth", markersize=5)
    ax0.plot(aoa_p, p_cl_p, "s--", color="tab:blue", label="Predicted (transformer)", markersize=4)
    ax0.plot(aoa_p, c_cl_p, "^-", color="tab:green", label="Corrected (MLP)", markersize=4)
    ax0.plot(aoa_p, nf_cl_p, "d:", color="tab:orange", label="NeuralFoil xxxlarge", markersize=4)
    ax0.set_ylabel("Cl")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax0.set_title(
        f"Airfoil index {idx}  |  Re={args.Re:.0e}  |  device={device}  |  {nf_geom_note}"
    )

    ax1.plot(aoa_p, gt_cd_p, "o-", color="0.2", label="Ground truth", markersize=5)
    ax1.plot(aoa_p, p_cd_p, "s--", color="tab:blue", label="Predicted (transformer)", markersize=4)
    ax1.plot(aoa_p, c_cd_p, "^-", color="tab:green", label="Corrected (MLP)", markersize=4)
    ax1.plot(aoa_p, nf_cd_p, "d:", color="tab:orange", label="NeuralFoil xxxlarge", markersize=4)
    ax1.set_ylabel("Cd")
    ax1.set_xlabel("AoA (°)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)
    ax1.set_xlim(-8.5, 8.5)

    fig.tight_layout()
    out = Path(args.out) if args.out else FIGURES / f"polar_corr_airfoil_{idx}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
