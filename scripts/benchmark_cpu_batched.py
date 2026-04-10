#!/usr/bin/env python3
"""Throughput comparison: batched GeomPolarTransformer + PolarCorrectionMLP vs NeuralFoil xxxlarge.

The PyTorch pipeline runs on the chosen device (default **mps**). NeuralFoil is NumPy-based and
always runs on **CPU**; only the batched NN core is timed there (Kulfan batching, N_cases=B×9).

Requires: torch, neuralfoil, aerosandbox.
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
from foilform.checkpoints import resolve_geom_polar_transformer, resolve_polar_correction  # noqa: E402
from foilform.geom_polar_transformer import GeomPolarTransformer  # noqa: E402
from foilform.paths import DATA_PROCESSED  # noqa: E402
from foilform.polar_correction_mlp import N_SLOTS, POLAR_DIM, PolarCorrectionMLP  # noqa: E402


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


def infer_transformer_n_layers(state_dict: dict) -> int:
    mx = -1
    for k in state_dict:
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                mx = max(mx, int(parts[1]))
    return mx + 1 if mx >= 0 else 8


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


def airfoil_to_kulfan_rows(
    coordinates: np.ndarray,
) -> tuple[dict[str, np.ndarray], float, float]:
    """Return kulfan dict with scalar entries and normalization for alpha/Re."""
    import aerosandbox as asb

    af_dict = asb.Airfoil(coordinates=coordinates).normalize(return_dict=True)
    norm_af = af_dict["airfoil"].to_kulfan_airfoil(n_weights_per_side=8, normalize_coordinates=False)
    kp = norm_af.kulfan_parameters
    da = float(af_dict["rotation_angle"])
    scale = float(af_dict["scale_factor"])
    return kp, da, scale


def build_batched_kulfan_inputs(
    coords_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    Re: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Stack B airfoils × 9 AoA into one vectorized NeuralFoil case (N_cases = B*9)."""
    B = coords_batch.shape[0]
    n_aoa = 9

    upper_lists: list[list[float]] = [[] for _ in range(8)]
    lower_lists: list[list[float]] = [[] for _ in range(8)]
    le_list: list[float] = []
    te_list: list[float] = []
    alpha_list: list[float] = []
    re_list: list[float] = []

    for b in range(B):
        kp, da, scale = airfoil_to_kulfan_rows(coords_batch[b])
        aoa = y_batch[b, :n_aoa, 2].astype(np.float64)
        for t in range(n_aoa):
            for i in range(8):
                upper_lists[i].append(float(kp["upper_weights"][i]))
                lower_lists[i].append(float(kp["lower_weights"][i]))
            le_list.append(float(kp["leading_edge_weight"]))
            te_list.append(float(kp["TE_thickness"]))
            alpha_list.append(float(aoa[t] + da))
            re_list.append(Re / scale)

    kulfan_parameters = {
        "upper_weights": [np.asarray(upper_lists[i], dtype=np.float64) for i in range(8)],
        "lower_weights": [np.asarray(lower_lists[i], dtype=np.float64) for i in range(8)],
        "leading_edge_weight": np.asarray(le_list, dtype=np.float64),
        "TE_thickness": np.asarray(te_list, dtype=np.float64),
    }
    alpha_arr = np.asarray(alpha_list, dtype=np.float64)
    re_arr = np.asarray(re_list, dtype=np.float64)
    return kulfan_parameters, alpha_arr, re_arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Batched throughput: ours (PyTorch) vs NeuralFoil xxxlarge")
    parser.add_argument("--device", type=str, default="mps", help="PyTorch device: mps, cuda, or cpu.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Airfoils per timed batch (same count for both).")
    parser.add_argument("--batches", type=int, default=1, help="Number of batches to time (after warmup).")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--Re", type=float, default=1e5)
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only_nine",
        action="store_true",
        default=True,
        help="Use only airfoils with 9 valid AoA (required for B×9 NeuralFoil batching).",
    )
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore")

    import neuralfoil as nf

    polars = np.load(DATA_PROCESSED / "polars.npy").astype(np.float32)
    geom_emb = np.load(DATA_PROCESSED / "geom_embeddings.npy").astype(np.float32)
    coords = np.load(DATA_PROCESSED / "coords.npy").astype(np.float32)
    y, mask = build_targets(polars)
    n = geom_emb.shape[0]
    s_steps = y.shape[1]

    perm = np.random.default_rng(args.seed).permutation(n)
    n_train = max(1, min(int(round(args.train_frac * n)), n - 1))
    val_idx = np.sort(perm[n_train:])

    if args.only_nine:
        nine_mask = np.array([int(np.isfinite(polars[i, :, 1]).sum()) == 9 for i in range(n)])
        val_idx = np.array([i for i in val_idx if nine_mask[i]], dtype=np.int64)
    col_indices = [
        [j for j in range(polars.shape[1]) if np.isfinite(polars[i, j, 1])] for i in range(n)
    ]

    need = args.batch_size * args.batches
    if val_idx.size == 0:
        raise RuntimeError("No validation airfoils with 9 AoA points; check --train_frac / data.")
    # Repeat val indices so one batch can exceed the pool size (e.g. 1024 with only ~561 unique val).
    rep = int(np.ceil(need / val_idx.size))
    bench_idx = np.tile(val_idx, rep)[:need]

    device = resolve_device(args.device)
    geom_ckpt = resolve_geom_polar_transformer()
    corr_ckpt = resolve_polar_correction()
    if geom_ckpt is None or corr_ckpt is None:
        raise FileNotFoundError(
            "Missing transformer or correction weights: add models/*.pt or train under runs/"
        )

    sd = torch.load(geom_ckpt, map_location=device, weights_only=False)
    nl = infer_transformer_n_layers(sd["model"])
    t_model = GeomPolarTransformer(n_layers=nl).to(device)
    t_model.load_state_dict(sd["model"], strict=True)
    t_model.eval()

    corr_model = PolarCorrectionMLP().to(device)
    corr_model.load_state_dict(
        torch.load(corr_ckpt, map_location=device, weights_only=False)["model"], strict=True
    )
    corr_model.eval()

    def run_ours_batch(idx_arr: np.ndarray) -> None:
        g = torch.from_numpy(geom_emb[idx_arr]).to(device)
        yt = torch.from_numpy(y[idx_arr]).to(device)
        with torch.no_grad():
            pred = t_model.decode_append(g, s_steps, aoa_ground_truth=yt[:, :, 2])
            pred_np = pred.detach().cpu().numpy()
            js_batch = [col_indices[i] for i in idx_arr]
            base34 = pred_to_polar34_batch(pred_np, js_batch)
            gy = torch.from_numpy(coords[idx_arr].astype(np.float32)).to(device)
            b34 = torch.from_numpy(base34).float().to(device)
            corr_model.predict(gy, b34)
        device_sync(device)

    def run_nf_batch(idx_arr: np.ndarray) -> None:
        c_b = coords[idx_arr]
        y_b = y[idx_arr]
        kp_in, alpha_arr, re_arr = build_batched_kulfan_inputs(c_b, y_b, Re=args.Re)
        nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=kp_in,
            alpha=alpha_arr,
            Re=re_arr,
            model_size="xxxlarge",
        )

    # Warmup
    for _ in range(args.warmup):
        run_ours_batch(bench_idx[: args.batch_size])
        run_nf_batch(bench_idx[: args.batch_size])

    # Time ours (PyTorch on device)
    device_sync(device)
    t0 = time.perf_counter()
    for bi in range(args.batches):
        sl = slice(bi * args.batch_size, (bi + 1) * args.batch_size)
        run_ours_batch(bench_idx[sl])
    t_ours = time.perf_counter() - t0

    # Time NeuralFoil (NumPy CPU, batched kulfan)
    t0 = time.perf_counter()
    for bi in range(args.batches):
        sl = slice(bi * args.batch_size, (bi + 1) * args.batch_size)
        run_nf_batch(bench_idx[sl])
    t_nf = time.perf_counter() - t0

    n_airfoils = args.batch_size * args.batches
    print(
        f"Batched benchmark  |  PyTorch device={device}  |  batch_size={args.batch_size}  |  batches={args.batches}"
    )
    print(f"Airfoils timed: {n_airfoils} (val, 9 AoA each)  |  Re={args.Re:g}")
    print(f"Transformer checkpoint: {geom_ckpt.parent.name} (n_layers={nl})")
    print("Note: NeuralFoil runs on CPU (NumPy); PyTorch pipeline uses --device.")
    print()
    print(f"{'Pipeline':<52} {'total_s':>10} {'ms/airfoil':>12} {'airfoils/s':>12}")
    print("-" * 90)
    ours_label = f"GeomPolarTransformer + MLP ({device.type.upper()} batched)"
    for name, tsec in [
        (ours_label, t_ours),
        ("NeuralFoil xxxlarge (CPU NumPy, N_cases=B×9)", t_nf),
    ]:
        print(
            f"{name:<52} {tsec:10.3f} {tsec / n_airfoils * 1000:12.3f} {n_airfoils / tsec:12.1f}"
        )
    print()
    if t_ours < t_nf:
        print(f"PyTorch pipeline is {t_nf / t_ours:.2f}x faster than NeuralFoil (wall time).")
    else:
        print(f"NeuralFoil is {t_ours / t_nf:.2f}x faster than PyTorch pipeline (wall time).")


if __name__ == "__main__":
    main()
