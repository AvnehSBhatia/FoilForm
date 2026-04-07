#!/usr/bin/env python3
"""Visualize geometry and aero tokenizer embedding spaces (PCA + optional UMAP).

Reads:
  geom_embeddings.npy  – (N_airfoils, 167, 8)  (or recomputes from checkpoint)
  aero_embeddings.npy  – (N_airfoils, 9, 8) with NaN for missing polars
  coords.npy, polars.npy – for colouring points by physical quantities

Examples:
  python scripts/visualize_embeddings.py
  python scripts/visualize_embeddings.py --output_dir figures --max_points 15000
  python scripts/visualize_embeddings.py --no_umap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
from foilform.paths import DATA_PROCESSED, FIGURES, ensure_dirs  # noqa: E402

N_PATCHES = 167
PATCH_PTS = 3


def extract_triplets(coords: np.ndarray) -> np.ndarray:
    """coords (N, 501, 2) → triplets (N, 167, 6)."""
    n = coords.shape[0]
    out = np.zeros((n, N_PATCHES, 6), dtype=np.float32)
    for k in range(N_PATCHES):
        i0 = k * PATCH_PTS
        out[:, k, 0:2] = coords[:, i0, :]
        out[:, k, 2:4] = coords[:, i0 + 1, :]
        out[:, k, 4:6] = coords[:, i0 + 2, :]
    return out


def _pca_2d(x: np.ndarray, seed: int) -> np.ndarray:
    """x (n, d) → (n, 2) via centered PCA (SVD)."""
    rng = np.random.default_rng(seed)
    n, d = x.shape
    if n < 3:
        return np.zeros((n, 2), dtype=np.float64)
    x64 = x.astype(np.float64)
    x64 = x64 - x64.mean(axis=0, keepdims=True)
    # Randomized subsample for covariance if huge
    if n > 50_000:
        idx = rng.choice(n, 50_000, replace=False)
        xc = x64[idx]
    else:
        xc = x64
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:2].T  # (d, 2)
    return x64 @ basis


def _maybe_umap(x: np.ndarray, random_state: int, n_neighbors: int, min_dist: float) -> np.ndarray | None:
    try:
        import umap
    except ImportError:
        return None
    n = x.shape[0]
    nn = min(n_neighbors, max(5, n // 40))
    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=nn,
        min_dist=min_dist,
    )
    return reducer.fit_transform(x.astype(np.float64))


def _scatter_panel(
    fig,
    axes,
    xy: np.ndarray,
    c0,
    c1,
    c2,
    titles: tuple[str, str, str],
    suptitle: str,
) -> None:
    import matplotlib.pyplot as plt

    sc0 = axes[0].scatter(xy[:, 0], xy[:, 1], c=c0, s=2, cmap="viridis", alpha=0.65, linewidths=0)
    plt.colorbar(sc0, ax=axes[0])
    axes[0].set_title(titles[0])
    sc1 = axes[1].scatter(xy[:, 0], xy[:, 1], c=c1, s=2, cmap="coolwarm", alpha=0.65, linewidths=0)
    plt.colorbar(sc1, ax=axes[1])
    axes[1].set_title(titles[1])
    sc2 = axes[2].scatter(xy[:, 0], xy[:, 1], c=c2, s=2, cmap="hsv", alpha=0.65, linewidths=0)
    plt.colorbar(sc2, ax=axes[2])
    axes[2].set_title(titles[2])
    for ax in axes:
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(DATA_PROCESSED))
    parser.add_argument("--output_dir", type=str, default=str(FIGURES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_points", type=int, default=25_000, help="Subsample per modality for speed.")
    parser.add_argument("--no_umap", action="store_true", help="PCA only (no umap-learn).")
    args = parser.parse_args()

    ensure_dirs()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geom_path = data_dir / "geom_embeddings.npy"
    aero_path = data_dir / "aero_embeddings.npy"
    coords_path = data_dir / "coords.npy"
    polars_path = data_dir / "polars.npy"

    if not geom_path.is_file() or not aero_path.is_file():
        raise FileNotFoundError(
            f"Need {geom_path.name} and {aero_path.name}. Run scripts/train_tokenizers.py first."
        )

    geom_emb = np.load(geom_path)  # (N, 167, 8)
    aero_emb = np.load(aero_path)  # (N, 9, 8)
    coords = np.load(coords_path)
    polars = np.load(polars_path)
    n = coords.shape[0]

    triplets = extract_triplets(coords)
    geom_flat = geom_emb.reshape(-1, geom_emb.shape[-1])
    trip_flat = triplets.reshape(-1, 6)
    patch_ids = np.tile(np.arange(N_PATCHES), n)

    rng = np.random.default_rng(args.seed)
    ng = geom_flat.shape[0]
    sub_g = min(args.max_points, ng)
    ig = rng.choice(ng, sub_g, replace=False)
    g_emb = geom_flat[ig]
    g_x1 = trip_flat[ig, 0]
    g_y1 = trip_flat[ig, 1]
    g_patch = patch_ids[ig]

    # Aero: valid cells only
    valid = np.isfinite(aero_emb[:, :, 0])
    ai, aj = np.where(valid)
    a_flat = aero_emb[valid]
    a_aoa = polars[ai, aj, 0]
    a_cl = polars[ai, aj, 1]
    a_cd = polars[ai, aj, 2]
    na = a_flat.shape[0]
    sub_a = min(args.max_points, na)
    ia = rng.choice(na, sub_a, replace=False)
    a_emb = a_flat[ia]
    a_aoa_s = a_aoa[ia]
    a_cl_s = a_cl[ia]
    a_cd_s = a_cd[ia]

    def run_and_save(name: str, z: np.ndarray, colors: tuple, titles: tuple[str, str, str], suptitle_base: str) -> None:
        pca_xy = _pca_2d(z, args.seed)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        _scatter_panel(
            fig,
            axes,
            pca_xy,
            colors[0],
            colors[1],
            colors[2],
            titles,
            f"{suptitle_base} — PCA-2D",
        )
        fig.savefig(out_dir / f"{name}_pca.png", dpi=160)
        plt.close(fig)

        if args.no_umap:
            return
        umap_xy = _maybe_umap(z, args.seed, n_neighbors=30, min_dist=0.1)
        if umap_xy is None:
            print("umap-learn not installed — skipped UMAP for", name)
            return
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        _scatter_panel(
            fig,
            axes,
            umap_xy,
            colors[0],
            colors[1],
            colors[2],
            titles,
            f"{suptitle_base} — UMAP-2D",
        )
        fig.savefig(out_dir / f"{name}_umap.png", dpi=160)
        plt.close(fig)

    run_and_save(
        "embed_geom",
        g_emb,
        (g_x1, g_y1, g_patch),
        ("x₁ (chord)", "y₁ (height)", "Patch index along contour"),
        "Geometry triplet embeddings (8-D)",
    )
    print(f"Saved {out_dir / 'embed_geom_pca.png'}")
    if not args.no_umap:
        print(f"Saved {out_dir / 'embed_geom_umap.png'} (if umap installed)")

    run_and_save(
        "embed_aero",
        a_emb,
        (a_aoa_s, a_cl_s, np.log10(a_cd_s + 1e-6)),
        ("AoA (°)", "Cl", "log₁₀(Cd)"),
        "Aero embeddings (8-D)",
    )
    print(f"Saved {out_dir / 'embed_aero_pca.png'}")
    if not args.no_umap:
        print(f"Saved {out_dir / 'embed_aero_umap.png'} (if umap installed)")

    # Summary: 8-D marginal histograms (geom)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for d in range(8):
        ax = axes[d // 4, d % 4]
        ax.hist(geom_flat[:, d], bins=60, color="#334155", alpha=0.85)
        ax.set_title(f"geom z[{d}]")
        ax.set_yticks([])
    fig.suptitle("Geometry embedding — per-dimension histograms (all triplets)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "embed_geom_dim_hist.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for d in range(8):
        ax = axes[d // 4, d % 4]
        ax.hist(a_flat[:, d], bins=60, color="#b45309", alpha=0.85)
        ax.set_title(f"aero z[{d}]")
        ax.set_yticks([])
    fig.suptitle("Aero embedding — per-dimension histograms (valid polars)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "embed_aero_dim_hist.png", dpi=140)
    plt.close(fig)
    print(f"Saved {out_dir / 'embed_geom_dim_hist.png'} and embed_aero_dim_hist.png")


if __name__ == "__main__":
    main()
