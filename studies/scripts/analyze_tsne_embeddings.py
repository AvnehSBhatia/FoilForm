#!/usr/bin/env python3
"""PCA/t-SNE on 167×8 geometry tokens for sampled airfoils."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_STUDIES = Path(__file__).resolve().parent.parent
_REPO = _STUDIES.parent
sys.path.insert(0, str(_STUDIES / "src"))

from foilform.paths import DATA_PROCESSED, FIGURES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-airfoils", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    geom = np.load(DATA_PROCESSED / "geom_embeddings.npy").astype(np.float32)
    n = min(args.n_airfoils, geom.shape[0])
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(geom.shape[0], size=n, replace=False)
    X = geom[idx].reshape(-1, 8)
    labels = np.repeat(np.arange(n), geom.shape[1])

    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        Z = TSNE(n_components=2, perplexity=30, random_state=args.seed).fit_transform(X)
    except Exception:
        from sklearn.decomposition import PCA

        Z = PCA(n_components=2).fit_transform(X)

    FIGURES.mkdir(parents=True, exist_ok=True)
    out_dir = FIGURES / "tsne_geom_tokens"
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=8, alpha=0.6)
    ax.set_title("Geometry tokens (167×8) 2D embedding")
    fig.colorbar(sc, ax=ax, label="airfoil sample id")
    fig.tight_layout()
    fig.savefig(out_dir / "tokens_2d.png", dpi=160)
    plt.close()
    print(f"Wrote {out_dir / 'tokens_2d.png'}")


if __name__ == "__main__":
    main()
