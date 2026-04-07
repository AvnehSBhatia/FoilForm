"""Load GeomPolarTransformer from a studies or legacy checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from foilform.geom_polar_transformer import GeomPolarTransformer


def infer_transformer_n_layers(state_dict: dict) -> int:
    mx = -1
    for k in state_dict:
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                mx = max(mx, int(parts[1]))
    return mx + 1 if mx >= 0 else 4


def load_geom_transformer(
    ckpt_path: Path,
    device: torch.device,
) -> GeomPolarTransformer:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["model"]
    args = state.get("args") or {}
    nl = int(state.get("n_layers", args.get("n_layers", infer_transformer_n_layers(sd))))
    bt = str(state.get("block_type", args.get("block_type", "pairwise"))).lower()
    drop = float(state.get("dropout", args.get("dropout", 0.05)))
    model = GeomPolarTransformer(n_layers=nl, dropout=drop, block_type=bt).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model
