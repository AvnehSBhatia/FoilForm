"""Geometry-triplet and aerodynamic tokenizer autoencoders.

Each maps raw observations to a compact 8-D embedding that can be decoded
back.  These serve as the tokenizers for the autoregressive airfoil generator:

  Geometry: (x1,y1, x2,y2, x3,y3) ⟶ 8-D ⟶ (x1,y1, x2,y2, x3,y3)
  Aero:     (AoA, Cl, Cd)          ⟶ 8-D ⟶ (AoA, Cl, Cd)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── Geometry triplet tokenizer ─────────────────────────────────────────────────

class TripletEncoder(nn.Module):
    """3 consecutive contour points → 8-D geometry embedding."""

    def __init__(self, d_embed: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, d_embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6) → (B, d_embed)"""
        return self.net(x)


class TripletDecoder(nn.Module):
    """8-D geometry embedding → 3 contour points."""

    def __init__(self, d_embed: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_embed) → (B, 6)"""
        return self.net(z)


# ── Aerodynamic tokenizer ─────────────────────────────────────────────────────

class AeroEncoder(nn.Module):
    """(AoA°, Cl, Cd) → 8-D aero embedding.

    AoA is Fourier-encoded (sin/cos at k=1..4) giving 8 periodic features.
    Input to the MLP is [Fourier(AoA) ‖ Cl ‖ Cd] = 10-D.
    """

    def __init__(self, d_embed: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, d_embed),
        )

    @staticmethod
    def aoa_fourier(aoa_deg: torch.Tensor) -> torch.Tensor:
        theta = aoa_deg * (math.pi / 180.0)
        parts = []
        for k in range(1, 5):
            parts.append(torch.sin(k * theta))
            parts.append(torch.cos(k * theta))
        return torch.stack(parts, dim=-1)

    def forward(self, polar: torch.Tensor) -> torch.Tensor:
        """polar: (B, 3) → (B, d_embed)"""
        fou = self.aoa_fourier(polar[:, 0])
        x = torch.cat([fou, polar[:, 1:3]], dim=-1)
        return self.net(x)


class AeroDecoder(nn.Module):
    """8-D aero embedding → (AoA°, Cl, Cd)."""

    def __init__(self, d_embed: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_embed) → (B, 3)"""
        return self.net(z)
