"""Polar correction MLP: conv geometry encoder + base polar → Δ correction.

Copy of repo src/foilform/polar_correction_mlp.py for studies imports (studies/src shadows repo src).
"""

from __future__ import annotations

import torch
import torch.nn as nn

GEOM_STATIONS = 501
GEOM_CHANNELS = 2   # x and y per station
N_SLOTS = 17
POLAR_DIM = 2 * N_SLOTS  # 34

EXPECTED_PARAMETER_COUNT = 368 + 2_592 + 20_864 + 8_256 + 2_210


class PolarCorrectionMLP(nn.Module):
    """
    1D-conv encoder for 501 (x, y) stations, merged with base polar, outputs Δ.

    Corrected polar = ``polar_in + forward(geom, polar_in)`` (see ``predict``).
    Output layer initialized near zero so the network starts at identity correction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(GEOM_CHANNELS, 16, kernel_size=11, stride=5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=3)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc1 = nn.Linear(32 * 4 + POLAR_DIM, 128)   # 162 → 128
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, POLAR_DIM)
        nn.init.uniform_(self.fc3.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, geom: torch.Tensor, polar_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geom: (B, 501, 2) x and y at each station.
            polar_in: (B, 34) base polar (Cl slots then Cd slots).

        Returns:
            Delta (B, 34) to add to ``polar_in``.
        """
        x = geom.transpose(1, 2)                      # (B, 2, 501)
        x = torch.tanh(self.conv1(x))                 # (B, 16, 99)
        x = torch.tanh(self.conv2(x))                 # (B, 32, 32)
        x = self.pool(x)                              # (B, 32, 4)
        x = x.flatten(1)                              # (B, 128)
        x = torch.cat([x, polar_in], dim=-1)          # (B, 162)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

    def predict(self, geom: torch.Tensor, polar_in: torch.Tensor) -> torch.Tensor:
        """``polar_in + forward(geom, polar_in)``."""
        return polar_in + self.forward(geom, polar_in)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def split_cl_cd(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split (..., 34) into Cl (..., 17) and Cd (..., 17)."""
    return y[..., :N_SLOTS], y[..., N_SLOTS:]
