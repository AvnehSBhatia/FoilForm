"""Resolve paths to trained weights.

Prefer ``models/<name>.pt`` (versioned defaults in git) over ``runs/*/best_*.pt``
or ``artifacts/*.pt`` from local training.
"""

from __future__ import annotations

from pathlib import Path

from foilform.paths import ARTIFACTS, MODELS, RUNS

# Filenames inside models/
GEOM_POLAR_TRANSFORMER = MODELS / "geom_polar_transformer.pt"
POLAR_CORRECTION = MODELS / "polar_correction.pt"
GEOM_TOKENIZER = MODELS / "geom_tokenizer.pt"
AERO_TOKENIZER = MODELS / "aero_tokenizer.pt"


def _latest_in_runs(glob_pat: str) -> Path | None:
    if not RUNS.is_dir():
        return None
    cands = list(RUNS.glob(glob_pat))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def resolve_geom_polar_transformer() -> Path | None:
    """Production GeomPolarTransformer; prefer ``models/geom_polar_transformer.pt``."""
    if GEOM_POLAR_TRANSFORMER.is_file():
        return GEOM_POLAR_TRANSFORMER
    return _latest_in_runs("*/best_geom_polar_transformer.pt")


def resolve_polar_correction() -> Path | None:
    """Polar correction MLP; prefer ``models/polar_correction.pt``."""
    if POLAR_CORRECTION.is_file():
        return POLAR_CORRECTION
    return _latest_in_runs("*/best_polar_correction.pt")


def resolve_geom_tokenizer() -> Path | None:
    """Geometry tokenizer; prefer ``models/geom_tokenizer.pt``, else ``artifacts/``."""
    if GEOM_TOKENIZER.is_file():
        return GEOM_TOKENIZER
    p = ARTIFACTS / "geom_tokenizer.pt"
    return p if p.is_file() else None


def resolve_aero_tokenizer() -> Path | None:
    """Aero tokenizer; prefer ``models/aero_tokenizer.pt``, else ``artifacts/``."""
    if AERO_TOKENIZER.is_file():
        return AERO_TOKENIZER
    p = ARTIFACTS / "aero_tokenizer.pt"
    return p if p.is_file() else None
