"""Repository layout: data/processed, artifacts, raw, runs, figures."""

from pathlib import Path

# src/foilform/paths.py → repo root is two levels up
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS = REPO_ROOT / "artifacts"
RAW = REPO_ROOT / "raw"
RUNS = REPO_ROOT / "runs"
FIGURES = REPO_ROOT / "figures"
# Canonical shipped weights (tracked in git). See foilform.checkpoints.resolve_*.
MODELS = REPO_ROOT / "models"


def ensure_dirs() -> None:
    for p in (DATA_PROCESSED, ARTIFACTS, RAW, RUNS, FIGURES, MODELS):
        p.mkdir(parents=True, exist_ok=True)
