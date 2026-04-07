"""Studies layout: artifacts under repo/studies/, data still at repo/data/processed."""

from pathlib import Path

# studies/src/foilform/paths.py → repo root is four levels up
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS = REPO_ROOT / "artifacts"
RAW = REPO_ROOT / "raw"
RUNS = REPO_ROOT / "studies" / "runs"
FIGURES = REPO_ROOT / "studies" / "figures"
STUDIES_ROOT = REPO_ROOT / "studies"


def ensure_dirs() -> None:
    STUDIES_ROOT.mkdir(parents=True, exist_ok=True)
    for p in (DATA_PROCESSED, ARTIFACTS, RAW, RUNS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
