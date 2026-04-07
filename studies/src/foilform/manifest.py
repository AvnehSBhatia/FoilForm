"""Append one JSON line per experiment to studies/results_manifest.jsonl."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from foilform.paths import STUDIES_ROOT, ensure_dirs


def manifest_path() -> Path:
    return STUDIES_ROOT / "results_manifest.jsonl"


def append_manifest(row: dict[str, Any]) -> None:
    ensure_dirs()
    path = manifest_path()
    if "ts_unix" not in row:
        row = {**row, "ts_unix": time.time()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
