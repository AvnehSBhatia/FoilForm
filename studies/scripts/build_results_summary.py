#!/usr/bin/env python3
"""Merge studies/results_manifest.jsonl into studies/results_summary.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_STUDIES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_STUDIES / "src"))

from foilform.manifest import manifest_path  # noqa: E402
from foilform.paths import STUDIES_ROOT  # noqa: E402


def main() -> None:
    path = manifest_path()
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    out = STUDIES_ROOT / "results_summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
