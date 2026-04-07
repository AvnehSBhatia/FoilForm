#!/usr/bin/env python3
"""Plot val MAE (from history at best epoch) vs N_LAYERS for n_layers_* runs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_STUDIES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_STUDIES / "src"))

from foilform.paths import FIGURES, RUNS  # noqa: E402


def main() -> None:
    points = []
    for d in sorted(RUNS.glob("geom_polar_n_layers_*")):
        m = re.match(r"geom_polar_n_layers_(\d+)$", d.name)
        if not m:
            continue
        nl = int(m.group(1))
        summ = d / "summary.json"
        hist = d / "history.json"
        if not summ.is_file() or not hist.is_file():
            continue
        s = json.loads(summ.read_text())
        be = s.get("best_epoch", -1)
        h = json.loads(hist.read_text())
        row = next((r for r in h if r.get("epoch") == be), None)
        if not row:
            continue
        points.append(
            {
                "n_layers": nl,
                "val_mae_cl": row.get("val_mae_cl"),
                "val_mae_cd": row.get("val_mae_cd"),
                "best_val_mse": s.get("best_val_mse"),
            }
        )
    points.sort(key=lambda x: x["n_layers"])
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_json = FIGURES / "n_layers_depth_curve.json"
    out_json.write_text(json.dumps(points, indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [p["n_layers"] for p in points]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, [p["val_mae_cl"] for p in points], "o-", label="val MAE Cl")
        ax.plot(xs, [p["val_mae_cd"] for p in points], "s-", label="val MAE Cd")
        ax.set_xlabel("N_LAYERS")
        ax.set_ylabel("val MAE (raw)")
        ax.legend()
        ax.set_title("Depth sensitivity (pairwise block)")
        fig.tight_layout()
        fig.savefig(FIGURES / "n_layers_vs_val_mae.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"Plot skipped: {e}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
