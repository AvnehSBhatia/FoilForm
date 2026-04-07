#!/usr/bin/env python3
"""Scatter params vs val MAE (e.g. n_layers sweep); reads studies/runs/*/summary.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_STUDIES = Path(__file__).resolve().parent.parent
_REPO = _STUDIES.parent
sys.path.insert(0, str(_STUDIES / "src"))

from foilform.paths import FIGURES, RUNS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob-subdir", type=str, default="geom_polar_*")
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(RUNS.glob(f"{args.glob_subdir}/summary.json")):
        with summary_path.open() as f:
            s = json.load(f)
        args_p = (summary_path.parent / "args.json")
        ap = {}
        if args_p.is_file():
            ap = json.loads(args_p.read_text())
        rows.append(
            {
                "run": summary_path.parent.name,
                "n_params": s.get("n_params", ap.get("n_params")),
                "best_val_mse": s.get("best_val_mse"),
                "val_mae_cl": None,
                "val_mae_cd": None,
            }
        )
        hist = summary_path.parent / "history.json"
        if hist.is_file():
            h = json.loads(hist.read_text())
            be = s.get("best_epoch", -1)
            for r in h:
                if r.get("epoch") == be:
                    rows[-1]["val_mae_cl"] = r.get("val_mae_cl")
                    rows[-1]["val_mae_cd"] = r.get("val_mae_cd")
                    break

    out_json = FIGURES / "pareto_n_layers_data.json"
    FIGURES.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["n_params"] for r in rows if r["n_params"] is not None]
        ys = [r["best_val_mse"] for r in rows if r["best_val_mse"] is not None]
        if len(xs) == len(ys) and xs:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(xs, ys, c="steelblue")
            ax.set_xlabel("parameters")
            ax.set_ylabel("best val balanced MSE")
            ax.set_title("Pareto: params vs val error")
            fig.tight_layout()
            fig.savefig(FIGURES / "pareto_params_vs_val_mse.png", dpi=160)
            plt.close()
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
