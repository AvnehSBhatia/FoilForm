#!/usr/bin/env python3
"""Plot train vs val balanced MSE for every history.csv under studies/runs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_STUDIES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_STUDIES / "src"))

from foilform.paths import FIGURES, RUNS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, default="**/history.csv")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out_root = FIGURES / "training_curves"
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable: {e}")
        return

    for hist_path in sorted(RUNS.glob(args.glob)):
        rows = []
        with hist_path.open() as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                rows.append(row)
        if not rows:
            continue
        ep = [int(r["epoch"]) for r in rows]
        tr = [float(r["train_bal_mse"]) for r in rows]
        va = [float(r["val_bal_mse"]) for r in rows if r.get("val_bal_mse") not in ("", "nan")]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ep, tr, label="train_bal_mse")
        if len(va) == len(ep):
            ax.plot(ep, va, label="val_bal_mse")
        ax.set_xlabel("epoch")
        ax.set_ylabel("balanced MSE")
        ax.legend()
        ax.set_title(hist_path.parent.name)
        fig.tight_layout()
        safe = hist_path.parent.name.replace("/", "_") + ".png"
        fig.savefig(out_root / safe, dpi=140)
        plt.close(fig)

    print(f"Saved plots under {out_root}")


if __name__ == "__main__":
    main()
