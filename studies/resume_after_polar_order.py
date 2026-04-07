#!/usr/bin/env python3
"""Continue studies after ``polar_order_reverse`` (manifest line ~20): corrector, eval, analysis, summary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_STUDIES = Path(__file__).resolve().parent
_REPO = _STUDIES.parent
PY = sys.executable
SCRIPTS = _STUDIES / "scripts"


def run(args: list[str]) -> None:
    cmd = [PY, *args]
    print("\n=== ", " ".join(cmd), " ===\n", flush=True)
    subprocess.check_call(cmd, cwd=str(_REPO))


def main() -> None:
    pair_ckpt = _STUDIES / "runs" / "geom_polar_ablation_pairwise" / "best_geom_polar_transformer.pt"
    if not pair_ckpt.is_file():
        raise FileNotFoundError(f"Missing {pair_ckpt}")

    run(
        [
            str(SCRIPTS / "train_polar_correction_mlp.py"),
            "--geom-checkpoint",
            str(pair_ckpt),
            "--run-name",
            "on_ablation_pairwise",
            "--experiment-id",
            "corr_on_ablation_pairwise",
            "--device",
            "cpu",
        ]
    )

    corr_ckpt = _STUDIES / "runs" / "polar_corr_on_ablation_pairwise" / "best_polar_correction.pt"
    run(
        [
            str(SCRIPTS / "eval_metrics.py"),
            "--geom-checkpoint",
            str(pair_ckpt),
            "--corr-checkpoint",
            str(corr_ckpt),
            "--output-json",
            str(_STUDIES / "figures" / "eval_main_val.json"),
            "--experiment-id",
            "eval_main_val",
            "--skip_nf",
        ]
    )

    run(
        [
            str(SCRIPTS / "analyze_val_distribution.py"),
            "--geom-checkpoint",
            str(pair_ckpt),
            "--corr-checkpoint",
            str(corr_ckpt),
        ]
    )
    run([str(SCRIPTS / "analyze_worst_best_airfoils.py"), "--geom-checkpoint", str(pair_ckpt)])
    run([str(SCRIPTS / "plot_pareto.py")])
    run([str(SCRIPTS / "plot_n_layers_depth.py")])
    run([str(SCRIPTS / "plot_training_curves.py")])
    run(
        [
            str(SCRIPTS / "benchmark_inference.py"),
            "--geom-checkpoint",
            str(pair_ckpt),
            "--output-json",
            str(_STUDIES / "figures" / "benchmark_inference.json"),
        ]
    )
    run(
        [
            str(SCRIPTS / "analyze_corrector_delta.py"),
            "--geom-checkpoint",
            str(pair_ckpt),
            "--corr-checkpoint",
            str(corr_ckpt),
        ]
    )
    run([str(SCRIPTS / "analyze_tsne_embeddings.py")])
    run(
        [
            str(SCRIPTS / "analyze_geometry_sensitivity.py"),
            "--geom-checkpoint",
            str(pair_ckpt),
        ]
    )
    run([str(SCRIPTS / "build_results_summary.py")])

    print("\nDone: corrector + eval + figures + results_summary.json\n")


if __name__ == "__main__":
    main()
