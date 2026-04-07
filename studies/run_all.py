#!/usr/bin/env python3
"""Run full publication studies suite (sequential, multi-day). Checkpoints under studies/runs/."""

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
    # --- Phase A: transformer ablations & sweeps (named runs) ---
    runs_geom = [
        ("ablation_pairwise", ["--block-type", "pairwise", "--n-layers", "4", "--experiment-id", "ablation_pairwise"]),
        ("ablation_standard_mlp", ["--block-type", "standard_mlp", "--n-layers", "4", "--experiment-id", "ablation_standard_mlp"]),
        ("ablation_no_mlp", ["--block-type", "no_mlp", "--n-layers", "4", "--experiment-id", "ablation_no_mlp"]),
        ("ablation_attention_only", ["--block-type", "attention_only", "--n-layers", "4", "--experiment-id", "ablation_attention_only"]),
        ("ablation_no_pairwise", ["--block-type", "no_pairwise", "--n-layers", "4", "--experiment-id", "ablation_no_pairwise"]),
    ]
    for name, extra in runs_geom:
        run(
            [
                str(SCRIPTS / "train_geom_polar_transformer.py"),
                "--run-name",
                name,
                *extra,
            ]
        )

    for nl in ("1", "2", "4", "8", "16"):
        run(
            [
                str(SCRIPTS / "train_geom_polar_transformer.py"),
                "--run-name",
                f"n_layers_{nl}",
                "--n-layers",
                nl,
                "--block-type",
                "pairwise",
                "--experiment-id",
                f"n_layers_{nl}",
            ]
        )

    for dr in ("0.0", "0.05", "0.1", "0.15"):
        run(
            [
                str(SCRIPTS / "train_geom_polar_transformer.py"),
                "--run-name",
                f"dropout_{dr.replace('.', '_')}",
                "--dropout",
                dr,
                "--experiment-id",
                f"dropout_{dr}",
            ]
        )

    for tf in ("0.4", "0.5", "0.6", "0.7", "0.8"):
        run(
            [
                str(SCRIPTS / "train_geom_polar_transformer.py"),
                "--run-name",
                f"train_frac_{tf.replace('.', '_')}",
                "--train_frac",
                tf,
                "--experiment-id",
                f"train_frac_{tf}",
            ]
        )

    run(
        [
            str(SCRIPTS / "train_geom_polar_transformer.py"),
            "--run-name",
            "polar_order_reverse",
            "--reverse-aoa-order",
            "--experiment-id",
            "polar_order_reverse",
        ]
    )

    # --- Phase B: correction MLP on baseline pairwise backbone ---
    pair_ckpt = _STUDIES / "runs" / "geom_polar_ablation_pairwise" / "best_geom_polar_transformer.pt"
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

    # --- Phase C: eval + metrics JSON ---
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

    # --- Phase D: analysis figures ---
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

    print("\nAll studies steps finished. See studies/runs, studies/figures, studies/results_manifest.jsonl\n")


if __name__ == "__main__":
    main()
