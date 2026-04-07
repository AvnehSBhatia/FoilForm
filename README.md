# FoilForm

**Airfoil geometry to aerodynamic polar prediction using a compact autoregressive transformer with pairwise interaction blocks.**

FoilForm learns to map raw airfoil contour geometry directly to lift and drag coefficients (C_l, C_d) across angle-of-attack sweeps. The full pipeline — tokenizer autoencoders, a causal transformer with a novel outer-product pairwise attention mechanism, and a residual correction MLP — fits in under **39k parameters** total and runs inference at **~0.3 ms per airfoil** on CPU.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Stage 1: Geometry and Aero Tokenizers](#stage-1-geometry-and-aero-tokenizers)
  - [Stage 2: GeomPolarTransformer](#stage-2-geompolartransformer)
  - [Stage 3: PolarCorrectionMLP](#stage-3-polarcorrectionmlp)
- [Data Pipeline](#data-pipeline)
- [Ablation Studies](#ablation-studies)
  - [Block Architecture Ablation](#block-architecture-ablation)
  - [Depth Sweep](#depth-sweep)
  - [Regularization Sensitivity](#regularization-sensitivity)
  - [Data Scaling](#data-scaling)
  - [AoA Ordering](#aoa-ordering)
  - [Correction MLP Impact](#correction-mlp-impact)
  - [NeuralFoil Comparison](#neuralfoil-comparison)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Scripts Reference](#scripts-reference)
- [License](#license)

---

## Overview

Traditional computational fluid dynamics (CFD) and panel-method codes (XFOIL, XFoil) produce accurate airfoil polars but are expensive for large-scale design sweeps. Neural surrogate models offer orders-of-magnitude speedup, but prior approaches typically require large networks, fixed-resolution inputs, or separate models per operating condition.

FoilForm takes a different approach:

1. **Tokenize** the airfoil contour into a sequence of compact 8-D geometry tokens (167 patches of 3 contour points each).
2. **Autoregressively decode** the polar curve: a causal transformer predicts C_l and C_d at each angle of attack, conditioning on its own prior predictions and ground-truth AoA.
3. **Refine** with a lightweight residual correction MLP that operates on the raw 501-point geometry and the transformer's base polar.

The result is a two-stage system with **4,470 transformer parameters** and **34,290 correction parameters** that achieves a validation MAE of **0.040 on C_l** and **0.0081 on C_d** across 1,178 held-out airfoils.

---

## Architecture

### Stage 1: Geometry and Aero Tokenizers

Before the transformer sees any data, raw coordinates and polar observations are compressed into 8-D embeddings by small autoencoder pairs trained with MSE reconstruction loss plus an L2 embedding penalty.

#### Geometry Tokenizer (TripletEncoder / TripletDecoder)

The airfoil contour is represented as 501 (x, y) stations. These are partitioned into **167 non-overlapping triplets** of 3 consecutive points, yielding a `(N, 167, 6)` tensor. Each 6-D triplet is independently encoded:

```
(x1, y1, x2, y2, x3, y3)  ──▶  Linear(6 → 64) → LN → GELU
                                  → Linear(64 → 64) → LN → GELU
                                  → Linear(64 → 8)            ──▶  8-D embedding
```

The decoder mirrors this with `8 → 64 → 64 → 6` (GELU activations, no LayerNorm). The encoder output for every airfoil is saved as `geom_embeddings.npy` with shape `(N, 167, 8)`.

#### Aero Tokenizer (AeroEncoder / AeroDecoder)

Each polar observation `(AoA°, C_l, C_d)` is encoded with **Fourier features on AoA** — `sin(k·θ)` and `cos(k·θ)` for k = 1..4 — concatenated with C_l and C_d to form a 10-D input:

```
[sin(θ), cos(θ), sin(2θ), cos(2θ), ..., sin(4θ), cos(4θ), Cl, Cd]
    ──▶  Linear(10 → 64) → LN → GELU → Linear(64 → 64) → LN → GELU → Linear(64 → 8)
```

The Fourier encoding allows the network to learn periodic AoA structure without manual feature engineering.

---

### Stage 2: GeomPolarTransformer

The core model. It consumes a sequence of 8-D geometry tokens and autoregressively predicts `(C_l, C_d)` at each AoA step. The full default configuration uses **d_model = 8**, **4 layers**, **dropout = 0.05**, and totals **4,470 trainable parameters**.

#### Input Fusion

At each sequence position, the geometry token embedding is concatenated with a **right-shifted** aero tuple embedding (so position *i* sees the prediction from step *i−1*, and position 0 sees zeros):

```
input_i = in_proj( [geom_token_i ‖ shifted_aero_{i-1}] )     # Linear(16 → 8)
        → LayerNorm → Dropout
```

The aero tuple embedding comes from a learned `Linear(3 → 8)` mapping of `[C_l, C_d, AoA°]` from the previous decoding step (with ground-truth AoA during training).

#### AttentionPairwiseBlock

Each layer is a residual block with four branches — causal self-attention, pairwise outer-product (×2), and a bottleneck MLP — all with pre-LayerNorm and dropout:

```
                    ┌─────────────────────────────────────────────────┐
                    │              AttentionPairwiseBlock              │
                    ├─────────────────────────────────────────────────┤
  x ───────────────▶│                                                 │
                    │  1. Causal Self-Attention                       │
                    │     LN → tanh → Q,K,V (d×d each)               │
                    │     scores = QKᵀ / √d + causal_mask             │
                    │     x = x + Dropout(softmax(scores) · V)        │
                    │                                                 │
                    │  2. Pairwise Outer-Product (first)              │
                    │     LN → tanh → outer[b,i,j,k] = x_j · x_k     │
                    │     T₁ = LeakyReLU(W_p1 @ outer + B_p1)        │
                    │     T₂ = LeakyReLU(W_p2 @ T₁ + B_p2)           │
                    │     x = x + Dropout(mean(T₂, dim=-1))           │
                    │                                                 │
                    │  3. Bottleneck MLP                              │
                    │     LN → Linear(8→16) → LeakyReLU               │
                    │        → Linear(16→8)                           │
                    │     x = x + Dropout(mlp_out)                    │
                    │                                                 │
                    │  4. Pairwise Outer-Product (second)             │
                    │     Same structure as (2) with separate weights  │
                    │     x = x + Dropout(mean(T₂', dim=-1))          │
  x' ◀─────────────│                                                 │
                    └─────────────────────────────────────────────────┘
```

The **pairwise outer-product** is the distinctive component: at each sequence position, it computes a `d × d` outer product of the token's feature vector with itself, then applies two learned bilinear transforms with LeakyReLU. This captures multiplicative feature interactions that standard attention and MLPs miss — analogous to pair energies in protein structure prediction (e.g., EvoFormer). The mean-pooled result is added as a residual.

Each block uses **4 LayerNorms** and **dropout** on every residual path.

#### Output Head

The last token's hidden state is projected through a three-stage linear head:

```
h = x[:, -1, :]                      # (B, 8)
h = h @ W_{8×8} + b_8                # (B, 8)
h = h @ W_{8×2}                      # (B, 2)
h = h @ W_{2×2} + b_2                # (B, 2) → [Cl, Cd]
```

#### Autoregressive Decoding (decode_append)

During training and inference, the model uses KV-cached autoregressive decoding:

1. **Context pass**: Run all 167 geometry tokens through the full stack with causal masking. Cache all K, V tensors per layer.
2. **Decode loop**: For each AoA step *t*:
  - Construct a synthetic token from `tuple_to_embed([pred_C_l, pred_C_d, AoA_t])`.
  - Run this single token through each block using the cached K, V (append-only).
  - Read `(C_l, C_d)` from the output head.
  - Append the new K, V to the cache for the next step.

During training, `AoA_t` comes from ground truth (`aoa_ground_truth`); at inference the same ground-truth AoA schedule is used since AoA is a user-specified operating condition, not a predicted quantity.

#### Parameter Breakdown (default 4-layer pairwise)


| Component                         | Parameters |
| --------------------------------- | ---------- |
| Attention Q, K, V (per block)     | 192        |
| Pairwise W, B ×4 (per block)      | 512        |
| Bottleneck MLP 8→16→8 (per block) | 280        |
| LayerNorm ×4 (per block)          | 64         |
| **Per-block total**               | **1,048**  |
| **All 4 blocks**                  | **4,192**  |
| tuple_to_embed (3 → 8)            | 32         |
| in_proj (16 → 8)                  | 136        |
| norm_in                           | 16         |
| Output head                       | 94         |
| **Grand total**                   | **4,470**  |


---

### Stage 3: PolarCorrectionMLP

A residual correction network that refines the transformer's base polar predictions. It operates on the **raw 501-point geometry** (not the tokenized form) and the transformer's 34-slot polar output (17 C_l values + 17 C_d values across the AoA grid).

```
Geometry encoder:
  (B, 2, 501) → Conv1d(2→16, k=11, s=5) → tanh      # → (B, 16, 99)
             → Conv1d(16→32, k=5, s=3) → tanh         # → (B, 32, 32)
             → AdaptiveAvgPool1d(4) → flatten           # → (B, 128)

Correction head:
  cat(geom_features, base_polar)                        # → (B, 162)
  → Linear(162→128) → tanh
  → Linear(128→64) → tanh
  → Linear(64→34)                                      # → (B, 34)  Δ correction

Corrected polar = base_polar + Δ
```

The output layer is initialized near zero (`Uniform(-0.01, 0.01)`) so the network starts as an identity correction and learns the residual. Total: **34,290 parameters**.

The conv encoder sees the full airfoil shape at original resolution, complementing the transformer which only sees the 167-patch tokenized form. This architectural split lets the correction MLP recover fine-grained geometric details that tokenization discards.

---

## Data Pipeline

### Raw Data

The dataset is compiled from airfoil coordinate files (`.dat`) and coefficient tables, merged into a single CSV (`raw/COMPILED AIRFOIL DATA.csv`). Each row contains:

- **Filename**: Airfoil identifier
- **Geometry**: JSON-encoded list of 501 (x, y) contour points
- **AoA, C_l, C_d**: JSON-encoded polar data at observed angles of attack

### Processing

`scripts/prepare_data.py` builds three NumPy arrays:


| Array          | Shape           | Description                                                             |
| -------------- | --------------- | ----------------------------------------------------------------------- |
| `coords.npy`   | `(N, 501, 2)`   | Contour coordinates (x, y) per station                                  |
| `polars.npy`   | `(N, n_aoa, 3)` | `[AoA, C_l, C_d]` on a shared AoA grid; missing entries marked with NaN |
| `aoa_grid.npy` | `(n_aoa,)`      | The common AoA values in degrees                                        |


`scripts/train_tokenizers.py` then produces:


| Array                 | Shape         | Description                                               |
| --------------------- | ------------- | --------------------------------------------------------- |
| `geom_embeddings.npy` | `(N, 167, 8)` | Geometry token embeddings from the trained TripletEncoder |
| `aero_embeddings.npy` | `(N, ...)`    | Aero token embeddings from the trained AeroEncoder        |


### Train / Validation Split

All splits are at the **airfoil level** (not the observation level) with a fixed seed (`seed=42`). The default `train_frac=0.6` yields ~1,178 validation airfoils. Every AoA step for a given airfoil belongs to the same split.

---

## Ablation Studies

All experiments are orchestrated by `studies/run_all.py` and logged to `studies/results_manifest.jsonl`. Training defaults: 120 epochs, batch size 64, AdamW (lr=3e-4, weight_decay=1e-4), cosine annealing, early stopping (patience=20, warmup=10).

### Block Architecture Ablation

Five block variants tested at 4 layers, dropout=0.05:


| Block Type                                           | Parameters | Val MAE C_l | Val MAE C_d |
| ---------------------------------------------------- | ---------- | ----------- | ----------- |
| **Pairwise** (attention + pairwise + MLP + pairwise) | 4,470      | **0.0747**  | 0.0089      |
| **Attention Only** (no pairwise, no MLP)             | 1,110      | 0.0748      | **0.0087**  |
| **Standard MLP** (attention + MLP, no pairwise)      | 2,294      | 0.0904      | 0.0086      |
| **No Pairwise** (attention + MLP)                    | 2,294      | 0.0904      | 0.0086      |
| **No MLP** (attention + pairwise, no MLP)            | 3,286      | 0.0955      | 0.0089      |


The pairwise blocks match the much smaller attention-only variant on C_l while the MLP-only variants consistently underperform, suggesting the pairwise outer-product interaction captures structure that a standard feedforward cannot.

### Depth Sweep

Pairwise blocks with varying layer count:


| Layers | Parameters | Val MAE C_l | Val MAE C_d |
| ------ | ---------- | ----------- | ----------- |
| 1      | 1,326      | 0.0923      | 0.0088      |
| 2      | 2,374      | 0.0927      | 0.0089      |
| **4**  | **4,470**  | **0.0747**  | **0.0089**  |
| 8      | 8,662      | 0.0826      | 0.0089      |
| 16     | 17,046     | 0.0779      | 0.0088      |


Four layers is the sweet spot. Deeper models (8, 16) show diminishing returns, likely due to optimization difficulty at this small model scale.

### Regularization Sensitivity

Dropout sweep at 4 layers, pairwise:


| Dropout  | Val MAE C_l | Val MAE C_d |
| -------- | ----------- | ----------- |
| 0.00     | 0.1972      | 0.0105      |
| **0.05** | **0.0747**  | **0.0089**  |
| 0.10     | 0.0796      | 0.0088      |
| 0.15     | 0.0862      | 0.0090      |


Dropout is critical. Without it (0.00), validation C_l MAE degrades by **2.6x** — the model severely overfits despite having only 4,470 parameters.

### Data Scaling

Training fraction sweep (pairwise, 4 layers, dropout=0.05):


| Train Fraction | Val MAE C_l | Val MAE C_d |
| -------------- | ----------- | ----------- |
| 0.4            | 0.1053      | 0.0092      |
| 0.5            | 0.0900      | 0.0091      |
| 0.6            | 0.0747      | 0.0089      |
| 0.7            | 0.0664      | 0.0088      |
| 0.8            | 0.0664      | 0.0087      |


Performance scales smoothly with training data, with C_l MAE dropping ~37% from 40% to 80% train fraction.

### AoA Ordering


| Ordering                | Val MAE C_l | Val MAE C_d |
| ----------------------- | ----------- | ----------- |
| Default (ascending AoA) | 0.0747      | 0.0089      |
| **Reverse AoA**         | **0.0714**  | **0.0087**  |


Reversing the AoA order in the autoregressive sequence provides a modest improvement, suggesting the model benefits from seeing high-AoA (more challenging) predictions first.

### Correction MLP Impact

End-to-end evaluation on 1,178 validation airfoils:


| Stage                | MAE C_l    | MAE C_d    | MAE L/D  |
| -------------------- | ---------- | ---------- | -------- |
| Transformer alone    | 0.0746     | 0.0089     | 6.01     |
| **+ Correction MLP** | **0.0404** | **0.0081** | **4.88** |


The correction MLP nearly halves C_l error (**-46%**) and meaningfully improves L/D prediction, validating the two-stage design.

### NeuralFoil Comparison

Comparison against [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) (xxxlarge model) on the same validation set:


| Model                    | MAE C_l   | MAE C_d   | MAE L/D  | ms/airfoil |
| ------------------------ | --------- | --------- | -------- | ---------- |
| **FoilForm (corrected)** | **0.040** | **0.008** | **4.88** | **~0.3**   |
| NeuralFoil (xxxlarge)    | 0.524     | 0.019     | 29.93    | ~1.6       |


FoilForm achieves **13x lower C_l error** and **6x lower L/D error** on this dataset. Note that NeuralFoil is a general-purpose pretrained model whereas FoilForm is trained on this specific dataset — the comparison demonstrates the value of dataset-specific training rather than a general superiority claim.

---

## Repository Structure

```
FoilForm/
├── src/foilform/                    # Core library
│   ├── __init__.py                  # Package exports and path constants
│   ├── paths.py                     # REPO_ROOT, DATA_PROCESSED, ARTIFACTS, etc.
│   ├── geom_polar_transformer.py    # GeomPolarTransformer + AttentionPairwiseBlock
│   ├── polar_correction_mlp.py      # PolarCorrectionMLP (Conv1d + MLP)
│   └── tokenizer_model.py           # TripletEncoder/Decoder, AeroEncoder/Decoder
│
├── scripts/                         # Training and evaluation scripts
│   ├── prepare_data.py              # CSV → coords.npy, polars.npy
│   ├── train_tokenizers.py          # Train geometry + aero autoencoders
│   ├── train_geom_polar_transformer.py
│   ├── train_polar_correction_mlp.py
│   ├── eval_polar_correction.py     # Batched MAE evaluation + timing
│   ├── benchmark_cpu_batched.py     # CPU inference benchmark
│   ├── plot_geom_polar_airfoil.py   # Qualitative airfoil polar plots
│   ├── plot_polar_correction_airfoil.py
│   └── visualize_embeddings.py      # Token embedding visualization
│
├── data/processed/                  # Processed arrays (generated)
│   ├── coords.npy                   # (N, 501, 2)
│   ├── polars.npy                   # (N, n_aoa, 3)
│   ├── geom_embeddings.npy          # (N, 167, 8)
│   └── aero_embeddings.npy
│
├── raw/                             # Raw airfoil data
│   ├── COMPILED AIRFOIL DATA.csv    # Master dataset
│   ├── AIRFOILS/                    # Individual .dat + coefficient files
│   └── bigfoil/                     # Additional coordinate files
│
├── artifacts/                       # Trained tokenizer checkpoints
│   ├── geom_tokenizer.pt
│   └── aero_tokenizer.pt
│
├── runs/                            # Training run outputs (dev)
├── figures/                         # Generated plots (dev)
├── tests/                           # Unit and integration tests
│
├── studies/                         # Reproducible ablation suite
│   ├── run_all.py                   # Orchestrates the full study pipeline
│   ├── src/foilform/                # Extended package (block ablations, manifest)
│   ├── scripts/                     # Study-specific train/eval/analysis scripts
│   ├── runs/                        # One directory per experiment
│   ├── figures/                     # Aggregated results (JSON, plots)
│   ├── results_manifest.jsonl       # Structured experiment log
│   └── results_summary.json         # Merged results
│
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.2+

### Installation

```bash
git clone https://github.com/AvnehSBhatia/FoilForm.git
cd FoilForm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Full Training Pipeline

```bash
# 1. Process raw data into arrays
python scripts/prepare_data.py

# 2. Train geometry and aero tokenizers
python scripts/train_tokenizers.py

# 3. Train the GeomPolarTransformer
python scripts/train_geom_polar_transformer.py

# 4. Train the correction MLP (requires transformer checkpoint)
python scripts/train_polar_correction_mlp.py --geom-checkpoint runs/<run>/best_geom_polar_transformer.pt

# 5. Evaluate
python scripts/eval_polar_correction.py
```

### Reproducing the Study

The full ablation suite can be reproduced with a single command:

```bash
python studies/run_all.py
```

This sequentially runs all block ablations, depth sweeps, dropout sweeps, data fraction sweeps, AoA ordering experiments, correction MLP training, evaluation, and analysis scripts. Results are written to `studies/runs/`, `studies/figures/`, and `studies/results_manifest.jsonl`.

---

## Scripts Reference

### Training


| Script                                    | Description                                                         |
| ----------------------------------------- | ------------------------------------------------------------------- |
| `scripts/prepare_data.py`                 | Parse raw CSV into `coords.npy` and `polars.npy`                    |
| `scripts/train_tokenizers.py`             | Train TripletEncoder/Decoder and AeroEncoder/Decoder                |
| `scripts/train_geom_polar_transformer.py` | Train the autoregressive geometry-to-polar transformer              |
| `scripts/train_polar_correction_mlp.py`   | Train the residual correction MLP on frozen transformer predictions |


### Evaluation


| Script                             | Description                                                 |
| ---------------------------------- | ----------------------------------------------------------- |
| `scripts/eval_polar_correction.py` | Compute validation MAE (C_l, C_d, L/D) and inference timing |
| `scripts/benchmark_cpu_batched.py` | CPU batching throughput benchmark                           |


### Visualization


| Script                                     | Description                                                    |
| ------------------------------------------ | -------------------------------------------------------------- |
| `scripts/plot_geom_polar_airfoil.py`       | Plot predicted vs. ground-truth polars for individual airfoils |
| `scripts/plot_polar_correction_airfoil.py` | Compare transformer-only vs. corrected predictions             |
| `scripts/visualize_embeddings.py`          | UMAP/t-SNE visualization of learned token embeddings           |


### Study Analysis


| Script                                            | Description                                              |
| ------------------------------------------------- | -------------------------------------------------------- |
| `studies/scripts/plot_pareto.py`                  | Parameter count vs. validation MAE Pareto front          |
| `studies/scripts/plot_n_layers_depth.py`          | Depth sweep visualization                                |
| `studies/scripts/plot_training_curves.py`         | Training/validation loss curves                          |
| `studies/scripts/analyze_val_distribution.py`     | Per-airfoil error distribution analysis                  |
| `studies/scripts/analyze_worst_best_airfoils.py`  | Best/worst predicted airfoils with contour overlays      |
| `studies/scripts/analyze_corrector_delta.py`      | Correction MLP delta histograms and heatmaps             |
| `studies/scripts/analyze_tsne_embeddings.py`      | PCA/t-SNE of geometry token space                        |
| `studies/scripts/analyze_geometry_sensitivity.py` | Polar sensitivity to geometric perturbations             |
| `studies/scripts/benchmark_inference.py`          | Inference timing comparison (transformer vs. NeuralFoil) |


---

## License

This project is licensed under the [MIT License](LICENSE).