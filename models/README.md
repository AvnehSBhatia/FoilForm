# Canonical model weights

These files are the **default checkpoints** used by evaluation scripts, plotting, and benchmarks. Code resolves them via `foilform.checkpoints` (prefer `models/`, then fall back to the newest file under `runs/` or `artifacts/`).

| File | Component |
|------|-----------|
| `geom_polar_transformer.pt` | `GeomPolarTransformer` (production pairwise block) |
| `polar_correction.pt` | `PolarCorrectionMLP` |
| `geom_tokenizer.pt` | Triplet geometry tokenizer |
| `aero_tokenizer.pt` | Aero tokenizer |

After training a better run, update the repo defaults:

```bash
cp runs/<run_id>/best_geom_polar_transformer.pt models/geom_polar_transformer.pt
cp runs/<run_id>/best_polar_correction.pt models/polar_correction.pt
cp artifacts/geom_tokenizer.pt models/geom_tokenizer.pt
cp artifacts/aero_tokenizer.pt models/aero_tokenizer.pt
```
