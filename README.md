# The Well: Code and Data for Foundation Model Paper

Reproducibility artifacts for "A Foundation Model for Physical World Models:
Large-Scale Simulation Pre-training, Neural Operator Theory, and Zero-Shot Transfer"
(submitted to JCP).

## Structure

```
code/
  model/          # Model architecture implementation
  experiments/    # All experiment scripts
  data/           # Data loading and preprocessing utilities
data/             # Experimental result artifacts
```

## Model

- `code/model/s1_well_pytorch.py` -- Unified tokenizer + physics-aware MoE backbone
  with spectral convolutions. Supports synthetic mode (no download) and The Well mode.

## Experiments

| Script | Purpose |
|--------|---------|
| `code/experiments/zero_shot_local.py` | Zero-shot cross-system generalization (Theorem 2 validation) |
| `code/experiments/scaling_study.py` | Pretraining convergence and scaling behavior grid |
| `code/experiments/run_supp.py` | Supplementary experiments (tokenizer ablation, complexity, sensitivity) |
| `code/experiments/run_m2.py` | Neural operator baseline comparison (FNO, DeepONet) |
| `code/experiments/zero_shot_exp.py` | Earlier zero-shot experiment variant |
| `code/experiments/zero_shot_v2.py` | Second zero-shot experiment variant |
| `code/experiments/extract_e4.py` | Spectral-truncation decay data extraction |
| `code/experiments/parse_e1_seeds.py` | Per-seed result parsing |
| `code/experiments/gap_fix.py` | Gap-fix: direct delta measurement (B2), PL constant estimation, reverse zero-shot |
| `code/experiments/gap_fix2.py` | Gap-fix round 2: joint multi-system pre-training, expert-capacity ablation |

## Data

| File | Description |
|------|-------------|
| `data/zero_shot_results_full.json` | Full-model zero-shot results (ch=32) |
| `data/scaling_results.json` | Scaling grid results (capacity x data size) |
| `data/zero_shot_results.json` | Earlier zero-shot results |
| `data/supp_results.json` | Supplementary experiment results |
| `data/supp_m2.json` | Baseline comparison results |
| `data/gap_fix_results.json` | Gap-fix results (delta=0.005, PL mu, reverse zero-shot VRMSE 0.83) |
| `data/gap_fix2_results.json` | Gap-fix round 2 results (joint multi-system, 4-expert ablation) |

## Data Loading

- `code/data/download_well_subset.py` -- Parallel chunked download for The Well datasets
- `code/data/extract_slice.py` -- Single-dataset slice extraction
- `code/data/extract_slice_multi.py` -- Multi-dataset slice extraction

## Data Source

The Well benchmark: https://github.com/PolymathicAI/the_well [19].
This repository contains only the authors' experimental code and result artifacts;
raw simulation data are hosted by PolymathicAI.

## Citation

If you use this code or data, please cite the manuscript.
