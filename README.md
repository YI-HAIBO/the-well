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
| `code/experiments/gap_fix3.py` | Gap-fix round 3: 128x128 resolution generalization |
| `code/experiments/gap_fix4.py` | Gap-fix round 4: capacity/data-regime scaling, UNet baseline |
| `code/experiments/gap_fix5.py` | Gap-fix round 5: per-sample gate audit, conservation diagnostic, hyperparameter extension |
| `code/experiments/gap_fix5b.py` | Gap-fix round 5b: B4 model field-integral conservation error vs data drift |
| `code/experiments/b3_online.py` | B3: finetuning-convergence validation of Theorem 3 (PL rate, init advantage) |
| `code/experiments/b1_third_system.py` | B1: third-system (Gray-Scott) zero-shot cross-system generalization |
| `code/experiments/c2_multisystem.py` | C2: multi-system scaling of the shared backbone (k=2..5) |
| `code/experiments/c2_k5.py` | C2: k=5 (viscoelastic) run, merged with the k=2..4 seed-0 results |
| `code/experiments/c1_c3.py` | C1: 256x256 native-resolution training; C3: online-deployment inference throughput |

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
| `data/gap_fix3_results.json` | Gap-fix round 3 results (128x128 in-domain + joint) |
| `data/gap_fix4_results.json` | Gap-fix round 4 results (capacity sweep, UNet, joint ch32) |
| `data/gap_fix5_results.json` | Gap-fix round 5 results (B5 gate audit, B4 data drift, B2 hyperparam) |
| `data/gap_fix5b_results.json` | Gap-fix round 5b results (B4 model conservation error) |
| `data/b3_results.json` | B3 results (finetuning convergence, rho, mu_emp, T* pre vs scratch) |
| `data/b1_results.json` | B1 results (3-system zero-shot VRMSE matrix) |
| `data/c2_results.json` | C2 results (k=2..5 mean held-out VRMSE per system) |
| `data/c1c3_results.json` | C1 (256x256 VRMSE, per-step time) + C3 (latency / throughput per resolution) |

## Data Loading

- `code/data/download_well_subset.py` -- Parallel chunked download for The Well datasets
- `code/data/extract_slice.py` -- Single-dataset slice extraction
- `code/data/extract_slice_multi.py` -- Multi-dataset slice extraction
- `code/data/fetch_well_data.py` -- Byte-range fetch of shear-flow and turbulent-radiative slices
- `code/data/fetch_multi.py` -- Byte-range fetch of Gray-Scott, Euler, and viscoelastic slices

## Data Source

The Well benchmark: https://github.com/PolymathicAI/the_well [19].
This repository contains only the authors' experimental code and result artifacts;
raw simulation data are hosted by PolymathicAI.

## Citation

If you use this code or data, please cite the manuscript.
