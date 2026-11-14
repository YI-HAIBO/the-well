# The Well: Code and Data for Foundation Model Paper

This repository contains the code and data artifacts supporting the manuscript
"A Foundation Model for Physical World Models: Large-Scale Simulation Pre-training,
Neural Operator Theory, and Zero-Shot Transfer" (submitted to JCP).

## Structure

- `code/` -- experiment source code
- `data/` -- experimental results and data manifests

## Code

- `zero_shot_local.py` -- zero-shot cross-system generalization experiment (Theorem 2 validation)
- `scaling_study.py` -- pretraining convergence and scaling behavior study

## Data

- `zero_shot_results_full.json` -- full-model zero-shot results (ch=32)
- `scaling_results.json` -- scaling grid results (model capacity x training data size)
- `zero_shot_results.json` -- earlier zero-shot results
- `supp_results.json` -- supplementary experiment results
- `supp_m2.json` -- neural operator baseline comparison results

## Data Source

The Well benchmark data is available at https://github.com/PolymathicAI/the-well.
This repository contains only the experimental code and result artifacts generated
by the authors; the raw simulation data is hosted by PolymathicAI.

## Citation

If you use this code or data, please cite the manuscript.
