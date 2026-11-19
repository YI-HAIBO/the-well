# Reproducing the paper's core results (minimal CPU pipeline)

This repository contains a one-command minimal reproduction of the headline
numbers of *Learning cross-system operators with lightweight mixture-of-experts*
(The Well / JCP submission).

## Requirements

- Python 3.10+
- `numpy`, `torch` (CPU is sufficient), `h5py` (only for the optional cross-system check)
- ~300 MB disk, ~3 minutes on a modern CPU

## Data

The pipeline expects the official The-Well shear_flow slices in `data/`.
If the local slices are missing, fetch them from the public `repro-data-v1`
release (the same files the paper's numbers were produced from):

```bash
python fetch_data.py all     # downloads slice_train/slice_test, reassembles turb
```

| file | content | official source |
|---|---|---|
| `data/slice_train.npz` | 4 train trajectories, 20 steps, 256×512 (`pressure`, `tracer`, `velocity`) | `https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well/datasets/shear_flow/data/train/shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5` |
| `data/slice_test.npz` | 4 test trajectories, same layout | `.../shear_flow/data/test/shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5` |
| `turb_rad_layer.hdf5` (optional) | 8 trajectories of `turbulent_radiative_layer_2D` for the cross-system check | The Well v1.0, `turbulent_radiative_layer_2D` |

The `slice_train.npz` and `slice_test.npz` assets are bundled directly in the
release; the 635 MB `turb_rad_layer.hdf5` is shipped as split parts
(`turb_rad_layer.hdf5.part_NN`) and `fetch_data.py` reassembles them
automatically. Full-resolution HDF5 files are available from the official
Flatiron mirror (13.4 GB for the full shear train file); the bundled NPZ slices
are fixed 4-trajectory extracts used for the CPU-scale experiments in the paper.

## Run

```bash
python reproduce_minimal.py
```

Fixed seeds (0/1/2); no GPU required. The script prints and saves
(`reproduce_minimal_results.json`):

1. **In-domain** (80/20 random windows) VRMSE
2. **Leave-one-trajectory-out (LOO)** VRMSE (per-sample-normalized protocol)
3. **Cross-system zero-shot** shear → turbulent_radiative_layer VRMSE (if the
   turb file is present)
4. **Cross-system few-shot adaptation** (16 target windows) VRMSE

VRMSE = sqrt(MSE / Var(y)); the mean predictor scores 1.0 by construction.

## Repository map

- `manuscript_en_jcp.tex` — paper source
- `code/` — experiment and paper-update scripts (`local_attack.py`, `attack2a_e3_e10.py`, `attack2b_e6_e8.py`, …)
- `data/` — official data slices + result JSONs
- `TheWell_全量短板清单_2026-11-17_final.md` — consolidated limitation tracker

All experiment scripts write JSON results with fixed seeds; re-running them
reproduces the numbers quoted in the paper's Limitations discussion.
