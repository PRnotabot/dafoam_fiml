# FIML with Experimental Data: Periodic Hill at Re=5600

This tutorial demonstrates Field Inversion and Machine Learning (FIML) applied to the periodic hill flow using **experimental velocity data** from Rapp & Manhart (2011) instead of numerical k-ε reference data. It then uses symbolic regression (PySR) to discover an **interpretable algebraic equation** for the turbulence model correction factor.

## Overview

The workflow consists of three main steps:

1. **Data Preparation** (`genExpData.py`): Generate experimental velocity profiles at probe locations
2. **Field Inversion** (`runScript_FI.py`): Optimize the turbulence correction factor β_FI to match experimental data
3. **Symbolic Regression** (`sr_training/runSR.py`): Discover algebraic expression β_FI = f(features)

## Setup

### Prerequisites
- OpenFOAM v1812 environment (with `$WM_PROJECT` set)
- DAFoam with adjoint capabilities
- Python 3.7+ with numpy, scipy, openmdao, mpi4py
- PySR for symbolic regression: `pip install pysr`

### Directory Structure
```
PeriodicHill_Exp/
├── 0.orig/                    # Initial conditions template
├── constant/                  # Mesh, properties, turbulence model
├── system/                    # Solver settings (controlDict, fvSchemes, fvSolution)
├── genExpData.py             # Generate experimental reference data
├── preProcessing.sh          # Setup script (copies BC files, generates UData)
├── Allclean.sh               # Cleanup script
├── runScript_FI.py           # Field inversion optimization
├── runScript_features.py     # Compute turbulence features after FI
├── sr_training/
│   └── runSR.py              # Symbolic regression on FI results
├── references.md             # All data sources and download links
└── README.md                 # This file
```

## Quick Start

### 1. Preprocessing
```bash
cd tutorials/PeriodicHill_Exp
source $DAFOAM_ROOT_PATH/loadDAFoam.sh
bash preProcessing.sh
```

This generates:
- `0/UData`: Experimental velocity field at 3500 mesh cells
- `probePointCoords.json`: Coordinates of 67 measurement points (5 streamwise stations)

### 2. Field Inversion
```bash
# Run optimization (default: IPOPT, 50 iterations)
mpirun --oversubscribe -np 4 python runScript_FI.py

# Or test with a single primal run
mpirun --oversubscribe -np 4 python runScript_FI.py -task run_model

# Verify gradients against finite-difference
mpirun --oversubscribe -np 4 python runScript_FI.py -task check_totals
```

**Output**: Optimized `betaFINuTilda` field in the final time directory (e.g., `4000/betaFINuTilda`)

### 3. Compute Features
```bash
mpirun --oversubscribe -np 4 python runScript_features.py
```

This runs one primal solve with regression features enabled, writing:
- `PoD`, `VoS`, `PSoSS`, `SCurv` (and other features) to the latest time directory

These features represent Galilean-invariant turbulence characteristics needed by symbolic regression.

### 4. Symbolic Regression
```bash
cd sr_training
python runSR.py
# Or with custom settings:
python runSR.py -niterations 200 -maxsize 30 -features PoD VoS PSoSS SCurv
```

**Output**:
- `results/equation.py`: Python callable function
- `results/equation.tex`: LaTeX format
- `results/equation.cpp`: C++ code for DAFoam integration
- `results/sr_report.json`: Metrics and Pareto front

## Case Details

### Experimental Data
- **Source**: Rapp, C., Manhart, M. (2011). "Flow over periodic hills: an experimental study." *Experiments in Fluids*, 51, 247-269.
- **Reynolds number**: Re_H = 5600 (based on hill height H and bulk velocity at crest)
- **Measurement technique**: PIV (2D) + LDA (1D point validation)
- **Measurement stations**: x/H = 0.5, 2.0, 4.0, 6.0, 8.0

### Domain Geometry
- Channel height: Ly = 3.035H
- Streamwise period: Lx = 9H (one hill)
- Mesh: 3500 cells (coarse for demo purposes)
- Spanwise: 0.1H (2D-like periodic channel)

### Physics
- **Solver**: DASimpleFoam (incompressible, steady SIMPLE algorithm)
- **Turbulence model**: Spalart-Allmaras (SA)
- **Correction mechanism**: Spatial multiplier β_FI on SA production term
- **Viscosity**: ν = 5e-6 m²/s
- **Bulk velocity**: U_b = 0.028 m/s

### Key Flow Features
- **Separation**: x/H ≈ 0.2 (pressure-induced from curved surface)
- **Recirculation**: x/H = 0.2 to 4.7
- **Reattachment**: x/H ≈ 4.7 (decreasing with increasing Re)
- **Shear layers**: Developing on both sides of separated zone

## Field Inversion Optimization

### Objectives
```
J = UProbeVar + 0.001 * betaVar
```

- **UProbeVar**: L² error between computed and experimental velocities at 67 probe points
- **betaVar**: Regularization penalty (keeps β near 1.0, prevents overfitting)

### Design Variables
- **β_FI field**: One scalar value per cell (3500 design variables)
- **Bounds**: [-5, 10] (allows ~1000x reduction or ~40x amplification)
- **Initial value**: 1.0 (no correction)

### Optimization Settings
- **Optimizer**: IPOPT (default) or SNOPT/SLSQP
- **Max iterations**: 50
- **Tolerance**: 1e-5
- **Gradient computation**: Discrete adjoint (automatic differentiation)

## Symbolic Regression

### Features Used
Default 4-feature set (all computed by DAFoam):

| Feature | Definition | Range | Meaning |
|---------|-----------|-------|---------|
| PoD | P_k/(P_k + D_k) | [0,1] | Production vs. dissipation |
| VoS | \|Ω\|/(\|Ω\|+\|S\|) | [0,1] | Vorticity vs. strain |
| PSoSS | \|∇p\|/(\|∇p\|+normal stress) | [0,1] | Pressure vs. Reynolds stress |
| SCurv | \|U·∇U\|/(\|U·U\|+\|U·∇U\|) | [0,1] | Streamline curvature |

All normalized to [0,1] to ensure Galilean invariance.

### PySR Configuration
- **Operators**: +, -, *, /, exp, log, tanh, sqrt, square, abs
- **Search space**: Complexity up to ~25 (tunable with `-maxsize`)
- **Populations**: 30 (parallel genetic algorithms)
- **Iterations**: 100 (tune with `-niterations`)
- **Training/validation split**: 80/20

### Output
Discovered equation appears as:
- Symbolic expression: `beta_FI = ...`
- Validation metrics: MSE, RMSE, R²
- Pareto front: All non-dominated solutions (complexity vs. accuracy trade-off)

## Cleanup

```bash
./Allclean.sh
```

Removes: `0/`, processor*, time directories, optimization logs, `probePointCoords.json`, etc.

## Data Sources & References

See `references.md` for:
- Complete citations (Rapp & Manhart, Breuer et al., Bidar et al.)
- Download links (NASA LARC database, ERCOFTAC QNET, GitHub)
- Contact information for original experimental data

## Extensions & Modifications

### Change Input Features
Edit `EXPERIMENTAL_PROFILES` in `genExpData.py` or modify `-features` arg in `runSR.py`:
```bash
python runSR.py -features PoD VoS PSoSS SCurv UOrth KoU2
```

### Change Experimental Data
Replace velocity profiles in `EXPERIMENTAL_PROFILES` dict (lines 155-213 in `genExpData.py`). Update from:
- ERCOFTAC QNET database
- NASA Turbulence Modeling Resource
- Original Rapp & Manhart (2011) supplementary data

### Mesh Refinement
Generate finer mesh and update `nCells` in `runScript_FI.py`, `runScript_features.py`, and `genExpData.py` (if using different mesh).

### Coupled NN Training
To train a neural network instead of pure field inversion, see the Ramp tutorial:
```
tutorials/Ramp/steady_SA/train/runScript.py
```

This demonstrates coupled training with regression model active from the start.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: dafoam` | Load OpenFOAM env: `source $DAFOAM_ROOT_PATH/loadDAFoam.sh` |
| `FileNotFoundError: polyMesh` | Run `preProcessing.sh` from the case directory |
| `RecursionError` in genExpData.py | This is fixed in the current version. Verify you have the latest code. |
| PySR not installed | `pip install pysr` (requires Julia backend) |
| Optimization stalls | Try reducing regularization weight in `runScript_FI.py`: change `0.001` to `0.0001` |
| Gradient check fails | Increase primal tolerance: `primalMinResTol: 1e-10` in `runScript_FI.py` |

## References

**Primary References**:
- Rapp, C., Manhart, M. (2011). Flow over periodic hills. *Experiments in Fluids*, 51, 247-269. https://doi.org/10.1007/s00348-011-1045-y
- Breuer, M., Peller, N., Rapp, C., Manhart, M. (2009). Flow over periodic hills. *Computers & Fluids*, 38(2), 433-457. https://doi.org/10.1016/j.compfluid.2008.05.002

**FIML & DAFoam**:
- Bidar, O. (2024). Data-driven Augmentation of Turbulence Models. PhD Thesis, University of Sheffield.
- He, P., Mader, C.A., Martins, J.R.R.A. (2018). DAFoam. *AIAA Journal*, 58(3), 1304-1319.
- DAFoam: https://dafoam.github.io

**Symbolic Regression**:
- PySR: https://github.com/MilesCranmer/PySR

See `references.md` for complete list of citations and data sources.
