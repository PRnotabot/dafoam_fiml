# Cylinder Re=3900 FIML Tutorial (Steady SA)

Decoupled Field Inversion and Machine Learning for flow around a cylinder at Re=3900 using the Spalart-Allmaras turbulence model.

## Overview

This tutorial demonstrates:
1. **Field Inversion (FI)**: Optimize betaFINuTilda field to match experimental Cp on cylinder surface
2. **Machine Learning (ML)**: Train MLP to predict beta from flow features (PoD, VoS, chiSA, PSoSS)

## Directory Structure

```
train/
├── c1/                      # Case directory
│   ├── 0_orig/              # Initial conditions (user provides)
│   ├── constant/
│   │   └── polyMesh/        # Mesh files (user provides)
│   └── system/              # OpenFOAM controls (user provides)
├── CpRef.json               # Experimental Cp data (user provides)
├── runScript_FI.py          # Field inversion script
├── createRefField.py        # Create pRef from experimental data
├── preProcessing.sh         # Setup script
└── tf_training/
    └── trainModel.py        # MLP training script
```

## Files You Need to Provide

### 1. Mesh (`c1/constant/polyMesh/`)

Standard OpenFOAM mesh files. The cylinder patch must be named `cylinder`.

Required boundary patches:
- `inlet` - velocity inlet
- `outlet` - pressure outlet
- `cylinder` - cylinder surface (no-slip wall)
- `top`, `bottom` - far-field or symmetry
- `front`, `back` - empty (2D) or cyclic (3D)

### 2. Initial Conditions (`c1/0_orig/`)

Required fields:
- `U` - velocity (inlet: [0.039, 0, 0] for Re=3900 with nu=1e-5, D=1)
- `p` - pressure
- `nuTilda` - SA turbulent viscosity
- `nut` - turbulent viscosity (derived)
- `betaFINuTilda` - initial beta field (set to uniform 1.0)

### 3. System Files (`c1/system/`)

- `controlDict` - runtime control
- `fvSchemes` - discretization schemes
- `fvSolution` - solver settings

Use settings similar to standard DAFoam tutorials.

### 4. Experimental Cp Data (`CpRef.json`)

JSON format with theta (degrees from stagnation point) and Cp values:

```json
{
    "theta": [0, 10, 20, 30, ..., 180],
    "Cp": [1.0, 0.95, 0.82, ..., -1.2]
}
```

**Theta convention**: 0° = front stagnation point, 90° = top/bottom, 180° = rear

Example (Norberg 1987 experimental data):
```json
{
    "theta": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180],
    "Cp": [1.0, 0.92, 0.72, 0.42, 0.05, -0.32, -0.65, -0.90, -1.05, -1.10, -1.08, -1.02, -0.95, -0.90, -0.87, -0.85, -0.84, -0.83, -0.82]
}
```

## Pre-FIML Steps (Standard CFD Setup)

1. **Generate mesh** using blockMesh, snappyHexMesh, or external tool
2. **Verify mesh quality**: `checkMesh`
3. **Run baseline SA simulation** to ensure convergence:
   ```bash
   cd c1
   mpirun -np 4 DASimpleFoam -parallel
   ```
4. **Validate baseline** against experimental Cp (expect significant deviation)

## FIML Workflow

### Step 1: Pre-processing

```bash
# From train/ directory
chmod +x preProcessing.sh
./preProcessing.sh
```

This creates the `pRef` field on the cylinder surface from your experimental data.

### Step 2: Field Inversion

```bash
# From train/ directory
mpirun -np 4 python runScript_FI.py -optimizer IPOPT -task run_driver
```

This optimizes betaFINuTilda to minimize:
- `CpVar`: Cp error on cylinder surface
- `betaVar`: Regularization on beta (keeps it close to 1.0)

**Output**: Inverted beta field in `c1/` directory

### Step 3: Prepare Training Data

```bash
# Copy inverted fields for ML training
cp -r c1 c1_data
```

### Step 4: Train MLP

```bash
cd tf_training

# Update nCells in trainModel.py to match your mesh
python trainModel.py
```

**Output**: Trained model in `tf_training/model/`

### Step 5: Verify Trained Model

```bash
cd c1
# Run with augmented model
python runPrimal.py -augmented True
```

Compare Cp with experimental data - should show improvement over baseline.

## Configuration Notes

### Adjusting nCells

Update `nCells` in both scripts after mesh generation:

```python
# In runScript_FI.py and trainModel.py
nCells = <actual_cell_count>
```

Get cell count: `grep nCells c1/constant/polyMesh/owner | head -1`

### Tuning Field Inversion

In `runScript_FI.py`:
- `CpVar.scale`: Weight of Cp error term (increase to match Cp more closely)
- `betaVar.scale`: Regularization weight (decrease for more aggressive inversion)
- `beta` bounds: `lower=-5.0, upper=10.0` (typical range)

### Flow Conditions

Default: Re=3900 with D=1.0, nu=1e-5 gives U0=0.039 m/s

Modify in both `runScript_FI.py` and `runPrimal.py`:
```python
Re = 3900.0
D = 1.0
nu = 1.0e-5
U0 = Re * nu / D
```

## Expected Results

- Baseline SA: Poor Cp prediction in separation region (theta > 80°)
- After FIML: Cp matches experimental data within ~10%
- Beta field: Values typically range [0.5, 2.0], with peaks near separation

## Troubleshooting

**Inversion not converging**:
- Reduce `betaVar.scale` (less regularization)
- Increase `max_iter` in optimizer settings

**ML model poor accuracy**:
- Increase epochs or neurons in `trainModel.py`
- Check feature normalization with `normalizer.mean` and `normalizer.variance`

**Augmented primal diverges**:
- Tighten beta bounds in field inversion
- Check `outputUpperBound`/`outputLowerBound` in regression model
