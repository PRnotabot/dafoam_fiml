# Symbolic Distillation Pipeline for FIML

Convert a black-box neural network turbulence correction into an interpretable algebraic equation through a 3-stage pipeline.

## Pipeline Overview

```
Stage 1                    Stage 2                    Stage 3
Coupled NN Training   -->  Coupled Compression   -->  Symbolic Regression
(runScript.py)             (runCompression.py)        (PySR)

[4 inputs, [20,20], 1]    [3 inputs, [5], 1]         beta_FI = f(PoD, VoS, chiSA)
541 parameters             21 parameters              Algebraic equation
```

**Stage 1** trains a large neural network correction factor using DAFoam's coupled adjoint optimization with physics constraints (pressure variance, velocity field matching, drag error).

**Stage 2** compresses the network by ranking input features by weight saliency, selecting the top-K, and re-training a smaller network initialized from the teacher's weights. The coupled adjoint ensures physics-consistency is maintained through compression.

**Stage 3** fits a symbolic expression to the compressed network using PySR. The small network (~20 parameters) makes symbolic regression tractable, and the discovered equation is validated against the NN.

## Prerequisites

1. DAFoam environment with OpenFOAM, PETSc, OpenMDAO
2. Field inversion completed (run `runScript_FI.py` for each case)
3. PySR installed (`pip install pysr`) for Stage 3

## Quick Start

```bash
# Run the full pipeline
cd tutorials/Ramp/steady_SA/train/symbolic_distillation
mpirun --oversubscribe -np 4 python runPipeline.py

# If Stage 1 (runScript.py) was already run:
mpirun --oversubscribe -np 4 python runPipeline.py -skip_stage1

# Only run symbolic regression (Stages 1 & 2 already done):
python runPipeline.py -skip_stage1 -skip_stage2

# Customize compression (keep 2 features, 4 hidden neurons):
mpirun --oversubscribe -np 4 python runPipeline.py -n_features 2 -hidden 4
```

### Run Individual Stages

```bash
# Stage 1 (from parent directory)
cd ../
mpirun --oversubscribe -np 4 python runScript.py

# Stage 2
cd symbolic_distillation/
mpirun --oversubscribe -np 4 python runCompression.py -n_features 3 -hidden 5

# Stage 3 only (after Stage 2)
python runPipeline.py -skip_stage1 -skip_stage2
```

## Output Files

| File | Stage | Description |
|------|-------|-------------|
| `../designVariable.json` | 1 | Trained teacher NN parameters (541 params) |
| `designVariable_compressed.json` | 2 | Compressed student NN parameters (~21 params) |
| `results/distillation_report.json` | 3 | Pipeline summary with equation and metrics |
| `results/equation.py` | 3 | Discovered equation as Python function |
| `results/nn_data.npz` | 3 | NN evaluation data (saved if PySR unavailable) |

## Using the Discovered Equation

The output equation can replace the neural network in `daOptions`:

```python
# Instead of the NN regression model, use the algebraic equation directly
# in a custom DAFoam function or as an analytical correction.
#
# Example discovered equation:
#   beta_FI = 1.0 + 0.5 * tanh(PoD - 2.0) * VoS
```

For DAFoam C++ integration, implement the equation in a custom `DARegression` model type or `DAFvSource`.

## Command-Line Options

### runPipeline.py

| Flag | Default | Description |
|------|---------|-------------|
| `-skip_stage1` | False | Skip Stage 1 |
| `-skip_stage2` | False | Skip Stage 2 |
| `-n_features` | 3 | Number of input features to keep |
| `-hidden` | 5 | Neurons in compressed hidden layer |
| `-optimizer` | IPOPT | Optimizer (IPOPT, SNOPT, SLSQP) |
| `-max_iter_s2` | 30 | Max optimizer iterations for Stage 2 |
| `-np` | 4 | Number of MPI processes |
| `-sr_iterations` | 100 | PySR search iterations |
| `-sr_maxsize` | 25 | Max equation complexity |
| `-output_dir` | results | Output directory for Stage 3 |

### runCompression.py

| Flag | Default | Description |
|------|---------|-------------|
| `-optimizer` | IPOPT | Optimizer |
| `-task` | run_driver | Task type (run_driver, run_model, compute_totals, check_totals) |
| `-n_features` | 3 | Features to keep |
| `-hidden` | 5 | Hidden neurons |
| `-max_iter` | 30 | Max iterations |
| `-teacher_dir` | .. | Path to Stage 1 results |
