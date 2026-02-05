# PySR Symbolic Regression for FIML Turbulence Modeling

This directory contains tools for discovering interpretable algebraic equations for the SA turbulence model correction factor using symbolic regression.

## Overview

Instead of using a neural network (black-box) to model the correction factor β_FI, symbolic regression discovers explicit mathematical equations that can be:
- Interpreted by domain experts
- Analyzed for physical meaning
- Easily implemented in any CFD solver
- Published in papers without proprietary dependencies

## Prerequisites

### 1. Install PySR

```bash
pip install pysr
```

PySR uses Julia internally. On first run, it will install required Julia packages automatically.

### 2. Run Field Inversion First

The symbolic regression requires training data from field inversion. Run:

```bash
cd ..
mpirun -np 4 python runScript_FI.py -index 0  # Generates c1_data/
mpirun -np 4 python runScript_FI.py -index 1  # Generates c2_data/
```

This creates directories containing:
- Feature fields: `PoD`, `VoS`, `chiSA`, `PSoSS`
- Target field: `betaFINuTilda` (the correction factor to learn)

## Usage

### Basic Training

```bash
cd sr_training
python trainModel_SR.py
```

### Training Options

```bash
python trainModel_SR.py [options]

Options:
  -niterations N    Number of PySR iterations (default: 100)
  -populations N    Number of populations in genetic algorithm (default: 30)
  -maxsize N        Maximum complexity of equations (default: 25)
  -maxdepth N       Maximum depth of expression tree (default: 6)
  -val_split F      Validation split fraction (default: 0.2)
  -seed N           Random seed for reproducibility (default: 0)
  -output_dir DIR   Output directory for results (default: results)
  -turbo            Enable turbo mode for faster convergence
```

### Example Commands

Quick test run:
```bash
python trainModel_SR.py -niterations 20 -populations 10 -turbo
```

Production run (longer, more thorough):
```bash
python trainModel_SR.py -niterations 200 -populations 50 -maxsize 30
```

## Feature Space

The four input features are bounded non-dimensional quantities:

| Feature | Physical Meaning | Range |
|---------|-----------------|-------|
| PoD | Production / (Production + Destruction) | [0, 1] |
| VoS | Vorticity / (Vorticity + Strain) | [0, 1] |
| chiSA | νtilde / (ν + νtilde) | [0, 1] |
| PSoSS | \|∇p\| / (\|∇p\| + normal stress) | [0, 1] |

## Operator Design Space

The symbolic regression searches over:

| Type | Operators | Constraints |
|------|-----------|-------------|
| Binary | +, -, *, / | Division: numerator ≤ 5, denominator ≤ 3 complexity |
| Unary | exp, log, tanh, sqrt, square, abs | exp/log arguments ≤ 3 complexity |
| Structure | maxsize=25, maxdepth=6 | No nested exp(exp(...)) or log(log(...)) |

## Output Files

After training, results are saved to `results/` (or custom `-output_dir`):

```
results/
├── equation_best.py      # Python callable function
├── equation_best.tex     # LaTeX equation for papers
├── equation_best_sympy.txt # SymPy expression
├── equation_best.cpp     # C++ code for DAFoam integration
├── pareto_front.json     # All Pareto-optimal equations
├── pareto_front.png      # Complexity vs accuracy plot
├── predictions.png       # Parity plot (predicted vs true)
├── sr_report.json        # Comprehensive metrics report
└── model.pkl             # Pickled PySR model for reuse
```

## Understanding Results

### Pareto Front

The Pareto front shows the tradeoff between equation complexity and accuracy. Points on the front represent equations where you cannot improve accuracy without increasing complexity (or vice versa).

### Choosing an Equation

Consider:
1. **Validation R²** - Higher is better; aim for R² > 0.7
2. **Complexity** - Lower is more interpretable; complexity < 20 preferred
3. **Physical meaning** - Does the equation make sense physically?

### Example Output

```
Best equation found:
  1.0 + 0.5*tanh(PoD - VoS) - 0.2*chiSA

Complexity: 12
Training loss (MSE): 1.234e-03

Validation metrics:
  R²: 0.85
  RMSE: 0.035
```

## Using Discovered Equations

### In Python

```python
from results.equation_best import beta_fi
import numpy as np

# Single point
beta = beta_fi(PoD=0.5, VoS=0.3, chiSA=0.7, PSoSS=0.2)

# Vectorized
PoD = np.array([0.5, 0.6, 0.7])
VoS = np.array([0.3, 0.4, 0.5])
chiSA = np.array([0.7, 0.6, 0.5])
PSoSS = np.array([0.2, 0.3, 0.4])
beta = beta_fi(PoD, VoS, chiSA, PSoSS)
```

### In DAFoam (C++)

The exported `equation_best.cpp` provides a template for integration into DAFoam's regression model framework.

### In Publications

Use the LaTeX output in `equation_best.tex` directly in your paper.

## Comparison with Neural Network

| Aspect | Neural Network | Symbolic Regression |
|--------|---------------|---------------------|
| Interpretability | Black box | Explicit equation |
| Accuracy | Usually higher | May be slightly lower |
| Generalization | Can overfit | Implicit regularization |
| Implementation | Requires framework | Simple algebraic expression |
| Publication | Hard to reproduce | Fully reproducible |

## Troubleshooting

### Julia Installation Issues

If PySR fails to start, try:
```bash
python -c "import pysr; pysr.install()"
```

### Out of Memory

Reduce data size or complexity:
```bash
python trainModel_SR.py -maxsize 15 -populations 20
```

### Slow Convergence

Use turbo mode for faster (but less thorough) search:
```bash
python trainModel_SR.py -turbo
```

## References

- [PySR Documentation](https://astroautomata.com/PySR/)
- [DAFoam FIML Documentation](../../docs/FIML_Decoupled_Training.md)
- Cranmer, M. (2023). "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl"




  Plan: PySR Symbolic Regression for Decoupled FIML                                                                     
                                                                                                                       
 Overview                                                                                                              
                                                                                                                       
 Use PySR symbolic regression on field inversion data to discover interpretable algebraic equations for the SA         
 turbulence model correction factor.                                                                                   
                                                                                                                       
 Goal: Discover interpretable equations: beta(PoD, VoS, chiSA, PSoSS) = <algebraic equation>                           
                                                                                                                       
 Scope: Decoupled training only (no DAFoam integration needed). Focus on:                                              
 1. Reading field inversion training data                                                                              
 2. Running PySR symbolic regression                                                                                   
 3. Exporting interpretable equations with validation metrics                                                          
                                                                                                                       
 ---                                                                                                                   
 File Structure                                                                                                        
                                                                                                                       
 tutorials/Ramp/steady_SA/train/                                                                                       
 ├── tf_training/           # Existing TensorFlow (preserved)                                                          
 │   └── trainModel.py                                                                                                 
 ├── sr_training/           # NEW: Symbolic Regression                                                                 
 │   ├── trainModel_SR.py   # Main PySR training script                                                                
 │   ├── sr_utils.py        # Export, validation, visualization utilities                                              
 │   └── README.md          # Documentation                                                                            
                                                                                                                       
 ---                                                                                                                   
 Implementation Tasks                                                                                                  
                                                                                                                       
 Task 1: Create sr_training/sr_utils.py                                                                                
                                                                                                                       
 Utility functions for:                                                                                                
 - export_equation_formats(model) - Export to SymPy, LaTeX, Python callable                                            
 - validate_equation(func, X, y) - Compute MSE, R², max error, correlation                                             
 - plot_pareto_front(model) - Complexity vs accuracy visualization                                                     
 - plot_prediction_comparison(y_true, y_pred) - Parity plots                                                           
 - save_equation_report(model, output_dir) - Save all outputs                                                          
                                                                                                                       
 Task 2: Create sr_training/trainModel_SR.py                                                                           
                                                                                                                       
 Main training script following existing TensorFlow pattern:                                                           
                                                                                                                       
 # Data loading (same as trainModel.py)                                                                                
 ofm = PYOFM(comm=MPI.COMM_WORLD)                                                                                      
 cases = ["c1_data", "c2_data"]                                                                                        
 features = ["PoD", "VoS", "chiSA", "PSoSS"]                                                                           
 # Read ~10,000 samples (5000 cells × 2 cases)                                                                         
                                                                                                                       
 # PySR configuration                                                                                                  
 model = PySRRegressor(                                                                                                
     binary_operators=["+", "-", "*", "/"],                                                                            
     unary_operators=["exp", "log", "tanh", "sqrt", "square", "abs"],                                                  
     maxsize=25,                                                                                                       
     maxdepth=6,                                                                                                       
     constraints={"/": (5, 3), "exp": 3, "log": 3, "sqrt": 3},                                                         
     nested_constraints={"exp": {"exp": 0}, "log": {"log": 0}},                                                        
     niterations=100,                                                                                                  
     populations=30,                                                                                                   
     variable_names=["PoD", "VoS", "chiSA", "PSoSS"],                                                                  
 )                                                                                                                     
                                                                                                                       
 # Train and export                                                                                                    
 model.fit(X_train, y_train)                                                                                           
 # Save: equation.py, equation.tex, pareto_front.json                                                                  
                                                                                                                       
 Task 3: Create sr_training/README.md                                                                                  
                                                                                                                       
 Documentation covering:                                                                                               
 - Prerequisites (run field inversion first)                                                                           
 - Usage instructions                                                                                                  
 - PySR configuration options                                                                                          
 - Output interpretation                                                                                               
                                                                                                                       
 ---                                                                                                                   
 Feature Space (Unchanged)                                                                                             
 ┌─────────┬─────────────────────────────────────────┬────────┐                                                        
 │ Feature │            Physical Meaning             │ Range  │                                                        
 ├─────────┼─────────────────────────────────────────┼────────┤                                                        
 │ PoD     │ Production / (Production + Destruction) │ [0, 1] │                                                        
 ├─────────┼─────────────────────────────────────────┼────────┤                                                        
 │ VoS     │ Vorticity / (Vorticity + Strain)        │ [0, 1] │                                                        
 ├─────────┼─────────────────────────────────────────┼────────┤                                                        
 │ chiSA   │ nuTilda / (nu + nuTilda)                │ [0, 1] │                                                        
 ├─────────┼─────────────────────────────────────────┼────────┤                                                        
 │ PSoSS   │                                         │ ∇p     │                                                        
 └─────────┴─────────────────────────────────────────┴────────┘                                                        
 Design Space (Operators)                                                                                              
 ┌───────────┬───────────────────────────────────┬──────────────────────────┐                                          
 │   Type    │             Operators             │       Constraints        │                                          
 ├───────────┼───────────────────────────────────┼──────────────────────────┤                                          
 │ Binary    │ +, -, *, /                        │ Division: num≤5, denom≤3 │                                          
 ├───────────┼───────────────────────────────────┼──────────────────────────┤                                          
 │ Unary     │ exp, log, tanh, sqrt, square, abs │ exp/log: arg≤3           │                                          
 ├───────────┼───────────────────────────────────┼──────────────────────────┤                                          
 │ Structure │ maxsize=25, maxdepth=6            │ No nested exp/log        │                                          
 └───────────┴───────────────────────────────────┴──────────────────────────┘                                          
 ---                                                                                                                   
 Workflow                                                                                                              
                                                                                                                       
 1. Prerequisites                                                                                                      
    └── Run field inversion: mpirun -np 4 python runScript_FI.py -index 0/1                                            
    └── Generates c1_data/, c2_data/ with feature fields                                                               
                                                                                                                       
 2. Symbolic Regression                                                                                                
    └── cd sr_training && python trainModel_SR.py                                                                      
    └── Outputs: results/equation_best.py, pareto_front.json, plots                                                    
                                                                                                                       
 3. Validation                                                                                                         
    └── Compare SR equation vs field inversion truth                                                                   
    └── Metrics: MSE, R², max error                                                                                    
                                                                                                                       
 4. Prediction (Optional)                                                                                              
    └── Use SR equation in predict/trained/ case                                                                       
    └── Compare against baseline SA and reference SST                                                                  
                                                                                                                       
 ---                                                                                                                   
 Key Files to Modify/Create                                                                                            
 ┌────────────────────────────────────┬────────────────────────────┐                                                   
 │                File                │           Action           │                                                   
 ├────────────────────────────────────┼────────────────────────────┤                                                   
 │ train/sr_training/trainModel_SR.py │ CREATE - Main PySR script  │                                                   
 ├────────────────────────────────────┼────────────────────────────┤                                                   
 │ train/sr_training/sr_utils.py      │ CREATE - Utility functions │                                                   
 ├────────────────────────────────────┼────────────────────────────┤                                                   
 │ train/sr_training/README.md        │ CREATE - Documentation     │                                                   
 └────────────────────────────────────┴────────────────────────────┘                                                   
 ---                                                                                                                   
 Verification                                                                                                          
                                                                                                                       
 1. Unit Test: SR script runs without error on synthetic data                                                          
 2. Data Loading: Successfully reads features from c1_data/c2_data                                                     
 3. Training: PySR converges and produces Pareto front                                                                 
 4. Export: Equation exported to all formats (SymPy, LaTeX, Python, C++)                                               
 5. Validation: R² > 0.8 on held-out data (reasonable fit)                                                             
 6. Comparison: Document SR vs NN accuracy tradeoffs                                                                   
                                                                                                                       
 ---                                                                                                                   
 Expected Outputs                                                                                                      
                                                                                                                       
 After running trainModel_SR.py:                                                                                       
                                                                                                                       
 sr_training/results/                                                                                                  
 ├── equation_best.py      # Python callable: def beta_fi(PoD, VoS, chiSA, PSoSS)                                      
 ├── equation_best.tex     # LaTeX equation for documentation                                                          
 ├── pareto_front.json     # All Pareto-optimal equations with metrics                                                 
 ├── pareto_front.png      # Complexity vs accuracy plot                                                               
 ├── predictions.png       # Parity plot (predicted vs true)                                                           
 └── model.pkl             # Pickled PySR model for reuse                                                              
                                                                                                                       
 Success Criteria                                                                                                      
                                                                                                                       
 1. PySR discovers equation with R² > 0.7 on validation set                                                            
 2. Pareto front shows meaningful complexity-accuracy tradeoff                                                         
 3. Best equation is interpretable (complexity < 20)                                                                   
 4. Outputs exported in SymPy, LaTeX, and Python callable formats 
