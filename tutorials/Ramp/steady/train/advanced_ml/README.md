# Advanced ML Tutorial: Feature Selection & Architecture Search for FIML

This tutorial extends the base `Ramp/steady/` FIML pipeline with systematic feature selection, neural network architecture exploration, and a comprehensive guide to the ML fundamentals behind each step.

## Overview

The base tutorial uses 4 hand-picked input features (`PoD`, `VoS`, `PSoSS`, `KoU2`) with a fixed `[50,50]` ReLU architecture. This advanced tutorial:

1. **Expands the feature space** to all 11 available features during field inversion
2. **Ranks features** using Spearman correlation and mutual information
3. **Searches over architectures** (6 hidden-layer configs x 3 activations = 18 combinations)
4. **Trains the best model** and exports it in the same format for coupled NN evaluation

## Prerequisites

- Completed the base `Ramp/steady/` field inversion tutorial (or have reference data)
- OpenFOAM environment loaded (`source $DAFOAM_ROOT_PATH/loadDAFoam.sh`)
- Python packages: `tensorflow`, `scipy`, `scikit-learn` (in addition to the standard DAFoam stack)

## Quick Start

### Step 1: Run Field Inversion with All 11 Features

From the `train/` directory (parent of `advanced_ml/`), run field inversion for both flow conditions. This uses the modified FI script with all 11 features in `inputNames`:

```bash
cd tutorials/Ramp/steady/train/

# Case c1 (U0 = 10 m/s)
mpirun --oversubscribe -np 4 python advanced_ml/runScript_FI.py -task run_driver -index 0

# Case c2 (U0 = 20 m/s)
mpirun --oversubscribe -np 4 python advanced_ml/runScript_FI.py -task run_driver -index 1
```

This produces `c1_data/` and `c2_data/` directories in `train/` containing all 11 feature fields plus the optimized `betaFIOmega` field.

### Step 2: Train with Feature Selection and Architecture Search

```bash
cd advanced_ml/
python trainModel.py
```

This runs three phases automatically:
- **Phase A**: Loads data, computes feature rankings, selects top 6 features
- **Phase B**: Tests 18 architecture combinations (300 epochs each)
- **Phase C**: Retrains the best architecture for 500 epochs

### Common Options

```bash
# Select top 4 features instead of 6
python trainModel.py --n_features 4

# Skip architecture search, use default [20,20] tanh
python trainModel.py --skip_search

# Manually specify features
python trainModel.py --features "PoD,VoS,KoU2,PSoSS"

# Train for more epochs
python trainModel.py --epochs 1000
```

## ML Fundamentals for CFD Users

### Feature Space

DAFoam computes 11 flow-derived features, each normalized to roughly [0, 1] using the form A/(A+B+epsilon):

| Feature | Physical Meaning |
|---------|-----------------|
| **VoS** | Vorticity magnitude relative to strain rate. High in shear layers and vortical regions. |
| **PoD** | Production-to-dissipation ratio of turbulent kinetic energy. Indicates equilibrium (near 1) vs. non-equilibrium turbulence. |
| **chiSA** | Ratio of eddy viscosity to molecular viscosity (SA-specific). Measures turbulence intensity. |
| **pGradStream** | Streamwise pressure gradient indicator. Distinguishes favorable (accelerating) from adverse (separating) gradients. |
| **PSoSS** | Pressure strain relative to shear stress. Captures pressure-strain redistribution effects. |
| **SCurv** | Streamline curvature indicator. Important near curved walls and in turning flows. |
| **UOrth** | Velocity non-orthogonality to wall. Captures flow separation and reattachment. |
| **KoU2** | Turbulent kinetic energy relative to mean kinetic energy. Overall turbulence intensity measure. |
| **ReWall** | Wall-distance-based Reynolds number. Captures near-wall vs. free-stream behavior. |
| **CoP** | Convection-to-production ratio. Indicates turbulence transport effects. |
| **TauoK** | Ratio of Reynolds stress to turbulent kinetic energy. Measures stress anisotropy. |

**Why not use all 11?** More features isn't always better:
- **Curse of dimensionality**: With ~10,000 training samples, a model with too many inputs can overfit by memorizing noise rather than learning physical relationships.
- **Redundant features**: Some features are highly correlated (e.g., `VoS` and `chiSA` may carry similar information). Redundant inputs make training harder without improving accuracy.
- **Generalization**: A model trained on fewer, physically meaningful features is more likely to generalize to unseen flow conditions.

### Feature Selection Methods

This tutorial uses two complementary methods:

**Spearman Rank Correlation** measures the strength of monotonic relationships between each feature and the target beta field. It computes the correlation between the rank-ordered values, so it captures any monotonic trend (linear or not). A Spearman |rho| near 1 means the feature has a strong, consistent monotonic relationship with beta. Advantages: fast, interpretable, robust to outliers. Limitation: misses non-monotonic relationships (e.g., beta peaks at intermediate feature values).

**Mutual Information (MI)** measures general statistical dependence, including nonlinear and non-monotonic relationships. It quantifies how much knowing the feature reduces uncertainty about beta. A high MI score means the feature carries information about beta, regardless of the relationship shape. Advantages: catches complex dependencies that Spearman misses. Limitation: requires more data to estimate accurately, and the absolute values are harder to interpret.

**Combined ranking**: Features are ranked by each method independently, then ranks are summed. This balances both perspectives — a feature that ranks well on both metrics is likely genuinely informative.

### Activation Functions

DAFoam's C++ regression module (`DARegression.C`) supports three activation functions:

**`tanh` (hyperbolic tangent)**
- Output range: [-1, 1]
- Smooth, zero-centered, bounded
- Good default for FIML because the features are already normalized to [0, 1], and beta values are typically O(1)
- Potential issue: gradients saturate for large inputs (vanishing gradient), but this is rarely a problem with normalized features and shallow networks

**`relu` (rectified linear unit)**
- Output range: [0, infinity)
- Fast to compute, enables sparse activations
- Can suffer from "dying ReLU" where neurons permanently output zero if they receive large negative inputs during training
- Common in deep learning, but the unbounded output can cause instability in FIML if beta predictions become very large

**`sigmoid` (logistic function)**
- Output range: (0, 1)
- Smooth, bounded, always positive
- Most susceptible to vanishing gradients (gradient max is 0.25 at input=0)
- Could be useful if beta is expected to always be positive

**Practical guidance**: Start with `tanh`. If training is slow or gets stuck, try `relu`. Use the architecture search to let the data decide.

### Hidden Layer Architecture

The hidden layer configuration controls model capacity — how complex a function the network can represent.

**Width** (neurons per layer): More neurons = more capacity per layer. A single wide layer (e.g., `[50]`) can approximate many functions but may need more parameters than a deeper alternative.

**Depth** (number of layers): More layers enable hierarchical feature composition. A `[20, 20]` network can learn compositions like "high VoS AND low ReWall implies beta > 1" that a single layer would need many more neurons to capture.

**Practical tradeoffs for FIML**:
- With ~10,000 training samples, networks with more than ~500 parameters risk overfitting
- 2 hidden layers is usually sufficient — the underlying physics (turbulence model correction) is smooth and doesn't require deep hierarchical representations
- A `[20, 20]` network with 4 inputs has ~541 parameters; with 11 inputs it has ~681 — still manageable
- Very deep networks (3+ layers) rarely help and can make training unstable

The architecture search tests several configurations and reports both training and validation MSE, making overfitting easy to detect (large gap between the two).

### Regularization

Regularization prevents the model from fitting noise in the training data.

**In the field inversion step**: The `betaVar` objective penalizes large deviations of beta from 1.0 (baseline). This is explicitly a regularization term — without it, the optimizer would find extreme beta values that perfectly match the reference data but have no physical meaning. Tuning the `scale` parameter on `betaVar` controls the smoothness/accuracy tradeoff.

**In the NN training step**: The `validation_split=0.2` withholds 20% of data for validation. If training MSE keeps decreasing but validation MSE starts increasing, the model is overfitting. The architecture search uses validation MSE (not training MSE) to select the best model, which inherently favors models that generalize.

**Early stopping** (not implemented here but easy to add): Stop training when validation loss hasn't improved for N epochs. Add to `model.fit()`:
```python
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
model.fit(..., callbacks=[callback])
```

### Practical Recommendations

1. **Start simple**: Run with `--skip_search` first to get a baseline, then run the full search
2. **Check the feature rankings**: If a feature has near-zero MI and Spearman scores, it's likely noise — don't include it
3. **Watch for overfitting**: If val MSE >> train MSE, reduce model size or add dropout
4. **Validate on unseen conditions**: The true test is whether the NN-augmented model improves predictions on flow conditions not in the training set
5. **Physical sanity checks**: Plot `betaFIOmegaNN` in ParaView — it should be smooth and near 1.0 in well-resolved regions

## Output Files

```
advanced_ml/
  model/                         # TensorFlow SavedModel (use for coupled NN evaluation)
  results/
    feature_analysis.txt         # Feature ranking table with Spearman and MI scores
    architecture_search.txt      # All 18 architectures ranked by validation MSE
    training_summary.txt         # Final model metrics (MSE, R^2, max error)
  betaFIOmegaNN_C1               # Predicted beta field for case c1 (OpenFOAM format)
  betaFIOmegaNN_C2               # Predicted beta field for case c2 (OpenFOAM format)
```

## References

- Duraisamy, K., Iaccarino, G., & Xiao, H. (2019). Turbulence modeling in the age of data. *Annual Review of Fluid Mechanics*, 51, 357-377.
- Parish, E. J., & Duraisamy, K. (2016). A paradigm for data-driven predictive modeling using field inversion and machine learning. *Journal of Computational Physics*, 305, 758-774.
- Holland, J. R., Baeder, J. D., & Duraisamy, K. (2019). Field inversion and machine learning with embedded neural networks: Physics-consistent neural network training. *AIAA Aviation Forum*.
- TensorFlow documentation: https://www.tensorflow.org/api_docs
- scikit-learn feature selection: https://scikit-learn.org/stable/modules/feature_selection.html
