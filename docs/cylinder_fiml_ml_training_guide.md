# Training Neural Networks for Cylinder FIML: A Practical Guide

**Audience:** Mechanical/Aerospace engineers with CFD experience but limited ML background.
**Case:** Flow around a circular cylinder at Re = 3900, Spalart-Allmaras turbulence model.
**Goal:** Train a neural network to predict the correction field beta(x) from local flow features.

---

## Table of Contents

1. [The Big Picture: Why ML for Turbulence Correction?](#1-the-big-picture)
2. [Understanding Your Data: The Beta Field](#2-understanding-your-data)
3. [Spatial Filtering: Not All Cells Are Equal](#3-spatial-filtering)
4. [Feature Engineering: What the Network Sees](#4-feature-engineering)
5. [Model Architecture: Choosing the Right Network](#5-model-architecture)
6. [Loss Functions: What "Accurate" Means](#6-loss-functions)
7. [Training Hyperparameters: Epochs, Batch Size, Learning Rate](#7-training-hyperparameters)
8. [Decoupled Training Walkthrough](#8-decoupled-training-walkthrough)
9. [Coupled Training: Physics in the Loop](#9-coupled-training)
10. [Evaluating Performance](#10-evaluating-performance)
11. [Common Pitfalls and Debugging](#11-common-pitfalls)
12. [Scaling to Larger Meshes](#12-scaling-to-larger-meshes)
13. [Quick Reference](#13-quick-reference)

---

## 1. The Big Picture

### What is FIML?

RANS turbulence models (like Spalart-Allmaras) use simplified equations to approximate
turbulence. These simplifications introduce errors that vary spatially -- the model might
be accurate in attached boundary layers but poor in separation regions or wakes.

**Field Inversion and Machine Learning (FIML)** is a two-stage correction strategy:

1. **Field Inversion (FI):** Given high-fidelity data (experiments or DNS/LES), find a
   spatially-varying correction field beta(x) that, when multiplied into the SA production
   term, makes the RANS solution match the data:

   ```
   Production = Cb1 * S_tilda * nuTilda * beta(x)
   ```

   When beta = 1.0, you recover the standard SA model. Where beta > 1, production is
   amplified (more turbulence). Where beta < 1, it is suppressed.

2. **Machine Learning (ML):** The per-cell beta field from FI is case-specific -- it cannot
   be transferred to a new geometry or flow condition. So we train a neural network:

   ```
   beta = NN(local_flow_features)
   ```

   The features are quantities like vorticity/strain ratio, production/destruction ratio,
   etc. -- things computable at any point in any flow. If the NN learns the right mapping,
   it generalizes to new cases.

### Two Training Approaches

| Approach | What It Does | Pros | Cons |
|----------|-------------|------|------|
| **Decoupled** | Train NN on (features, beta) pairs from FI | Fast (~minutes), simple | No physics enforcement during training |
| **Coupled** | NN sits inside CFD solver; optimizer adjusts weights to match data | Physics-consistent, best accuracy | Expensive (~hours), requires adjoint solver |

This guide covers both, with emphasis on the decoupled approach since that is where most
ML tuning decisions live.

---

## 2. Understanding Your Data

### Where Does the Training Data Come From?

After field inversion (`runScript_FI.py`), each mesh cell has:
- A **beta value** (the target the NN must predict)
- **Flow feature values** (computed from the converged flow solution with that beta)

For the cylinder at Re=3900 with ~25,000 cells, field inversion produces 25,000
(feature_vector, beta) training pairs.

### What Does the Beta Field Look Like?

For cylinder flow, the inverted beta field has a distinctive spatial structure:

```
                    Flow direction -->

                      beta ~ 1.0 (freestream, no correction needed)
                    .........................
                   .                         .
                  .    beta ~ 1.0 (upstream   .
                 .     boundary layer is OK)   .
                 .          ___                .
                 .         /   \    beta > 1   .
                  .       | CYL |   (separation.region, more production needed)
                   .       \___/             .
                    .         |             .
                     .   wake region       .
                      .  beta != 1.0      .
                       .    |            .
                        .   v           .
                         ...............
                    beta ~ 1.0 (far downstream, correction fades)
```

**Key observation:** The vast majority of cells (freestream, far-field) have beta ~ 1.0.
Only cells near the cylinder surface and in the wake have meaningful corrections. This
is the central challenge for ML training on this case.

### Inspecting Your Beta Field

Before training, always visualize the FI output in ParaView:

1. Open the case in ParaView
2. Color by `betaFINuTilda`
3. Check the range -- typical values are [0.5, 2.0] for well-regularized inversions
4. Look for:
   - **Smooth variations** (good) vs. **cell-to-cell oscillations** (noisy -- increase
     regularization `betaVar.scale` in `runScript_FI.py`)
   - **Extreme values** (beta < 0 or beta > 5) suggest weak regularization or poor
     data constraints
   - **Thin layers of correction** vs. **broad regions** -- this tells you how many
     cells carry useful signal

---

## 3. Spatial Filtering: Not All Cells Are Equal

### The Class Imbalance Problem

This is the single most important concept for cylinder FIML training. Consider a ~25,000
cell mesh:

| Region | Approximate Cell Count | Beta Value | Contains Useful Signal? |
|--------|----------------------|------------|------------------------|
| Freestream | ~18,000 (72%) | 1.0 (exactly) | No -- NN should output 1.0 here |
| Near-wake (1-5D downstream) | ~3,000 (12%) | 0.7 - 2.0 | **Yes** |
| Cylinder boundary layer | ~2,000 (8%) | 0.8 - 1.5 | **Yes** |
| Far wake (>5D) | ~2,000 (8%) | ~1.0 | Marginal |

If you train on all 25,000 cells equally, the network sees ~72% "boring" samples where
the answer is trivially 1.0. It will learn to output ~1.0 everywhere and report a low
MSE -- because getting 72% of samples perfectly right is easy. But it will fail on the
~28% of cells that actually matter.

**This is called class imbalance** in ML terminology -- the distribution of your targets
is heavily skewed toward one value.

### Strategy 1: Spatial Filtering (Region Selection)

Train only on cells where beta deviates meaningfully from 1.0:

```python
# After loading all cells
threshold = 0.05  # cells where |beta - 1| > threshold
mask = np.abs(outputs.flatten() - 1.0) > threshold
inputs_filtered = inputs[mask]
outputs_filtered = outputs[mask]

print(f"Kept {mask.sum()} of {len(mask)} cells ({100*mask.sum()/len(mask):.1f}%)")
# For cylinder: expect ~5,000-8,000 cells (20-32%)
```

**How to choose the threshold:**
- `threshold = 0.01`: Very inclusive, keeps most cells with any correction. Low risk
  of discarding useful signal but still includes many near-1.0 cells.
- `threshold = 0.05`: Moderate. Good starting point for cylinder flow.
- `threshold = 0.10`: Aggressive. Only keeps cells with significant correction. Risks
  losing cells in the transition zone between corrected and uncorrected regions.

**Recommendation for the cylinder case:** Start with `threshold = 0.05`. This typically
retains 5,000-8,000 cells from a 25K mesh, focusing on the separation region, near-wake,
and shear layers where the SA model needs correction.

### Strategy 2: Sample Weighting

Instead of discarding cells, assign higher weight to cells with larger corrections:

```python
# Weight proportional to |beta - 1|
weights = np.abs(outputs.flatten() - 1.0)
weights = weights / weights.mean()  # normalize so mean weight = 1

# In TensorFlow:
model.fit(inputs, outputs, sample_weight=weights, ...)
```

This keeps all data but makes the optimizer pay more attention to cells where beta
deviates. The freestream cells still contribute (reinforcing that the NN should output
1.0 for "boring" features) but don't dominate the gradient.

### Strategy 3: Combined Approach (Recommended)

Use mild spatial filtering + sample weighting:

```python
# Step 1: Remove clearly uninteresting cells (freestream)
threshold = 0.02
mask = np.abs(outputs.flatten() - 1.0) > threshold
X = inputs[mask]
y = outputs[mask]

# Step 2: Weight remaining cells
w = np.abs(y.flatten() - 1.0)
w = w / w.mean()

# Step 3: Clip extreme weights to avoid instability
w = np.clip(w, 0.1, 10.0)
```

### What About Geometry-Based Filtering?

An alternative to beta-based filtering is to select cells by geometric region:

```python
# Select cells within a bounding box around the cylinder and wake
# Cylinder center at (0, 0), diameter D = 1.0
x_min, x_max = -1.0 * D, 10.0 * D   # 1D upstream to 10D downstream
y_min, y_max = -2.0 * D,  2.0 * D    # 2D above and below

# If you have cell center coordinates:
geo_mask = (
    (cell_x >= x_min) & (cell_x <= x_max) &
    (cell_y >= y_min) & (cell_y <= y_max)
)
```

This is useful when you want to define the region of interest *a priori* (before seeing
beta), but it may include cells where beta = 1.0 (e.g., in the potential flow region
upstream of the cylinder). The beta-based threshold is generally more targeted.

### Impact on the Current trainModel.py

The existing `trainModel.py` (`tutorials/Cylinder_Re3900/steady_SA/train/tf_training/`)
trains on **all cells** with no filtering:

```python
# Line 41-44: Reads entire field, no filtering
field = np.zeros(nCells)
ofm.readField("betaFINuTilda", "volScalarField", case, field)
output_data = field.reshape(-1, 1)
```

This is the first thing to change for better cylinder performance. See
[Section 8](#8-decoupled-training-walkthrough) for the modified script.

---

## 4. Feature Engineering: What the Network Sees

### The 11 Available Features

DAFoam computes turbulence-aware features in `DARegression.C` (lines 164-351). Each
feature captures a different aspect of the local flow state:

| Feature | Physical Meaning | Formula Pattern | Range |
|---------|-----------------|-----------------|-------|
| **VoS** | Vorticity vs. Strain rate | \|Omega\| / (\|Omega\| + \|S\|) | [0, 1] |
| **PoD** | SA Production vs. Destruction | Prod / (Prod + Destr) | [0, 1] |
| **chiSA** | Turbulent-to-molecular viscosity ratio | nu_t / (nu + nu_t) | [0, 1] |
| **PSoSS** | Pressure gradient vs. Reynolds stress | \|grad_p\| / (\|grad_p\| + \|stress\|) | [0, 1] |
| **pGradStream** | Pressure gradient along streamline | (U . grad_p) / normalize | Real |
| **SCurv** | Streamline curvature | \|U . grad_U\| / (\|U\|^2 + \|U . grad_U\|) | [0, 1] |
| **UOrth** | Flow non-orthogonality | see DARegression.C | [0, 1] |
| **KoU2** | Turbulence intensity | k / (0.5*U^2 + k) | [0, 1] |
| **ReWall** | Wall-distance Reynolds number | sqrt(k)*y / (50*nu + sqrt(k)*y) | [0, 1] |
| **CoP** | TKE convection vs. production | Conv / (Conv + Prod) | [0, 1] |
| **TauoK** | Reynolds stress anisotropy | \|tau\| / (k + \|tau\|) | [0, 1] |

All features use the **ratio normalization** pattern `A / (A + B + epsilon)`, which:
- Maps values to a bounded range (usually [0, 1])
- Is dimensionless (can be compared across cases)
- Is Galilean invariant (independent of reference frame)
- Avoids division by zero (epsilon = 1e-16)

### Which Features for Cylinder Flow?

The tutorial default uses 4 features: **PoD, VoS, chiSA, PSoSS**. These capture:

- **PoD** (Production/Destruction): Directly related to where beta acts. Where the SA
  model over/under-predicts production relative to destruction.
- **VoS** (Vorticity/Strain): Distinguishes rotational flow (shear layers, vortices)
  from irrotational straining. Important in the cylinder wake.
- **chiSA** (Viscosity ratio): Encodes the local turbulence level. High in the wake,
  low in laminar regions.
- **PSoSS** (Pressure gradient / Stress): Captures adverse pressure gradients driving
  separation on the cylinder rear.

**For cylinder flow specifically, consider adding:**
- **SCurv** (Streamline curvature): The flow curves sharply around the cylinder.
  Curvature effects are a known weakness of SA.
- **ReWall** (Wall Reynolds number): Helps distinguish near-wall from far-field
  behavior. Important for separation prediction.

**General guidance on feature count:**
- **3-4 features:** Simpler network, easier to train, less overfitting risk. Start here.
- **5-7 features:** More expressive. Only useful if 4 features give poor results.
- **8+ features:** Diminishing returns. Features become redundant, and the network
  has more ways to overfit.

### Feature Normalization

Even though features are already bounded [0, 1], the `Normalization` layer in TensorFlow
further standardizes them (zero mean, unit variance) before feeding to the network. This
is good practice -- it makes all features contribute equally at initialization, regardless
of their actual distributions.

```python
# trainModel.py lines 59-60: The normalizer adapts to your specific data
normalizer = layers.Normalization(input_shape=[len(features)], axis=None)
normalizer.adapt(inputs)
```

The normalizer computes and stores `mean` and `variance` from your training data. During
inference, it applies: `x_normalized = (x - mean) / sqrt(variance)`.

**Important:** If you apply spatial filtering (Section 3), adapt the normalizer on the
*filtered* data, not the full field. The statistics of features in the wake are different
from the full domain.

### Feature Diagnostics

Enable feature printing in the coupled training config to understand your data:

```python
"printInputInfo": True,  # Prints min/max/mean/std of each feature per iteration
"writeFeatures": True,   # Saves feature fields to disk for ParaView visualization
```

For decoupled training, do this manually:

```python
for i, name in enumerate(features):
    col = inputs[:, i]
    print(f"{name}: min={col.min():.4f}, max={col.max():.4f}, "
          f"mean={col.mean():.4f}, std={col.std():.4f}")
```

Look for:
- **Features with near-zero variance:** The network cannot learn from these -- they
  are approximately constant across all cells. Consider removing them.
- **Features with outliers:** Values far from the mean can dominate training. Check
  if these correspond to real physics (e.g., stagnation point) or numerical artifacts.

---

## 5. Model Architecture: Choosing the Right Network

### What Is a Multilayer Perceptron (MLP)?

An MLP is the simplest type of neural network: layers of neurons connected in sequence.
Each neuron computes:

```
output = activation(w1*x1 + w2*x2 + ... + wN*xN + bias)
```

For FIML, the MLP maps flow features to beta:

```
[PoD, VoS, chiSA, PSoSS] --> Hidden Layer 1 --> Hidden Layer 2 --> [beta]
     4 inputs                  N neurons          N neurons        1 output
```

### Architecture Decisions

#### Number of Hidden Layers

| Layers | Network Type | Capacity | Recommendation |
|--------|-------------|----------|----------------|
| 1 | Shallow | Limited -- can only learn simple mappings | Only for very simple beta fields |
| 2 | Standard | Good balance of capacity and trainability | **Default choice for FIML** |
| 3+ | Deep | High capacity but harder to train, overfitting risk | Rarely needed for FIML |

**Why 2 layers is the sweet spot:** The beta field is a smooth, continuous function of
the features. Two hidden layers can approximate any continuous function (universal
approximation theorem). Adding more layers increases the risk of overfitting to noise
in the FI data without improving the underlying mapping.

#### Neurons Per Layer

| Neurons | Total Parameters (4 inputs, 2 layers) | Capacity | Risk |
|---------|---------------------------------------|----------|------|
| 10 | 10*(4+1) + 10*(10+1) + 1*(10+1) = 171 | Low | Underfitting |
| 20 | 20*(4+1) + 20*(20+1) + 1*(20+1) = 541 | Medium | **Good default** |
| 50 | 50*(4+1) + 50*(50+1) + 1*(50+1) = 2,851 | High | Overfitting risk |
| 100 | 100*(4+1) + 100*(100+1) + 1*(100+1) = 10,701 | Very high | Very likely to overfit |

**Counting parameters:** For a layer with `n_in` inputs and `n_out` neurons:
`params = n_out * (n_in + 1)` (the +1 is the bias term per neuron).

**Rule of thumb:** Your number of *effective* training samples should be at least
10-50x your parameter count. With ~7,000 filtered cells and 541 parameters, the ratio
is ~13x -- workable but not generous. With 2,851 parameters (50-neuron layers), the
ratio drops to ~2.5x -- almost certain overfitting.

**Recommendation for cylinder (~25K cells, ~7K after filtering):**
- Start with `[20, 20]` (541 params) -- this is the DAFoam coupled training default
- If underfitting (training loss plateaus high), try `[30, 30]` (1,021 params)
- The existing `trainModel.py` uses `[50, 50]` (2,851 params) -- this is likely
  **too large** for a filtered cylinder dataset. Consider reducing.

#### Activation Functions

| Function | Formula | Range | Properties |
|----------|---------|-------|------------|
| **tanh** | (e^x - e^{-x}) / (e^x + e^{-x}) | (-1, 1) | Smooth, centered at zero. Default for coupled training. |
| **ReLU** | max(0, x) | [0, inf) | Fast, simple. Default for decoupled (TensorFlow). Can "die" (output stuck at 0). |
| **sigmoid** | 1 / (1 + e^{-x}) | (0, 1) | Output bounded. Less common for hidden layers. |

**For decoupled training (TensorFlow):** ReLU is fine. It trains faster and works well
with the Adam optimizer.

**For coupled training (DAFoam adjoint):** tanh is preferred. The DAFoam C++ implementation
uses tanh by default because its bounded output helps adjoint stability. The adjoint
(gradient computation) flows backward through the network; unbounded activations like
ReLU can cause gradient issues in the coupled CFD-NN system.

**Practical note:** The activation function usually matters less than the architecture
size and data quality. If you are unsure, use tanh for consistency with the coupled setup.

### Decoupled vs. Coupled Architecture

| Setting | Decoupled (`trainModel.py`) | Coupled (DAFoam `regressionModel`) |
|---------|---------------------------|-------------------------------------|
| Layers | `[50, 50]` | `[20, 20]` |
| Activation | ReLU | tanh |
| Normalizer | TF `Normalization` layer | `inputShift` / `inputScale` |
| Output processing | None (linear output) | `outputShift=1.0`, `outputScale=1.0` |
| Bounds | None | `outputUpperBound=10`, `outputLowerBound=-10` |

**Important mismatch:** The decoupled model uses a TF `Normalization` layer which
computes mean/variance from data. The coupled model uses manual `inputShift` and
`inputScale`. If you train decoupled and later want to use those weights in the coupled
framework, you must transfer the normalization parameters:

```python
# After decoupled training:
mean = normalizer.mean.numpy()      # shape: (4,)
var = normalizer.variance.numpy()   # shape: (4,)

# Convert to DAFoam format:
inputShift = (-mean / np.sqrt(var + 1e-7)).tolist()
inputScale = (1.0 / np.sqrt(var + 1e-7)).tolist()
```

---

## 6. Loss Functions: What "Accurate" Means

### Mean Squared Error (MSE) -- Decoupled Training

The decoupled `trainModel.py` uses the simplest loss function:

```python
# trainModel.py line 74
loss = "mean_squared_error"

# Equivalent to:
# L = (1/N) * sum_i (beta_predicted[i] - beta_true[i])^2
```

MSE treats all errors equally -- a 0.1 error in the freestream (where beta should be
1.0) counts the same as a 0.1 error in the separation region (where beta might be 0.7).

### Weighted MSE -- Better for Cylinder

With spatial filtering or sample weighting (Section 3), the effective loss becomes:

```
L = (1/N) * sum_i  w[i] * (beta_predicted[i] - beta_true[i])^2
```

where `w[i]` is higher for cells with significant corrections. This directly addresses
the class imbalance problem.

### Multi-Objective Loss -- Coupled Training

The coupled training in DAFoam uses a physics-informed loss that is fundamentally
different from the decoupled MSE:

```python
# From runScript_FI.py (field inversion, analogous structure for coupled NN training)
J = CpVar + 0.01 * betaVar
```

Where:
- **CpVar** = surface pressure error: `sum_faces (p_sim - p_exp)^2`
- **betaVar** = regularization: `0.01 * sum_cells (beta - 1)^2`

**Key difference:** The coupled loss does not compare `beta_predicted` to `beta_true`.
It compares the *flow solution produced by that beta* to experimental data. The network
learns to produce a beta field that makes the CFD output correct, not necessarily a
beta field that matches the FI result cell-by-cell.

This is why coupled training produces better physics -- even if the per-cell beta differs
from the FI result, the resulting flow field matches the data.

### Regularization in the Loss Function

Regularization prevents the network from learning extreme, non-physical corrections.
There are two types relevant to FIML:

**1. L2 regularization on weights (weight decay):**

```python
# Penalize large NN weights to prevent overfitting
layers.Dense(units=50, activation="relu",
             kernel_regularizer=tf.keras.regularizers.l2(1e-4))
```

This adds `lambda * sum(w^2)` to the loss. It makes the network prefer simple,
smooth mappings over complex ones. Recommended for cylinder training where the filtered
dataset is small.

**2. Beta regularization (in coupled training):**

```python
# From DAFoam config (runScript_FI.py line 78-85)
"betaVar": {
    "type": "variance",
    "scale": 0.01,  # weight of regularization term
    "varName": "betaFINuTilda",
    ...
}
```

This penalizes `||beta - 1||^2` -- keeping the correction close to the unmodified SA
model. The `scale` parameter controls the trade-off: larger values produce smoother
beta fields (better for ML) but may not match data as closely.

---

## 7. Training Hyperparameters: Epochs, Batch Size, Learning Rate

### What Are These?

If you are new to ML training, think of it as an optimization problem (you are
familiar with those from CFD):

- **Epoch:** One complete pass through all training samples. Analogous to one
  "iteration" through the entire dataset.
- **Batch size:** Number of samples processed before updating the network weights.
  The dataset is split into batches; each batch computes a loss and gradient update.
- **Learning rate:** Step size for the optimizer. Analogous to a relaxation factor
  in iterative solvers.

### Epochs

```
                  Loss
                   |
                   |  *
                   |    *
                   |      * *
                   |          * * *
             train |              * * * * * * *          (converged)
                   |                    * * * * * * *    (validation)
                   |
                   +------------------------------------> Epochs
                   0         100       200       300
```

**How many epochs?**

| Dataset Size | Recommended Epochs | Why |
|-------------|-------------------|-----|
| ~5,000 (filtered) | 500 - 1,000 | Small dataset needs more passes |
| ~25,000 (full) | 200 - 500 | More data per epoch, converges faster |

The current `trainModel.py` uses **500 epochs** -- reasonable for ~25K cells but
may need to increase to 800-1000 if you filter down to ~5K cells.

**When to stop:**
- Watch the **validation loss** (not training loss). TensorFlow prints both.
- If validation loss stops decreasing for 50+ epochs, training is done.
- If validation loss *increases* while training loss decreases, the network is
  **overfitting** (memorizing data rather than learning the pattern).

**Early stopping (recommended):**

```python
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',     # watch validation loss
    patience=50,            # wait 50 epochs for improvement
    restore_best_weights=True  # go back to the best model
)

model.fit(..., callbacks=[callback])
```

This automatically stops training when the network starts overfitting, and restores
the best weights.

### Batch Size

| Batch Size | Gradient Quality | Training Speed | Memory |
|-----------|-----------------|----------------|--------|
| 32 | Noisy (stochastic) | Slow per epoch | Low |
| 128 | Moderate | Balanced | Moderate |
| 500 | Smooth | Fast per epoch | Higher |
| Full dataset | Exact (batch gradient descent) | Fastest per epoch | Highest |

The current `trainModel.py` uses **batch_size=500** for ~25K cells, giving ~50 batches
per epoch (with 80% training split = 20,000 samples / 500 = 40 batches).

**For filtered data (~5,000-8,000 cells):**
- `batch_size=500` gives ~8-13 batches -- still reasonable
- `batch_size=128` gives ~31-50 batches -- finer gradient updates, often better
- `batch_size=32` might be too noisy for this smooth problem

**Recommendation:** Use `batch_size=128` for filtered cylinder data. If training is
unstable (loss jumps around), increase to 256. If training is too slow to converge,
decrease to 64.

**Intuition:** Smaller batches add "noise" to the gradient, which can actually help
escape poor local minima. But for FIML, the mapping is typically smooth and a
single local minimum is likely, so moderate batch sizes work well.

### Learning Rate

The learning rate is the most impactful single hyperparameter.

```python
# trainModel.py line 73
optimizer=tf.optimizers.Adam(learning_rate=0.001)
```

The Adam optimizer adapts the learning rate per-parameter, so the initial value is
less critical than for simpler optimizers (like SGD). Still:

| Learning Rate | Effect |
|--------------|--------|
| 0.01 | Aggressive. Fast initial progress but may overshoot. |
| **0.001** | Standard default. Works for most FIML cases. |
| 0.0001 | Conservative. Use if training is unstable. |

**Learning rate scheduling (for better convergence):**

```python
# Reduce learning rate when validation loss plateaus
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,      # halve the learning rate
    patience=30,     # wait 30 epochs
    min_lr=1e-6
)

model.fit(..., callbacks=[lr_scheduler, early_stopping])
```

This starts with the standard 0.001 rate for broad exploration, then automatically
reduces it for fine-tuning as training converges.

### Validation Split

```python
# trainModel.py line 81
validation_split=0.2  # 80% train, 20% validation
```

The validation set is data the network **never trains on** -- it only evaluates
performance on this set after each epoch. This is your overfitting detector.

**0.2 (20%) is standard.** For small filtered datasets (~5K cells), you might reduce
to 0.1 to keep more training data, but you lose some overfitting detection ability.

**Important caveat for spatial data:** Random splitting (TensorFlow's default) mixes
cells from all spatial regions into both train and validation sets. Adjacent cells
have very similar features and beta values, so the "validation" set is not truly
independent -- it leaks information. For a more honest assessment:

```python
# Spatial validation: hold out a region (e.g., downstream wake)
# This tests actual generalization, not interpolation between neighbors
val_mask = cell_x > 5.0 * D  # hold out far wake
train_mask = ~val_mask

X_train, y_train = inputs[train_mask], outputs[train_mask]
X_val, y_val = inputs[val_mask], outputs[val_mask]

model.fit(X_train, y_train, validation_data=(X_val, y_val), ...)
```

This is more stringent but gives a realistic estimate of how well the NN will
perform on unseen flow regions.

---

## 8. Decoupled Training Walkthrough

### Modified trainModel.py for Cylinder

Below is an improved version of the training script incorporating the recommendations
from Sections 3-7. Changes from the original are annotated with comments.

```python
#!/usr/bin/env python
"""
Improved MLP training for cylinder FIML.
Key changes from original trainModel.py:
  1. Spatial filtering (remove freestream cells)
  2. Sample weighting (emphasize large corrections)
  3. Smaller network (avoid overfitting on filtered data)
  4. Learning rate scheduling and early stopping
  5. L2 regularization
  6. Diagnostic plots
"""

import numpy as np
from mpi4py import MPI
from pyofm import PYOFM
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

# ====================================================================
# Configuration -- ADJUST THESE FOR YOUR CASE
# ====================================================================
nCells = 25000           # <-- Update to your actual cell count
cases = ["c1_data"]      # Single case for cylinder
features = ["PoD", "VoS", "chiSA", "PSoSS"]

# Spatial filtering
BETA_THRESHOLD = 0.05    # Keep cells where |beta - 1| > this value
USE_SAMPLE_WEIGHTS = True

# Architecture
HIDDEN_LAYERS = [20, 20]     # Match DAFoam coupled default
ACTIVATION = "tanh"          # Match DAFoam coupled default
L2_REG = 1e-4                # Weight regularization

# Training
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
PATIENCE_EARLY_STOP = 80     # Stop if no improvement for this many epochs
PATIENCE_LR_REDUCE = 30      # Halve LR if no improvement for this many epochs

# ====================================================================
# Load data
# ====================================================================
ofm = PYOFM(comm=MPI.COMM_WORLD)

inputs = None
outputs = None

for case in cases:
    input_data = []
    for feature in features:
        field = np.zeros(nCells)
        ofm.readField(feature, "volScalarField", case, field)
        input_data.append(field)
    input_data = np.asarray(input_data).transpose()  # (nCells, nFeatures)

    field = np.zeros(nCells)
    ofm.readField("betaFINuTilda", "volScalarField", case, field)
    output_data = field.reshape(-1, 1)

    if inputs is None:
        inputs = np.copy(input_data)
        outputs = np.copy(output_data)
    else:
        inputs = np.concatenate((inputs, input_data), axis=0)
        outputs = np.concatenate((outputs, output_data), axis=0)

print(f"Raw data: {inputs.shape[0]} cells, {inputs.shape[1]} features")

# ====================================================================
# Feature diagnostics
# ====================================================================
for i, name in enumerate(features):
    col = inputs[:, i]
    print(f"  {name}: min={col.min():.4f}, max={col.max():.4f}, "
          f"mean={col.mean():.4f}, std={col.std():.4f}")

print(f"  beta: min={outputs.min():.4f}, max={outputs.max():.4f}, "
      f"mean={outputs.mean():.4f}, std={outputs.std():.4f}")

# ====================================================================
# Spatial filtering
# ====================================================================
deviation = np.abs(outputs.flatten() - 1.0)
mask = deviation > BETA_THRESHOLD

print(f"\nSpatial filtering (threshold={BETA_THRESHOLD}):")
print(f"  Kept {mask.sum()} of {len(mask)} cells ({100*mask.sum()/len(mask):.1f}%)")

inputs_filtered = inputs[mask]
outputs_filtered = outputs[mask]

# ====================================================================
# Sample weights (optional)
# ====================================================================
if USE_SAMPLE_WEIGHTS:
    weights = np.abs(outputs_filtered.flatten() - 1.0)
    weights = weights / weights.mean()  # Normalize mean to 1
    weights = np.clip(weights, 0.1, 10.0)  # Prevent extreme weights
    print(f"  Sample weights: min={weights.min():.2f}, max={weights.max():.2f}")
else:
    weights = None

# ====================================================================
# Build model
# ====================================================================
# Normalization layer adapts to FILTERED data
normalizer = layers.Normalization(
    input_shape=[len(features)],
    axis=None,
)
normalizer.adapt(inputs_filtered)

# Build the sequential model
layer_list = [normalizer]
for n_neurons in HIDDEN_LAYERS:
    layer_list.append(
        layers.Dense(
            units=n_neurons,
            activation=ACTIVATION,
            kernel_regularizer=regularizers.l2(L2_REG),
        )
    )
layer_list.append(layers.Dense(units=1))  # Linear output

model = tf.keras.Sequential(layer_list)
model.summary()

n_params = model.count_params()
n_samples = inputs_filtered.shape[0]
print(f"\nParameter-to-sample ratio: {n_samples}/{n_params} = {n_samples/n_params:.1f}x")
if n_samples / n_params < 10:
    print("  WARNING: Ratio < 10x. High overfitting risk. Consider reducing neurons.")

# ====================================================================
# Compile
# ====================================================================
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mean_squared_error",
)

# ====================================================================
# Callbacks
# ====================================================================
cb_early = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE_EARLY_STOP,
    restore_best_weights=True,
    verbose=1,
)

cb_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=PATIENCE_LR_REDUCE,
    min_lr=1e-6,
    verbose=1,
)

# ====================================================================
# Train
# ====================================================================
history = model.fit(
    inputs_filtered,
    outputs_filtered,
    sample_weight=weights,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[cb_early, cb_lr],
    verbose=1,
)

# ====================================================================
# Evaluate
# ====================================================================
# MSE on filtered data (what we trained on)
pred_filtered = model.predict(inputs_filtered, verbose=0)[:, 0]
mse_filtered = np.mean((outputs_filtered.flatten() - pred_filtered) ** 2)

# MSE on ALL cells (the full picture)
pred_all = model.predict(inputs, verbose=0)[:, 0]
mse_all = np.mean((outputs.flatten() - pred_all) ** 2)

# MSE only on freestream (should be near zero -- NN should output ~1.0)
mse_freestream = np.mean((outputs[~mask].flatten() - pred_all[~mask]) ** 2)

print(f"\n{'='*50}")
print(f"Final MSE (filtered cells):   {mse_filtered:.6f}")
print(f"Final MSE (all cells):        {mse_all:.6f}")
print(f"Final MSE (freestream only):  {mse_freestream:.6f}")

# ====================================================================
# Diagnostic plots
# ====================================================================
# 1. Training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.semilogy(history.history["loss"], label="Train")
ax1.semilogy(history.history["val_loss"], label="Validation")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (log scale)")
ax1.set_title("Training History")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Parity plot (predicted vs true)
ax2.scatter(outputs_filtered.flatten(), pred_filtered, s=1, alpha=0.3)
ax2.plot([outputs_filtered.min(), outputs_filtered.max()],
         [outputs_filtered.min(), outputs_filtered.max()],
         "r--", label="Perfect prediction")
ax2.set_xlabel("Beta (FI truth)")
ax2.set_ylabel("Beta (NN predicted)")
ax2.set_title("Parity Plot (filtered cells)")
ax2.legend()
ax2.set_aspect("equal")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_diagnostics.png", dpi=150)
print("Saved training_diagnostics.png")

# 3. Beta histogram comparison
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(outputs_filtered.flatten(), bins=50, alpha=0.5, label="FI truth", density=True)
ax.hist(pred_filtered, bins=50, alpha=0.5, label="NN predicted", density=True)
ax.set_xlabel("Beta value")
ax.set_ylabel("Density")
ax.set_title("Beta Distribution: FI vs NN")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("beta_distribution.png", dpi=150)
print("Saved beta_distribution.png")

# ====================================================================
# Save model
# ====================================================================
model.save("model")
print("Model saved to ./model")

# Write full predicted beta field for ParaView visualization
ofm.writeField("betaFINuTildaNN", "volScalarField", pred_all[:nCells])
print("Predicted beta field written to betaFINuTildaNN")
```

### What Changed and Why

| Change | Original | Improved | Reason |
|--------|----------|----------|--------|
| Spatial filtering | None | threshold=0.05 | Remove ~72% freestream cells that teach nothing |
| Sample weights | None | w = \|beta-1\| | Emphasize cells with significant corrections |
| Architecture | [50, 50] = 2,851 params | [20, 20] = 541 params | Avoid overfitting on smaller filtered dataset |
| Activation | ReLU | tanh | Consistency with coupled DAFoam framework |
| L2 regularization | None | 1e-4 | Prevent overfitting |
| Early stopping | None | patience=80 | Stop before overfitting |
| LR scheduling | Fixed 0.001 | ReduceLROnPlateau | Fine-tune in later epochs |
| Batch size | 500 | 128 | Better gradient updates for filtered dataset |
| Diagnostics | Print MSE only | Plots + multi-region MSE | Understand performance |

---

## 9. Coupled Training: Physics in the Loop

### How It Differs from Decoupled

In coupled training, the neural network lives *inside* the CFD solver. Each "iteration"
of the optimizer:

1. **Forward pass:** The NN produces a beta field from flow features
2. **Primal solve:** The CFD solver converges with this beta field
3. **Objective evaluation:** Compare the CFD solution to experimental data (Cp, wake
   velocities, etc.)
4. **Adjoint solve:** Compute dJ/d(NN_weights) via reverse-mode automatic differentiation
   through the entire CFD solver
5. **Weight update:** The optimizer adjusts NN weights to reduce the objective

```
                    NN Weights (design variables)
                         |
                         v
        Features <-- [Neural Network] --> beta(x)
                         |
                         v
                  [CFD Solver (SIMPLE)]
                         |
                         v
                  [Flow Solution: U, p, nuTilda]
                         |
                         v
                  [Objective: ||Cp_sim - Cp_exp||^2]
                         |
                         v
                  [Adjoint: dJ/d(weights)]
                         |
                         v
                  [Optimizer: update weights]
```

### Configuration for Coupled Cylinder Training

The coupled training script would look similar to `runScript_FI.py` (from
`tutorials/Cylinder_Re3900/steady_SA/train/`) but with `regressionModel` active and
NN parameters as design variables instead of per-cell beta:

```python
daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    ...
    "regressionModel": {
        "active": True,
        "model1": {
            "modelType": "neuralNetwork",
            "inputNames": ["PoD", "VoS", "chiSA", "PSoSS"],
            "outputName": "betaFINuTilda",
            "hiddenLayerNeurons": [20, 20],
            "inputShift": [0.0, 0.0, 0.0, 0.0],
            "inputScale": [1.0, 1.0, 1.0, 1.0],
            "outputShift": 1.0,     # Baseline: NN output 0 -> beta = 1
            "outputScale": 1.0,
            "activationFunction": "tanh",
            "defaultOutputValue": 1.0,
            "outputUpperBound": 10.0,
            "outputLowerBound": -10.0,
            "writeFeatures": True,
        },
    },
    "inputInfo": {
        "model1": {
            "type": "regressionPar",
            "components": ["solver", "function"],
        },
    },
    "function": {
        "CpVar": {
            "type": "variance",
            "source": "patchToFace",
            "patches": ["cylinder"],
            "scale": 1.0,
            "mode": "surface",
            "varName": "p",
            "varType": "scalar",
        },
        "betaVar": {
            "type": "variance",
            "source": "allCells",
            "scale": 0.01,
            "mode": "field",
            "varName": "betaFINuTilda",
            "varType": "scalar",
        },
    },
}
```

Key differences from the FI script:
- **`inputInfo`** uses `"type": "regressionPar"` (NN weights) instead of
  `"type": "field"` (per-cell beta)
- **`regressionModel`** is active -- the NN computes beta from features during the
  primal solve
- **Design variables** are ~541 NN parameters, not ~25,000 cell values

### Why You Don't Need Spatial Filtering in Coupled Training

The coupled loss (CpVar) only cares about **surface pressure on the cylinder**. Cells
far from the cylinder contribute negligibly to Cp. The adjoint solver automatically
computes which cells are important through the sensitivity dJ/d(beta) -- cells in the
separation zone and near-wake have large sensitivities; freestream cells have near-zero
sensitivities.

The optimizer sees these sensitivities (via dJ/d(weights)) and naturally focuses the
NN's capacity on the important regions. Spatial filtering is built into the physics.

### Multi-Case Coupled Training

For better generalization, train on multiple cases simultaneously (different Re, angles
of attack, etc.):

```python
# In the OpenMDAO problem setup
cases = [
    {"name": "c1", "Re": 3900, "U0": 0.039},
    {"name": "c2", "Re": 10000, "U0": 0.1},
]

# Each case uses the SAME NN weights but different flow conditions
# Total objective = sum of per-case objectives
```

This forces the NN to learn a generalizable mapping, not one tuned to a single case.
See `tutorials/Ramp/steady_SA/train/runScript.py` for a working multi-case example.

### When to Use Coupled vs. Decoupled

| Criterion | Choose Decoupled | Choose Coupled |
|-----------|-----------------|----------------|
| Quick prototyping | Yes | No |
| Limited compute | Yes | No |
| Best physics accuracy | No | Yes |
| Multiple cases | Manual per-case | Built-in multi-case |
| Feature/architecture exploration | Yes (fast iteration) | No (too slow) |
| Final production model | No | Yes |
| Interpretability/symbolic regression | Use decoupled for PySR | N/A |

**Practical workflow:** Start decoupled for architecture selection and feature screening.
Once you find a good configuration, switch to coupled training for the final model.

---

## 10. Evaluating Performance

### Metrics for Decoupled Training

**1. MSE on filtered cells (primary metric):**

```python
mse = np.mean((beta_true[mask] - beta_predicted[mask]) ** 2)
```

Target: < 0.01 for well-inverted cylinder data. Below 0.005 is excellent.

**2. R-squared (coefficient of determination):**

```python
ss_res = np.sum((beta_true - beta_predicted) ** 2)
ss_tot = np.sum((beta_true - beta_true.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
```

- R^2 = 1.0: Perfect prediction
- R^2 = 0.0: Predicts the mean everywhere
- R^2 < 0: Worse than predicting the mean

Target: R^2 > 0.85 on filtered cells. Above 0.95 is excellent.

**3. Maximum absolute error:**

```python
max_err = np.max(np.abs(beta_true - beta_predicted))
```

Important for safety -- even if MSE is low, a few extreme mispredictions can cause
the CFD solver to diverge. Target: max_err < 1.0.

**4. Parity plot:**

The scatter plot of predicted vs. true beta should cluster around the y = x diagonal.
Look for:
- **Systematic bias:** Points consistently above/below the diagonal
- **Heteroscedasticity:** Errors increasing with beta magnitude
- **Outlier clusters:** Groups of cells the NN consistently gets wrong

### Metrics for Coupled Training

The coupled loss is physics-based, so evaluate the *flow solution*, not beta itself:

**1. Cp error vs. experiment:**

```python
# Compare surface pressure coefficient
Cp_error = np.mean((Cp_sim - Cp_exp) ** 2)
```

This is the actual goal of FIML. The beta field is a means, not an end.

**2. Wake velocity profiles:**

Compare streamwise velocity (Ux) at wake cross-sections against experimental data.
This checks if the NN-corrected model predicts the right wake recovery.

**3. Integrated forces:**

Drag coefficient (Cd) should improve from baseline SA toward the experimental value.

### Diagnostic Checklist

After training, verify:

- [ ] Training and validation loss curves converge (no diverging gap = no overfitting)
- [ ] Parity plot shows points near the diagonal (good fit)
- [ ] Beta histogram (NN) matches beta histogram (FI) (correct distribution)
- [ ] Freestream MSE is very small (NN outputs ~1.0 outside the correction region)
- [ ] When plugged back into CFD: Cp improves over baseline SA
- [ ] No extreme beta values cause solver divergence

---

## 11. Common Pitfalls and Debugging

### Pitfall 1: Training on All Cells Without Filtering

**Symptom:** Low overall MSE but poor performance in the wake. Parity plot shows a
dense cluster at (1.0, 1.0) with scattered points elsewhere.

**Cause:** The 72% freestream cells dominate the loss. The network learns "always
output 1.0" because that minimizes error on most cells.

**Fix:** Apply spatial filtering (Section 3). Verify by checking MSE separately on
filtered vs. freestream cells.

### Pitfall 2: Overfitting (Network Too Large)

**Symptom:** Training loss very low but validation loss much higher. Parity plot looks
good on training data but poor on validation.

**Cause:** The 2,851 parameters of [50, 50] is too many for ~5,000 filtered cells.
The network memorizes the training data including noise.

**Fix:**
- Reduce to `[20, 20]` (541 parameters)
- Add L2 regularization (`kernel_regularizer=l2(1e-4)`)
- Add early stopping
- Increase data (add more cases if available)

### Pitfall 3: Noisy Beta Field from FI

**Symptom:** Poor ML accuracy even with correct architecture and filtering. Parity
plot shows high scatter. Features at similar values map to very different beta values.

**Cause:** The field inversion produced a noisy, non-smooth beta field -- typically
from too-weak regularization (`betaVar.scale` too low) or too-few data constraints.

**Fix:** Go back to field inversion:
- Increase `betaVar.scale` from 0.01 to 0.05 or 0.1
- Add more data constraints (wake velocity probes, multiple Cp locations)
- The resulting beta field should be smooth and monotonic in the wake

### Pitfall 4: Feature Dead Zones

**Symptom:** One or more features have near-zero variance across all cells.

**Cause:** For the cylinder at this Re, some features may be nearly constant. A constant
feature cannot help predict anything -- it just adds a parameter for the network to waste.

**Fix:** Check feature statistics (Section 4). Remove features with `std < 0.01` from
the input list. A network with 3 informative features will outperform one with 4 features
where one is dead.

### Pitfall 5: Normalization Mismatch Between Decoupled and Coupled

**Symptom:** Decoupled model shows good training metrics but diverges or gives wrong
results when used in the coupled DAFoam framework.

**Cause:** TensorFlow's `Normalization` layer applies `(x - mean) / sqrt(var)` while
DAFoam's coupled framework uses `(x + inputShift) * inputScale`. If these are not
equivalent, the NN sees different inputs in each context.

**Fix:** Transfer normalization parameters explicitly:

```python
mean = normalizer.mean.numpy()
var = normalizer.variance.numpy()

# DAFoam equivalent:
inputScale = 1.0 / np.sqrt(var + 1e-7)
inputShift = -mean  # then multiplied by inputScale
```

### Pitfall 6: Learning Rate Too High

**Symptom:** Loss oscillates wildly or increases after initial decrease.

**Fix:** Reduce from 0.001 to 0.0001. Or use `ReduceLROnPlateau` callback.

### Pitfall 7: Beta Bounds Too Tight

**Symptom:** Converged beta values "clamp" at the bounds (-5 or +10 in many cells).

**Cause:** In field inversion, the optimizer wants larger corrections but hits the
imposed bounds. The resulting beta field has flat regions at the limits -- these are
artificial and the NN cannot learn them well.

**Fix:** Widen bounds in `runScript_FI.py` (e.g., `lower=-10, upper=20`) or, more
likely, check if your experimental data is consistent with the CFD setup (units,
reference frame, etc.).

---

## 12. Scaling to Larger Meshes

### Medium Meshes (~50K cells)

Most of the guidance above applies directly. Key adjustments:

- **Filtered dataset:** Expect ~10,000-15,000 cells after filtering (same percentage).
  This supports slightly larger networks: `[30, 30]` (1,021 params) with 10-15x ratio.
- **Batch size:** 256 works well. Enough samples for stable gradients.
- **Training time:** ~2-5 minutes on CPU with TensorFlow. No GPU needed.
- **Feature computation:** The coupled DAFoam solver handles all feature computation on
  the mesh. Decoupled training reads pre-computed feature fields with `ofm.readField()`
  so mesh size mainly affects I/O time (negligible).

### Fine Meshes (~100K+ cells)

- **Filtered dataset:** ~25,000-30,000 cells. Now you have plenty of data.
- **Architecture:** Can safely use `[50, 50]` (2,851 params) with ~10x ratio. Even
  `[30, 30, 30]` (3 layers) becomes viable.
- **Batch size:** 512 or 1024 for faster epoch time.
- **Data loading:** OpenFOAM field files become large. Consider using binary format
  for faster I/O (`writeFormat binary` in `controlDict`).
- **Spatial filtering becomes even more important:** With 100K cells, possibly 80K+
  are freestream. Training on all cells without filtering is almost guaranteed to fail.
- **Consider mini-batch sampling from the region of interest** -- rather than loading
  all cells into memory, sample 10K-20K cells from the filtered region each epoch.

### 3D Meshes (~1M+ cells)

For 3D cylinder (spanwise resolution) or complex 3D geometries:

- **Memory:** The full training arrays may not fit in RAM. Use TensorFlow's `Dataset`
  API for batched loading.
- **Architecture:** Same `[20, 20]` to `[50, 50]` range. The complexity of the beta
  mapping does not increase with mesh size (it is a local, feature-based mapping).
- **Feature computation:** Can become expensive in coupled training due to gradient
  computation through all cells. Consider subset-based training where only cells near
  the body contribute to the objective.

---

## 13. Quick Reference

### Recommended Settings for Cylinder Re=3900 (~25K cells)

```python
# Spatial filtering
BETA_THRESHOLD = 0.05      # Keep ~7,000 cells

# Architecture
HIDDEN_LAYERS = [20, 20]   # 541 parameters
ACTIVATION = "tanh"
L2_REG = 1e-4

# Training
EPOCHS = 1000              # With early stopping
BATCH_SIZE = 128
LEARNING_RATE = 0.001      # With ReduceLROnPlateau
VALIDATION_SPLIT = 0.2
PATIENCE_EARLY_STOP = 80
PATIENCE_LR_REDUCE = 30

# Features
FEATURES = ["PoD", "VoS", "chiSA", "PSoSS"]
```

### Decision Flowchart

```
Start
  |
  v
Run Field Inversion (runScript_FI.py)
  |
  v
Visualize beta field in ParaView
  |
  +--> Noisy/oscillatory? --> Increase betaVar.scale, re-run FI
  |
  v
Smooth beta field obtained
  |
  v
Check feature statistics (print min/max/std)
  |
  +--> Any feature with std ~ 0? --> Remove that feature
  |
  v
Apply spatial filtering (threshold = 0.05)
  |
  v
Train with [20, 20], tanh, LR=0.001
  |
  v
Check training/validation curves
  |
  +--> Val loss >> Train loss? --> Overfitting: add L2 reg, reduce neurons
  |
  +--> Both loss high? --> Underfitting: increase neurons or add features
  |
  v
Check parity plot and beta histogram
  |
  +--> Good? --> Validate in CFD (run augmented primal)
  |
  +--> Poor in specific region? --> Check FI quality in that region
  |
  v
Cp improvement over baseline? --> Done!
```

### Key File Locations in DAFoam

| Component | Path | Purpose |
|-----------|------|---------|
| NN forward pass (C++) | `src/adjoint/DARegression/DARegression.C` | Feature computation, network evaluation |
| NN parameter exposure | `src/adjoint/DAInput/DAInputRegressionPar.C` | Maps optimizer vars to NN weights |
| Variance objective | `src/adjoint/DAFunction/DAFunctionVariance.C` | Data mismatch loss (surface/field/probe) |
| SA beta integration | `src/adjoint/DAModel/DATurbulenceModel/DASpalartAllmaras.C:457` | Beta multiplies production term |
| Cylinder FI script | `tutorials/Cylinder_Re3900/steady_SA/train/runScript_FI.py` | Field inversion setup |
| Cylinder ML script | `tutorials/Cylinder_Re3900/steady_SA/train/tf_training/trainModel.py` | Decoupled TF training |
| Ramp coupled training | `tutorials/Ramp/steady_SA/train/runScript.py` | Multi-case coupled NN training |
| Ramp decoupled TF | `tutorials/Ramp/steady_SA/train/tf_training/trainModel.py` | Reference decoupled script |
| Ramp symbolic regression | `tutorials/Ramp/steady_SA/train/sr_training/trainModel_SR.py` | PySR equation discovery |

---

## Appendix A: Glossary for CFD Engineers

| ML Term | CFD Analogy | Meaning |
|---------|-------------|---------|
| **Epoch** | One sweep through all cells | Complete pass through training data |
| **Batch** | Subset of cells per update | Group of samples for one gradient step |
| **Learning rate** | Relaxation factor | Step size for weight updates |
| **Loss function** | Objective function | What the optimizer minimizes |
| **Overfitting** | Capturing noise | Model memorizes data instead of learning patterns |
| **Underfitting** | Over-smoothing | Model too simple to capture the true pattern |
| **Validation set** | Held-out test case | Data not used in training, tests generalization |
| **Regularization** | Smoothing penalty | Prevents overly complex solutions |
| **Activation function** | Transfer function | Nonlinearity applied at each neuron |
| **Gradient descent** | Steepest descent optimization | Iteratively update weights to reduce loss |
| **Backpropagation** | Adjoint method | Efficient gradient computation through the network |

## Appendix B: Understanding Backpropagation via the Adjoint Analogy

If you are familiar with adjoint methods in CFD, you already understand backpropagation.
They are mathematically identical:

| Step | Adjoint CFD | Neural Network Backpropagation |
|------|------------|-------------------------------|
| Forward solve | Solve N-S equations: R(U, alpha) = 0 | Forward pass: y = NN(x; theta) |
| Objective | J(U, alpha) | L(y, y_true) |
| Backward solve | Solve adjoint: (dR/dU)^T * psi = -(dJ/dU)^T | Backward pass: dL/d(theta) via chain rule |
| Gradient | dJ/d(alpha) = (dR/d(alpha))^T * psi + dJ/d(alpha) | dL/d(theta) = accumulated through layers |
| Update | alpha_new = alpha - step * gradient | theta_new = theta - lr * gradient |

In coupled FIML training, the adjoint propagates through both the neural network AND the
CFD solver -- giving you dJ/d(NN_weights) that accounts for how weight changes affect
the flow solution.
