# TensorTrace: FIML Pipeline for Spalart-Allmaras Model Tuning

**Pipeline:** Field Inversion → Coupled NN Training → NN Compression/Distillation → Symbolic Regression
**Turbulence Model:** Spalart-Allmaras (SA)
**Scenarios:** 2 cases — Backward-facing ramp at U₀ = 10 m/s (c1) and U₀ = 20 m/s (c2)
**Mesh:** 5,000 cells per case
**Reference data:** High-fidelity (DNS/LES) velocity field and surface pressure

---

## Notation Table

| Symbol | Meaning | Shape (symbolic) | Shape (concrete) |
|--------|---------|------------------|------------------|
| N_c | Number of mesh cells | — | 5,000 |
| N_s | Number of state DOFs per cell (U,V,p,nuTilda,phi) | — | 5 |
| N_w | Total state vector size | N_c × N_s | 25,000 |
| N_f | Number of NN input features | — | 4 |
| N_h | Hidden layer neurons | — | [20, 20] (teacher) or [5] (student) |
| N_p | Total NN parameters | — | 541 (teacher) or 21 (student) |
| N_ref | Number of reference data points | — | varies by objective |
| W | State vector | N_w × 1 | 25,000 × 1 |
| β | Correction field (betaFINuTilda) | N_c × 1 | 5,000 × 1 |
| η | Feature fields (PoD, VoS, chiSA, PSoSS) | N_c × N_f | 5,000 × 4 |
| θ | NN weight/bias vector (flat) | N_p × 1 | 541 × 1 |
| J | Objective function (scalar) | 1 | 1 |
| R(W, β) | PDE residual vector | N_w × 1 | 25,000 × 1 |
| ψ | Adjoint vector | N_w × 1 | 25,000 × 1 |
| dR/dW | State Jacobian | N_w × N_w | 25,000 × 25,000 |
| dR/dβ | Residual sensitivity to beta | N_w × N_c | 25,000 × 5,000 |
| dβ/dθ | NN Jacobian (beta w.r.t. weights) | N_c × N_p | 5,000 × 541 |
| dJ/dθ | Total gradient of objective w.r.t. NN weights | N_p × 1 | 541 × 1 |

---

## Pipeline Overview

```
STAGE 1: Field Inversion (Data Assimilation)
    β(5000) ──primal──> W*(25000) ──eval──> J_FI(1)
    J_FI ──adjoint──> dJ/dβ(5000)
    optimizer updates β

STAGE 2: Coupled NN Training
    θ(541) ──NN──> β(5000) ──primal──> W*(25000) ──eval──> J_NN(1)
    J_NN ──adjoint──> ψ(25000) ──chain rule──> dJ/dθ(541)
    optimizer updates θ

STAGE 3: NN Compression (Knowledge Distillation)
    θ_student(21) ──compressed NN──> β(5000) ──primal──> W*(25000) ──eval──> J_comp(1)
    J_comp ──adjoint──> dJ/dθ_student(21)
    optimizer refines θ_student

STAGE 4: Symbolic Regression
    Evaluate compressed NN on feature grid → (X, y) dataset
    PySR: discovers β = f(features; a,b,c,...) with 3-10 coefficients
```

---

## Stage 1: Field Inversion (Data Assimilation)

### 1.1 Problem Statement

The SA turbulence model contains a production term:

```
P_nuTilda = C_b1 * Ŝ * nuTilda * β
```

where β = 1 in the standard model. Field inversion finds the spatially-varying β(x) that minimizes the mismatch between the RANS solution and reference data.

**Optimization problem:**
```
min_β  J(β) = Σᵢ scale_i * ||u_i(β) - u_i_ref||² / N_ref_i
subject to: R(W, β) = 0  (converged RANS equations)
```

**Code reference:** `tutorials/Ramp/steady_SA/train/runScript_FI.py`

### Step 1.1: Initialize beta field

**Input tensor(s):**
- β₀ (5000 × 1): initialized to 1.0 everywhere (no correction)

**Operation:**
```
β₀[i] = 1.0    for all i ∈ {0, ..., N_c - 1}
```

**Output tensor(s):**
- β₀ (5000 × 1): uniform ones vector

**Code:** `runScript_FI.py:177`
```python
self.dvs.add_output("beta", val=np.ones(nCells), distributed=False)
```

### Step 1.2: SIMPLE primal solve with beta

**Input tensor(s):**
- β (5000 × 1): current correction field
- W_prev (25000 × 1): previous iteration state (or initial condition)

**Operation:**
The SIMPLE algorithm solves the coupled RANS + SA system iteratively. At each SIMPLE iteration:

```
1. Momentum predictor:     A_U * U* = H - ∇p
2. Pressure correction:    ∇·(1/A_U * ∇p') = ∇·U*
3. Velocity correction:    U = U* - (1/A_U) * ∇p'
4. SA transport equation:  ∇·(U * nuTilda) - ∇·((nu + nuTilda) * ∇nuTilda)
                           = C_b1 * Ŝ * nuTilda * β[cell]     ← beta enters here
                           - C_w1 * f_w * (nuTilda/y)²
5. Update turbulent viscosity: nut = f_v1 * nuTilda
```

The state vector W = [U_x, U_y, p, nuTilda, phi] is 5 DOFs per cell.

**Structural insight:** Beta multiplies ONLY the production term of the SA equation. It does not appear in momentum, pressure, or any other equation. This means dR/dβ is sparse — only the nuTilda rows have non-zero entries.

**Output tensor(s):**
- W* (25000 × 1): converged state vector satisfying R(W*, β) ≈ 0

**Code:** `src/adjoint/DASolver/DASimpleFoam/DASimpleFoam.C` (primal solve), `src/adjoint/DAModel/DATurbulenceModel/DASpalartAllmaras.C:457` (beta insertion)
```cpp
== Cb1_ * phase_ * rho * Stilda * nuTilda_ * betaFINuTilda_
```

### Step 1.3: Evaluate objective function

**Input tensor(s):**
- W* (25000 × 1): converged state
- u_ref (varies): reference data from DNS/LES/experiment

**Operation (variance objective):**
```
J = Σ_objectives [ scale_k * (1/N_ref_k) * Σ_j (u_computed_j - u_ref_j)² ]
```

For the tutorial case, the composite objective is:
```
J = J_UFieldVar + J_dragVar + J_betaVar

J_UFieldVar = (0.1 / N_field) * Σ_cells Σ_{c∈{x,y}} (U_c[cell] - U_c_ref[cell])²
J_dragVar   = (1.0) * (C_D - C_D_ref)²
J_betaVar   = (1.0 / N_c) * Σ_cells (β[cell] - 1.0)²    ← regularization
```

**Structural insight:** `J_betaVar` is a Tikhonov regularization term that penalizes deviation of β from 1.0. Without it, the field inversion would overfit — β could take extreme values in cells far from measurement points because the objective doesn't constrain them. The regularization ensures β stays near the baseline model where data provides no guidance.

**Output tensor(s):**
- J (scalar): total objective value

**Code:** `src/adjoint/DAFunction/DAFunctionVariance.C:336-616`
```cpp
scalar varDif = (var[cellI] - refValue_[timeIndex - 1][cellI]);
functionValue += geoWeight * scale_ * varDif * varDif;
```

### Step 1.4: Adjoint solve — dJ/dβ for all cells simultaneously

**Input tensor(s):**
- dR/dW (25000 × 25000): state Jacobian — SPARSE
- dJ/dW (25000 × 1): partial derivative of objective w.r.t. state — SPARSE (mostly zeros)

**Operation (adjoint equation):**

```
(dR/dW)ᵀ * ψ = -(dJ/dW)ᵀ

 ───────           ──────
 25000×25000       25000×1
 sparse,           sparse,
 ~5-7 non-zeros    non-zero only at
 per row (FVM      cells/faces where
 stencil)          objective is evaluated
```

**Sparsity of dR/dW (state Jacobian):**

The state Jacobian has a block structure from the FVM discretization. For cell-based state ordering [U_x, U_y, p, nuTilda, phi]_cell:

```
Block structure (one cell's coupling to its ~6 neighbors in 2D):

For cell i coupled to neighbor j:
        U_x_j  U_y_j  p_j   nuT_j  phi_j
U_x_i  [  ##    ##    ##     .      .    ]
U_y_i  [  ##    ##    ##     .      .    ]
p_i    [  ##    ##    ##     .      .    ]
nuT_i  [  ##    ##    .      ##     .    ]   ← β enters this row
phi_i  [  ##    ##    ##     .      ##   ]

. = zero, # = non-zero
```

Each cell couples to itself plus ~5-6 face neighbors (hex mesh), giving ~6-7 blocks per row. The total matrix has:
- Size: 25,000 × 25,000
- Non-zeros: ~25,000 × 5 × 6 = 750,000 (out of 625,000,000 possible) → ~0.12% fill
- Structure: block-sparse with 5×5 blocks

**Sparsity pattern (5-cell 1D illustration):**
```
State ordering: [U,V,p,nuT,phi]_cell0, [U,V,p,nuT,phi]_cell1, ...

         cell0    cell1    cell2    cell3    cell4
cell0  [ #####    #####    .....    .....    ..... ]
cell1  [ #####    #####    #####    .....    ..... ]
cell2  [ .....    #####    #####    #####    ..... ]
cell3  [ .....    .....    #####    #####    ##### ]
cell4  [ .....    .....    .....    #####    ##### ]

Each ##### is a 5×5 dense block (intra-cell coupling between all state variables)
```

**Why this structure?** The FVM discretization uses compact stencils — each cell's equation only depends on its face-neighbor cells. This creates a band-like sparsity pattern. The bandwidth depends on the mesh topology and cell ordering; graph-based reordering (RCM, natural) reduces bandwidth for efficient ILU preconditioning.

**Solver:** GMRES with ILU(1) preconditioner
- `adjEqnOption.gmresRelTol`: 1e-6
- `adjEqnOption.pcFillLevel`: 1 (ILU(1) — one level of fill)

**Output tensor(s):**
- ψ (25000 × 1): adjoint vector

**Code:** `src/adjoint/DALinearEqn/` (GMRES + ILU solver), `src/adjoint/DAJacCon/` (Jacobian construction)

### Step 1.5: Total derivative assembly — dJ/dβ

**Input tensor(s):**
- ψ (25000 × 1): adjoint vector
- ∂R/∂β (25000 × 5000): partial derivative of residual w.r.t. beta — VERY SPARSE
- ∂J/∂β (5000 × 1): direct partial of objective w.r.t. beta

**Operation:**
```
dJ/dβ = ∂J/∂β + ψᵀ * ∂R/∂β
         ─────   ─────────────
         5000×1   (1×25000)(25000×5000) = 1×5000 → transpose to 5000×1
```

**Sparsity of ∂R/∂β:**

```
∂R/∂β has shape 25000 × 5000 (N_w × N_c)

Since β only appears in the SA production term:
- Rows for U_x, U_y, p, phi: ALL ZERO
- Rows for nuTilda: DIAGONAL (each cell's nuTilda equation depends only on its own β)

Structure (5-cell example, 25 rows × 5 cols):

        β_0  β_1  β_2  β_3  β_4
U_x_0 [  .    .    .    .    .  ]
U_y_0 [  .    .    .    .    .  ]
p_0   [  .    .    .    .    .  ]
nuT_0 [  #    .    .    .    .  ]  ← C_b1 * Ŝ₀ * nuTilda₀
phi_0 [  .    .    .    .    .  ]
U_x_1 [  .    .    .    .    .  ]
U_y_1 [  .    .    .    .    .  ]
p_1   [  .    .    .    .    .  ]
nuT_1 [  .    #    .    .    .  ]  ← C_b1 * Ŝ₁ * nuTilda₁
phi_1 [  .    .    .    .    .  ]
 ...
```

Non-zeros: exactly N_c = 5,000 (one per cell, in the nuTilda row)
Fill: 5,000 / (25,000 × 5,000) = 0.004%

**Why so sparse?** Beta enters the PDE at exactly one location: the production term of the SA equation. Each cell's production depends only on its own beta value (no spatial coupling through beta). This makes the matrix extremely sparse — diagonal in the nuTilda block, zero everywhere else.

**Physical meaning of dJ/dβ[i]:**
The gradient tells the optimizer: "if I increase β at cell i by a small amount, how much does the objective change?" Large gradients indicate cells where the correction is most impactful — typically in separation regions, recirculation zones, and reattachment points.

**Direct term ∂J/∂β:**
From the regularization `J_betaVar = (1/N_c) * Σ (β_i - 1)²`:
```
∂J_betaVar/∂β_i = (2/N_c) * (β_i - 1.0) * scale_betaVar
```

**Output tensor(s):**
- dJ/dβ (5000 × 1): gradient of objective w.r.t. correction field

### Step 1.6: Optimizer update

**Operation:**
```
β_new = optimizer_step(β, dJ/dβ)
```

IPOPT (interior-point method) or SNOPT (SQP) uses the gradient and L-BFGS Hessian approximation to compute the update.

**Design variable bounds:** β ∈ [-5, 10] per cell.

**Forward vs. Reverse mode cost:**
```
Forward mode:  N_dv = 5,000 linear solves (one per beta component)
Reverse mode:  N_obj = 1 linear solve   (one adjoint)

Speedup: 5,000x

This is WHY adjoint methods exist. For field inversion with thousands of design
variables and one objective, reverse mode is dramatically cheaper.
```

### Step 1.7: Result

After optimization converges, we have:
- β* (5000 × 1): optimal correction field for this case
- This β* is case-specific — it does NOT generalize to other geometries/conditions

---

## Stage 2: Coupled Neural Network Training

### 2.1 Problem Statement

Replace the per-cell β with a neural network that predicts β from local flow features:

```
β_i = NN(η₁(x_i), η₂(x_i), η₃(x_i), η₄(x_i); θ)   for each cell i

where:
  η₁ = PoD  (Production over Destruction)
  η₂ = VoS  (Vorticity over Strain)
  η₃ = chiSA (SA viscosity ratio)
  η₄ = PSoSS (Pressure over normal Stress)
```

**Key design change:** The design variables switch from N_c = 5,000 per-cell β values to N_p = 541 NN parameters (weights + biases). The NN provides spatial coherence and generalization.

**Multi-case training:** Both cases (c1, c2) are trained simultaneously:
```
min_θ  J(θ) = J_c1(θ) + J_c2(θ)
```

**Code reference:** `tutorials/Ramp/steady_SA/train/runScript.py`

### Step 2.1: Feature computation — η(W)

**Input tensor(s):**
- W* (25000 × 1): converged state vector for a given iteration

**Operation (for each of the 4 features, computed cell-by-cell):**

```
Feature: VoS (Vorticity over Strain)
  Ω = skew(∇U)           — vorticity tensor
  S = symm(∇U)           — strain rate tensor
  VoS[cell] = |Ω| / (|Ω| + |S| + ε)    where ε = 1e-16
  → range: [0, 1]

Feature: PoD (Production over Destruction)
  Obtained from DATurbulenceModel::getTurbProdOverDestruct()
  PoD[cell] = P_k / (P_k + D_k + ε)
  → range: [0, 1]

Feature: chiSA
  chiSA[cell] = nuTilda / (ν + nuTilda + ε)
  → range: [0, 1]

Feature: PSoSS (Pressure gradient over Normal Stress)
  PSoSS[cell] = |∇p| / (|∇p| + |3 * mean(U · diag(∇U))| + ε)
  → range: [0, 1]
```

After raw computation, each feature is shifted and scaled:
```
η_normalized[cell] = (η_raw[cell] + inputShift) * inputScale
```

In the tutorial, shift = 0 and scale = 1, so features are used as-is.

**Structural insight:** All features follow the normalization pattern `A / (A + B + ε)`. This maps values to [0, 1], is Galilean invariant (frame-independent), and avoids division-by-zero. The pattern is crucial for NN training stability — without it, features would have wildly different scales and the NN would struggle to learn.

**Output tensor(s):**
- η (5000 × 4): feature matrix, one row per cell, one column per feature

**Code:** `src/adjoint/DARegression/DARegression.C:164-351`
```cpp
features_[modelName][idxI][cellI] = (magOmega[cellI] / (magS[cellI] + magOmega[cellI] + 1e-16)
    + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
```

### Step 2.2: NN forward pass — θ → β

**Input tensor(s):**
- θ (541 × 1): flat parameter vector (all weights and biases)
- η (5000 × 4): feature matrix

**Operation (cell-by-cell, fully connected MLP):**

Architecture: 4 inputs → [20, 20] hidden (tanh) → 1 output

```
For each cell i:
  counterI = 0  (walks through flat parameter array)

  # Hidden layer 0: 4 inputs → 20 neurons
  For neuron j = 0..19:
    z_j = Σ_{k=0}^{3} η[i,k] * θ[counterI++]   (4 weights)
    z_j += θ[counterI++]                          (1 bias)
    a_j = tanh(z_j)                               (activation)

  # Hidden layer 1: 20 inputs → 20 neurons
  For neuron j = 0..19:
    z_j = Σ_{k=0}^{19} a_prev[k] * θ[counterI++]  (20 weights)
    z_j += θ[counterI++]                             (1 bias)
    a_j = tanh(z_j)                                  (activation)

  # Output layer: 20 inputs → 1 output (no activation)
  output = Σ_{k=0}^{19} a_last[k] * θ[counterI++]  (20 weights)
  output += θ[counterI++]                             (1 bias)

  β[i] = outputScale * (output + outputShift)
       = 1.0 * (output + 1.0)
```

**Parameter count:**
```
Layer 0: 20 neurons × (4 weights + 1 bias) = 100
Layer 1: 20 neurons × (20 weights + 1 bias) = 420
Output:  1 neuron × (20 weights + 1 bias)  = 21
                                     Total = 541
```

**Parameter memory layout (flat array):**
```
Index:  [0..3]  [4]   [5..8]  [9]   ... [99]    [100..119] [120]  ... [519]  [520..539] [540]
        W_00    b_0   W_01    b_1   ... b_19    W_10       b_20   ... b_39   W_out      b_out
        ├── neuron 0 ──┤├── neuron 1──┤  ...     ├── layer 1 neurons ──────┤  ├─ output ─┤
        ├───────────── Layer 0 (100 params) ─────┤├──── Layer 1 (420) ─────┤├── Out (21)─┤
```

**Structural insight:** The same θ vector produces different β fields for different cells because the INPUT features differ per cell. The weights are shared across all cells — this is what provides spatial coherence and generalization. A cell in a separation zone and a cell in the freestream see different feature values, so the NN maps them to different β values, but using the same learned function.

**Output tensor(s):**
- β (5000 × 1): correction field predicted by NN

**Code:** `src/adjoint/DARegression/DARegression.C:354-490`
```cpp
layerVals[layerI][neuronI] += features_[modelName][neuronJ][cellI] * parameters_[modelName][counterI];
counterI++;
// ...
outputField[cellI] = outputScale_[modelName] * (outputVal + outputShift_[modelName]);
```

### Step 2.3: Primal solve with NN-predicted beta

Same as Stage 1, Step 1.2, but β comes from the NN instead of being a direct design variable.

### Step 2.4: Evaluate objective

Same as Stage 1, Step 1.3. For coupled training, the composite objective sums across cases:
```
J_total = J_c1(θ) + J_c2(θ)
        = pVar_c1 + pVar_c2
```

**Code:** `runScript.py:175,189`
```python
self.add_subsystem("obj", om.ExecComp("value=c1+c2"))
self.connect("%s.aero_post.pVar" % case, "obj.%s" % case)
```

### Step 2.5: Adjoint solve and gradient chain — dJ/dθ

This is the core of the FIML gradient computation. The total derivative uses the chain rule through three stages:

```
dJ/dθ = ∂J/∂θ  +  ψᵀ * ∂R/∂θ
         ─────     ──────────────
         direct     indirect (through flow change)
```

But ∂R/∂θ and ∂J/∂θ are themselves chain rules through β:

```
∂R/∂θ = (∂R/∂β) * (∂β/∂θ)
          ─────     ────────
          25000×5000  5000×541
          (sparse)    (dense but structured)
```

So the full chain is:

```
dJ/dθ = ∂J/∂β * ∂β/∂θ  +  ψᵀ * ∂R/∂β * ∂β/∂θ
       = [∂J/∂β + ψᵀ * ∂R/∂β] * ∂β/∂θ
       = dJ/dβ * ∂β/∂θ
         ──────   ────────
         1×5000    5000×541
```

**The NN Jacobian ∂β/∂θ:**

Shape: N_c × N_p = 5000 × 541

```
        θ_0   θ_1   θ_2   ...  θ_540
β_0   [  #     #     #          #    ]
β_1   [  #     #     #          #    ]
β_2   [  #     #     #          #    ]
 ...
β_4999[  #     #     #          #    ]
```

**DENSE.** Every β_i depends on every θ_j (fully connected network). However, note that:
- All rows have the SAME sparsity pattern (same NN architecture applied to each cell)
- Different rows have DIFFERENT values (because input features differ per cell)

**In practice, DAFoam does NOT form ∂β/∂θ explicitly.** Instead, CoDiPack's reverse-mode AD computes the product `dJ/dβ * ∂β/∂θ` directly during the adjoint pass through the NN forward code. This is much cheaper than forming the full 5000 × 541 matrix.

**AD propagation cost:** For reverse mode, the cost of propagating through the NN is proportional to the cost of one forward evaluation of the NN (with a small constant factor ~3-5x). Since the NN evaluation is cheap compared to the SIMPLE solve, the adjoint cost is dominated by the linear solve, not the NN backpropagation.

**Forward vs. Reverse mode cost comparison:**
```
Forward mode:  N_dv = 541 adjoint-like solves (one per NN parameter)
Reverse mode:  N_obj = 1 adjoint solve (one per objective)

Speedup: 541x

This is WHY reverse-mode AD and adjoint methods are essential for FIML.
For 541 design variables and 1 objective, reverse mode is 541× cheaper.
```

**For multi-case training (c1 + c2):**
```
Reverse mode: 2 adjoint solves (one per case), then sum gradients
Forward mode: 2 × 541 = 1,082 forward-mode passes

The gradients combine as:
dJ_total/dθ = dJ_c1/dθ + dJ_c2/dθ
```

**Output tensor(s):**
- dJ/dθ (541 × 1): total gradient of objective w.r.t. NN parameters

### Step 2.6: Optimizer update

```
θ_new = IPOPT_step(θ, dJ/dθ)
```

IPOPT uses the L-BFGS approximation to the Hessian (with history length 10).

**Output:** After convergence, the trained NN parameters are saved:
```python
opt_dv = {"parameter1": prob.get_val("parameter1").tolist()}
with open("designVariable.json", "w") as f:
    json.dump(opt_dv, f)
```

---

## Stage 3: NN Compression (Knowledge Distillation)

### 3.1 Problem Statement

Compress the 541-parameter teacher network (4 inputs → [20,20] → 1) into a 21-parameter student network (3 inputs → [5] → 1) while maintaining physics consistency.

**Code reference:** `tutorials/Ramp/steady_SA/train/symbolic_distillation/runCompression.py`

### Step 3.1: Feature importance ranking

**Input tensor(s):**
- W₀_teacher (20 × 4): first-layer weight matrix of trained teacher NN

**Operation:**
```
importance[k] = (1/20) * Σ_{j=0}^{19} |W₀[j, k]|    for each input feature k

Rankings (example): PoD > chiSA > VoS > PSoSS
→ Keep top 3: PoD, VoS, chiSA   (drop PSoSS)
```

The weight matrix W₀ is extracted from the flat parameter array using the memory layout:
```
θ[0:4]   = weights for neuron 0 (4 inputs)
θ[4]     = bias for neuron 0
θ[5:9]   = weights for neuron 1
...
```

**Structural insight:** First-layer weight saliency is a rough but effective proxy for feature importance. If the mean absolute weight connecting feature k to all hidden neurons is small, the NN has learned to mostly ignore that feature. This heuristic is fast (no CFD needed) and provides a principled starting point for compression.

**Output tensor(s):**
- rankings (4 × 1): feature indices sorted by importance [descending]
- top_features (3 × 1): indices of features to keep

**Code:** `distillation_utils.py:99-112`
```python
importance = np.mean(np.abs(W0), axis=0)
rankings = np.argsort(importance)[::-1]
```

### Step 3.2: Student network initialization (knowledge distillation)

**Input tensor(s):**
- θ_teacher (541 × 1): trained teacher parameters
- top_features (3 × 1): indices [e.g., 0, 1, 2] of kept features

**Operation:**

```
Teacher: 4 inputs → [20, 20] → 1   (541 params)
Student: 3 inputs → [5]     → 1   (21 params)

Student parameter count:
  Layer 0: 5 neurons × (3 weights + 1 bias) = 20
  Output:  1 neuron  × (5 weights + 1 bias)  = 6
  Total: 26  ... wait, let me recount.

  Layer 0: 5 × 3 = 15 weights + 5 biases = 20
  Output:  1 × 5 = 5 weights + 1 bias = 6
  Total: 26 parameters

  (With the DAFoam memory layout: each neuron stores weights then bias)
  = 5 * (3 + 1) + 1 * (5 + 1) = 20 + 6 = 26
```

**Correction: With 3 inputs and [5] hidden:**
```
N_p = 3*5 + 5*1 + 5 + 1 = 15 + 5 + 5 + 1 = 26
```

Knowledge distillation transfer:
```
1. First layer: W₀_student[j, :] = W₀_teacher[top_neurons[j], top_features]
   - top_neurons: the 5 neurons with highest weight magnitude on kept features
   - Shape: (5, 3) extracted from (20, 4) by selecting 3 columns and 5 rows

2. Output layer: W_out_student = W_out_teacher[:, top_neurons]
   - Shape: (1, 5) extracted from (1, 20) by selecting 5 columns

3. Biases: b_student = b_teacher[top_neurons]
```

**Output tensor(s):**
- θ_student_init (26 × 1): initialized student parameters

**Code:** `distillation_utils.py:115-222`

### Step 3.3: Coupled refinement of student network

The student network is then fine-tuned using the same coupled adjoint optimization as Stage 2, but with the smaller architecture:

```
min_θ_student  J(θ_student) = pVar_c1(θ_student) + pVar_c2(θ_student)
```

**Forward vs. Reverse cost:**
```
Forward: 26 linear solves per case (one per student parameter)
Reverse: 1 linear solve per case

Speedup: 26×  (less dramatic than Stage 2, but still significant)
```

The small size of the student network means the adjoint gradient computation is faster and the optimization landscape is simpler (fewer local minima).

**Output:**
- θ_student_opt (26 × 1): optimized compressed NN parameters saved to `designVariable_compressed.json`

---

## Stage 4: Symbolic Regression

### 4.1 Problem Statement

Discover an interpretable algebraic expression β = f(features) that approximates the compressed NN. This is a **decoupled** step — SR is performed outside the CFD solver.

**Code reference:** `tutorials/Ramp/steady_SA/train/symbolic_distillation/runPipeline.py:117-271` (Stage 3 of the pipeline script)

### Step 4.1: Generate training data from compressed NN

**Input tensor(s):**
- θ_student (26 × 1): compressed NN parameters
- X_grid (10000 × 3): random samples of feature values

**Operation:**
```
For each sample i = 0..9999:
  y[i] = NN_student(X_grid[i, 0], X_grid[i, 1], X_grid[i, 2]; θ_student)
```

This evaluates the compressed NN on a grid of feature values to produce (input, output) pairs for SR.

**Output tensor(s):**
- X (10000 × 3): feature samples
- y (10000 × 1): NN output values (β predictions)

**Code:** `runPipeline.py:149-167`
```python
X = np.zeros((n_samples, args.n_features))
for i, name in enumerate(selected_names):
    lo, hi = feature_ranges[name]
    X[:, i] = np.random.uniform(lo, hi, n_samples)
y = evaluate_nn(compressed_params, args.n_features, student_hidden, X)
```

### Step 4.2: PySR symbolic regression

**Input tensor(s):**
- X_train (8000 × 3): training feature samples (80% split)
- y_train (8000 × 1): corresponding NN outputs

**Operation (genetic programming):**

PySR evolves a population of symbolic expressions using:
```
Binary operators: +, -, *, /
Unary operators:  exp, log, tanh, sqrt, square, abs

Constraints:
  - Max complexity: 25 nodes in expression tree
  - Max depth: 6
  - Parsimony penalty: 0.0032 per complexity unit
  - Nested constraint: no exp(exp(...)) or log(log(...))
```

Each generation:
1. Evaluate all candidate expressions on training data
2. Compute fitness = MSE + parsimony * complexity
3. Select best expressions (tournament selection)
4. Mutate and crossover to produce next generation
5. Periodically optimize constants in each expression (BFGS on the scalar coefficients)

**Output:** Pareto front of expressions trading off complexity vs. accuracy

Example result:
```
Complexity 3:  β = 1.02               (constant, loss = 2.3e-1)
Complexity 7:  β = 1 + 0.3*VoS       (linear, loss = 4.1e-2)
Complexity 13: β = 1 + 0.3*tanh(2.1*VoS - 0.8*PoD)  (loss = 1.2e-3)
Complexity 19: β = 1 + 0.28*tanh(2.1*VoS - 0.82*PoD + 0.15*chiSA)  (loss = 3.5e-4)
```

**Output tensor(s) (for the best expression):**
- Expression coefficients (3-10 scalars): e.g., {0.3, 2.1, 0.8}
- Expression structure: tree representation

### Step 4.3: Validation

The best expression is evaluated on the validation set:
```
y_pred = f_SR(X_val)
R² = 1 - Σ(y_val - y_pred)² / Σ(y_val - mean(y_val))²
```

The expression is then exported to multiple formats:
- **Python** (numpy-compatible callable)
- **C++** (for insertion into `DARegression.C` as a new modelType)
- **LaTeX** (for publication)
- **SymPy** (for symbolic manipulation)

**Code:** `sr_utils.py:14-149`

---

## Tensor Inventory: Complete Summary

### Stage 1: Field Inversion

| Tensor | Shape (symbolic) | Shape (concrete) | Structure | Non-zeros | Description |
|--------|-----------------|-----------------|-----------|-----------|-------------|
| β | N_c × 1 | 5,000 × 1 | dense | 5,000 | Correction field (design variable) |
| W | N_w × 1 | 25,000 × 1 | dense | 25,000 | RANS state vector [U,V,p,nuTilda,phi] |
| R(W,β) | N_w × 1 | 25,000 × 1 | dense | 25,000 | PDE residual vector |
| J | 1 | 1 | scalar | 1 | Composite objective |
| u_ref | varies | varies | dense | varies | Reference data (DNS/LES) |
| dR/dW | N_w × N_w | 25,000 × 25,000 | block-sparse | ~750,000 | State Jacobian (FVM stencil) |
| dR/dβ | N_w × N_c | 25,000 × 5,000 | ultra-sparse | 5,000 | Only nuTilda rows, diagonal in β |
| dJ/dW | N_w × 1 | 25,000 × 1 | sparse | ~N_ref | Non-zero only at measured locations |
| ψ | N_w × 1 | 25,000 × 1 | dense | 25,000 | Adjoint vector |
| dJ/dβ | N_c × 1 | 5,000 × 1 | dense | 5,000 | Total gradient for field inversion |

### Stage 2: Coupled NN Training

| Tensor | Shape (symbolic) | Shape (concrete) | Structure | Non-zeros | Description |
|--------|-----------------|-----------------|-----------|-----------|-------------|
| θ | N_p × 1 | 541 × 1 | dense | 541 | NN parameters (flat array) |
| η | N_c × N_f | 5,000 × 4 | dense | 20,000 | Feature matrix (4 features per cell) |
| β(θ) | N_c × 1 | 5,000 × 1 | dense | 5,000 | NN-predicted correction field |
| ∂β/∂θ | N_c × N_p | 5,000 × 541 | dense | 2,705,000 | NN Jacobian (never formed explicitly) |
| dJ/dθ | N_p × 1 | 541 × 1 | dense | 541 | Total gradient for NN training |
| W₀ | N_h[0] × N_f | 20 × 4 | dense | 80 | First hidden layer weights |
| W₁ | N_h[1] × N_h[0] | 20 × 20 | dense | 400 | Second hidden layer weights |
| W_out | 1 × N_h[1] | 1 × 20 | dense | 20 | Output layer weights |
| b₀ | N_h[0] × 1 | 20 × 1 | dense | 20 | First hidden layer biases |
| b₁ | N_h[1] × 1 | 20 × 1 | dense | 20 | Second hidden layer biases |
| b_out | 1 | 1 | scalar | 1 | Output bias |

### Stage 3: Compression

| Tensor | Shape (symbolic) | Shape (concrete) | Structure | Non-zeros | Description |
|--------|-----------------|-----------------|-----------|-----------|-------------|
| importance | N_f × 1 | 4 × 1 | dense | 4 | Feature importance scores |
| θ_student | N_p_s × 1 | 26 × 1 | dense | 26 | Compressed NN parameters |
| W₀_student | 5 × 3 | 5 × 3 | dense | 15 | Student first-layer weights |
| W_out_student | 1 × 5 | 1 × 5 | dense | 5 | Student output weights |

### Stage 4: Symbolic Regression

| Tensor | Shape (symbolic) | Shape (concrete) | Structure | Non-zeros | Description |
|--------|-----------------|-----------------|-----------|-----------|-------------|
| X | N_samples × N_f_s | 10,000 × 3 | dense | 30,000 | Feature grid samples |
| y | N_samples × 1 | 10,000 × 1 | dense | 10,000 | NN evaluation on grid |
| coefficients | N_coeff × 1 | ~5 × 1 | dense | ~5 | SR expression coefficients |

---

## Complete Data Flow

```
FORWARD:
  Stage 1: β(5000) ──SIMPLE──> W*(25000) ──eval──> J_FI(1)
  Stage 2: θ(541) ──NN──> β(5000) ──SIMPLE──> W*(25000) ──eval──> J_NN(1)
  Stage 3: θ_s(26) ──small_NN──> β(5000) ──SIMPLE──> W*(25000) ──eval──> J_comp(1)
  Stage 4: features(3) ──PySR──> β = f(PoD,VoS,chiSA) with ~5 coefficients

BACKWARD:
  Stage 1: J_FI(1) ──adjoint──> ψ(25000) ──chain──> dJ/dβ(5000)
  Stage 2: J_NN(1) ──adjoint──> ψ(25000) ──chain──> dJ/dβ(5000) ──AD──> dJ/dθ(541)
  Stage 3: J_comp(1) ──adjoint──> ψ(25000) ──chain──> dJ/dβ(5000) ──AD──> dJ/dθ_s(26)
  Stage 4: No backward pass (genetic programming is gradient-free)

KEY DIMENSIONAL REDUCTION:
  FI: 5,000 design vars → NN: 541 design vars → Compressed: 26 vars → SR: ~5 coefficients
```

---

## Structural Analysis

### Why dR/dW is block-sparse

The FVM discretization creates a compact stencil: each cell's equation depends on the cell itself and its face neighbors (typically 4-6 in 2D, 6-26 in 3D). This creates a sparse band structure. The 5×5 dense blocks arise because all state variables within a cell are coupled (momentum depends on pressure, pressure depends on velocity, etc.). The off-diagonal blocks are sparser — not all state variables couple across cell boundaries (e.g., phi only couples to its face neighbors' phi).

**Computational exploitation:**
- Graph coloring: The Jacobian is computed via colored finite differences. With a compact stencil, ~7-12 colors suffice to perturb all cells simultaneously. This means dR/dW requires ~7-12 residual evaluations instead of 25,000.
- ILU preconditioning: The ILU(k) factorization with k=1 (one level of fill) provides an effective preconditioner because the matrix is already sparse and banded.
- Cell reordering: `adjEqnOption.jacMatReOrdering` controls the cell numbering. "natural" uses the mesh ordering; RCM (reverse Cuthill-McKee) minimizes bandwidth for more efficient ILU.

### Why dR/dβ is ultra-sparse

β appears in exactly ONE term of ONE equation (SA production). This structural sparsity has profound implications:
1. The product `ψᵀ * ∂R/∂β` extracts only the nuTilda components of ψ at each cell.
2. The gradient `dJ/dβ[i] = ∂J/∂β[i] + ψ_nuTilda[i] * C_b1 * Ŝ[i] * nuTilda[i]`
3. This is a per-cell scalar product — no matrix assembly needed.

### Why the NN Jacobian ∂β/∂θ is dense but cheap to handle

Every β_i depends on all 541 parameters because the NN is fully connected. The full matrix would be 5000 × 541 ≈ 2.7M entries. But:
1. **Reverse AD avoids forming it.** CoDiPack propagates the seed vector `dJ/dβ` backward through the NN code, directly producing `dJ/dθ` — a vector of 541 entries. Cost: O(N_c × forward_pass_cost).
2. **The NN is applied cell-by-cell.** Each cell's contribution to dJ/dθ is independent. This is embarrassingly parallel — perfect for vectorized/GPU execution.

### Graph coloring for state Jacobian

For the 5000-cell mesh with ~6 neighbors per cell:
```
Coloring: ~12 colors (2D hex mesh)
Cost of dR/dW computation: 12 residual evaluations (vs. 25,000 column-by-column)
Speedup: ~2,000x from coloring alone
```

The coloring algorithm assigns colors to cells such that no two same-colored cells share a face. This allows all same-colored cells to be perturbed simultaneously without interference.

**Code:** `src/adjoint/DAColoring/`

---

## Key Structural Insight

The entire FIML pipeline is an exercise in **dimensional reduction through physically-informed parameterization:**

| Stage | Design Variables | Meaning |
|-------|-----------------|---------|
| Field Inversion | 5,000 (per cell) | Maximum expressiveness, zero generalization |
| Coupled NN | 541 (NN weights) | Spatial coherence via shared function, generalization |
| Compressed NN | 26 (pruned weights) | Key features only, interpretable architecture |
| Symbolic Regression | ~5 (expression coefficients) | Fully interpretable, publishable, portable |

Each stage trades expressiveness for generalization, interpretability, and computational efficiency. The adjoint method makes this tractable — without it, computing gradients of a CFD objective w.r.t. 541 NN parameters would require 541 CFD solves per optimization iteration. With the adjoint, it requires exactly 1 (per case).
