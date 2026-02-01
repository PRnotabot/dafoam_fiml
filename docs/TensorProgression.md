# TensorProgression

**Tracing every vector, gradient, Jacobian, and tensor through the FIML pipeline**

*From Navier-Stokes to neural network weight updates, on a 5-cell convergent channel.*

---

## Notation

| Symbol | Meaning | Shape |
|--------|---------|-------|
| **W** | State vector (all unknowns) | N_s x 1 |
| **R(W)** | Residual vector (discrete PDE = 0 at convergence) | N_s x 1 |
| dR/dW | State Jacobian | N_s x N_s |
| **J** | Scalar objective function | 1 |
| dJ/dW | Objective sensitivity to states | 1 x N_s |
| psi | Adjoint vector | N_s x 1 |
| beta | Correction field (one value per cell) | N_c x 1 |
| phi | Feature vector (inputs to NN, per cell) | N_f x 1 |
| theta | NN weight vector (all weights + biases) | N_p x 1 |

Where for our 5-cell example with DASimpleFoam + SA:
- N_c = 5 (cells)
- N_s = 5 x 5 = 25 (3 velocity components + pressure + nuTilda per cell)
- N_f = 7 (VoS, PoD, chiSA, pGradStream, PSoSS, SCurv, UOrth)
- N_p = 71 (for [7]--[5]--[5]--[1] architecture; derived below)

---

## Stage 0: The 5-Cell Convergent Channel

```
                  WALL (no-slip)
    ┌─────────┬─────────┬─────────┬─────────┬─────────┐
    │         │         │         │         │         │
    │  cell0  │  cell1  │  cell2  │  cell3  │  cell4  │
    │         │         │         │         │         │
INLET         │         │         │         │        OUTLET
U=10 │  f1    │   f2    │   f3    │   f4    │         p=0
    │         │         │         │         │
    └─────────┴─────────┴─────────┴─────────┴─────────┘
                  WALL (no-slip)
      ◄─────── channel narrows ──────►
```

This matches `runRegTests_DASimpleFoamRegPar.py` which uses:
- `"solverName": "DASimpleFoam"` -- see `DASimpleFoam.C:123`
- SA turbulence model with `betaFINuTilda` correction
- Inlet: U = [10, 0, 0], nuTilda = 4.5e-5
- Outlet: p = 0

**Illustrative converged state** (1D projection):

| Cell | Ux (m/s) | p (m^2/s^2) | nuTilda (m^2/s) |
|------|----------|-------------|-----------------|
| 0 | 10.0 | 48.5 | 4.50e-5 |
| 1 | 10.8 | 36.2 | 4.10e-5 |
| 2 | 11.9 | 24.3 | 3.70e-5 |
| 3 | 13.2 | 12.1 | 3.20e-5 |
| 4 | 14.8 | 0.0 | 2.80e-5 |

*(Velocity increases, pressure drops -- Bernoulli in action.)*

---

## Stage 1: CFD Forward Pass (Primal Solve)

### 1.1 The State Vector W

The state vector is a single column that packs every unknown in every cell:

```
         ┌──────────┐
         │ Ux_0     │ ◄── cell 0, velocity x
         │ Uy_0     │
         │ Uz_0     │
         │ p_0      │ ◄── cell 0, pressure
         │ nuT_0    │ ◄── cell 0, SA variable
         ├──────────┤
         │ Ux_1     │ ◄── cell 1
         │ Uy_1     │
         │ ...      │
         ├──────────┤
         │   ...    │
         ├──────────┤
         │ Ux_4     │ ◄── cell 4
         │ Uy_4     │
         │ Uz_4     │
         │ p_4      │
         │ nuT_4    │
         └──────────┘
            25 x 1
```

**Concrete values** (full state vector, illustrative):

```
W = [10.0, 0.0, 0.0, 48.5, 4.50e-5,     ← cell 0
     10.8, 0.0, 0.0, 36.2, 4.10e-5,     ← cell 1
     11.9, 0.0, 0.0, 24.3, 3.70e-5,     ← cell 2
     13.2, 0.0, 0.0, 12.1, 3.20e-5,     ← cell 3
     14.8, 0.0, 0.0,  0.0, 2.80e-5]^T   ← cell 4
```

**Code:** State ordering is managed by `DAIndex` (`src/adjoint/DAIndex/DAIndex.C`).
Field objects live in OpenFOAM's object registry and are accessed via:
```cpp
// DASimpleFoam/UEqnSimple.H:27 — the velocity field
const volVectorField& U = db.lookupObject<volVectorField>("U");
```

### 1.2 The Governing Equations (PDE Level)

**Momentum (Navier-Stokes, incompressible):**
```
   ∇·(UU) = -∇p + ∇·[(ν + νt)∇U]
   ─────    ────   ──────────────
  convection  pressure   viscous diffusion
   gradient
```

**Continuity:**
```
   ∇·U = 0
```

**SA turbulence model** (with FIML beta correction):
```
   ∇·(U·nuTilda) = Cb1·S_tilde·nuTilda·β  -  Cw1·fw·(nuTilda/d)^2  +  (1/σ)·∇·[(ν+nuTilda)·∇nuTilda]
                    ─────────────────────     ──────────────────────     ─────────────────────────────────
                    production × β            destruction                diffusion
```

The **beta field** (β) is exactly where FIML injects its correction. When β = 1 everywhere, you get standard SA. When β ≠ 1, you're augmenting the turbulence model.

**Code:** `DASpalartAllmaras.C:454-458`
```cpp
== Cb1_ * phase_ * rho * Stilda * nuTilda_ * betaFINuTilda_   // production × β
    - fvm::Sp(Cw1_ * phase_ * rho * fw(Stilda) * nuTilda_ / sqr(y_), nuTilda_)  // destruction
```

### 1.3 FVM Discretization: How PDEs Become Matrices

**Key FVM Insight 1: Divergence → Face Fluxes → Owner/Neighbour → Sparse Matrix**

The divergence ∇·(UU) is computed via Gauss's theorem as a sum of face fluxes:

```
   ∇·(UU)|_cell_i  ≈  Σ_faces  (F_f · U_f · A_f)
```

Each internal face connects exactly 2 cells (owner and neighbour).
The face value U_f is interpolated from the owner and neighbour cell values.

For a 5-cell 1D mesh, the stencil of cell 2 is:

```
         cell 1     cell 2     cell 3
           │           │           │
           ▼           ▼           ▼
      ───[face2]───[face3]───
```

Cell 2's residual depends on cell 1, cell 2, and cell 3 -- **tridiagonal-like sparsity**.

**Key FVM Insight 2: SIMPLE Segregation → Block-Sequential Solves**

Instead of solving the coupled 25x25 system directly, SIMPLE solves three smaller systems sequentially in each iteration:

```
SIMPLE Iteration Loop:    ┌──────────────────────────────────────────┐
                          │  1. Solve momentum: A_U · U* = b_U - ∇p │
                          │     (UEqnSimple.H:27)                    │
                          │                                          │
                          │  2. Solve pressure: ∇²p' = ∇·(U*/A_U)   │
                          │     (pEqnSimple.H:44)                    │
                          │                                          │
                          │  3. Correct: U = U* - (1/A_U)·∇p'       │
                          │     (pEqnSimple.H:72)                    │
                          │                                          │
                          │  4. Solve turbulence: A_ν · ν~ = b_ν    │
                          │     (DASpalartAllmaras.C:454-460)        │
                          │                                          │
                          │  5. Update β: DARegression::compute()    │
                          │     (DASimpleFoam.C:167)                 │
                          │                                          │
                          │  6. Check convergence                    │
                          │     (DASolver.C:184)                     │
                          └──────────────────────────────────────────┘
```

Each sub-system is 5x5 (one equation per cell) -- much more tractable than the coupled 25x25. The price is outer iterations to converge the coupling.

**Code:** The full loop is in `DASimpleFoam.C:139`:
```cpp
while (this->loop(runTime))  // outer SIMPLE loop
{
    #include "UEqnSimple.H"   // momentum predictor
    #include "pEqnSimple.H"   // pressure corrector
    // turbulence: daTurbulenceModelPtr_->correct()
    regModelFail_ = daRegressionPtr_->compute();  // beta field update
}
```

### 1.4 The Residual Vector R(W)

At convergence, R(W*) = 0. During iteration, R(W) ≠ 0 measures "how far from the solution we are."

The residual for cell *i* of the momentum equation:

```
   R_Ux,i = Σ_faces(F_f · Ux_f · A_f) + (∇p)_i · V_i - Σ_faces(ν_eff · (∇Ux)_f · A_f)
```

**Concrete residual vector during iteration 50** (illustrative):

```
R(W) = [2.3e-6, 0.0, 0.0, -1.1e-7, 8.4e-9,     ← cell 0 residuals
        1.8e-6, 0.0, 0.0, -8.2e-8, 7.1e-9,     ← cell 1
        ...                                       ← cells 2-3
        9.1e-7, 0.0, 0.0, -3.4e-8, 2.3e-9]^T   ← cell 4
```

**Code:** Residual computation in `DASimpleFoam.C:1140-1237` (`calcLduResiduals`):
```cpp
// Momentum residual: R = A*U - source + V*gradP - H(U)
URes[i] = UDiag[i] * U[i] - USource[i] + U.mesh().V()[i] * gradP[i];
URes.primitiveFieldRef() -= UEqn.lduMatrix::H(U);  // off-diagonal contributions
```

---

## Stage 2: The Jacobian ∂R/∂W and Its Structure

### 2.1 What the Jacobian Looks Like

The Jacobian ∂R/∂W is the matrix of partial derivatives: entry (i,j) = ∂R_i/∂W_j.

For our 5-cell problem: **25 x 25**, but most entries are zero.

```
         W:   Ux0 Uy0 Uz0  p0  ν0  Ux1 Uy1 Uz1  p1  ν1  ... Ux4 Uy4 Uz4  p4  ν4
             ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬─── ───┬───┬───┬───┬───┐
R_Ux0       │ ██│ ██│   │ ██│   │ ██│   │   │ ██│   │       │   │   │   │   │
R_Uy0       │ ██│ ██│   │   │   │   │ ██│   │   │   │       │   │   │   │   │
R_Uz0       │   │   │ ██│   │   │   │   │ ██│   │   │       │   │   │   │   │
R_p0        │ ██│   │   │ ██│   │ ██│   │   │ ██│   │       │   │   │   │   │
R_ν0        │ ██│   │   │   │ ██│ ██│   │   │   │ ██│       │   │   │   │   │
            ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤       │   │   │   │   │
R_Ux1       │ ██│   │   │ ██│   │ ██│ ██│   │ ██│   │ ██    │   │   │   │   │
R_Uy1       │   │ ██│   │   │   │ ██│ ██│   │   │   │       │   │   │   │   │
...         │   │   │   │   │   │   │   │   │   │   │  ...  │   │   │   │   │
            │   │   │   │   │   │   │   │   │   │   │       │   │   │   │   │
R_ν4        │   │   │   │   │   │   │   │   │   │   │  ...  │ ██│   │   │   │ ██│
            └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴─── ───┴───┴───┴───┴───┘
```

**Pattern: band-diagonal** with bandwidth = (number of coupled variables) x (stencil width).

Here, stencil width = 3 (cell + 2 neighbours in 1D) and we couple 5 variables per cell, giving a band of about 15 entries per row. Out of 625 total entries (25x25), only ~375 are non-zero: **60% sparse**.

In a real 3D mesh with 100,000 cells: N_s = 500,000. The Jacobian would be 500,000 x 500,000 = 2.5 x 10^11 entries. But only ~10-20 are non-zero per row. This sparsity is what makes the adjoint tractable.

### 2.2 How DAFoam Builds It: Coloring + Finite Differences

**The problem:** Computing 25 columns of dR/dW naively needs 25 residual evaluations (perturb each state variable, recompute R, take finite difference).

**The trick:** Graph coloring. If two columns have no row in common, perturb both simultaneously.

```
   25x25 Jacobian          With 8 colors
   ┌──────────────┐        ┌──────────────┐
   │ ██  ██  ██   │        │ c1  c2  c3   │    8 perturbed residual
   │ ██  ██  ██   │  ───►  │ c1  c2  c3   │    evaluations instead
   │ ██  ██  ██   │        │ c1  c2  c3   │    of 25.
   │   ██  ██  ██ │        │   c4  c5  c6 │
   │   ██  ██  ██ │        │   c4  c5  c6 │    ~3x speedup.
   │     ██  ██  ██│       │     c7  c8  c7│
   └──────────────┘        └──────────────┘

   For real 3D meshes: ~15 colors vs. 500,000 columns → ~33,000x speedup.
```

**Code:** The distance-2 coloring algorithm is in `DAColoring.C:32-784` (parallel implementation).

The colored finite difference is in `DAPartDeriv.C:350-473`:
```cpp
for (label color = 0; color < nColors; color++) {
    this->perturbStates(jacConColors, normStatePerturbVec, color, delta, wVecNew);
    daResidual.masterFunction(mOptions, xvVec, wVecNew, resVec);     // R(W + δe_color)
    VecAXPY(resVec, -1.0, resVecRef);                                 // R(W+δ) - R(W)
    VecScale(resVec, 1.0/delta);                                      // / δ
    daJacCon_.calcColoredColumns(color, coloredColumn);               // which columns?
    this->setPartDerivMat(resVec, coloredColumn, transposed, jacMat); // assign to matrix
}
```

The connectivity pattern is built in `DAJacCon.C:2039-2332` by walking the mesh stencil:
```cpp
// For each cell, add all neighbour cells at each connectivity level
addConMatCell(...)           // level 0: cell itself
addConMatNeighbourCells(...) // level 1: face neighbours
// ... up to level 3 for higher-order schemes
```

---

## Stage 3: The Adjoint — Why Reverse Mode Wins

### 3.1 The Total Derivative Problem

We want: **dJ/d(alpha)** -- how does the objective change when we change design variables?

The chain rule gives:

```
   dJ       ∂J        ∂J     dW
   ── = ────── + ────── · ──────
   dα     ∂α       ∂W      dα

                     ▲
                     │
        This requires solving the full
        nonlinear system implicitly:
        R(W(α), α) = 0  ⟹  dW/dα = -(∂R/∂W)^{-1} · ∂R/∂α
```

Substituting:

```
   dJ     ∂J                      ∂R  -1   ∂R
   ── = ──── - (∂J/∂W) · (────────)   · ────
   dα     ∂α               ∂W           ∂α
```

### 3.2 Forward vs. Reverse: The Counting Argument

**Forward mode:** For each design variable alpha_k, solve:

```
   ∂R     dW       ∂R
   ── · ────── = - ──────     →  one linear solve per design variable
   ∂W    dα_k      ∂α_k
```

Cost: N_dv linear solves (one per design variable).

**Reverse mode (adjoint):** For each objective J_m, solve:

```
            T
   (∂R/∂W)  · ψ_m = -(∂J_m/∂W)^T     →  one linear solve per objective
```

Then:
```
   dJ_m        ∂J_m        T    ∂R
   ──── = ────── + ψ_m  · ────
   dα       ∂α              ∂α
```

Cost: N_obj linear solves (one per objective).

**For FIML:**
- N_dv = N_p = 71 (NN weights) -- many!
- N_obj = 1 (variance objective) -- just one!

```
   Forward: 71 linear solves   ◄── expensive
   Reverse:  1 linear solve    ◄── 71x cheaper!
```

**This is why DAFoam uses reverse-mode AD and the adjoint method.**

### 3.3 The Adjoint Equation

```
            T
   (∂R/∂W)  · ψ  =  -(∂J/∂W)^T
   ─────────         ───────────
    25x25 matrix      25x1 vector
    (transpose of     (how objective
     state Jacobian)   depends on states)

              ψ = 25x1 adjoint vector
```

**Concrete example** (illustrative adjoint vector for UVar objective):

```
                                ∂J/∂W:
                                (objective sensitivity)
  ∂J                            ┌─────────┐
  ──── for each state:          │ 2·(Ux0 - Ux0_ref)·scale │ ← if cell 0 in box
  ∂W_i                          │ 2·(Uy0 - Uy0_ref)·scale │
                                │ 2·(Uz0 - Uz0_ref)·scale │
   For UVar with boxToCell,     │ 0                        │ ← pressure doesn't enter J
   only cells inside the box    │ 0                        │ ← nuTilda doesn't enter J
   have non-zero ∂J/∂W.         │ ...                      │
                                └─────────┘

  The adjoint solve produces:    ψ = (∂R/∂W)^{-T} · (∂J/∂W)^T

  ψ = [0.34, 0.0, 0.0, -0.021, 0.0018,   ← adjoint for cell 0
       0.29, 0.0, 0.0, -0.016, 0.0014,   ← adjoint for cell 1
       ...                                 ← cells 2-3
       0.11, 0.0, 0.0, -0.004, 0.0005]^T  ← adjoint for cell 4
```

**Code:** The adjoint solve is in `mphys_dafoam.py:426-567`:
```python
# solve the adjoint equation [dRdW]^T * Psi = dFdW
dFdWArray = d_outputs[self.stateName]              # ∂J/∂W
dFdW = DASolver.array2Vec(dFdWArray)

# Krylov solve: (dR/dW)^T * psi = -dFdW
fail = DASolver.solverAD.solveLinearEqn(DASolver.ksp, dFdW, self.psi)  # line 540
```

The matrix-vector product (dR/dW)^T * v is computed **matrix-free** via reverse-mode AD:
```python
# mphys_dafoam.py:394-406
DASolver.solverAD.calcJacTVecProduct(    # reverse AD: computes J^T * v
    self.stateName, "stateVar",           # input type
    jacInput,                             # input values (for linearization point)
    self.residualName, "residual",        # output type
    seed,                                 # the vector v
    product,                              # result: J^T * v
)
```

### 3.4 The Jacobian Transpose: Why It's the Same Cost

A beautiful property: computing **A^T * v** via reverse-mode AD costs the same as computing **A * v** via forward-mode AD, regardless of matrix size.

```
   Forward AD:   seed → [forward pass] → A * seed         O(N_s) work
   Reverse AD:   seed → [backward pass] → A^T * seed      O(N_s) work

   We never form the full N_s x N_s matrix. We only need matrix-vector products.
```

The preconditioner (the explicit matrix `dRdWTPC`) is assembled via colored finite differences for the Krylov solver's preconditioning step (`mphys_dafoam.py:517`):
```python
DASolver.solver.calcdRdWT(1, DASolver.dRdWTPC)  # build PC matrix
```

### 3.5 The Total Derivative via Adjoint

After solving for psi, the total derivative for any design variable alpha is:

```
   dJ     ∂J        T    ∂R
   ── = ──── + ψ  · ────
   dα     ∂α         ∂α
         ────        ────
         direct      indirect
         effect      (through flow change)
```

**Code:** `mphys_dafoam.py:409-424`:
```python
for inputName in list(inputs.keys()):
    inputType = inputDict[inputName]["type"]
    product = np.zeros_like(jacInput)
    DASolver.solverAD.calcJacTVecProduct(  # computes ψ^T · ∂R/∂α via reverse AD
        inputName, inputType,               # which design variable
        jacInput,                           # linearization point
        self.residualName, "residual",
        seed,                               # seed = adjoint vector ψ
        product,                            # result: ψ^T · ∂R/∂α
    )
    d_inputs[inputName] += product
```

---

## Stage 4: Field Inversion — β as Design Variable

### 4.1 The β Field

Beta is a scalar field with one value per cell. It multiplies the production term in the SA equation:

```
   Production_augmented = Cb1 · S_tilde · nuTilda · β_i     (for cell i)
```

When β = 1: standard SA (no correction).
When β > 1: enhanced production → more turbulence.
When β < 1: reduced production → less turbulence.

**Concrete beta field** (illustrative, after field inversion):

```
   β = [1.12, 1.08, 1.00, 0.95, 0.91]^T     ← 5 x 1
        ────  ────  ────  ────  ────
        near   ↕     no   slight  near
        inlet        corr. reduct. outlet
```

**Code:** Beta is initialized in `DASpalartAllmaras.C:95-103`:
```cpp
betaFINuTilda_(
    IOobject("betaFINuTilda", mesh.time().timeName(), mesh,
             IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE),
    mesh,
    dimensionedScalar("betaFINuTilda", dimensionSet(0,0,0,0,0,0,0), 1.0),  // default = 1
    "zeroGradient")
```

### 4.2 How β Enters the Residual

The SA residual for cell *i* depends on β_i through the production term:

```
   R_νi(W, β) = [convection + diffusion]_i  -  Cb1 · S_tilde_i · nuTilda_i · β_i  +  destruction_i
```

Therefore:

```
   ∂R_νi
   ───── = -Cb1 · S_tilde_i · nuTilda_i     (a scalar, one per cell)
    ∂β_i
```

**Key structural insight:** ∂R/∂β is **diagonal** -- each cell's residual depends only on its own beta.

```
          ∂R/∂β :

          β0      β1      β2      β3      β4
         ┌─────┬─────┬─────┬─────┬─────┐
R_Ux0    │  0  │  0  │  0  │  0  │  0  │   ← momentum not directly coupled to β
R_Uy0    │  0  │  0  │  0  │  0  │  0  │
R_Uz0    │  0  │  0  │  0  │  0  │  0  │
R_p0     │  0  │  0  │  0  │  0  │  0  │   ← pressure not directly coupled to β
R_ν0     │ -d0 │  0  │  0  │  0  │  0  │   ← SA residual IS coupled
R_Ux1    │  0  │  0  │  0  │  0  │  0  │
...      │     │     │     │     │     │
R_ν1     │  0  │ -d1 │  0  │  0  │  0  │
...      │     │     │     │     │     │
R_ν4     │  0  │  0  │  0  │  0  │ -d4 │
         └─────┴─────┴─────┴─────┴─────┘
           25 x 5 (very sparse: only 5 non-zeros in a 25x5 matrix)

  where d_i = Cb1 · S_tilde_i · nuTilda_i  (positive scalar)
```

### 4.3 dJ/dβ via the Adjoint

```
   dJ     ∂J        T    ∂R
   ── = ──── + ψ  · ────
   dβ     ∂β         ∂β
```

Since J (variance) depends on U, not on β directly: ∂J/∂β = 0.

So:

```
   dJ        T    ∂R
   ── = ψ  · ────  =  [ψ_ν0 · (-d0),  ψ_ν1 · (-d1),  ...,  ψ_ν4 · (-d4)]
   dβ         ∂β
```

**Only the nuTilda components of the adjoint vector contribute to the beta gradient.**

**Concrete gradient** (illustrative):

```
   dJ/dβ = [-0.0018 · (-d0), -0.0014 · (-d1), ..., -0.0005 · (-d4)]
         = [0.0018·d0, 0.0014·d1, ..., 0.0005·d4]

   If d ≈ [0.42, 0.38, 0.35, 0.31, 0.28]:

   dJ/dβ ≈ [7.6e-4, 5.3e-4, ..., 1.4e-4]^T    ← 5 x 1 gradient
```

This gradient tells the optimizer: "to reduce J, increase β at cell 0 the most, cell 4 the least."

---

## Stage 5: FIML — Neural Network Replaces β

### 5.1 The Feature Vector φ (Per Cell)

Instead of optimizing N_c = 5 beta values directly, FIML trains a neural network to predict beta from local flow features.

For each cell, compute 7 scalar features:

```
   Feature    │ Formula                          │ Physical meaning           │ Range
   ───────────┼──────────────────────────────────┼────────────────────────────┼───────
   VoS        │ |Ω| / (|S| + |Ω| + ε)           │ Vorticity/Strain ratio     │ [0, 1]
   PoD        │ Production / (Prod + Dest + ε)   │ Production/Destruction     │ [0, 1]
   chiSA      │ nuTilda / (ν + nuTilda + ε)      │ Eddy viscosity ratio       │ [0, 1]
   pGradStream│ (U·∇p) / (|U||∇p| + |U·∇p| + ε)│ Pressure grad along stream │ [-1,1]
   PSoSS      │ |∇p| / (|∇p| + |3·diag(U·∇U)| + ε) │ Normal/shear stress    │ [0, 1]
   SCurv      │ |U·∇U| / (|U·U| + |U·∇U| + ε)  │ Streamline curvature       │ [0, 1]
   UOrth      │ |U·∇U·U|/(|U||∇U·U|+|U·∇U·U|+ε)│ Velocity non-orthogonality │ [0, 1]
```

**Design choice:** All features use the normalization pattern `A/(A+B+ε)` which maps to [0,1] or [-1,1], ensures Galilean invariance, and avoids division by zero.

**Code:** `DARegression.C:164-352` (`calcInputFeatures`):
```cpp
if (inputName == "VoS")
{
    volScalarField magOmega = mag(skew(gradU));       // |Ω|
    volScalarField magS = mag(symm(gradU));            // |S|
    features_[modelName][idxI][cellI] =
        (magOmega[cellI] / (magS[cellI] + magOmega[cellI] + 1e-16)  // A/(A+B+ε)
         + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
}
```

**Concrete feature matrix** (illustrative, 5 cells x 7 features):

```
              VoS    PoD    chiSA   pGrad  PSoSS  SCurv  UOrth
   cell 0:  [0.48,  0.62,  0.75,  -0.31,  0.44,  0.12,  0.08]
   cell 1:  [0.51,  0.58,  0.71,  -0.22,  0.41,  0.15,  0.11]
   cell 2:  [0.50,  0.55,  0.67,  -0.15,  0.38,  0.13,  0.09]
   cell 3:  [0.49,  0.52,  0.63,  -0.08,  0.36,  0.11,  0.07]
   cell 4:  [0.47,  0.49,  0.60,  -0.03,  0.33,  0.10,  0.06]
```

### 5.2 The NN Forward Pass — Cell by Cell

Architecture from the test case: 7 inputs → [5] → [5] → 1 output, with tanh activation.

```
  φ = [φ1, φ2, ..., φ7]^T   (7 x 1 input)
        │
        ▼
  ┌─────────────┐
  │  Layer 1:    │   h1 = tanh(W1 · φ + b1)     W1: 5x7, b1: 5x1
  │  5 neurons   │   h1: 5 x 1
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Layer 2:    │   h2 = tanh(W2 · h1 + b2)    W2: 5x5, b2: 5x1
  │  5 neurons   │   h2: 5 x 1
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Output:     │   y = W3 · h2 + b3            W3: 1x5, b3: 1x1
  │  1 neuron    │   y: scalar
  │  (linear)    │
  └──────┬──────┘
         │
         ▼
  β_i = outputScale · (y + outputShift)           (outputShift=1.0 → β centered at 1)
```

### 5.3 Parameter Count

```
   Layer     │ Weights        │ Biases  │ Subtotal
   ──────────┼────────────────┼─────────┼─────────
   Input→H1  │ 7 × 5 = 35    │ 5       │ 40
   H1→H2     │ 5 × 5 = 25    │ 5       │ 30
   H2→Output │ 5 × 1 = 5     │ 1       │ 6
   ──────────┼────────────────┼─────────┼─────────
   Total     │ 65             │ 11      │ N_p = 71
```

**Code:** `DARegression.C:652-689` (`nParameters`):
```cpp
label nParameters = nInputs * hiddenLayerNeurons_[modelName][0];  // 7*5=35
for (layerI = 1; ...) {
    nParameters += hiddenLayerNeurons_[layerI] * hiddenLayerNeurons_[layerI-1];  // 5*5=25
}
nParameters += hiddenLayerNeurons_[nHiddenLayers-1] * 1;  // 5*1=5
// biases: 5 + 5 + 1 = 11
// Total: 35 + 25 + 5 + 11 = 71 ✓
```

### 5.4 The Flat Parameter Vector θ

All 71 weights and biases are stored in a **single flat array**. This is the design variable vector.

```
   θ = [w1_1,1, w1_1,2, ..., w1_1,7, b1_1,        ← neuron 1 of layer 1 (8 params)
        w1_2,1, w1_2,2, ..., w1_2,7, b1_2,        ← neuron 2 of layer 1
        ...,                                        ← neurons 3-5 of layer 1
        w2_1,1, ..., w2_1,5, b2_1,                 ← neuron 1 of layer 2
        ...,                                        ← neurons 2-5 of layer 2
        w3_1, ..., w3_5, b3]^T                      ← output layer (6 params)
                                                      ────────────────
                                                      71 x 1 total
```

**Code:** The forward pass walks this array with a counter (`DARegression.C:414`):
```cpp
label counterI = 0;                                    // flat index into parameters_
for (label layerI = 0; layerI < nHiddenLayers; layerI++) {
    for (label neuronI = 0; neuronI < nNeurons; neuronI++) {
        if (layerI == 0) {
            forAll(inputNames, neuronJ) {
                layerVals[layerI][neuronI] +=
                    features_[modelName][neuronJ][cellI]   // input feature
                    * parameters_[modelName][counterI];     // weight θ[counterI]
                counterI++;
            }
        }
        layerVals[layerI][neuronI] += parameters_[modelName][counterI];  // bias
        counterI++;
        // activation: tanh
        layerVals[layerI][neuronI] = tanh(layerVals[layerI][neuronI]);
    }
}
// output layer (no activation):
outputVal += layerVals[last][neuronJ] * parameters_[modelName][counterI];
counterI++;
outputVal += parameters_[modelName][counterI];  // output bias
outputField[cellI] = outputScale * (outputVal + outputShift);
```

### 5.5 Concrete Forward Pass for Cell 0

Input features for cell 0: φ = [0.48, 0.62, 0.75, -0.31, 0.44, 0.12, 0.08]

With all weights initialized to 0.005 (as in the test case):

```
   Layer 1, neuron 1:
     z = 0.005·0.48 + 0.005·0.62 + 0.005·0.75 + 0.005·(-0.31) + 0.005·0.44
       + 0.005·0.12 + 0.005·0.08 + 0.005      (bias)
     z = 0.005·(0.48+0.62+0.75-0.31+0.44+0.12+0.08) + 0.005
     z = 0.005·2.18 + 0.005 = 0.0109 + 0.005 = 0.0159
     h1_1 = tanh(0.0159) = 0.01590 (≈ linear for small input)

   All 5 neurons in layer 1 produce the same value (same weights):
     h1 = [0.01590, 0.01590, 0.01590, 0.01590, 0.01590]

   Layer 2, neuron 1:
     z = 5 · (0.005 · 0.01590) + 0.005 = 5·7.95e-5 + 0.005 = 0.005398
     h2_1 = tanh(0.005398) = 0.005397

   Output:
     y = 5 · (0.005 · 0.005397) + 0.005 = 5·2.699e-5 + 0.005 = 0.005135
     β_0 = 1.0 · (0.005135 + 1.0) = 1.005135

   ➤ With uniform 0.005 weights, β ≈ 1.005 — barely different from 1.0.
   ➤ The optimizer will adjust these weights to fit the reference data.
```

### 5.6 The Full Dependency Chain

```
   θ (NN weights)
   │
   │ DAInputRegressionPar::run()          [DAInputRegressionPar.C]
   │ ↳ daRegression.setParameter(θ[i])    [DARegression.H:176]
   ▼
   DARegression::compute()                [DARegression.C:354]
   │ ↳ calcInputFeatures(W)              [DARegression.C:164] → φ(W)
   │ ↳ NN forward pass with θ            [DARegression.C:416-484] → β
   ▼
   β (correction field)
   │
   │ Used in SA equation:
   │   Cb1 · S_tilde · nuTilda · β_i
   │   [DASpalartAllmaras.C:457]
   ▼
   R(W, β(θ, W)) = 0                     [DASimpleFoam.C:139 loop]
   │
   │ Primal solve: iterate SIMPLE until R = 0
   ▼
   W* (converged flow solution)
   │
   │ Evaluate objective:
   │   J = Σ_i scale · (U_i - U_ref_i)²
   │   [DAFunctionVariance.C:460-518]
   ▼
   J (scalar objective value)
```

### 5.7 dJ/dθ — How DAFoam Computes the Gradient

**There is no explicit NN Jacobian computation.** DAFoam uses end-to-end reverse-mode AD through the entire chain.

The gradient dJ/dθ is computed as:

```
   dJ        T    ∂R
   ── = ψ  · ────
   dθ         ∂θ
```

where ∂R/∂θ is the partial derivative of the residual with respect to NN weights. This captures the full chain: θ → NN → β → SA equation → R.

**Code:** `mphys_dafoam.py:409-424`:
```python
for inputName in list(inputs.keys()):     # inputName = "reg_model"
    inputType = inputDict[inputName]["type"]  # inputType = "regressionPar"
    DASolver.solverAD.calcJacTVecProduct(
        inputName, inputType,       # "reg_model", "regressionPar"
        jacInput,                   # current NN weights
        self.residualName, "residual",
        seed,                       # adjoint vector ψ
        product,                    # result: ψ^T · ∂R/∂θ = dJ/dθ
    )
```

**What happens inside `calcJacTVecProduct`:**

1. The CoDiPack reverse-mode AD tape records every operation:
   - `DAInputRegressionPar::run()` sets each weight → recorded on tape
   - `DARegression::compute()` computes features and NN forward pass → recorded
   - Turbulence model uses β → recorded
   - Residual computation → recorded

2. The tape is played backwards with the adjoint vector as seed:
   ```
   seed (ψ) → ∂R/∂β → ∂β/∂h2 → ∂h2/∂h1 → ∂h1/∂θ → product (dJ/dθ)
   ```

3. This gives the **exact** gradient — no finite-difference approximation for the NN part.

**Concrete gradient** (illustrative, 71 entries):

```
   dJ/dθ = [∂J/∂w1_1,1, ∂J/∂w1_1,2, ..., ∂J/∂w1_1,7, ∂J/∂b1_1,    ← layer 1, neuron 1
            ∂J/∂w1_2,1, ...,                                           ← layer 1, neurons 2-5
            ...,
            ∂J/∂w3_1, ..., ∂J/∂w3_5, ∂J/∂b3]^T                       ← output layer

   Example magnitudes (early optimization):
   dJ/dθ ≈ [3.2e-3, 4.1e-3, 4.9e-3, -2.0e-3, 2.8e-3, 0.8e-3, 0.5e-3, 2.1e-3,  ← layer 1
            ...
            1.1e-3, 0.9e-3, 0.8e-3, 0.7e-3, 0.6e-3, 0.3e-3]^T                    ← output
```

### 5.8 Shape Summary: Every Tensor in the Chain

```
   Tensor            │ Shape   │ Description
   ──────────────────┼─────────┼──────────────────────────────────────
   θ                 │ 71×1    │ NN weights (design variables)
   φ_i               │ 7×1     │ Feature vector for cell i
   W1, b1            │ 5×7, 5  │ Layer 1 weights and biases
   h1_i              │ 5×1     │ Hidden layer 1 output for cell i
   W2, b2            │ 5×5, 5  │ Layer 2 weights and biases
   h2_i              │ 5×1     │ Hidden layer 2 output for cell i
   W3, b3            │ 1×5, 1  │ Output layer weights and bias
   β                 │ 5×1     │ Correction field (all cells)
   W                 │ 25×1    │ State vector (all cells, all vars)
   R(W,β)            │ 25×1    │ Residual vector
   ∂R/∂W             │ 25×25   │ State Jacobian (sparse, ~60% non-zero)
   (∂R/∂W)^T         │ 25×25   │ Transpose (same sparsity)
   ∂J/∂W             │ 1×25    │ Objective sensitivity to states
   ψ                 │ 25×1    │ Adjoint vector
   ∂R/∂β             │ 25×5    │ Residual sensitivity to beta (very sparse)
   ∂R/∂θ             │ 25×71   │ Residual sensitivity to NN weights (via AD)
   dJ/dθ             │ 71×1    │ Total gradient (what optimizer needs)
```

---

## Stage 6: Optimization — Closing the Loop

### 6.1 The Optimization Problem

```
   minimize   J(θ) = Σ_cells |U(W*(θ)) - U_ref|²
      θ

   subject to:  R(W*(θ), β(θ, W*(θ))) = 0     (physics constraint, solved implicitly)
                θ_lower ≤ θ ≤ θ_upper            (weight bounds)
```

**Code:** `runRegTests_DASimpleFoamRegPar.py:130-134`:
```python
self.add_design_var("reg_model", lower=-100.0, upper=100.0, scaler=1.0, indices=[0, 50])
self.add_objective("scenario.aero_post.UVar", scaler=1.0)
self.add_constraint("scenario.aero_post.PVar", equals=0.3)
```

### 6.2 One Optimization Iteration

```
   Iteration k:

   1. Set NN weights:     θ_k → DARegression
      ┌──────────────────────────────────────┐
      │ DASolver.setSolverInput("reg_model",  │
      │   "regressionPar", θ_k)               │
      │ [DASolver.C:1477-1513]                │
      └──────────┬───────────────────────────┘
                 │
   2. Primal solve:       R(W, β(θ_k)) = 0 → W*_k
      ┌──────────┴───────────────────────────┐
      │ DASimpleFoam::solvePrimal()          │
      │ [DASimpleFoam.C:123-185]             │
      │                                       │
      │ Inside each SIMPLE iteration:         │
      │   • daRegressionPtr_->compute() ← β  │
      │   • UEqn, pEqn, turbulence           │
      └──────────┬───────────────────────────┘
                 │
   3. Evaluate:           J_k = J(W*_k)
      ┌──────────┴───────────────────────────┐
      │ DAFunctionVariance::calcFunction()   │
      │ [DAFunctionVariance.C:460-518]       │
      │ J = Σ scale·(U_i - U_ref_i)²        │
      └──────────┬───────────────────────────┘
                 │
   4. Adjoint solve:      (∂R/∂W)^T ψ = -(∂J/∂W)^T
      ┌──────────┴───────────────────────────┐
      │ DAFoamSolver.solve_linear()          │
      │ [mphys_dafoam.py:426-567]            │
      │                                       │
      │ Uses Krylov solver with:              │
      │   • Matrix-free (dR/dW)^T·v via AD   │
      │   • Explicit PC matrix from coloring  │
      └──────────┬───────────────────────────┘
                 │
   5. Gradient:           dJ/dθ_k = ψ^T · ∂R/∂θ
      ┌──────────┴───────────────────────────┐
      │ calcJacTVecProduct("reg_model",       │
      │   "regressionPar", ...)               │
      │ [mphys_dafoam.py:415-424]             │
      └──────────┬───────────────────────────┘
                 │
   6. Update:             θ_{k+1} = θ_k - α · H^{-1} · dJ/dθ_k
      ┌──────────┴───────────────────────────┐
      │ Optimizer (SNOPT/IPOPT via pyOptSparse│
      │ or OpenMDAO) uses dJ/dθ_k to compute │
      │ the next iterate θ_{k+1}.             │
      └──────────────────────────────────────┘
```

### 6.3 What the Optimizer Sees

From the optimizer's perspective, the interface is simple:

```
   Input:  θ ∈ R^71          ← "reg_model" design variables
   Output: J ∈ R^1           ← "UVar" objective
           C ∈ R^2           ← "PVar" and "UProbe" constraints
           dJ/dθ ∈ R^71      ← gradient of objective
           dC/dθ ∈ R^{2×71}  ← gradient of constraints
```

All the CFD, turbulence modeling, adjoint solving, and AD happens **inside** this black box. The optimizer just sees a smooth function from 71 inputs to 3 outputs, with exact gradients.

---

## Stage 7: The Tensor Progression Summary

Here is the complete data flow, with every tensor's shape and role:

```
θ ∈ R^71 ──► φ(W) ∈ R^{5×7} ──► NN(φ;θ) ──► β ∈ R^5 ──► R(W,β) = 0 ──► W* ∈ R^25 ──► J ∈ R
  NN weights    features from       forward       correction    SIMPLE         converged      objective
                flow solution       pass          field         loop           states

                            ◄── BACKWARD (adjoint) ───◄

J ∈ R ──► ∂J/∂W ∈ R^{1×25} ──► ψ ∈ R^25 ──► dJ/dθ ∈ R^71
  objective   hand-derivable      adjoint      gradient for
              or via AD           solve:       optimizer
                                  (∂R/∂W)^T ψ
                                  = -(∂J/∂W)^T
```

**The key insight**: in the forward direction, information flows from 71 weights through millions of operations to a single scalar. In the reverse direction, the adjoint propagates sensitivity from that single scalar back through the same operations to all 71 weights in **one pass**. This asymmetry -- 71 inputs, 1 output -- is precisely where the adjoint method provides its massive computational advantage.

---

## Appendix A: Why the Jacobian Structure Matters

### A.1 Sparsity → Coloring → Efficiency

```
   Real mesh (100K cells):

   N_s = 500,000 state variables
   Jacobian: 500,000 × 500,000 = 2.5 × 10^11 entries
   Non-zeros per row: ~15 (from FVM stencil)

   Without coloring: 500,000 residual evaluations
   With coloring:    ~15 residual evaluations        (~33,000x faster)

   This works because FVM stencil ≈ 3 cells wide in each direction,
   so you only need ~15 colors (one per stencil position × variable).
```

### A.2 Block Structure → Segregated Preconditioner

```
   The SIMPLE algorithm naturally decomposes the system:

   ┌──────────┬──────────┬──────────┐
   │  A_UU    │  A_Up    │    0     │    U block (momentum)
   ├──────────┼──────────┼──────────┤
   │  A_pU    │  A_pp    │    0     │    p block (pressure)
   ├──────────┼──────────┼──────────┤
   │  A_νU    │    0     │  A_νν   │    ν block (turbulence)
   └──────────┴──────────┴──────────┘

   The upper-right block A_Uν (how U depends on ν) is weak,
   the lower-left A_νU (how ν depends on U) is moderate.

   This block structure means:
   1. ILU preconditioners work well (fill stays local)
   2. The adjoint transpose has the SAME block structure
   3. Fixed-point iteration converges well
```

### A.3 Diagonal ∂R/∂β → Cheap Field Inversion Gradient

```
   Since ∂R/∂β has only N_c non-zeros (one per cell, in the ν-equation row),
   computing ψ^T · ∂R/∂β is just N_c scalar multiplications.

   No matrix assembly, no coloring, no finite differences needed for this part.

   The expensive part is the adjoint solve for ψ, which is O(N_s) regardless
   of how many design variables you have.
```

---

## Appendix B: Mapping to the DAFoam Test Case

The test case `runRegTests_DASimpleFoamRegPar.py` uses exactly this pipeline:

| Concept | Test Case Setting | Code Reference |
|---------|-------------------|----------------|
| Solver | `DASimpleFoam` | `DASimpleFoam.C:123` |
| Turb model | Spalart-Allmaras | `DASpalartAllmaras.C:457` |
| Beta output | `betaFINuTilda` | `DASpalartAllmaras.C:95-103` |
| Features | 7 features (VoS, PoD, ...) | `DARegression.C:182-284` |
| Architecture | [7]→[5]→[5]→[1] | `DARegression.C:403-484` |
| Activation | tanh | `DARegression.C:453-456` |
| N_parameters | 71 | `DARegression.C:652-689` |
| Objective | UVar (variance) | `DAFunctionVariance.C:460-518` |
| Design vars | indices [0, 50] of θ | `mphys_dafoam.py:409-424` |
| AD mode | Reverse (CoDiPack) | `mphys_dafoam.py:394-406` |
| Adjoint solve | Krylov (GMRES) | `mphys_dafoam.py:455-540` |
| PC matrix | Colored FD | `DAPartDeriv.C:350-473` |

To run:
```bash
cd tests
mpirun --oversubscribe -np 4 python runRegTests_DASimpleFoamRegPar.py
```
