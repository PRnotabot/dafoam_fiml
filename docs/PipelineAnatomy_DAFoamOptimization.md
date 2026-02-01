# DAFoam Optimization Pipeline

**End-to-end gradient-based and surrogate-based CFD shape optimization using discrete adjoints on OpenFOAM.**

*Codebase: DAFoam (Discrete Adjoint with OpenFOAM) | Domain: Aerodynamic / Multidisciplinary Design Optimization*

---

## Notation Table

| Symbol | Meaning | Shape/Type | Code Variable |
|--------|---------|------------|---------------|
| W | State vector (U, p, phi, nuTilda) | N_s x 1 | `states`, `wVec` |
| X_v | Volume mesh coordinates | 3*N_cells x 1 | `volCoords`, `xvVec` |
| X_s | Surface mesh coordinates | 3*N_surf x 1 | `x_aero`, `surfCoords` |
| X_dv | Design variables (shape, twist, BC) | N_dv x 1 | `shape`, `twist`, `patchV` |
| R(W) | PDE residual vector | N_s x 1 | `resVec`, `URes`, `pRes` |
| F(W) | Objective function (CD, CL, etc.) | scalar | `funcName`, `functionValue` |
| dR/dW | State Jacobian | N_s x N_s (sparse) | `dRdWT`, `dRdWTPC` |
| psi | Adjoint variable | N_s x 1 | `self.psi` |
| dF/dX | Total derivative of F w.r.t. design | N_dv x 1 | `product`, `d_inputs` |

---

## Pipeline Overview Diagram

```
                          OPTIMIZATION LOOP
    +----------------------------------------------------------+
    |                                                          |
    v                                                          |
[Design Variables X_dv]                                        |
    |                                                          |
    v                                                          |
+-----------+     +-------------+     +------------------+     |
| Stage 1   |---->| Stage 2     |---->| Stage 3          |     |
| Geometry  |     | Mesh Warp   |     | Primal CFD Solve |     |
| & Config  |     | (IDWarp)    |     | (SIMPLE/PIMPLE)  |     |
+-----------+     +-------------+     +------------------+     |
                                             |                 |
                                             v                 |
                  +------------------+  +-----------+          |
                  | Stage 5          |<-| Stage 4   |          |
                  | Adjoint Solve    |  | Function  |          |
                  | (dR/dW)^T*psi   |  | Evaluation|          |
                  | = dF/dW         |  | (CD, CL)  |          |
                  +------------------+  +-----------+          |
                        |                                      |
                        v                                      |
                  +------------------+                         |
                  | Stage 6          |                         |
                  | Total Derivative |                         |
                  | dF/dX_dv         |                         |
                  +------------------+                         |
                        |                                      |
                        v                                      |
                  +------------------+                         |
                  | Stage 7          |-------------------------+
                  | Optimizer Update |
                  | (IPOPT/SNOPT/EGO)|
                  +------------------+
```

---

## Stage 1: Configuration and Geometry Setup

### What it does
Takes user-specified options (solver type, boundary conditions, objective functions, design variable definitions) and initializes the DAFoam solver, mesh warping, and geometry parameterization objects. Input: Python options dict. Output: initialized `PYDAFOAM` and `DAFoamBuilder` objects ready for optimization.

### Algorithm

```
1. User script defines daOptions dict and meshOptions dict
2. DAFoamBuilder.__init__() stores options, determines scenario type
3. DAFoamBuilder.initialize(comm):
   +---> PYDAFOAM(options, comm)
   |       +---> _solverRegistry()        -- register available solvers
   |       +---> _initializeOptions()     -- merge user options with DAOPTION defaults
   |       +---> _initSolver()            -- load C++ .so libraries via Cython
   |       |       +---> pyDASolvers(args, options)  -- C++ bridge
   |       |       +---> solver.initSolver()         -- OpenFOAM field init
   |       +---> setPrimalBoundaryConditions()
   |       +---> _readMeshInfo()
   +---> USMesh(mesh_options)             -- IDWarp mesh warping
   +---> DASolver.setMesh(mesh)           -- link mesh to solver
```

### Key insight
The three compilation modes (Original, ADR, ADF) coexist as separate `.so` libraries. The Original mode runs the primal and constructs preconditioner matrices. The ADR (reverse) mode handles adjoint solves and Jacobian-vector products via CoDiPack automatic differentiation. This separation means primal performance is unaffected by AD overhead.

### Data flow

```
[daOptions dict] + [meshOptions dict] + [OpenFOAM case files]
    |
    v
[Stage 1: Configuration]
    |
    v
[PYDAFOAM object] + [USMesh object] + [C++ solver initialized]
```

### Code mapping

**User script entry:**
`tests/runRegTests_DASimpleFoamRegPar.py:33-45` -- defines `daOptions`

**Builder initialization:**
`dafoam/mphys/mphys_dafoam.py:67-78`
```python
def initialize(self, comm):
    self.DASolver = PYDAFOAM(options=self.options, comm=comm)
    mesh = USMesh(options=self.mesh_options, comm=comm)
    self.DASolver.setMesh(mesh)
```

**C++ bridge loading:**
`dafoam/pyDAFoam.py:1418-1438` (inside `_initSolver`)
```python
from .libs.pyDASolvers import pyDASolvers       # Original mode
from .libs.ADR.pyDASolvers import pyDASolvers    # Reverse AD mode
self.solver.initSolver()
self.solverAD.initSolver()
```

**Cython bridge:**
`src/pyDASolvers/pyDASolvers.pyx:114-156` -- Python wrapper class
`src/pyDASolvers/DASolvers.H:58-61` -- C++ delegation to `DASolver::initSolver()`

**DAOPTION defaults:**
`dafoam/pyDAFoam.py:60-650` -- ~600 lines of default configuration values

---

## Stage 2: Mesh Warping

### What it does
Deforms the volume mesh based on updated surface coordinates from the geometry parameterization (FFD). Input: surface coordinates `x_aero` from geometry component. Output: deformed volume coordinates `aero_vol_coords` for CFD solve.

### Algorithm

```
1. Geometry component (pyGeo FFD) perturbs surface mesh based on design variables
2. DAFoamWarper receives new surface coordinates
3. IDWarp inverse-distance weighting propagates surface deformation into volume:

   X_v_new = X_v_base + W * (X_s_new - X_s_base)

   where W is the inverse-distance weighting matrix
4. Volume mesh quality is checked before proceeding
```

### Key insight
IDWarp uses an algebraic mesh warping approach (not PDE-based like Laplacian smoothing). This makes it fast enough for optimization loops but requires careful radius-of-influence parameters to avoid inverted cells. The mesh quality check at `mphys_dafoam.py:325` acts as a guard -- if mesh quality fails, the optimization iteration is flagged and the optimizer can reduce step size.

### Data flow

```
[X_dv design variables] --> [pyGeo FFD] --> [X_s surface coords, 3*N_surf x 1]
    |
    v
[Stage 2: IDWarp]
    |
    v
[X_v volume coords, 3*N_cells x 1]
```

### Code mapping

**OpenMDAO component:**
`dafoam/mphys/mphys_dafoam.py:797-840` -- `DAFoamWarper` class

```python
def compute(self, inputs, outputs):
    x_a = inputs["x_aero"].reshape((-1, 3))          # line ~825
    DASolver.setSurfaceCoordinates(x_a)               # line ~826
    DASolver.mesh.warpMesh()                          # line ~827
    outputs["dafoam_vol_coords"] = DASolver.mesh.getSolverGrid()  # line ~829
```

**Mesh quality check (before primal):**
`dafoam/mphys/mphys_dafoam.py:325`
```python
meshOK = DASolver.solver.checkMesh()
```
Routes to: `src/adjoint/DASolver/DASolver.H:554` -- `daCheckMeshPtr_->run()`

---

## Stage 3: Primal CFD Solve

### What it does
Solves the steady-state incompressible Navier-Stokes equations using the SIMPLE algorithm to obtain converged flow fields (U, p, phi, nuTilda). Input: volume mesh + boundary conditions. Output: converged state vector W.

### Governing equations

For incompressible steady-state flow (DASimpleFoam):

```
Momentum:    div(U U) = -grad(p) + div(nu_eff * grad(U))
             --------   --------   -----------------------
             convection  pressure   diffusion (incl. turbulent viscosity)

Continuity:  div(U) = 0
             ------
             mass conservation

Turbulence:  Transport equation for nuTilda (Spalart-Allmaras)
             div(U nuTilda) = ... (production - destruction + diffusion)
```

### Algorithm (SIMPLE)

```
Loop:  +--> Step 1: Assemble & solve momentum eqn (UEqnSimple.H)
       |              H * U* = -grad(p_old) + sources
       |    Step 2: Solve pressure correction eqn (pEqnSimple.H)
       |              div(1/A * grad(p')) = div(H/A)
       |    Step 3: Correct velocity and flux
       |              U = H/A - (1/A)*grad(p)
       |              phi = flux(U) + pressure correction
       |    Step 4: Solve turbulence model equations (TEqnSimple.H)
       |    Step 5: Evaluate objective functions
       |    Step 6: Check convergence (residual < primalMinResTol)
       +--- If not converged, repeat
```

### Key insight
DAFoam's SIMPLE implementation includes the `#include` pattern from OpenFOAM where the equation assembly is in separate `.H` files (`UEqnSimple.H`, `pEqnSimple.H`). This same structure is later reused for the adjoint: the fixed-point adjoint mirrors the SIMPLE iteration but operates on the transpose operators. The segregated nature of SIMPLE (solving U and p separately) is exploited in the adjoint to avoid assembling the full coupled Jacobian.

### Data flow

```
[X_v volume mesh] + [BCs: U0, p0, nuTilda0] --> [Stage 3: SIMPLE loop]
    |
    v
[W = {U, p, phi, nuTilda}, N_s x 1]  +  [function values: CD, CL]
```

### Code mapping

**Python trigger:**
`dafoam/pyDAFoam.py:789-810`
```python
def __call__(self):
    self.primalFail = self.solver.solvePrimal()
```

**OpenMDAO wrapper:**
`dafoam/mphys/mphys_dafoam.py:314-361`
```python
def solve_nonlinear(self, inputs, outputs):
    DASolver.set_solver_input(inputs, self.DVGeo)      # line 321
    meshOK = DASolver.solver.checkMesh()                # line 325
    DASolver()                                          # line 334 - runs primal
    states = DASolver.getStates()                       # line 353
    outputs[self.stateName] = states                    # line 354
```

**C++ SIMPLE loop:**
`src/adjoint/DASolver/DASimpleFoam/DASimpleFoam.C:123-185`
```cpp
label DASimpleFoam::solvePrimal() {
    // SIMPLE loop (line 149-157):
    #include "UEqnSimple.H"     // momentum equation
    #include "pEqnSimple.H"     // pressure correction
    #include "TEqnSimple.H"     // energy equation (optional)
    turbulencePtr_->correct();  // line 160
    this->calcAllFunctions(1);  // line 163 - evaluate objectives
}
```

**Residual computation:**
`src/adjoint/DAResidual/DAResidualSimpleFoam.H:35-45` -- stores `URes_`, `pRes_`, `phiRes_`

---

## Stage 4: Function Evaluation

### What it does
Computes scalar objective and constraint values (drag, lift, moment, etc.) from the converged flow field. Input: state vector W. Output: scalar function values.

### Algorithm

```
For each function in daOptions["function"]:
    1. Identify function type (force, moment, massFlowRate, etc.)
    2. Select mesh faces/cells via source selection (patchToFace, boxToCell, etc.)
    3. Compute function value from fields:
       - Force: F = integral_patch (p * n + tau) . d
       - Moment: M = integral_patch (r x (p * n + tau)) . axis
       - MassFlowRate: mdot = integral_patch (rho * U . n)
    4. Apply scaling factor
    5. Apply time operation (final, average, sum, variance)
```

### Key insight
Each function type is a separate class inheriting from `DAFunction` and registered via OpenFOAM's runtime selection table. This lets users define arbitrary combinations of objectives and constraints in the options dict without modifying C++ code. The `patchToFace` source selection pattern means functions are computed only on relevant boundary faces, keeping cost proportional to surface size rather than volume size.

### Data flow

```
[W state vector] + [function definition from daOptions]
    |
    v
[Stage 4: DAFunction::calcFunction()]
    |
    v
[F_i scalar values: CD=0.0275, CL=0.501, ...]
```

### Code mapping

**Python evaluation:**
`dafoam/pyDAFoam.py:906-928`
```python
def evalFunctions(self, funcs):
    for funcName in list(self.getOption("function").keys()):
        functionValue = self.solver.getTimeOpFuncVal(funcName)
        funcs[funcName] = functionValue
```

**OpenMDAO component:**
`dafoam/mphys/mphys_dafoam.py:723-736` -- `DAFoamFunctions.compute()`

**C++ function evaluation:**
`src/adjoint/DASolver/DASolver.H:534` -- `double calcFunction(const word functionName)`
`src/adjoint/DAFunction/DAFunction.H:139` -- `virtual scalar calcFunction() = 0`

**Example: Force function:**
`src/adjoint/DAFunction/DAFunctionForce.H:28-67`
```cpp
class DAFunctionForce : public DAFunction {
    vector forceDir_;      // projection direction
    word dirMode_;         // "parallelToFlow" or "fixedDirection"
    virtual scalar calcFunction();  // integral of (p*n + tau) . dir
};
```

34+ function types available in `src/adjoint/DAFunction/`.

---

## Stage 5: Adjoint Solve

### What it does
Solves the adjoint equation to compute the sensitivity of the objective function with respect to all state variables. Input: dF/dW (function sensitivity to states). Output: adjoint vector psi satisfying [dR/dW]^T * psi = dF/dW.

### Governing equation

```
Adjoint equation:   [dR/dW]^T * psi = dF/dW

where:
  dR/dW   = Jacobian of residual R w.r.t. states W (N_s x N_s, sparse)
  dF/dW   = partial derivative of objective F w.r.t. states W (N_s x 1)
  psi     = adjoint variable / Lagrange multiplier (N_s x 1)
```

### Algorithm

Two solution methods are available:

**Method A: Krylov (default)**
```
1. Compute preconditioner matrix:  PC = dR/dW^T  (coloring-based FD)
2. Create matrix-free KSP with MLR (Multi-Level Richardson)
3. Matrix-vector products via AD:  y = [dR/dW]^T * x  (CoDiPack reverse)
4. Solve with PETSc GMRES:
   Loop:  +--> Step 1: Compute r = dF/dW - [dR/dW]^T * psi_k
          |    Step 2: Apply PC^{-1} to r (Boomer AMG)
          |    Step 3: Update psi via Krylov subspace
          |    Step 4: Check convergence
          +--- If ||r|| / ||r_0|| > rtol, repeat
```

**Method B: Fixed-Point**
```
Mirrors SIMPLE structure on the adjoint:
Loop:  +--> Step 1: Compute adjoint residual
       |    Step 2: Solve adjoint U via inverse transpose of momentum operator
       |    Step 3: Solve adjoint p via inverse transpose of pressure operator
       |    Step 4: Update phi adjoint
       |    Step 5: Check convergence
       +--- If ||R_adj|| / ||R_adj_0|| > fpRelTol, repeat
```

### Key insight
The Krylov method uses a matrix-free approach: the Jacobian-vector product [dR/dW]^T * x is computed via CoDiPack reverse-mode AD rather than assembling and storing the full Jacobian. Only the preconditioner matrix PC is explicitly assembled using graph-coloring-accelerated finite differences. Graph coloring reduces the number of residual evaluations from N_states (millions) to N_colors (typically 50-100), making preconditioner construction tractable. The PC is reused across multiple optimization iterations via `adjPCLag`.

### Data flow

```
[dF/dW, N_s x 1] + [W converged states] + [PC matrix]
    |
    v
[Stage 5: PETSc KSP or Fixed-Point]
    |
    v
[psi adjoint vector, N_s x 1]
```

### Code mapping

**OpenMDAO adjoint solve:**
`dafoam/mphys/mphys_dafoam.py:426-563`
```python
def solve_linear(self, d_outputs, d_residuals, mode):
    dFdW = DASolver.array2Vec(d_outputs[self.stateName])        # line 448
    # Krylov path:
    DASolver.solver.calcdRdWT(1, DASolver.dRdWTPC)              # line 517
    DASolver.solverAD.createMLRKSPMatrixFree(DASolver.dRdWTPC, DASolver.ksp)  # line 522
    fail = DASolver.solverAD.solveLinearEqn(DASolver.ksp, dFdW, self.psi)     # line 540
    # Fixed-point path:
    fail = DASolver.solverAD.runFPAdj(dFdW, self.psi)           # line 553
    d_residuals[self.stateName] = DASolver.vec2Array(self.psi)   # line 563
```

**C++ Krylov infrastructure:**
`src/adjoint/DASolver/DASolver.H:279` -- `void calcdRdWT(const label isPC, Mat dRdWT)`
`src/adjoint/DASolver/DASolver.H:384` -- `static PetscErrorCode dRdWTMatVecMultFunction(...)` (matrix-free matvec)
`src/adjoint/DASolver/DASolver.H:289` -- `label solveLinearEqn(const KSP ksp, ...)`

**C++ Fixed-point adjoint:**
`src/adjoint/DASolver/DASimpleFoam/DASimpleFoam.C:189-851` -- `runFPAdj(Vec dFdW, Vec psi)`

**Graph coloring:**
`src/adjoint/DAJacCon/DAJacCon.H:262` -- `void calcJacConColoring()`
`src/adjoint/DAPartDeriv/DAPartDeriv.H:118` -- `void calcPartDerivMat(...)` (coloring-accelerated FD)

---

## Stage 6: Total Derivative Computation

### What it does
Combines the adjoint solution with partial derivatives to compute the total derivative of the objective function with respect to design variables. Input: adjoint vector psi + partial derivatives. Output: gradient dF/dX_dv for the optimizer.

### Governing equation

```
Total derivative via adjoint method:

dF/dX = partial(F)/partial(X) - psi^T * partial(R)/partial(X)
        ----------------------   ---------------------------------
        direct effect            indirect effect through states

This is computed as a Jacobian-transpose-vector product:
  product = [dOutput/dInput]^T * seed

For each design variable input X_i:
  dF/dX_i = [dF/dX_i]_direct + [dR/dX_i]^T * psi
```

### Algorithm

```
For each objective function F:
  For each design variable X_i (shape, twist, BC, etc.):
    1. Compute [dF/dX_i]^T * seed  via reverse-mode AD
    2. Compute [dR/dX_i]^T * psi   via reverse-mode AD
    3. Total: dF/dX_i = step_1 + step_2
    4. Pass gradient to OpenMDAO which routes it to the optimizer
```

### Key insight
The adjoint method computes gradients with respect to ALL design variables in cost proportional to ONE adjoint solve, regardless of the number of design variables. This is the fundamental advantage over finite differences (which require N_dv+1 primal solves) and forward-mode AD (which requires N_dv forward solves). For shape optimization with hundreds of FFD control points, this makes gradient computation practical.

### Data flow

```
[psi, N_s x 1] + [partial derivatives via AD]
    |
    v
[Stage 6: JacTVecProduct for each input/output pair]
    |
    v
[dF/dX_dv gradient, N_dv x 1] -- passed to optimizer
```

### Code mapping

**OpenMDAO Jacobian-vector products:**
`dafoam/mphys/mphys_dafoam.py:368-424` -- `DAFoamSolver.apply_linear()`
```python
def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
    # dR/dW^T * seed (state Jacobian product)
    DASolver.solverAD.calcJacTVecProduct(
        self.stateName, "stateVar", jacInput,
        self.residualName, "residual", seed, product)    # line ~398
    # dR/dX^T * seed (for each input)
    DASolver.solverAD.calcJacTVecProduct(
        inputName, inputType, jacInput,
        self.residualName, "residual", seed, product)    # line ~415
```

**Function sensitivities:**
`dafoam/mphys/mphys_dafoam.py:739-794` -- `DAFoamFunctions.compute_jacvec_product()`
```python
DASolver.solverAD.calcJacTVecProduct(
    inputName, inputType, jacInput,
    outputName, outputType, seed, product)
```

**C++ AD-based JacTVecProduct:**
`src/pyDASolvers/DASolvers.H:348-355`
```cpp
void calcJacTVecProduct(
    const word inputName, const word inputType, const double* input,
    const word outputName, const word outputType,
    const double* seed, double* product);
```

**Cython bridge:**
`src/pyDASolvers/pyDASolvers.pyx:205-232`

---

## Stage 7: Optimizer Update

### What it does
Takes the objective value and gradient from the adjoint pipeline and updates design variables using a nonlinear programming algorithm. Input: F, dF/dX_dv, constraint values and gradients. Output: updated design variables X_dv for next iteration.

### Algorithm

Two optimization paths are supported:

**Path A: Gradient-Based (pyOptSparse)**
```
Uses IPOPT or SNOPT interior-point / SQP methods:
Loop:  +--> Step 1: Evaluate F(X), g(X) via Stages 1-4
       |    Step 2: Compute dF/dX, dg/dX via Stages 5-6
       |    Step 3: Optimizer updates X:
       |              IPOPT: X_{k+1} = X_k + alpha * d  (barrier method)
       |              SNOPT: X_{k+1} via QP subproblem
       |    Step 4: Check KKT optimality conditions
       +--- If not converged, repeat from Stage 1
```

**Path B: Surrogate-Based (EGO)**
```
Efficient Global Optimization with Kriging surrogate:
1. Generate initial DOE (Design of Experiments) points
2. Evaluate CFD at each DOE point (parallel)
3. Build Kriging surrogate model
Loop:  +--> Step 4: Maximize Expected Improvement (EI) criterion
       |    Step 5: Evaluate CFD at suggested point
       |    Step 6: Update Kriging model
       +--- Repeat for n_iter iterations
7. Run final primal on best point
```

### Key insight
The gradient-based path (Path A) is more efficient for smooth problems with many design variables because each iteration only requires one primal + one adjoint solve. The surrogate-based path (Path B) is useful when the objective landscape is noisy or has multiple local optima, but scales poorly with dimension since it does not use gradient information -- each iteration requires a full primal CFD evaluation. The `surrogateOptimization` class coordinates parallel CFD evaluations across MPI ranks, with rank 0 running the EGO algorithm and worker ranks executing CFD solves.

### Data flow

```
[F, dF/dX_dv, constraints, constraint gradients]
    |
    v
[Stage 7: IPOPT/SNOPT/EGO]
    |
    v
[X_dv_new updated design variables] --> back to Stage 1
```

### Code mapping

**Gradient-based optimizer setup:**
`tests/runRegTests_AeroOpt.py:203-225`
```python
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = "IPOPT"
prob.driver.opt_settings = {"max_iter": 2, "tol": 1e-5, ...}
prob.run_driver()   # line 225 -- executes full optimization loop
```

**Surrogate-based optimization:**
`dafoam/pyDAFoam.py:2392-2709` -- `surrogateOptimization` class
```python
def run_optimization(self):
    # Rank 0: run EGO
    self.EGO()  # line 2531
    # Workers: listen for points, run CFD
    self.om_prob.run_model()  # line 2548

def EGO(self):
    ego = EGO(surrogate=KRG(design_space=design_space), ...)  # line 2611
    x_opt = ego.optimize(fun=self.obj_val)  # line 2662
```

**Feasible design finder:**
`dafoam/mphys/mphys_dafoam.py:1118-1240` -- `OptFuncs.findFeasibleDesign()`
Uses Newton's method to find initial design satisfying constraints.

---

## Connection Diagram (Full Pipeline)

```
[daOptions]   [meshOptions]   [FFD file]
     |              |              |
     v              v              v
+--------------------------------------------+
| Stage 1: Configuration & Geometry Setup    |
| DAFoamBuilder.initialize()                 |
| PYDAFOAM(options) + USMesh(mesh_options)   |
+--------------------------------------------+
     |
     | PYDAFOAM object, USMesh object
     v
+--------------------------------------------+
| Stage 2: Mesh Warping (DAFoamWarper)       |
| IDWarp: X_s --> X_v                        |
| Input: surface coords from FFD             |
| Output: volume coords for solver           |
+--------------------------------------------+
     |
     | X_v (3*N_cells doubles)
     v
+--------------------------------------------+
| Stage 3: Primal CFD Solve (DAFoamSolver)   |
| SIMPLE iteration: UEqn -> pEqn -> turb     |
| Input: X_v + BCs                           |
| Output: W = {U, p, phi, nuTilda}           |
+--------------------------------------------+
     |
     | W (N_s doubles, distributed via MPI)
     v
+--------------------------------------------+
| Stage 4: Function Evaluation               |
| DAFunction: CD, CL, moment, etc.           |
| Input: W                                   |
| Output: F_i scalar values                  |
+----+---------------------------------------+
     |                          |
     | F_i values               | dF/dW (via AD)
     v                          v
+--------------------------------------------+
| Stage 5: Adjoint Solve                     |
| [dR/dW]^T * psi = dF/dW                   |
| Krylov (GMRES + AMG PC) or Fixed-Point     |
| Input: dF/dW, PC matrix                    |
| Output: psi (N_s x 1)                      |
+--------------------------------------------+
     |
     | psi (N_s doubles)
     v
+--------------------------------------------+
| Stage 6: Total Derivative                  |
| dF/dX = dF/dX_direct + psi^T * dR/dX      |
| AD-based JacTVecProduct for each X_i       |
| Output: gradient N_dv x 1                  |
+--------------------------------------------+
     |
     | dF/dX_dv (N_dv doubles)
     v
+--------------------------------------------+
| Stage 7: Optimizer Update                  |
| IPOPT/SNOPT (gradient) or EGO (surrogate)  |
| Output: X_dv_new for next iteration        |
+----+---------------------------------------+
     |
     +-----------> Back to Stage 2 (or Stage 1 if BC changes)
```

---

## Summary Table

| Stage | Input(s) | Output(s) | Key Algorithm | Code Entry |
|-------|----------|-----------|---------------|------------|
| 1. Config | daOptions, meshOptions | PYDAFOAM, USMesh | Option merging, C++ init | `mphys_dafoam.py:67` |
| 2. Mesh Warp | X_s surface coords | X_v volume coords | Inverse-distance weighting | `mphys_dafoam.py:797` |
| 3. Primal | X_v, BCs | W = {U,p,phi,nuTilda} | SIMPLE (segregated P-V) | `pyDAFoam.py:789` -> `DASimpleFoam.C:123` |
| 4. Functions | W | CD, CL, etc. | Surface integration | `pyDAFoam.py:906` -> `DAFunction.H:139` |
| 5. Adjoint | dF/dW, PC | psi | GMRES + AMG or Fixed-Point | `mphys_dafoam.py:426` -> `DASolver.H:289` |
| 6. Gradient | psi, partial derivs | dF/dX_dv | AD JacTVecProduct | `mphys_dafoam.py:368` -> `DASolvers.H:348` |
| 7. Optimize | F, dF/dX_dv | X_dv_new | IPOPT/SNOPT or EGO+Kriging | `prob.run_driver()` or `pyDAFoam.py:2531` |

---

## Appendix: Mapping to Example Test Case

The simplest test exercising the full gradient pipeline is `runRegTests_DASimpleFoamRegPar.py` (ConvergentChannel with neural network regression parameters).

| Concept | Example Setting | Code Reference |
|---------|----------------|----------------|
| Solver | DASimpleFoam (incompressible steady) | `tests/runRegTests_DASimpleFoamRegPar.py:35` |
| Test geometry | ConvergentChannel mesh | `tests/runRegTests_DASimpleFoamRegPar.py:20` |
| Design variables | Neural network regression weights | `tests/runRegTests_DASimpleFoamRegPar.py:38` |
| Objective functions | PVar, UProbe, UVar | `tests/runRegTests_DASimpleFoamRegPar.py` (options) |
| Boundary conditions | U0=10.0, p0=0.0, nuTilda0=4.5e-5 | `tests/runRegTests_DASimpleFoamRegPar.py:29-31` |
| Adjoint verification | Forward AD vs Reverse adjoint comparison | Reference: `tests/refs/DAFoam_Test_DASimpleFoamRegParRef.txt` |
| Reference CD derivative | -0.0000193426695999 (adjoint) vs -0.0000193426691065 (forward AD) | `tests/refs/DAFoam_Test_DASimpleFoamRegParRef.txt:9-11` |
| Full optimization test | AeroOpt with IPOPT, NACA0012, shape+patchV DVs | `tests/runRegTests_AeroOpt.py` |
| Surrogate optimization | AeroOptSBO with EGO+Kriging | `tests/runRegTests_AeroOptSBO.py` |

---

## Appendix: Python-C++ Bridge Architecture

The bridge between Python and C++ uses Cython:

```
Python (pyDAFoam.py)
    |
    | imports pyDASolvers module
    v
Cython (pyDASolvers.pyx)
    |
    | wraps C++ class with numpy array <-> C pointer conversion
    v
C++ Wrapper (DASolvers.H/C)
    |
    | delegates to actual solver via autoPtr<DASolver>
    v
C++ Core (DASolver -> DASimpleFoam, etc.)
    |
    | uses OpenFOAM fields, PETSc matrices
    v
OpenFOAM + PETSc
```

Key bridge files:
- `src/pyDASolvers/pyDASolvers.pyx:114-469` -- Cython wrapper (60+ methods)
- `src/pyDASolvers/DASolvers.H:1-578` -- C++ delegation class
- `src/pyDASolvers/DASolvers.C:15-23` -- Constructor: `DASolver::New(argsAll, pyOptions)`

Three compiled `.so` libraries coexist in `dafoam/libs/`:
- `pyDASolvers.so` (Original) -- primal solve, PC matrix construction
- `ADR/pyDASolvers.so` (Reverse AD) -- adjoint solve, JacTVecProduct
- `ADF/pyDASolvers.so` (Forward AD, optional) -- forward-mode verification
