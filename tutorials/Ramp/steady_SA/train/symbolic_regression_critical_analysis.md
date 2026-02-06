# Critical Analysis of Symbolic Regression Methods for FIML Turbulence Modeling

**Date:** 2026-02-06

---

## 1. Overview of Current SR Approaches in FIML

Symbolic regression (SR) discovers interpretable algebraic expressions from data. In the FIML context, SR aims to replace the black-box neural network correction `beta = NN(features)` with an explicit formula like `beta = 1 + 0.3*tanh(2.1*VoS - 0.8*PoD)`. Three distinct paradigms currently exist:

| Paradigm | Training | Physics Consistency | Interpretability |
|----------|----------|---------------------|------------------|
| **Decoupled SR** | FI → extract beta → fit SR to beta(features) | None during SR fit | High |
| **Coupled NN → Decoupled SR** (Distillation) | Coupled NN training → compress → fit SR | Only in NN stage | High |
| **Coupled SR** (CFD-in-the-loop) | SR expression evaluated inside CFD solver during training | Full | Highest |

---

## 2. Decoupled Symbolic Regression

### Method
Field inversion produces an optimal beta field for a given case. SR (typically PySR with genetic programming) then fits `beta = f(features)` as a supervised learning problem, independent of the CFD solver.

This is the approach implemented in DAFoam's `tutorials/Ramp/steady_SA/train/sr_training/`.

### Pros

- **Simplicity.** SR is a standalone post-processing step. No CFD solver modifications needed. PySR runs in Python/Julia with no OpenFOAM dependency.
- **Speed.** Fitting SR to a dataset of ~10,000 (feature, beta) pairs takes minutes, versus hours/days for coupled optimization.
- **Pareto front.** PySR produces a complexity-vs-accuracy frontier, letting the user choose the simplest acceptable expression.
- **Portability.** The resulting algebraic formula can be implemented in any CFD solver trivially—no NN inference infrastructure.
- **Mature tooling.** PySR (Cranmer, 2023) is well-maintained, supports constraint specification (operator nesting, complexity limits, unit constraints), and exports to Python/C++/LaTeX/SymPy.

### Cons

- **Model inconsistency.** The SR expression is fit to the FI beta field, not to the CFD objective. An expression that approximates beta well pointwise may produce a poor flow solution when inserted into the solver, because small errors in beta compound through the nonlinear PDE. This is the fundamental limitation.
- **FI beta is non-unique.** The field inversion optimization has many local minima—different initializations or regularization strengths produce different beta fields. SR inherits this ambiguity. The "true" correction is not well-defined.
- **Feature selection bottleneck.** PySR struggles with more than 4-5 input variables (combinatorial explosion of expression trees). The 11 DAFoam features must be pre-filtered, introducing human bias about which features matter.
- **Limited operator vocabulary.** Genetic programming searches over a fixed set of operators (`+, -, *, /, exp, tanh, sqrt`). If the true correction involves operators outside this set (e.g., conditional branches, min/max, special functions), SR will produce a poor approximation.
- **No uncertainty quantification.** A single symbolic expression provides no indication of reliability or extrapolation risk.

---

## 3. Hybrid Distillation (Coupled NN → Compressed NN → SR)

### Method
Train a large NN with coupled adjoint optimization (physics-consistent). Prune and compress the NN while maintaining physics consistency via coupled fine-tuning. Apply SR to the compressed NN (small input dimension, few parameters).

DAFoam implements this in `tutorials/Ramp/steady_SA/train/symbolic_distillation/`.

### Pros

- **Physics consistency preserved through compression.** The compressed NN is still coupled to the CFD solver, so pruning doesn't silently degrade the flow solution.
- **SR on a tractable target.** Compressing from [4 inputs, [20,20] hidden] to [3 inputs, [5] hidden] (541 → 21 parameters) makes the NN simple enough for SR to approximate accurately.
- **Feature importance as a byproduct.** Weight saliency analysis during compression reveals which features actually matter—more principled than manual selection.
- **Staged validation.** Each stage can be independently validated: large NN accuracy → compressed NN accuracy → SR expression accuracy, all against the CFD objective.

### Cons

- **Pipeline complexity.** Three sequential stages, each requiring configuration and validation. The total computational cost is dominated by the coupled training stages.
- **Compression may lose critical nonlinearities.** If the correction requires a specific nonlinear interaction between features, aggressive pruning may eliminate it. The coupled fine-tuning partially mitigates this, but cannot recover information destroyed by pruning.
- **SR still decoupled in the final step.** The last stage (SR fit to compressed NN) breaks physics consistency. The symbolic expression is only validated a posteriori by running it in the solver.
- **Diminishing returns on interpretability.** If the compressed NN already has 21 parameters and matches the CFD objective well, the additional step to SR may sacrifice accuracy for marginal interpretability gain.

---

## 4. Coupled (CFD-in-the-Loop) Symbolic Regression

### Method
Embed the symbolic expression directly inside the CFD solver. The expression's coefficients (and potentially its structure) are optimized by minimizing the CFD objective with the adjoint method. Recent work by Zhao et al. (2025) and Schmelzer et al. (2020) explores this direction, though practical implementations remain limited.

### Pros

- **Full model consistency.** The expression is trained against the actual CFD solution, not a proxy. Every coefficient is optimized to minimize the flow prediction error.
- **Fewest design variables.** A symbolic expression like `beta = 1 + a*tanh(b*VoS + c*PoD)` has 3 parameters, versus ~100+ for an NN. This drastically reduces the optimization problem size.
- **No black-box components.** The entire correction is transparent from training to deployment.
- **Guaranteed solver stability.** If the expression is bounded by construction (e.g., using tanh), the correction cannot produce extreme beta values that crash the solver.

### Cons

- **Structure search is the hard problem.** Gradient-based optimization can tune coefficients of a fixed expression, but cannot discover the expression structure. Genetic programming (PySR) is gradient-free and cannot be coupled to the adjoint. This creates a fundamental tension: the tool for structure search (GP) and the tool for physics-consistent training (adjoint) operate in incompatible optimization paradigms.
- **Combinatorial explosion.** The space of possible expressions grows super-exponentially with complexity. Exhaustive search is infeasible; heuristic search (GP) is noisy and sensitive to operator set, population size, and mutation rates.
- **Solver convergence sensitivity.** Evaluating each candidate expression requires a full CFD solve. Poor expressions crash the solver, wasting computational budget. Unlike NN training where gradients are smooth, SR fitness landscapes are discontinuous (a single operator change can flip the expression from converging to diverging).
- **Limited current implementations.** No mature, open-source framework exists for coupled CFD-SR. The Zhao et al. (2025) mutually coupled framework is the closest, but it uses data assimilation rather than direct adjoint optimization.
- **Scalability.** Each GP generation requires O(population_size) CFD solves. With population=30 and niterations=100, that is ~3,000 CFD solves—orders of magnitude more expensive than coupled NN training.

---

## 5. Can SR Be Directly Coupled in FIML?

This is the central forward-looking question. The answer is: **yes, partially, via a hybrid approach that separates structure search from coefficient optimization.**

### Proposed Architecture: Adjoint-Coupled Parametric SR

**Concept:** Use genetic programming to search over expression *templates* (structure), but optimize the coefficients of each template using the adjoint method (gradient-based, physics-consistent).

```
GP Outer Loop (gradient-free, structure search):
  For each candidate expression template T_k(features; a, b, c, ...):
    │
    ├── Adjoint Inner Loop (gradient-based, coefficient optimization):
    │     1. Insert T_k into CFD solver as beta = T_k(features; a,b,c)
    │     2. Run primal solve
    │     3. Evaluate objective J = ||u - u_ref||^2
    │     4. Run adjoint solve → dJ/da, dJ/db, dJ/dc
    │     5. Optimize coefficients via L-BFGS/SNOPT (few iterations)
    │     6. Return optimized J* for template T_k
    │
    └── GP uses J* as fitness for T_k
        Select, mutate, crossover → next generation of templates
```

### Why This Is Feasible

1. **Coefficient optimization is cheap with adjoints.** For a 3-5 parameter expression, the adjoint provides exact gradients. L-BFGS converges in ~10-20 iterations. Each iteration is one primal + one adjoint solve.
2. **Structure search space is manageable.** With 4 input features, ~6 operators, and max complexity 15-20, the search space is large but tractable for GP with a population of 20-50.
3. **DAFoam already has the infrastructure.** The `DAInputRegressionPar` mechanism treats NN weights as design variables. A symbolic expression's coefficients can be exposed the same way. The `DARegression::compute()` method would need to evaluate a symbolic expression instead of an NN, but the adjoint propagation through CoDiPack handles arbitrary differentiable expressions.
4. **Reduced design variables minimize FI non-uniqueness.** With only 3-5 coefficients, the optimization landscape has far fewer local minima than FI with N_cells variables. This directly addresses the non-uniqueness problem.

### Implementation Sketch for DAFoam

1. **New `modelType` in DARegression:** `"symbolicExpression"` alongside `"neuralNetwork"` and `"radialBasisFunction"`.
2. **Expression representation:** A fixed expression template stored as a string in `daOptions`, e.g., `"1.0 + p0 * tanh(p1 * VoS + p2 * PoD)"`. Parameters `p0, p1, p2` are the design variables.
3. **Expression evaluation:** Parse the template at initialization, evaluate per-cell using the feature fields. CoDiPack AD through the evaluation gives exact gradients.
4. **GP wrapper in Python:** A PySR-like outer loop that generates template strings, passes them to DAFoam for coupled coefficient optimization, and uses the optimized objective as fitness.

### Challenges and Mitigations

| Challenge | Mitigation |
|-----------|------------|
| GP outer loop requires many CFD evaluations | Warm-start coefficient optimization from parent expressions; use coarse mesh for GP screening, fine mesh for top candidates |
| Expression may crash solver | Wrap in bounded functions (tanh, sigmoid); reject candidates with beta outside [0.1, 5.0]; use short primal solve (100 iterations) for screening |
| Adjoint not available for non-differentiable operators (abs, max, min) | Restrict operator set to smooth operators; use smooth approximations (softmax, softabs) |
| Multi-case training multiplies cost | Parallelize across cases (MPI); use representative subset for GP, full set for final coefficient optimization |

---

## 6. How Coupled SR Reduces Design Variables in Field Inversion

Standard field inversion has N_cells design variables (one beta per cell). Coupled NN-FIML reduces this to ~100-500 NN weights. Coupled SR reduces it further to **3-10 expression coefficients**.

This has three consequences:

1. **Regularization is built-in.** With 5 parameters, the expression cannot overfit to noise in the reference data. The low-dimensional parameterization acts as an implicit regularizer far stronger than any explicit penalty term.

2. **Optimization is more robust.** Gradient-based optimizers (SNOPT, IPOPT) are designed for problems with O(10) variables. The adjoint Hessian is small enough to compute and invert, enabling Newton-type convergence.

3. **Multi-case training becomes tractable.** Training across 10 cases with 5 parameters is a 5-variable optimization problem. With NN weights (500 parameters), it requires careful scaler tuning and many iterations. With FI (50,000 variables across 10 cases), it is often infeasible.

The tradeoff is expressiveness: 5 parameters cannot capture the same complexity as 500 NN weights. But for many flows, the correction is actually simple—separation regions need more production, attached regions need less. A well-chosen symbolic template can capture this with far fewer parameters than an NN.

---

## 7. Further Advancements

### 7.1 LLM-Guided Expression Search

Recent work (Nature Scientific Reports, 2026) integrates LLMs with symbolic regression: the LLM proposes physically motivated expression templates based on turbulence theory, and GP refines the coefficients. This replaces blind combinatorial search with informed hypothesis generation. For FIML, an LLM could propose templates based on known deficiencies of the SA or SST model (e.g., curvature corrections, pressure gradient sensitivity).

### 7.2 Multi-Fidelity Coupled SR

Use cheap RANS-on-coarse-mesh evaluations to screen candidate expressions (GP outer loop), then validate top candidates with expensive fine-mesh coupled optimization. This could reduce the computational cost of coupled SR by 10-100x.

### 7.3 Constrained SR for Realizability

Enforce physical constraints directly in the expression structure:
- **Positivity:** `beta = 1 + softplus(expression)` ensures beta > 1 (only increases production)
- **Boundedness:** `beta = 1 + a*tanh(expression)` bounds correction magnitude
- **Limiting behavior:** Require `beta → 1` as `ReWall → 0` (no correction at wall) or as all features → 0 (freestream)
- **Unit consistency:** Recent work (Acta Mechanica, 2025) applies dimensional constraints to SR, rejecting dimensionally inconsistent expressions before evaluation

### 7.4 Differentiable SR via Relaxed Expression Trees

An emerging approach represents expression trees as continuous relaxations (soft operator selection via softmax over operator set). This makes the entire SR process differentiable, enabling end-to-end gradient-based training with the adjoint. The discrete expression is recovered by argmax at convergence. This eliminates the GP outer loop entirely, though current implementations (e.g., DSR, uDSR) have not been applied to FIML.

### 7.5 Symbolic Expressions as Priors for Bayesian FIML

Use the SR expression as a prior mean function in a Gaussian process correction: `beta(x) = f_SR(features) + GP(features)`. The SR expression captures the dominant correction, and the GP captures residual case-specific adjustments with calibrated uncertainty. This combines interpretability (SR) with flexibility (GP) and uncertainty quantification.

---

## 8. Summary: Recommended Path Forward

| Priority | Action | Rationale |
|----------|--------|-----------|
| **Immediate** | Implement adjoint-coupled coefficient optimization for fixed expression templates in DAFoam | Low implementation effort (extend `DARegression` with a `symbolicExpression` type); directly tests model-consistent SR |
| **Near-term** | Build GP-over-adjoint wrapper in Python | Automates structure search while maintaining physics consistency; leverages existing PySR infrastructure |
| **Medium-term** | Explore differentiable SR (relaxed expression trees) | Eliminates the GP/adjoint incompatibility; requires new research but has highest potential payoff |
| **Longer-term** | LLM-guided + multi-fidelity coupled SR | Combines domain knowledge with computational efficiency; depends on maturity of LLM-SR integration |

The key insight is that **the bottleneck is not SR itself, but the decoupling between expression discovery and physics-consistent training.** Any advance that tightens this coupling—whether through adjoint-optimized coefficients, differentiable expression trees, or GP-over-adjoint architectures—will yield disproportionate improvements in both accuracy and generalization.

---

## References

1. Parish, E.J. & Duraisamy, K. (2016). "A paradigm for data-driven predictive modeling using field inversion and machine learning." *J. Comput. Phys.*, 305, 758-774.
2. Cranmer, M. (2023). "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl." [arXiv:2305.01582](https://arxiv.org/abs/2305.01582)
3. Wu, J. et al. (2025). "Development of a Generalizable Data-driven Turbulence Model: Conditioned Field Inversion and Symbolic Regression." [AIAA Journal](https://arc.aiaa.org/doi/10.2514/1.J064416)
4. Zhao et al. (2025). "Data-driven turbulence modeling: A mutually coupled framework for symbolic regression and data assimilation." [Physics of Fluids](https://pubs.aip.org/aip/pof/article/37/7/075211/3356289)
5. Schmelzer, M. et al. (2020). "CFD-driven Symbolic Identification of Algebraic Reynolds-Stress Models." [arXiv:2104.09187](https://arxiv.org/pdf/2104.09187)
6. Mandler et al. (2025). "Interpretable data-driven turbulence modeling for separated flows using symbolic regression with unit constraints." [Acta Mechanica](https://link.springer.com/article/10.1007/s00707-025-04325-6)
7. Zhao et al. (2023). "A coupled framework for symbolic turbulence models from deep-learning." [Int. J. Heat Fluid Flow](https://www.sciencedirect.com/science/article/pii/S0142727X23000395)
8. Nature Scientific Reports (2026). "Knowledge integration for physics-informed symbolic regression using pre-trained large language models." [Nature](https://www.nature.com/articles/s41598-026-35327-6)
9. Weatheritt, J. & Sandberg, R. (2016). "A novel evolutionary algorithm applied to algebraic modifications of the RANS stress-strain relationship." *J. Comput. Phys.*, 325, 22-37.
10. He, P. et al. (2020). "DAFoam: An open-source adjoint framework for multidisciplinary design optimization with OpenFOAM." *AIAA Journal*.
