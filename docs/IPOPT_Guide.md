# IPOPT: A Comprehensive Technical Guide

## The Complete Inner Workings of a Primal-Dual Interior Point Optimizer

> *Based on the foundational work of Andreas Wächter (IBM Research) and Lorenz T. Biegler (Carnegie Mellon University).*
> *Primary references: [Wächter & Biegler (2006) Mathematical Programming](https://link.springer.com/article/10.1007/s10107-004-0559-y) and [IPOPT Documentation](https://coin-or.github.io/Ipopt/).*

---

## Table of Contents

1. [What IPOPT Solves](#1-what-ipopt-solves)
2. [The Big Picture: Interior Point as a Strategy](#2-the-big-picture-interior-point-as-a-strategy)
3. [Problem Reformulation: Barriers and Slacks](#3-problem-reformulation-barriers-and-slacks)
4. [The Primal-Dual System and KKT Conditions](#4-the-primal-dual-system-and-kkt-conditions)
5. [Computing the Newton Step](#5-computing-the-newton-step)
6. [Inertia Correction and Regularization](#6-inertia-correction-and-regularization)
7. [Step Acceptance: The Filter Line Search](#7-step-acceptance-the-filter-line-search)
8. [Second-Order Correction and the Maratos Effect](#8-second-order-correction-and-the-maratos-effect)
9. [Barrier Parameter Update Strategies](#9-barrier-parameter-update-strategies)
10. [Feasibility Restoration Phase](#10-feasibility-restoration-phase)
11. [Convergence Criteria](#11-convergence-criteria)
12. [Hessian Handling: Exact vs. Quasi-Newton](#12-hessian-handling-exact-vs-quasi-newton)
13. [Linear Solver Options and Their Impact](#13-linear-solver-options-and-their-impact)
14. [Scaling: How and Why It Matters](#14-scaling-how-and-why-it-matters)
15. [Warm Starting: Challenges and Approaches](#15-warm-starting-challenges-and-approaches)
16. [Key Options and Their Mathematical Effects](#16-key-options-and-their-mathematical-effects)
17. [Diagnostic Reading: Understanding IPOPT Output](#17-diagnostic-reading-understanding-ipopt-output)
18. [Common Pitfalls and Troubleshooting](#18-common-pitfalls-and-troubleshooting)
19. [IPOPT vs. SNOPT: A Detailed Comparison](#19-ipopt-vs-snopt-a-detailed-comparison)
20. [IPOPT vs. SLSQP, L-BFGS-B, Adam, and Other Solvers](#20-ipopt-vs-slsqp-l-bfgs-b-adam-and-other-solvers)
21. [IPOPT in Practice: pyOptSparse and OpenMDAO](#21-ipopt-in-practice-pyoptsparse-and-openmdao)
22. [References](#22-references)

---

## 1. What IPOPT Solves

IPOPT (Interior Point OPTimizer) solves the general nonlinear programming problem:

```
                    minimize    f(x)
                       x

                    subject to  g_L <= g(x) <= g_U       (general constraints)
                                x_L <=  x   <= x_U       (bound constraints)
```

where:
- `x in R^n` is the vector of decision variables
- `f: R^n -> R` is a smooth (at least twice continuously differentiable) objective function
- `g: R^n -> R^m` is a vector of smooth constraint functions
- `g_L, g_U` are lower/upper bounds on constraints (set `g_L_i = g_U_i` for equality, `-inf` or `+inf` to remove a bound)
- `x_L, x_U` are lower/upper bounds on variables

**Key assumptions:**
1. All functions are **smooth** (C^2 continuous). IPOPT is *not* designed for non-smooth, noisy, or discontinuous problems.
2. **First derivatives** (gradients, Jacobians) are available — ideally **second derivatives** (Hessians) as well.
3. The constraint Jacobian and Hessian of the Lagrangian are **sparse**. IPOPT exploits sparsity throughout its linear algebra.

### What Distinguishes IPOPT

Unlike active-set methods (SNOPT, SLSQP) that work on the boundary of the feasible region, IPOPT approaches the solution **from the interior**. Variables are kept strictly within their bounds at every iteration, converging to the boundary only at the solution. This fundamental difference shapes every aspect of the algorithm.

### The Lagrangian

The Lagrangian function, central to IPOPT's formulation, is:

```
         L(x, lambda, z^L, z^U) = f(x) + lambda^T g(x)
                                   - (z^L)^T (x - x_L) - (z^U)^T (x_U - x)
```

where:
- `lambda in R^m` are the Lagrange multipliers for general constraints
- `z^L, z^U >= 0` are the multipliers for lower and upper bound constraints

At a local optimum satisfying constraint qualifications, the KKT conditions hold:
```
         nabla_x L = 0              (stationarity)
         g(x) feasible               (primal feasibility)
         z^L, z^U >= 0               (dual feasibility)
         z^L_i (x_i - x_L_i) = 0    (complementarity, lower bounds)
         z^U_i (x_U_i - x_i) = 0    (complementarity, upper bounds)
```

---

## 2. The Big Picture: Interior Point as a Strategy

The interior point method (also called the barrier method) is a strategy for solving constrained NLPs by converting bound constraints into a smooth penalty term in the objective. The key insight: instead of tracking which constraints are active (as SQP methods do), replace the hard boundary with a logarithmic barrier that becomes infinitely repulsive at the boundary.

### The Interior Point Philosophy

```
         ┌──────────────────────────────────────────────────────────────┐
         │                    ITERATION k                               │
         │                                                              │
         │  1. At current point x_k (strictly interior to bounds):      │
         │     - Evaluate f(x_k), g(x_k)                               │
         │     - Evaluate gradients: nabla f, nabla g (Jacobian)        │
         │     - Evaluate Hessian of Lagrangian (or L-BFGS approx)      │
         │                                                              │
         │  2. Form the primal-dual KKT system:                         │
         │     - Augmented system with barrier terms                    │
         │     - Include diagonal contribution Sigma from bounds        │
         │                                                              │
         │  3. Factor and solve the KKT system:                         │
         │     - Check inertia (correct eigenvalue signature)           │
         │     - Apply regularization if needed                         │
         │     - Iterative refinement for accuracy                      │
         │                                                              │
         │  4. Line search with filter:                                 │
         │     - Fraction-to-the-boundary rule (stay interior)          │
         │     - Filter acceptance or Armijo condition                   │
         │     - Second-order correction if needed                      │
         │                                                              │
         │  5. Update barrier parameter mu:                             │
         │     - Monotone: decrease after subproblem solved              │
         │     - Adaptive: recompute at every iteration                 │
         │                                                              │
         │  6. Update: x_{k+1} = x_k + alpha_p * d_x                   │
         │             lambda_{k+1} = lambda_k + alpha_d * d_lambda     │
         │                                                              │
         │  7. Check convergence -> if not, goto step 1                 │
         └──────────────────────────────────────────────────────────────┘
```

### Why the Barrier Approach?

The logarithmic barrier transforms bound constraints into a smooth penalty:

```
         x >= 0   becomes   -mu * ln(x)   added to the objective
```

As `mu -> 0`, the barrier solutions trace the **central path** converging to the constrained optimum. The beauty is that:
- No combinatorial active-set identification is needed
- The iteration count is largely **independent** of the number of inequality constraints
- Each iteration has a predictable, structured linear algebra cost

### Contrast with SQP (SNOPT)

```
         ┌──────────────────────────────────┬────────────────────────────────┐
         │     Interior Point (IPOPT)        │     SQP (SNOPT)                │
         ├──────────────────────────────────┼────────────────────────────────┤
         │ Iterates stay strictly interior   │ Iterates can be on boundaries  │
         │ Solve Newton system per iter      │ Solve QP subproblem per iter   │
         │ Cost ~ independent of # ineq     │ Cost grows with active set     │
         │ Barrier parameter mu -> 0         │ No barrier parameter           │
         │ Filter line search               │ Merit function line search      │
         │ Natural for many inequalities     │ Natural for few active constr  │
         └──────────────────────────────────┴────────────────────────────────┘
```

---

## 3. Problem Reformulation: Barriers and Slacks

### Slack Variable Introduction

IPOPT first introduces slack variables to convert inequality constraints to equalities. For a constraint:

```
         g_L_i <= g_i(x) <= g_U_i
```

IPOPT introduces slacks and writes:

```
         g_i(x) - s_i = 0,     g_L_i <= s_i <= g_U_i
```

After this transformation, the problem has only equality constraints plus bound constraints on both original variables and slacks.

### The Barrier Subproblem

Replace all bound constraints with logarithmic barrier terms. For variables `x_i` with bounds `x_L_i <= x_i <= x_U_i`:

```
         minimize    phi_mu(x, s) = f(x) - mu * sum_i ln(x_i - x_L_i)
                                         - mu * sum_i ln(x_U_i - x_i)
                                         - mu * sum_j ln(s_j - g_L_j)
                                         - mu * sum_j ln(g_U_j - s_j)

         subject to  g(x) - s = 0
```

where `mu > 0` is the **barrier parameter**. The logarithmic terms enforce strict interiority:
- As `x_i -> x_L_i`: `ln(x_i - x_L_i) -> -infinity`, penalty `-> +infinity`
- The barrier creates an invisible wall that keeps iterates away from bounds

### The Central Path

As `mu` decreases from a large value to zero, the sequence of barrier minimizers `{x*(mu)}` traces a smooth curve called the **central path**:

```
         mu = 10:     x*(mu) deep in interior (far from boundaries)
         mu = 1:      x*(mu) moving toward boundaries
         mu = 0.01:   x*(mu) close to constrained optimum
         mu -> 0:     x*(mu) -> x* (solution of original NLP)

         Feasible Region (2D example):

              ┌───────────────────────┐
              │     mu=10             │
              │       ●               │
              │        ╲              │
              │         ●  mu=1      │
              │          ╲            │
              │           ●  mu=0.01  │
              │            ╲          │
              │             ◆ x*      │   ◆ = constrained optimum (on boundary)
              └───────────────────────┘
```

### Why Logarithmic Barriers?

1. **Self-concordance**: The logarithmic barrier is self-concordant, ensuring well-behaved Newton steps
2. **Complementarity**: At the barrier optimum, `mu / (x_i - x_L_i) = z^L_i`, naturally generating multiplier estimates
3. **Smooth path**: The central path is smooth, enabling efficient path-following

---

## 4. The Primal-Dual System and KKT Conditions

### Perturbed KKT Conditions

Applying first-order optimality conditions to the barrier subproblem yields the **perturbed KKT system**:

```
         (1)  nabla f(x) + J(x)^T lambda - z^L + z^U = 0     (stationarity)
         (2)  g(x) - s = 0                                     (primal feasibility)
         (3)  X^L Z^L e = mu e                                 (perturbed complementarity, lower)
         (4)  X^U Z^U e = mu e                                 (perturbed complementarity, upper)
```

where:
- `J(x) = nabla g(x)` is the `m x n` constraint Jacobian
- `X^L = diag(x_i - x_L_i)`, `X^U = diag(x_U_i - x_i)` are diagonal slack matrices
- `Z^L = diag(z^L_i)`, `Z^U = diag(z^U_i)` are diagonal bound multiplier matrices
- `e = (1, 1, ..., 1)^T`

When `mu = 0`, condition (3) becomes `x_i z^L_i = 0` — exact complementarity — recovering the standard KKT conditions.

### The Newton Step

Applying Newton's method to the perturbed KKT system (1)-(4) and eliminating the bound multiplier directions from conditions (3)-(4), we obtain the **augmented system**:

```
         [ W + Sigma    J^T ] [ d_x     ]   [ r_d ]
         [                   ] [         ] = [     ]
         [ J             0  ] [ d_lambda ]   [ r_p ]
```

where:
- `W = nabla^2_{xx} L(x, lambda)` is the **Hessian of the Lagrangian**:
  ```
  W = nabla^2 f(x) + sum_{i=1}^m lambda_i * nabla^2 g_i(x)
  ```
- `Sigma` is a **positive diagonal matrix** from the barrier terms:
  ```
  Sigma = (X^L)^{-1} Z^L + (X^U)^{-1} Z^U
  ```
  This encodes the "stiffness" of each bound constraint. Near an active bound, `Sigma_ii -> infinity`.
- `J` is the constraint Jacobian
- `r_d = -(nabla f + J^T lambda - z^L + z^U)` is the dual residual
- `r_p = -(g(x) - s)` is the primal residual

### Recovering Bound Multiplier Directions

After solving the augmented system for `(d_x, d_lambda)`, the bound multiplier directions are computed explicitly:

```
         d_z^L = (X^L)^{-1} (mu e - Z^L X^L e - Z^L d_x^L)
         d_z^U = (X^U)^{-1} (mu e - Z^U X^U e + Z^U d_x^U)
```

where `d_x^L` and `d_x^U` are the components of `d_x` corresponding to variables with lower and upper bounds, respectively.

### The Role of Sigma

The diagonal `Sigma` is IPOPT's mechanism for handling bounds without an active set:

```
         Variable far from bound:     Sigma_ii ≈ 0      (bound has little effect)
         Variable near lower bound:   Sigma_ii ≈ z^L_i / (x_i - x_L_i) >> 1
         Variable near upper bound:   Sigma_ii ≈ z^U_i / (x_U_i - x_i) >> 1
```

As `mu -> 0` and variables approach their active bounds, `Sigma_ii -> infinity` for those variables, effectively **eliminating them** from the Newton system — the interior-point analog of fixing variables at bounds in an active-set method.

### Geometric Intuition

```
         Full space R^n
         ┌────────────────────────────────────────┐
         │                                         │
         │    W + Sigma ≈ W   (far from bounds)   │
         │    Newton step ≈ standard Newton        │
         │                                         │
         │              Near bound x_i = x_L_i:    │
         │              Sigma_ii -> infinity        │
         │              d_x_i -> 0 (forced small)  │
         │              Variable "frozen" at bound  │
         │                                         │
         └────────────────────────────────────────┘
```

---

## 5. Computing the Newton Step

### The Augmented System

The core linear system at each IPOPT iteration is:

```
         K = [ W + Sigma    J^T ]     (n+m) x (n+m) symmetric matrix
             [ J             0  ]
```

This is a **symmetric indefinite** matrix. At a solution with the correct inertia:
- The (1,1) block `W + Sigma` should be positive definite on the null space of `J`
- The (2,2) block is zero
- `K` has exactly `n` positive and `m` negative eigenvalues

### Factorization Approaches

IPOPT uses **symmetric indefinite factorization**:

```
         K = P L D L^T P^T
```

where:
- `P` is a permutation matrix (from fill-reducing ordering: AMD, METIS, etc.)
- `L` is unit lower triangular
- `D` is block diagonal with `1x1` and `2x2` blocks

The inertia (number of positive, negative, zero eigenvalues) is determined from `D`.

### Why Not Normal Equations?

An alternative is to eliminate `d_x` and form the **condensed (normal equations) system**:

```
         J (W + Sigma)^{-1} J^T d_lambda = ...     (m x m system)
```

This is smaller but:

```
         ┌─────────────────────────────────────────────────────────────┐
         │  Augmented System              │  Normal Equations           │
         ├─────────────────────────────────┼─────────────────────────── │
         │  Size: (n+m) x (n+m)           │  Size: m x m               │
         │  Condition: kappa(K)            │  Condition: kappa(K)^2     │
         │  Handles rank-deficient J       │  Fails if J rank-deficient │
         │  ~40% more work per factor      │  Cheaper factorization     │
         │  More numerically stable        │  Less numerically stable   │
         │  Gives inertia information      │  Inertia not clear         │
         └─────────────────────────────────┴─────────────────────────── ┘
```

IPOPT uses the augmented system by default for its superior numerical properties.

### Iterative Refinement

After the initial solve `K * d = r`, IPOPT improves accuracy via iterative refinement:

```
         1. Compute residual:  r_new = r - K * d
         2. Solve:             K * delta_d = r_new
         3. Update:            d <- d + delta_d
         4. Repeat if needed
```

Controlled by `min_refinement_steps` (default 1) and `max_refinement_steps` (default 10). More refinement steps help with ill-conditioned systems but increase cost.

### Fraction-to-the-Boundary Rule

After computing the Newton direction, the step size is limited to maintain strict interiority:

```
         alpha_p^max = max { alpha in (0, 1] : x + alpha * d_x >= (1 - tau) * x_L
                                                x + alpha * d_x <= x_U - (1 - tau) * (x_U - x) }

         alpha_d^max = max { alpha in (0, 1] : z^L + alpha * d_z^L >= (1 - tau) * z^L
                                                z^U + alpha * d_z^U >= (1 - tau) * z^U }
```

where `tau = max(tau_min, 1 - mu)` ensures that variables and multipliers stay strictly positive. As `mu -> 0`, `tau -> 1`, allowing the step to approach the boundary more closely.

**Note:** IPOPT uses **separate step sizes** for primal (`alpha_p`) and dual (`alpha_d`) variables. This primal-dual approach is more aggressive than a pure primal method and is key to IPOPT's efficiency.

---

## 6. Inertia Correction and Regularization

### Why Inertia Matters

For the Newton direction to be a descent direction for the barrier objective, the augmented matrix `K` must have the correct inertia:

```
         Required inertia of K:  (n positive, m negative, 0 zero)

         ┌────────────────────────────────────────────────────────────┐
         │  If inertia is (n, m, 0):   Correct — good search dir     │
         │  If fewer than n positive:  W+Sigma not pos. def. on      │
         │                             null(J) — direction may ascend │
         │  If fewer than m negative:  J is rank-deficient —          │
         │                             constraints are degenerate     │
         │  If zero eigenvalues:       System is singular             │
         └────────────────────────────────────────────────────────────┘
```

### The Correction Mechanism

When the inertia is wrong, IPOPT modifies the augmented system:

```
         K_delta = [ W + Sigma + delta_x I    J^T          ]
                   [ J                        -delta_c I    ]
```

where:
- `delta_x > 0` is **Hessian regularization** — makes the (1,1) block more positive definite
- `delta_c >= 0` is **constraint regularization** — handles rank-deficient Jacobians by perturbing the (2,2) block

### Perturbation Schedule

IPOPT uses an adaptive schedule to find the smallest regularization that corrects the inertia:

```
         ┌─────────────────────────────────────────────────────────────┐
         │  Step 1: Try delta_x = 0, delta_c = 0                      │
         │          Factor K and check inertia                         │
         │                                                             │
         │  Step 2: If wrong inertia:                                  │
         │          If first time:  delta_x = first_hessian_perturbation│
         │          If recurring:   delta_x *= perturb_inc_fact        │
         │                                                             │
         │  Step 3: If still wrong:                                    │
         │          Keep increasing delta_x until correct inertia      │
         │          or delta_x > max_hessian_perturbation (give up)    │
         │                                                             │
         │  Step 4: If zero eigenvalues in (2,2) block:                │
         │          Set delta_c > 0 to regularize constraints          │
         │                                                             │
         │  Next iteration: If previous delta_x worked,                │
         │          Try delta_x_new = delta_x_old * perturb_dec_fact   │
         │          (attempt to reduce perturbation over time)          │
         └─────────────────────────────────────────────────────────────┘
```

### Key Regularization Parameters

| Option | Default | Effect |
|--------|---------|--------|
| `min_hessian_perturbation` | 1e-20 | Floor for delta_x |
| `max_hessian_perturbation` | 1e40 | Ceiling for delta_x |
| `first_hessian_perturbation` | 1e-4 | Initial delta_x when correction is needed |
| `perturb_inc_fact` | 8 | Multiplicative increase when correction is insufficient |
| `perturb_dec_fact` | 1/3 | Multiplicative decrease when correction is no longer needed |
| `perturb_inc_fact_first` | 100 | Faster increase on the very first correction |

### Diagnostic: The `lg(rg)` Column

In IPOPT's iteration output, the column `lg(rg)` shows `log10(delta_x)`:

```
         lg(rg) = -         No regularization needed (ideal)
         lg(rg) = -4        delta_x = 1e-4 (mild regularization)
         lg(rg) = -1        delta_x = 0.1 (significant regularization)
         lg(rg) =  2        delta_x = 100 (heavy regularization — trouble)
```

**Healthy optimization**: `lg(rg) = -` most iterations, possibly small values early on.
**Unhealthy optimization**: persistent or growing `lg(rg)` indicates the Hessian is indefinite or the problem is poorly conditioned.

### Inertia-Free Alternative

For linear solvers that do not provide inertia information (e.g., some iterative solvers), IPOPT offers an **inertia-free mode** using curvature tests:

```
         If d^T (W + Sigma) d < neg_curv_test_tol * ||d||^2:
            → negative curvature detected → add regularization
```

This is controlled by `neg_curv_test_tol` and enables IPOPT to work with a wider range of linear solvers.

---

## 7. Step Acceptance: The Filter Line Search

### The Bi-Objective Perspective

IPOPT's default globalization strategy treats the problem as a **bi-objective optimization** with two competing goals:

```
         Goal 1: Minimize the barrier objective     phi_mu(x)
         Goal 2: Minimize constraint violation       theta(x) = ||g(x) - s||
```

Rather than combining these into a single scalar merit function (as SNOPT does), IPOPT uses a **filter** — a set of previously visited points that define what "acceptable progress" means.

### The Filter

A filter `F` is a set of pairs `{(theta_j, phi_j)}` representing constraint violation and objective value at previous iterates. A new trial point `(theta^trial, phi^trial)` is **acceptable to the filter** if it is not dominated:

```
         (theta^trial, phi^trial) is acceptable to F if, for ALL (theta_j, phi_j) in F:

             theta^trial < (1 - gamma_theta) * theta_j
                        OR
             phi^trial < phi_j - gamma_phi * theta_j
```

where `gamma_theta` and `gamma_phi` are small margins (typically 1e-5).

### Geometric Picture of the Filter

```
         phi (barrier objective)
          |
          |  ╳ rejected (dominated by filter entry)
          |
          |  ● filter entry 1
          |  │·····
          |  │    ·  ● filter entry 2
          |  │    ·  │·····
          |  │    ·  │    ·
          |  │    ·  │    ·  ✓ accepted (not dominated)
          |  │    ·  │    ·
          └──┴────·──┴────·──────── theta (constraint violation)
```

Points below and to the left of all filter entries are accepted. The filter prevents cycling by remembering past performance.

### Two Acceptance Modes

At each iteration, IPOPT determines whether to use **f-type** (objective-focused) or **h-type** (constraint-focused) acceptance:

**f-type (switching condition satisfied):** When the current iterate is nearly feasible (`theta_k < theta_min`) and the objective can be improved:

```
         Armijo condition:  phi(x_k + alpha * d) <= phi(x_k) + eta_phi * alpha * nabla phi^T d

         where eta_phi = 10^{-4} (default)
```

**h-type (otherwise):** Accept any step that passes the filter — reduces either `theta` or `phi` sufficiently.

### Backtracking

If a trial step is rejected:

```
         alpha <- alpha * alpha_red_factor     (default: 0.5)
```

Retry with the reduced step. Continue until the step is accepted or `alpha` becomes too small (triggering the restoration phase).

### Why a Filter Instead of a Merit Function?

```
         ┌──────────────────────────┬───────────────────────────────┐
         │  Merit Function (SNOPT)   │  Filter (IPOPT)               │
         ├──────────────────────────┼───────────────────────────────┤
         │ Single scalar measure     │ Bi-objective: (theta, phi)    │
         │ Requires penalty param    │ No penalty parameters needed  │
         │ Penalty tuning can fail   │ Self-adjusting acceptance     │
         │ Smooth (augmented Lagr.)  │ Set-based (no smoothness)     │
         │ Well-studied theory       │ Newer, equally rigorous       │
         │ Maratos effect (mild)     │ Maratos effect (handled by    │
         │                          │ second-order correction)       │
         └──────────────────────────┴───────────────────────────────┘
```

The filter eliminates the need to tune penalty parameters, which can be a significant practical advantage.

---

## 8. Second-Order Correction and the Maratos Effect

### The Maratos Effect

Near the solution, the Newton step `alpha = 1` may be rejected because it temporarily **increases constraint violation** even while making progress toward the optimum. This is the Maratos effect:

```
         x_k is nearly feasible and nearly optimal
         Newton step d_k points toward x*
         But: theta(x_k + d_k) > theta(x_k)  (constraint violation increases!)

         The filter/merit function rejects this good step
         → convergence stalls near the solution
```

### The Second-Order Correction (SOC)

When the full Newton step is rejected, IPOPT computes a correction that fixes the linearization error:

```
         Step 1: Compute the constraint violation at the trial point:
                 c_trial = g(x_k + d_k) - s_k - d_s

         Step 2: Solve the corrected system:
                 [ W + Sigma    J^T ] [ d_x^soc     ]   [ r_d           ]
                 [ J             0  ] [ d_lambda^soc ] = [ r_p - c_trial ]

         Step 3: Try the corrected step:  x^soc = x_k + d_k + d_x^soc
```

The SOC step corrects the constraint violation without changing the objective improvement, effectively eliminating the Maratos effect.

### SOC in Practice

- Up to `max_soc` (default 4) correction steps are attempted
- Each SOC step requires one additional constraint evaluation and one linear solve (but **not** a new factorization — same matrix, different right-hand side)
- SOC is essential for fast convergence near the solution

### When SOC Activates

```
         Iteration output tag:
           alpha_pr = 1.0  f    →  full step accepted by filter (no SOC needed)
           alpha_pr = 1.0  F    →  full step accepted after SOC correction
           alpha_pr = 0.5  f    →  half step accepted by filter
           alpha_pr = ...  R    →  entering restoration phase (SOC failed)
```

---

## 9. Barrier Parameter Update Strategies

The barrier parameter `mu` controls the trade-off between following the central path and making progress toward the solution. IPOPT supports two fundamentally different strategies.

### Monotone Strategy (`mu_strategy = "monotone"`, default)

The classical **Fiacco-McCormick** approach:

```
         ┌───────────────────────────────────────────────────────────────┐
         │  OUTER LOOP: for mu = mu_0, mu_1, mu_2, ... -> 0             │
         │                                                               │
         │    INNER LOOP: solve barrier subproblem for current mu        │
         │      - Take Newton steps until barrier KKT error < mu * k_eps│
         │      - May require many iterations per mu value               │
         │                                                               │
         │    UPDATE: mu_{k+1} = min(kappa_mu * mu_k, mu_k^{theta_mu})  │
         │      - kappa_mu = mu_linear_decrease_factor (default 0.2)    │
         │      - theta_mu = mu_superlinear_decrease_power (default 1.5)│
         │      - Ensures superlinear convergence of mu -> 0             │
         └───────────────────────────────────────────────────────────────┘
```

**Pros**: Simple, well-understood convergence theory.
**Cons**: Can waste iterations solving intermediate barrier subproblems to high accuracy.

**Key parameters:**
| Option | Default | Effect |
|--------|---------|--------|
| `mu_init` | 0.1 | Starting barrier parameter |
| `mu_linear_decrease_factor` | 0.2 | Linear decrease rate (kappa_mu) |
| `mu_superlinear_decrease_power` | 1.5 | Superlinear decrease exponent (theta_mu) |
| `barrier_tol_factor` | 10 | Inner loop tolerance = mu * this factor |

### Adaptive Strategy (`mu_strategy = "adaptive"`)

Recomputes `mu` at **every iteration** using a quality function or probing heuristic. This is often faster in practice.

**Quality Function Oracle** (`mu_oracle = "quality-function"`, default for adaptive):

```
         1. Evaluate how well the current iterate satisfies the
            perturbed KKT conditions for various mu values

         2. Choose mu that minimizes a quality function:
            Q(mu) = measure of "how close are we to the central path for this mu?"

         3. The quality function considers both the complementarity error
            and the barrier objective decrease
```

**Probing Oracle** (`mu_oracle = "probing"`):

Adapted from Mehrotra's predictor-corrector for LP:

```
         1. Compute an affine scaling step (predictor) with mu = 0:
            → How far can we go toward the pure Newton step?

         2. Measure the complementarity at the predicted point:
            mu_aff = (x_pred - x_L)^T z^L_pred / n

         3. Compute centering parameter:
            sigma = (mu_aff / mu_current)^3

         4. Set mu_new = sigma * mu_current
            → Aggressive: if predictor step is good, decrease mu fast
            → Conservative: if predictor step is poor, keep mu large
```

**Safeguards:**
- `mu_min` (default 1e-11): prevents premature reduction
- `mu_max` (default 1e5): upper bound
- If adaptive fails, IPOPT **automatically falls back** to monotone

### Comparing Strategies

```
         ┌──────────────────────┬────────────────────┬──────────────────┐
         │                      │ Monotone            │ Adaptive          │
         ├──────────────────────┼────────────────────┼──────────────────┤
         │ mu updates           │ After subproblem    │ Every iteration   │
         │ Inner iterations     │ Multiple per mu     │ One per mu value  │
         │ Typical iter count   │ Higher              │ Lower             │
         │ Robustness           │ More robust         │ Slightly less     │
         │ For hard problems    │ Better              │ Riskier           │
         │ For easy problems    │ Wastes iterations   │ Much faster       │
         │ Default              │ YES                 │ No                │
         └──────────────────────┴────────────────────┴──────────────────┘
```

**Practical recommendation**: Try `mu_strategy = "adaptive"` first. If it struggles, fall back to `"monotone"`.

---

## 10. Feasibility Restoration Phase

### When Restoration Activates

The restoration phase is IPOPT's safety net, activated when the main algorithm cannot make progress:

```
         Triggers:
         1. Backtracking line search exhausts all step sizes
            (alpha_p < alpha_p_min without acceptance)

         2. Linear system factorization fails catastrophically

         3. User requests via start_with_resto = "yes"
```

### What Restoration Does

The restoration phase temporarily abandons the original objective and solves a **feasibility problem**:

```
         minimize    sum_i (p_i + n_i)  +  zeta/2 ||x - x_R||^2
          x, p, n

         subject to  g(x) = s + p - n
                     p, n >= 0
                     x_L <= x <= x_U
```

where:
- `p_i, n_i >= 0` are positive and negative deviations from constraint satisfaction
- `x_R` is the iterate when restoration was entered (regularization term)
- `zeta` is a small regularization weight

This is itself solved by IPOPT's interior-point method — a "solver within a solver."

### Exiting Restoration

The restoration phase returns to the main algorithm when:

```
         theta(x_restored) < required_infeasibility_reduction * theta(x_entry)

         where required_infeasibility_reduction = 0.9 (default)
```

After exiting, the bound multipliers are recomputed with a Newton step for complementarity.

### Infeasibility Detection

If the restoration phase converges to a point that **minimizes constraint violation but is not feasible**:

```
         IPOPT returns: "Infeasible_Problem_Detected"
         Message: "Converged to a point of local infeasibility.
                   Problem may be infeasible."
```

This provides a **certificate of local infeasibility**: IPOPT has found a local minimizer of `||g(x) - s||` that has nonzero residual.

### Restoration in the Iteration Output

```
         iter    objective    inf_pr   inf_du  lg(mu)  ||d||  lg(rg)  alpha_du  alpha_pr  ls
          25r   1.234e+02   5.67e-01  1.23e+01  -3.8   2.3e+00   -    1.00e+00  1.00e+00   1

         The 'r' suffix on the iteration number indicates restoration phase.
         The 'R' tag on alpha_pr indicates entering restoration.
```

### Key Restoration Options

| Option | Default | Effect |
|--------|---------|--------|
| `expect_infeasible_problem` | "no" | Enable heuristics for rapid infeasibility detection |
| `required_infeasibility_reduction` | 0.9 | Required theta reduction before exiting |
| `max_resto_iter` | 3000000 | Maximum restoration iterations |
| `resto_penalty_parameter` | 1000 | Penalty weight in restoration objective |

---

## 11. Convergence Criteria

### The Scaled NLP Error

IPOPT measures convergence using a **scaled NLP error** based on the KKT conditions. The algorithm terminates when all of the following are satisfied:

```
         Dual infeasibility:      ||nabla f + J^T lambda - z^L + z^U||_inf / s_d  <=  tol
         Primal feasibility:      ||g(x) - s||_inf                                 <=  tol
         Complementarity:         ||X^L Z^L e||_inf / s_c                           <=  tol
```

where the scaling factors are:

```
         s_d = max(s_max, (||lambda||_1 + ||z||_1) / (m + n)) / s_max
         s_c = max(s_max, ||z||_1 / n) / s_max

         s_max = 100 (default, controlled by s_max option)
```

These scaling factors prevent large multipliers from masking true infeasibility.

### Tolerance Hierarchy

IPOPT has a two-tier convergence system:

```
         ┌──────────────────────────────────────────────────────┐
         │  TIGHT TOLERANCES (primary convergence)              │
         │                                                      │
         │  tol = 1e-8 (default)                                │
         │  constr_viol_tol = 1e-4                              │
         │  dual_inf_tol = 1                                    │
         │  compl_inf_tol = 1e-4                                │
         │                                                      │
         │  Result: "Solve_Succeeded" (optimal solution found)  │
         ├──────────────────────────────────────────────────────┤
         │  LOOSE TOLERANCES (acceptable convergence)           │
         │                                                      │
         │  acceptable_tol = 1e-6                               │
         │  acceptable_constr_viol_tol = 0.01                   │
         │  acceptable_dual_inf_tol = 1e10                      │
         │  acceptable_compl_inf_tol = 0.01                     │
         │                                                      │
         │  Triggers after acceptable_iter (15) consecutive     │
         │  iterations meeting these tolerances                 │
         │                                                      │
         │  Result: "Solved_To_Acceptable_Level"                │
         └──────────────────────────────────────────────────────┘
```

The acceptable convergence provides a **fallback**: if the solver is making progress but cannot reach the tight tolerances (e.g., due to numerical noise), it declares a "good enough" solution rather than failing.

### Termination Limits

| Option | Default | Effect |
|--------|---------|--------|
| `max_iter` | 3000 | Maximum iterations |
| `max_wall_time` | 1e20 | Wall clock time limit (seconds) |
| `max_cpu_time` | 1e20 | CPU time limit (seconds) |
| `diverging_iterates_tol` | 1e20 | Detects unbounded problems (`||x|| > tol`) |

### Comparison with SNOPT Convergence

```
         ┌────────────────────────────────┬─────────────────────────────────┐
         │  IPOPT                          │  SNOPT                          │
         ├────────────────────────────────┼─────────────────────────────────┤
         │  Single tol for scaled error    │  Separate Major feas/opt tol    │
         │  Includes complementarity       │  Complementarity implicit       │
         │  Two-tier: tight + acceptable   │  Single tier                    │
         │  Default tol = 1e-8             │  Default tol = 1e-6             │
         │  Scales by multiplier norms     │  Scales by constraint norms     │
         └────────────────────────────────┴─────────────────────────────────┘
```

---

## 12. Hessian Handling: Exact vs. Quasi-Newton

### Option 1: Exact Hessian (`hessian_approximation = "exact"`, default)

IPOPT expects the user to provide the **Hessian of the Lagrangian**:

```
         W(x, sigma_f, lambda) = sigma_f * nabla^2 f(x) + sum_{i=1}^m lambda_i * nabla^2 g_i(x)
```

**Interface requirements:**
1. **Sparsity structure**: Row/column indices of nonzero entries in the lower triangle of `W` (symmetric, so only lower triangle needed)
2. **Numerical values**: Given `(x, sigma_f, lambda)`, return the nonzero values of `W`

**Benefits:**

```
         ┌────────────────────────────────────────────────────────────────┐
         │  With exact Hessian:                                           │
         │    - Fewer iterations (typically 2-5x fewer than L-BFGS)      │
         │    - Superlinear local convergence                             │
         │    - Proper inertia detection (crucial for non-convex)         │
         │    - More robust: "significantly more robust than              │
         │      first-derivative-only operation" (Wächter & Biegler)     │
         │    - Better handling of negative curvature                     │
         │                                                                │
         │  Cost:                                                         │
         │    - Must implement Hessian evaluation                         │
         │    - Hessian evaluation can be expensive                       │
         │    - Sparsity structure must be correct                        │
         └────────────────────────────────────────────────────────────────┘
```

### Option 2: Limited-Memory BFGS (`hessian_approximation = "limited-memory"`)

When exact Hessians are unavailable or too expensive, IPOPT uses L-BFGS:

```
         Store m most recent pairs:  {s_k, y_k}  where:
           s_k = x_{k+1} - x_k
           y_k = nabla_x L_{k+1} - nabla_x L_k

         Approximate Hessian B_k via two-loop recursion:
           B_k ≈ nabla^2_xx L  using the stored pairs
```

**Key parameters:**

| Option | Default | Effect |
|--------|---------|--------|
| `limited_memory_max_history` | 6 | Number of stored `{s, y}` pairs |
| `limited_memory_update_type` | "bfgs" | `"bfgs"` (positive definite) or `"sr1"` (can capture neg. curvature) |
| `limited_memory_initialization` | "scalar1" | How to initialize `B_0` at each iteration |
| `limited_memory_max_skipping` | 2 | Max consecutive skipped updates before reset |

**The BFGS skipping criterion:**

```
         If s_k^T y_k > 0:  Accept the pair, update B_k
         If s_k^T y_k <= 0: Skip the update (positive curvature condition violated)

         After limited_memory_max_skipping consecutive skips:
            → Reset the L-BFGS history (start fresh)
```

### Subspace Selection

`hessian_approximation_space` controls which variables receive the L-BFGS approximation:
- `"nonlinear-variables"` (default): Only variables appearing nonlinearly. Linear variables contribute zero to the Hessian, so no approximation is needed.
- `"all-variables"`: All variables use L-BFGS (simpler but potentially less accurate).

### Exact vs. L-BFGS Performance Comparison

```
         ┌───────────────────────────────────────────────────────────────┐
         │                    Exact Hessian    │    L-BFGS               │
         ├───────────────────────────────────── ┼────────────────────────┤
         │  Iterations:        20-50            │    50-300+              │
         │  Per-iteration cost: Higher (Hess)   │    Lower (no Hessian)  │
         │  Total cost:        Usually lower    │    Usually higher       │
         │  Convergence rate:  Superlinear      │    Linear               │
         │  Robustness:        More robust      │    Less robust          │
         │  Non-convex:        Handles well     │    May struggle         │
         │  Implementation:    Harder           │    Easier               │
         └──────────────────────────────────────┴────────────────────────┘
```

### Derivative Verification

IPOPT includes a built-in derivative checker (`derivative_test` option):

| Setting | What it checks | Cost |
|---------|---------------|------|
| `"none"` | Nothing (default) | 0 |
| `"first-order"` | Gradient and Jacobian vs. finite differences | O(n) function evals |
| `"second-order"` | Also checks the Hessian | O(n) gradient evals |

Output format:
```
         grad_f[    3] = -2.5000e+00    ~  -2.5000e+00  [ 1.234e-09]
         jac_g [1,  5] =  3.1416e+00    ~   3.1416e+00  [ 5.678e-10]
         h     [2,  3] =  1.0000e+00    ~   9.9999e-01  [ 1.234e-05]  ***
```

The `***` flag indicates a potential error (relative deviation exceeds `derivative_test_tol`).

---

## 13. Linear Solver Options and Their Impact

The linear solver is the **computational bottleneck** of IPOPT. Each iteration requires factoring the `(n+m) x (n+m)` augmented system. The choice of linear solver dramatically affects both speed and robustness.

### Available Linear Solvers

#### MA27 (HSL)

```
         Type:         Sparse symmetric indefinite, direct
         Parallelism:  Sequential only
         Ordering:     Built-in minimum degree
         Best for:     Small-to-medium problems (< 5000 variables)
         License:      HSL (free for academic, paid for commercial)
         Notes:        Historical default. Very robust, well-tested.
                       Provides inertia information.
```

#### MA57 (HSL)

```
         Type:         Sparse symmetric indefinite, direct
         Parallelism:  Sequential (METIS for ordering)
         Ordering:     METIS nested dissection or AMD
         Best for:     Small-medium problems, ill-conditioned systems
         License:      HSL
         Notes:        Better pivoting than MA27. Recommended for
                       ill-conditioned problems with scaling enabled.
```

#### HSL_MA86 (HSL)

```
         Type:         Sparse symmetric indefinite, direct
         Parallelism:  Multi-threaded (OpenMP)
         Ordering:     METIS
         Best for:     Large problems where factorization > 1 second
         License:      HSL
         Notes:        Significant speedup over MA27/MA57 on large
                       problems. Exploits multi-core parallelism.
```

#### HSL_MA97 (HSL)

```
         Type:         Sparse symmetric indefinite, direct
         Parallelism:  Multi-threaded (OpenMP)
         Ordering:     METIS
         Best for:     Large problems, modern hardware
         License:      HSL
         Notes:        Newest HSL solver. Often comparable or superior
                       to MA86. Good parallel scalability.
```

#### MUMPS (default when HSL unavailable)

```
         Type:         Sparse symmetric indefinite, direct
         Parallelism:  Multi-threaded, MPI capable
         Ordering:     METIS, AMD, or SCOTCH
         Best for:     When open-source is required
         License:      Open source (CeCILL-C)
         Notes:        Generally slower than HSL solvers (often 2-5x).
                       Free and widely available via conda.
                       Default when no HSL solvers are found.
```

#### Pardiso

```
         Type:         Sparse symmetric indefinite, direct
         Parallelism:  Multi-threaded
         Two variants: pardiso-project.org (academic) and Intel MKL
         Best for:     Large problems, when HSL is unavailable
         License:      Academic free / Intel oneAPI (free)
         Notes:        Performance comparable to HSL solvers.
                       Intel MKL version is easiest to obtain.
```

### Practical Recommendations

```
         ┌────────────────────────────┬───────────────────────────────────┐
         │  Scenario                   │  Recommended Solver               │
         ├────────────────────────────┼───────────────────────────────────┤
         │  Small (< 1000 vars)       │  MA27 or MA57                     │
         │  Medium, ill-conditioned    │  MA57 with mc19 scaling           │
         │  Large, factorization > 1s  │  MA86, MA97, or Pardiso          │
         │  Open-source required       │  MUMPS                            │
         │  Multi-core available       │  MA86, MA97, or Pardiso          │
         │  Quickest setup             │  MUMPS (ships with conda IPOPT)  │
         └────────────────────────────┴───────────────────────────────────┘
```

### Linear System Scaling

| Option | Effect |
|--------|--------|
| `linear_system_scaling = "mc19"` | Uses HSL MC19 routine to scale the augmented system (default) |
| `linear_system_scaling = "slack-based"` | Alternative scaling based on slack variable values |
| `linear_system_scaling = "none"` | No scaling of the linear system |
| `linear_scaling_on_demand = "yes"` | Only apply scaling when iterative refinement suggests it's needed (default) |

---

## 14. Scaling: How and Why It Matters

### The Scaling Problem

Poor scaling is one of the most common causes of IPOPT convergence difficulties. When objective and constraint functions have vastly different magnitudes, the KKT system becomes ill-conditioned:

```
         Well-scaled problem:                 Poorly-scaled problem:

         nabla f     ~ O(1)                   nabla f     ~ O(1e6)
         g(x)        ~ O(1)                   g(x)        ~ O(1e-3)
         Jacobian     ~ O(1)                   Jacobian     ~ O(1e-3)

         KKT system well-conditioned           KKT system ill-conditioned
         → fast convergence                    → slow or failed convergence
```

### NLP Scaling Method (`nlp_scaling_method`)

**Gradient-based scaling (default):**

```
         At the initial point x_0:

         1. Compute nabla f(x_0) and nabla g_i(x_0) for all constraints

         2. For objective: if max|nabla f(x_0)| > nlp_scaling_max_gradient:
               sigma_f = nlp_scaling_max_gradient / max|nabla f(x_0)|
            Solve with sigma_f * f(x) instead of f(x)

         3. For each constraint g_i: similarly compute sigma_i:
               sigma_i = nlp_scaling_max_gradient / max|nabla g_i(x_0)|
            Solve with sigma_i * g_i(x) instead of g_i(x)
```

**Key parameters:**
| Option | Default | Effect |
|--------|---------|--------|
| `nlp_scaling_max_gradient` | 100 | Target maximum gradient magnitude |
| `nlp_scaling_min_value` | 1e-8 | Floor for scaling factors |
| `obj_scaling_factor` | 1 | Manual objective scaling (set to -1 to maximize) |

**Other scaling methods:**
- `"none"`: No scaling. Use when the problem is already well-scaled.
- `"user-scaling"`: User provides explicit scaling factors through the interface.
- `"equilibration-based"`: Matrix equilibration approach.

### Scaling Pitfalls

```
         ┌───────────────────────────────────────────────────────────────┐
         │  WARNING: Gradient-based scaling depends on the initial       │
         │  point! If x_0 is far from the solution, the gradients at    │
         │  x_0 may not represent the problem's true character.         │
         │                                                               │
         │  Common failure mode:                                         │
         │    - nabla f(x_0) ≈ 0 at initial point (e.g., at a saddle)  │
         │    - Scaling factor becomes very large                        │
         │    - Scaled problem is WORSE than unscaled                    │
         │                                                               │
         │  Recommendation: For well-formulated problems, try            │
         │  nlp_scaling_method = "none" first. Add scaling only if       │
         │  convergence is poor.                                         │
         └───────────────────────────────────────────────────────────────┘
```

### User-Side Scaling Best Practices

Even without IPOPT's internal scaling, users affect scaling through problem formulation:

```
         1. Variable units:    x in [0.001, 0.01] → rescale to [1, 10]
         2. Objective:         f ~ 1e8 → normalize as f/f_0
         3. Constraints:       c ~ 1e-6 → normalize as c/c_ref
         4. Bound magnitudes:  x_L = 1e-10 → consider reformulating
```

---

## 15. Warm Starting: Challenges and Approaches

### Why Warm Starting Is Hard for Interior Point Methods

Warm starting is a well-known weakness of interior-point methods compared to active-set methods like SNOPT. The fundamental challenges are:

```
         ┌───────────────────────────────────────────────────────────────┐
         │  Challenge 1: Strict Interiority                              │
         │                                                               │
         │  Previous solution x* has some variables ON their bounds.     │
         │  But interior-point methods require x strictly INTERIOR.      │
         │  Starting from x* violates the barrier formulation.           │
         │                                                               │
         │  SNOPT: No problem! Active-set methods naturally live         │
         │         on boundaries. Warm start = pass the basis.           │
         ├───────────────────────────────────────────────────────────────┤
         │  Challenge 2: Complementarity                                 │
         │                                                               │
         │  At x*: x_i * z^L_i = 0 (exact complementarity)             │
         │  But barrier needs: x_i * z^L_i = mu > 0                    │
         │  Starting from exact complementarity with small mu           │
         │  → severely ill-conditioned KKT system                       │
         ├───────────────────────────────────────────────────────────────┤
         │  Challenge 3: Bound Multipliers                               │
         │                                                               │
         │  Need z^L, z^U from previous solve. These are not always     │
         │  available or meaningful when the problem changes.            │
         └───────────────────────────────────────────────────────────────┘
```

### IPOPT's Warm-Starting Implementation

Enable with `warm_start_init_point = "yes"`. IPOPT adjusts the initial point:

```
         1. Push x away from bounds:
            x_i = max(x_i, x_L_i + warm_start_bound_push)
            x_i = min(x_i, x_U_i - warm_start_bound_push)

         2. Adjust slacks similarly

         3. Clip multipliers:
            z^L_i = max(z^L_i, warm_start_mult_bound_push)
            z^L_i = min(z^L_i, warm_start_mult_init_max)
```

### Key Warm-Start Parameters

| Option | Default | Recommended for Warm Start |
|--------|---------|---------------------------|
| `warm_start_init_point` | "no" | "yes" |
| `warm_start_bound_push` | 0.001 | 1e-6 to 1e-9 |
| `warm_start_bound_frac` | 0.001 | 1e-6 to 1e-9 |
| `warm_start_slack_bound_push` | 0.001 | 1e-6 to 1e-9 |
| `warm_start_slack_bound_frac` | 0.001 | 1e-6 to 1e-9 |
| `warm_start_mult_bound_push` | 0.001 | 1e-6 |
| `warm_start_mult_init_max` | 1e6 | 1e6 |
| `mu_init` | 0.1 | 1e-6 (small, close to target) |

### Practical Warm-Starting Recipe

```python
# For sequential solves where active set changes little:
opt.setOption("warm_start_init_point", "yes")
opt.setOption("warm_start_bound_push", 1e-9)
opt.setOption("warm_start_bound_frac", 1e-9)
opt.setOption("warm_start_slack_bound_push", 1e-9)
opt.setOption("warm_start_slack_bound_frac", 1e-9)
opt.setOption("warm_start_mult_bound_push", 1e-9)
opt.setOption("mu_init", 1e-6)
opt.setOption("mu_strategy", "adaptive")
```

### Warm Start Performance: IPOPT vs. SNOPT

```
         Scenario: Solve problem P1, then solve P2 (small perturbation of P1)

         SNOPT: Cold start P1 → 30 major iterations
                Warm start P2 → 1-5 major iterations (!!!)
                Speedup: 6-30x

         IPOPT: Cold start P1 → 50 iterations
                Warm start P2 → 15-30 iterations
                Speedup: 1.5-3x

         Bottom line: If you need sequential solves with small perturbations,
                      SNOPT's warm start is dramatically superior.
```

---

## 16. Key Options and Their Mathematical Effects

### Comprehensive Options Reference

#### Barrier and Convergence Options

| Option | Default | Mathematical Effect |
|--------|---------|-------------------|
| `tol` | 1e-8 | Overall convergence tolerance for scaled NLP error |
| `max_iter` | 3000 | Maximum iterations |
| `mu_strategy` | "monotone" | Barrier parameter update: "monotone" or "adaptive" |
| `mu_init` | 0.1 | Initial barrier parameter |
| `mu_min` | 1e-11 | Minimum barrier parameter (adaptive only) |
| `mu_oracle` | "quality-function" | Adaptive mu selection method |
| `barrier_tol_factor` | 10 | Inner loop tolerance = mu * this factor (monotone) |

#### Convergence Tolerances

| Option | Default | Mathematical Effect |
|--------|---------|-------------------|
| `tol` | 1e-8 | Primary scaled NLP error tolerance |
| `dual_inf_tol` | 1 | Absolute dual infeasibility tolerance |
| `constr_viol_tol` | 1e-4 | Absolute constraint violation tolerance |
| `compl_inf_tol` | 1e-4 | Absolute complementarity tolerance |
| `acceptable_tol` | 1e-6 | Acceptable (fallback) tolerance |
| `acceptable_iter` | 15 | Consecutive acceptable iterations before declaring convergence |

#### Hessian Options

| Option | Default | Mathematical Effect |
|--------|---------|-------------------|
| `hessian_approximation` | "exact" | `"exact"` (user provides) or `"limited-memory"` (L-BFGS) |
| `limited_memory_max_history` | 6 | Number of L-BFGS pairs stored (increase for better approximation) |
| `limited_memory_update_type` | "bfgs" | `"bfgs"` (positive definite) or `"sr1"` (indefinite, captures neg. curvature) |
| `limited_memory_initialization` | "scalar1" | Initial Hessian approximation strategy |

#### Line Search Options

| Option | Default | Mathematical Effect |
|--------|---------|-------------------|
| `max_soc` | 4 | Maximum second-order correction steps |
| `watchdog_shortened_iter_trigger` | 10 | Shortened steps before watchdog activates |
| `alpha_red_factor` | 0.5 | Step size reduction factor in backtracking |
| `accept_every_trial_step` | "no" | Accept all steps (disable line search — dangerous!) |

#### Scaling Options

| Option | Default | Mathematical Effect |
|--------|---------|-------------------|
| `nlp_scaling_method` | "gradient-based" | "gradient-based", "none", "user-scaling", "equilibration-based" |
| `nlp_scaling_max_gradient` | 100 | Target max gradient magnitude after scaling |
| `obj_scaling_factor` | 1 | Manual objective multiplier (set to -1 to maximize) |

#### Linear Solver Options

| Option | Default | Mathematical Effect |
|--------|---------|-------------------|
| `linear_solver` | "mumps" | Linear algebra backend |
| `linear_system_scaling` | "mc19" | Linear system scaling method |
| `min_refinement_steps` | 1 | Minimum iterative refinement steps |
| `max_refinement_steps` | 10 | Maximum iterative refinement steps |

#### Output Options

| Option | Default | Effect |
|--------|---------|--------|
| `print_level` | 5 | Verbosity: 0 (silent) to 12 (debug) |
| `output_file` | "" | File for detailed output |
| `print_timing_statistics` | "no" | Print timing breakdown |
| `print_info_string` | "no" | Print additional info per iteration |

---

## 17. Diagnostic Reading: Understanding IPOPT Output

### The Iteration Output

IPOPT's default output (print_level 5) shows one line per iteration:

```
iter    objective    inf_pr   inf_du   lg(mu)  ||d||   lg(rg)  alpha_du  alpha_pr  ls
   0  1.2345678e+02 1.23e+01 5.67e+00  -1.0  0.00e+00    -     0.00e+00  0.00e+00   0
   1  8.9012345e+01 3.45e+00 2.34e+00  -1.0  2.34e+00    -     1.00e+00  1.00e+00f  1
   2  5.2345678e+01 1.23e+00 1.56e+00  -1.7  1.56e+00    -     1.00e+00  1.00e+00f  1
   3  2.3456789e+01 4.56e-01 8.90e-01  -1.7  9.87e-01    -     1.00e+00  1.00e+00h  1
   4  1.2345678e+01 1.23e-02 3.45e-01  -2.5  5.43e-01   -4     1.00e+00  1.00e+00h  1
   5  9.8765432e+00 2.34e-04 1.23e-02  -3.8  2.10e-01    -     1.00e+00  1.00e+00f  1
```

### Column Meanings

| Column | Meaning | What to Watch |
|--------|---------|---------------|
| `iter` | Iteration number (`r` suffix = restoration) | Growing `r` count = trouble |
| `objective` | Unscaled objective value | Should decrease (eventually) |
| `inf_pr` | Max constraint violation `theta(x)` | Should decrease to `< tol` |
| `inf_du` | Scaled dual infeasibility | Should decrease to `< tol` |
| `lg(mu)` | `log10` of barrier parameter | Should decrease over time |
| `||d||` | Infinity norm of primal step | Should decrease near solution |
| `lg(rg)` | `log10` of Hessian regularization (`-` = none) | `-` is ideal |
| `alpha_du` | Dual step size | 1.0 is ideal |
| `alpha_pr` | Primal step size (with tag) | 1.0 is ideal |
| `ls` | Number of backtracking line search steps | 1 is ideal |

### The `alpha_pr` Tags

| Tag | Meaning |
|-----|---------|
| `f` | Filter acceptance (f-type, Armijo condition) |
| `F` | Filter acceptance after second-order correction |
| `h` | Filter acceptance (h-type, constraint reduction) |
| `H` | h-type acceptance after SOC |
| `R` | Entering restoration phase |
| `w` | Watchdog procedure active |
| `r` | Restored previous iterate (watchdog failed) |

### What Healthy Optimization Looks Like

```
    ┌─────────────────────────────────────────────────────────────────┐
    │ HEALTHY OPTIMIZATION                                            │
    │                                                                 │
    │ - alpha_pr ≈ 1.0 most iterations (full Newton steps)           │
    │ - inf_pr decreasing steadily → 0                                │
    │ - inf_du decreasing steadily → 0                                │
    │ - lg(mu) decreasing (monotonically or adaptively)              │
    │ - lg(rg) = - (no regularization needed)                        │
    │ - ls = 1 (full step accepted on first try)                     │
    │ - No 'r' iterations (no restoration)                            │
    │ - objective stabilizes to final value                           │
    ├─────────────────────────────────────────────────────────────────┤
    │ UNHEALTHY OPTIMIZATION                                          │
    │                                                                 │
    │ - alpha_pr << 1 repeatedly: step rejected, backtracking        │
    │ - inf_pr oscillating or not decreasing: constraint issues       │
    │ - inf_du stagnating: wrong derivatives or bad Hessian           │
    │ - lg(rg) large and persistent: severe indefiniteness           │
    │ - ls > 5: line search struggling                                │
    │ - Many 'r' iterations: restoration phase dominating            │
    │ - Objective not improving: wrong search direction               │
    └─────────────────────────────────────────────────────────────────┘
```

### The Final Summary

```
         Number of Iterations....: 47

                                           (scaled)                 (unscaled)
         Objective...............:   9.8500000000e-01    9.8500000000e-01
         Dual infeasibility......:   2.3456789012e-09    2.3456789012e-09
         Constraint violation....:   1.2345678901e-10    1.2345678901e-10
         Complementarity.........:   3.4567890123e-09    3.4567890123e-09
         Overall NLP error.......:   3.4567890123e-09    3.4567890123e-09

         Number of objective function evaluations             = 52
         Number of objective gradient evaluations             = 48
         Number of equality constraint evaluations            = 52
         Number of inequality constraint evaluations          = 0
         Number of equality constraint Jacobian evaluations   = 48
         Number of inequality constraint Jacobian evaluations = 0
         Number of Lagrangian Hessian evaluations             = 47
         Total wall-clock secs in IPOPT (w/o function evaluations) = 2.345
         Total wall-clock secs in NLP function evaluations         = 1.234

         EXIT: Optimal Solution Found.
```

---

## 18. Common Pitfalls and Troubleshooting

### Problem: "Restoration Failed" or Repeated Restoration

**Causes:**
1. Incorrect derivatives (most common)
2. Problem is locally infeasible near the current iterate
3. Constraint qualification (LICQ/MFCQ) violated
4. Problem is inherently very difficult

**Diagnosis and Fixes:**
```
         1. Run with derivative_test = "first-order" (or "second-order")
            → Check for *** flagged derivatives

         2. Try a different starting point
            → The current starting region may be pathological

         3. Relax tolerances:
            tol = 1e-6, constr_viol_tol = 1e-3

         4. Try adaptive mu strategy:
            mu_strategy = "adaptive"

         5. Check for redundant constraints:
            → Rank-deficient Jacobian causes trouble
```

### Problem: Very Slow Convergence (Many Iterations)

**Causes:**
1. L-BFGS Hessian with too few stored pairs
2. Poor scaling
3. Barrier parameter decreasing too slowly

**Diagnosis and Fixes:**
```
         1. If using L-BFGS:
            limited_memory_max_history = 15-25 (increase from default 6)

         2. Check scaling:
            nlp_scaling_method = "gradient-based"  (or "none" if already well-scaled)

         3. Try adaptive mu:
            mu_strategy = "adaptive"
            mu_oracle = "probing"  (more aggressive)

         4. If exact Hessian is available, use it:
            hessian_approximation = "exact"
            → Typically 2-5x fewer iterations
```

### Problem: "Solved to Acceptable Level" Instead of "Optimal Solution Found"

**Causes:**
1. Tight tolerances not achievable with current precision
2. Function noise preventing tight convergence
3. Near-degenerate constraint system

**Diagnosis and Fixes:**
```
         1. Check if the acceptable solution is good enough:
            → Often acceptable_tol = 1e-6 is sufficient

         2. Tighten acceptable tolerances if needed:
            acceptable_constr_viol_tol = 1e-6
            acceptable_compl_inf_tol = 1e-6

         3. Improve function precision:
            → Use tighter solver tolerances in inner solvers (CFD, FEA)

         4. Try more iterations:
            max_iter = 10000
```

### Problem: "Diverging Iterates" or Unbounded Objective

**Causes:**
1. Problem is truly unbounded
2. Missing bound constraints
3. Incorrect problem formulation

**Diagnosis and Fixes:**
```
         1. Add bound constraints on all variables:
            x_L <= x <= x_U  with reasonable physical bounds

         2. Check objective sign:
            obj_scaling_factor = -1  if you accidentally minimize instead of maximize

         3. Verify problem formulation:
            → Missing constraints often cause unboundedness
```

### Problem: "Maximum Iterations Exceeded"

**Diagnosis and Fixes:**
```
         1. Check if the solver is making progress:
            → If objective/infeasibility are improving, just need more iterations:
              max_iter = 10000

         2. If NOT making progress:
            → Check derivatives (derivative_test)
            → Try different mu_strategy
            → Try different linear_solver
            → Check scaling

         3. If oscillating:
            → Problem may be non-smooth or discontinuous
            → IPOPT is not designed for such problems
```

### Problem: Large `lg(rg)` Values (Heavy Regularization)

**Causes:**
1. Hessian of Lagrangian is indefinite (non-convex problem)
2. Poor L-BFGS approximation
3. Near-singular KKT system

**Diagnosis and Fixes:**
```
         1. If using L-BFGS:
            → Switch to exact Hessian if possible
            → Increase limited_memory_max_history

         2. For non-convex problems:
            → Some regularization is expected and normal
            → Only concerning if lg(rg) > 0 persistently

         3. Check for near-singular Jacobian:
            → Remove redundant constraints
            → Better starting point
```

---

## 19. IPOPT vs. SNOPT: A Detailed Comparison

### Fundamental Algorithmic Differences

```
         ┌──────────────────────────────────────────────────────────────────┐
         │                    IPOPT                │      SNOPT              │
         ├──────────────────────────────────────────┼─────────────────────── │
         │  Algorithm: Primal-dual interior point   │  Algorithm: SQP        │
         │  Subproblem: Newton system (linear)      │  Subproblem: QP        │
         │  Handles bounds: Barrier function        │  Handles bounds: Active│
         │                                          │  set (B/S/N partition) │
         │  Iterates: Strictly interior             │  Iterates: Can be on   │
         │                                          │  boundaries            │
         │  Globalization: Filter line search       │  Globalization: Aug.   │
         │                                          │  Lagrangian merit fn   │
         │  Hessian: Exact (default) or L-BFGS      │  Hessian: L-BFGS on   │
         │                                          │  reduced space         │
         │  License: Open source (EPL)              │  License: Commercial   │
         └──────────────────────────────────────────┴────────────────────── ┘
```

### Cost Per Iteration

```
         IPOPT per iteration:
           - 1 function evaluation
           - 1 gradient/Jacobian evaluation
           - 1 Hessian evaluation (if exact; free if L-BFGS)
           - 1 sparse (n+m) x (n+m) symmetric factorization
           - 1+ backsolves (including iterative refinement)
           - Possible SOC steps (re-solve, no re-factor)

         SNOPT per major iteration:
           - 1 function evaluation
           - 1 gradient/Jacobian evaluation
           - 0 Hessian evaluations (quasi-Newton from gradient differences)
           - Multiple minor iterations (QP solves, each involving
             basis updates via LUSOL)
           - Each minor: O(m) for basis update + O(n_S^2) for reduced Hessian
```

### When IPOPT Wins

```
         ┌───────────────────────────────────────────────────────────────┐
         │  IPOPT is better when:                                        │
         │                                                               │
         │  1. Many inequality constraints (100s-1000s)                  │
         │     → IPOPT cost ~ independent of # inequalities              │
         │     → SNOPT's QP cost grows with active set                   │
         │                                                               │
         │  2. Many degrees of freedom (superbasics > 2000)              │
         │     → SNOPT's reduced Hessian becomes O(n_S^2) storage        │
         │     → IPOPT has no such limitation                            │
         │                                                               │
         │  3. Exact Hessian is available                                 │
         │     → IPOPT achieves superlinear convergence                  │
         │     → SNOPT uses L-BFGS (linear convergence)                  │
         │                                                               │
         │  4. Open-source/free license required                         │
         │     → IPOPT is EPL (free for all use)                         │
         │     → SNOPT requires paid commercial license                  │
         │                                                               │
         │  5. Very large problems (100k+ variables)                     │
         │     → IPOPT with parallel linear solvers scales well          │
         │     → SNOPT struggles if many superbasics                     │
         └───────────────────────────────────────────────────────────────┘
```

### When SNOPT Wins

```
         ┌───────────────────────────────────────────────────────────────┐
         │  SNOPT is better when:                                        │
         │                                                               │
         │  1. Function evaluations are expensive (CFD, FEA)             │
         │     → SNOPT typically needs fewer total evaluations            │
         │     → SNOPT's nonderivative line search skips gradient evals  │
         │                                                               │
         │  2. Sequential solves with warm starting                      │
         │     → SNOPT warm start: 1-5 iterations (6-30x speedup)       │
         │     → IPOPT warm start: 15-30 iterations (1.5-3x speedup)    │
         │                                                               │
         │  3. Few degrees of freedom at the solution                    │
         │     → SNOPT's reduced Hessian is tiny → very efficient        │
         │     → Think: many constraints, few free variables             │
         │                                                               │
         │  4. Need exact active set identification                      │
         │     → SNOPT returns variables exactly on bounds               │
         │     → IPOPT returns variables near (but not exactly on) bounds│
         │                                                               │
         │  5. Problem has a known favorable sparsity structure           │
         │     → SNOPT's LUSOL basis factorization exploits this         │
         └───────────────────────────────────────────────────────────────┘
```

### Benchmark Results

On standard benchmark suites:

```
         CUTEst benchmarks (429 small NLPs):
           SNOPT:       ~82.6% success rate
           SLSQP:       ~80%   success rate
           trust-constr: ~65.7% success rate (scipy interior point)

         Mittelmann benchmarks (~50 larger NLPs):
           Knitro:   Best overall
           IPOPT:    2nd-3rd place
           SNOPT:    Lower ranking on large problems

         Interpretation:
           - SNOPT excels on moderate-sized problems
           - IPOPT scales better to large problems
           - Both are robust and reliable for well-posed problems
```

### Combined Strategy

A powerful approach when both solvers are available:

```
         1. Use IPOPT for the initial cold-start solve
            (robust, good for large problems, no license cost)

         2. Use SNOPT for sequential restarts and parameter sweeps
            (superior warm starting, fewer function evaluations)

         3. If one solver fails, try the other
            (different algorithms fail on different problems)
```

---

## 20. IPOPT vs. SLSQP, L-BFGS-B, Adam, and Other Solvers

### Algorithm Class Overview

```
         ┌─────────────────────────────────────────────────────────────────┐
         │  Solver     │ Class          │ Constraints │ Hessian   │ Scale  │
         ├─────────────┼────────────────┼─────────────┼───────────┼────────┤
         │  IPOPT      │ Interior Point │ Full NLP    │ Exact/LBFGS│ 1M+   │
         │  SNOPT      │ SQP            │ Full NLP    │ L-BFGS    │ ~50k  │
         │  SLSQP      │ SQP (dense)    │ Full NLP    │ Dense BFGS│ ~2k   │
         │  L-BFGS-B   │ Quasi-Newton   │ Bounds only │ L-BFGS    │ 1M+   │
         │  Adam       │ 1st-order      │ None        │ None      │ 1B+   │
         │  SGD        │ 1st-order      │ None        │ None      │ 1B+   │
         └─────────────┴────────────────┴─────────────┴───────────┴────────┘
```

### IPOPT vs. SLSQP

```
         SLSQP (scipy.optimize.minimize, method='SLSQP'):
           - Dense O(n^2) storage, O(n^3) factorization
           - No sparsity exploitation
           - Impractical for n > 2000-3000
           - Ships with scipy (zero setup)
           - Known issues: may violate bounds during iterations
           - No parallel computation

         IPOPT:
           - Sparse storage, sparse factorization
           - Full sparsity exploitation
           - Practical for n > 100,000
           - Requires installation (conda install -c conda-forge cyipopt)
           - Strict bound satisfaction (barrier keeps iterates interior)
           - Parallel linear solvers available

         When to use SLSQP:  Quick prototyping with < 1000 variables
         When to use IPOPT:  Anything larger or production quality
```

### IPOPT vs. L-BFGS-B

```
         L-BFGS-B (scipy.optimize.minimize, method='L-BFGS-B'):
           - Only handles bound constraints (NO equality/inequality)
           - O(mn) storage (m = memory, n = variables)
           - Very fast per iteration
           - Superlinear convergence for smooth unconstrained problems
           - Gradient projection for bound handling

         IPOPT:
           - Handles full NLP (equality + inequality + bounds)
           - O(nnz) storage for sparse KKT system
           - More expensive per iteration
           - Superlinear convergence with exact Hessians

         When to use L-BFGS-B:  Unconstrained or bound-only, large smooth problems
         When to use IPOPT:     Any problem with general constraints
```

### IPOPT/SNOPT vs. Adam/SGD

These are fundamentally different tools for different problems:

```
         ┌───────────────────────────────────────────────────────────────┐
         │  Why Adam/SGD dominate deep learning (NOT IPOPT/SNOPT):       │
         │                                                               │
         │  1. SCALE: Neural nets have millions-billions of parameters   │
         │     → IPOPT needs O(n^2) for Hessian or O(nnz) factorization │
         │     → Adam needs O(n) per step (just element-wise ops)        │
         │                                                               │
         │  2. STOCHASTIC GRADIENTS: ML uses mini-batches                │
         │     → Noisy gradients corrupt L-BFGS curvature estimates     │
         │     → Adam is designed for noisy gradients                    │
         │                                                               │
         │  3. NON-CONVEXITY: Neural net landscapes are highly nonconvex │
         │     → SGD noise helps escape shallow local minima             │
         │     → Acts as implicit regularization                         │
         │                                                               │
         │  4. GENERALIZATION: SGD finds flatter minima that generalize  │
         │     → Exact optimization can overfit to training data         │
         │                                                               │
         │  5. NO CONSTRAINTS: ML loss functions are unconstrained       │
         │     → IPOPT's constraint machinery is unused overhead         │
         └───────────────────────────────────────────────────────────────┘

         ┌───────────────────────────────────────────────────────────────┐
         │  Why IPOPT/SNOPT dominate engineering optimization:            │
         │                                                               │
         │  1. CONSTRAINTS ARE ESSENTIAL: Physical laws, safety margins  │
         │     → Adam cannot handle equality/inequality constraints      │
         │     → IPOPT handles thousands of constraints naturally        │
         │                                                               │
         │  2. PRECISE CONVERGENCE NEEDED: Engineering tolerances        │
         │     → Adam converges to a neighborhood (sublinear O(1/√T))   │
         │     → IPOPT converges to machine precision (superlinear)      │
         │                                                               │
         │  3. DETERMINISTIC GRADIENTS: Adjoint/analytic derivatives     │
         │     → Full gradient available → 2nd-order methods excel       │
         │     → No mini-batch noise                                     │
         │                                                               │
         │  4. MODERATE SIZE: 100-100k variables typical in MDO          │
         │     → Well within IPOPT/SNOPT's efficient range               │
         └───────────────────────────────────────────────────────────────┘
```

### When L-BFGS Is Used in ML

L-BFGS can outperform Adam in specific ML scenarios:

```
         - Full-batch optimization (no mini-batch noise)
         - Small networks or fine-tuning stages
         - Physics-informed neural networks (PINNs)
         - Neural style transfer
         - L-BFGS converges in ~5 epochs vs Adam ~300 epochs (small problems)

         Hybrid approach:
           1. Train with Adam for initial rough convergence
           2. Switch to L-BFGS for final tight convergence
```

### Summary Comparison Table

```
   ┌──────────┬─────────────┬──────────┬──────────┬──────────┬──────────┐
   │ Property │ IPOPT       │ SNOPT    │ SLSQP    │ L-BFGS-B │ Adam     │
   ├──────────┼─────────────┼──────────┼──────────┼──────────┼──────────┤
   │ Max vars │ 1M+         │ ~50k     │ ~2k      │ 1M+      │ Billions │
   │ Constr.  │ Full NLP    │ Full NLP │ Full NLP │ Bounds   │ None     │
   │ Sparsity │ Full        │ Full     │ None     │ Implicit │ N/A      │
   │ Warm st. │ Limited     │ Excellent│ Poor     │ Poor     │ Easy     │
   │ License  │ Free (EPL)  │ Commercial│ Free    │ Free     │ Free     │
   │ Conv.    │ Superlinear │ Superlin.│ Superlin.│ Superlin.│ Sublinear│
   │ Cost/iter│ High (fact.)│ Moderate │ High     │ Low      │ Very low │
   │ Best for │ Large NLP   │ MDO      │ Prototype│ Bound-   │ Deep     │
   │          │             │          │          │ constrained│ learning │
   └──────────┴─────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 21. IPOPT in Practice: pyOptSparse and OpenMDAO

### Installation

**Conda (recommended):**
```bash
conda install -c conda-forge cyipopt
# or within pyOptSparse:
conda install -c conda-forge pyoptsparse
```

**From source:**
```bash
# Build IPOPT with HSL for best performance:
./configure --with-hsl --with-metis
make && make install
pip install cyipopt
```

### Basic pyOptSparse Usage

```python
from pyoptsparse import Optimization, IPOPT

# Define the optimization problem
def objfunc(xdict):
    x = xdict["x"]
    funcs = {}
    funcs["obj"] = x[0]**2 + x[1]**2
    funcs["con"] = x[0] + x[1] - 1.0
    fail = False
    return funcs, fail

# Set up the problem
optProb = Optimization("myProblem", objfunc)
optProb.addVarGroup("x", 2, lower=-10, upper=10, value=[5.0, 5.0])
optProb.addConGroup("con", 1, lower=0.0, upper=0.0)  # equality constraint
optProb.addObj("obj")

# Create optimizer and set options
opt = IPOPT()
opt.setOption("tol", 1e-6)
opt.setOption("max_iter", 500)
opt.setOption("print_level", 5)
opt.setOption("mu_strategy", "adaptive")
opt.setOption("limited_memory_max_history", 15)

# Solve
sol = opt(optProb, sens="CS")  # Complex-step derivatives

print(sol.fStar)   # Optimal objective
print(sol.xStar)   # Optimal variables
```

### Important pyOptSparse Limitation: No Hessian Callback

```
         ┌───────────────────────────────────────────────────────────────┐
         │  CRITICAL: pyOptSparse does NOT support the Hessian callback  │
         │  to IPOPT. This means:                                        │
         │                                                               │
         │  hessian_approximation is AUTOMATICALLY set to "limited-memory"│
         │                                                               │
         │  → You CANNOT use exact Hessians through pyOptSparse          │
         │  → IPOPT will use L-BFGS (fewer stored pairs = worse approx) │
         │  → Increase limited_memory_max_history for better performance │
         │                                                               │
         │  To use exact Hessians, use IPOPT directly via cyipopt:       │
         │    from cyipopt import minimize_ipopt                          │
         │  Or the native C/Fortran interface.                           │
         └───────────────────────────────────────────────────────────────┘
```

### Sensitivity Methods in pyOptSparse

| Method | Step Size | Accuracy | Cost |
|--------|-----------|----------|------|
| `"FD"` (finite differences) | `sensStep` (default 1e-6) | O(h) or O(h^2) | n+1 function evals per gradient |
| `"CS"` (complex step) | `sensStep` (default 1e-40j) | Machine precision | n+1 complex function evals |
| User function | N/A | Depends on implementation | Adjoint: O(1) |

**Complex step** is strongly recommended when the function supports complex arithmetic — it gives machine-precision gradients with no truncation error.

### Recommended pyOptSparse IPOPT Configuration

```python
# General-purpose configuration
opt = IPOPT()
opt.setOption("tol", 1e-6)
opt.setOption("max_iter", 1000)
opt.setOption("mu_strategy", "adaptive")
opt.setOption("limited_memory_max_history", 15)
opt.setOption("acceptable_tol", 1e-5)
opt.setOption("acceptable_iter", 10)
opt.setOption("print_level", 5)

# If problem is well-scaled:
opt.setOption("nlp_scaling_method", "none")

# If starting from a previous solution:
opt.setOption("warm_start_init_point", "yes")
opt.setOption("warm_start_bound_push", 1e-9)
opt.setOption("warm_start_bound_frac", 1e-9)
opt.setOption("warm_start_slack_bound_push", 1e-9)
opt.setOption("warm_start_slack_bound_frac", 1e-9)
opt.setOption("warm_start_mult_bound_push", 1e-9)
opt.setOption("mu_init", 1e-6)
```

### IPOPT in the MDO Context

```
         ┌───────────────────────────────────────────────────────────────┐
         │  IPOPT in MDO (pyOptSparse / OpenMDAO / Aviary):              │
         │                                                               │
         │  Role:     Recommended FREE alternative to SNOPT              │
         │  Status:   Auto-installed with pyOptSparse via conda          │
         │  Ranking:  Try IPOPT before ParOpt (simpler to configure)    │
         │                                                               │
         │  Typical MDO configuration:                                   │
         │    - mu_strategy = "adaptive" (faster convergence)            │
         │    - limited_memory_max_history = 10-20                       │
         │    - tol = 1e-6 to 1e-8                                      │
         │    - nlp_scaling_method depends on problem                    │
         │                                                               │
         │  For expensive function evaluations (CFD/FEA):                │
         │    - IPOPT lacks SNOPT's nonderivative line search            │
         │    - Each backtracking step requires gradient evaluation      │
         │    - This makes IPOPT more expensive per iteration for        │
         │      simulation-based optimization                            │
         │    - Mitigate with accept_every_trial_step = "yes" (risky)   │
         │      or just accept the extra cost                            │
         └───────────────────────────────────────────────────────────────┘
```

### Hot Starting in pyOptSparse

```python
# First solve
sol1 = opt(optProb, sens="CS", storeHistory="history.hst")

# Subsequent solve (must have identical problem structure)
sol2 = opt(optProb, sens="CS", hotStart="history.hst")
```

The history file stores variable values and optimizer state. The problem structure (number of variables, constraints, bounds) must be **identical** between the original and hot-started solve.

---

## 22. References

### Primary Sources

1. **Wächter, A., Biegler, L.T.** (2006). "On the Implementation of an Interior-Point Filter Line-Search Algorithm for Large-Scale Nonlinear Programming." *Mathematical Programming*, 106(1), 25-57. [DOI: 10.1007/s10107-004-0559-y](https://link.springer.com/article/10.1007/s10107-004-0559-y)

2. **Wächter, A., Biegler, L.T.** (2005). "Line Search Filter Methods for Nonlinear Programming: Motivation and Global Convergence." *SIAM Journal on Optimization*, 16(1), 1-31.

3. **Wächter, A., Biegler, L.T.** (2005). "Line Search Filter Methods for Nonlinear Programming: Local Convergence." *SIAM Journal on Optimization*, 16(1), 32-48.

4. **Nocedal, J., Wächter, A., Waltz, R.A.** (2009). "Adaptive Barrier Update Strategies for Nonlinear Interior Methods." *SIAM Journal on Optimization*, 19(4), 1674-1693.

### Software Documentation

5. **IPOPT Documentation.** [https://coin-or.github.io/Ipopt/](https://coin-or.github.io/Ipopt/)

6. **IPOPT Options Reference.** [https://coin-or.github.io/Ipopt/OPTIONS.html](https://coin-or.github.io/Ipopt/OPTIONS.html)

7. **IPOPT GitHub Repository (COIN-OR).** [https://github.com/coin-or/Ipopt](https://github.com/coin-or/Ipopt)

8. **pyOptSparse IPOPT Documentation.** [https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/optimizers/IPOPT.html](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/optimizers/IPOPT.html)

9. **GAMS IPOPT Documentation.** [https://www.gams.com/latest/docs/S_IPOPT.html](https://www.gams.com/latest/docs/S_IPOPT.html)

### Background Theory

10. **Nocedal, J., Wright, S.J.** (2006). *Numerical Optimization*, 2nd Edition. Springer. (Chapters 14, 19 on interior-point methods)

11. **Fletcher, R., Leyffer, S.** (2002). "Nonlinear Programming Without a Penalty Function." *Mathematical Programming*, 91(2), 239-269. (Filter method)

12. **Fiacco, A.V., McCormick, G.P.** (1968). *Nonlinear Programming: Sequential Unconstrained Minimization Techniques*. John Wiley & Sons. (Barrier methods)

### Comparison References

13. **Gill, P.E., Murray, W., Saunders, M.A.** (2005). "SNOPT: An SQP Algorithm for Large-Scale Constrained Optimization." *SIAM Review*, 47(1), 99-131.

14. **Aviary Optimization Algorithms Guide.** [https://openmdao.github.io/Aviary/theory_guide/optimization_algorithms.html](https://openmdao.github.io/Aviary/theory_guide/optimization_algorithms.html)

15. **Mittelmann, H.D.** Benchmarks for Optimization Software. [https://plato.asu.edu/sub/benchm.html](https://plato.asu.edu/sub/benchm.html)

---

## Appendix A: IPOPT Algorithm Flowchart

```
    ┌──────────────────┐
    │   Problem Setup   │
    │  (bounds, x_0)   │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Initialize:      │──── Push x_0 strictly interior to bounds
    │  mu_0, z^L, z^U  │     Set initial multipliers
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                     MAIN ITERATION LOOP                          │
    │                                                                  │
    │   ┌──────────────────┐                                          │
    │   │ Evaluate f, g    │                                          │
    │   │ Evaluate nabla f,│                                          │
    │   │ J, W (Hessian)   │                                          │
    │   └────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │   ┌──────────────────┐                                          │
    │   │ Check convergence│     YES                                   │
    │   │ (scaled NLP error│──────────► EXIT (Optimal Solution Found)  │
    │   │  < tol?)         │                                          │
    │   └────────┬─────────┘                                          │
    │            │ NO                                                   │
    │            ▼                                                     │
    │   ┌──────────────────┐                                          │
    │   │ Form augmented   │  K = [W+Sigma, J^T; J, 0]               │
    │   │ system           │                                          │
    │   └────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │   ┌──────────────────┐                                          │
    │   │ Factor K, check  │  If wrong: add delta_x * I               │
    │   │ inertia          │  to (1,1) block, re-factor               │
    │   └────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │   ┌──────────────────┐                                          │
    │   │ Solve for d_x,   │  Apply iterative refinement              │
    │   │ d_lambda         │  Recover d_z^L, d_z^U                    │
    │   └────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │   ┌──────────────────┐                                          │
    │   │ Fraction-to-the- │  alpha_p, alpha_d: stay interior         │
    │   │ boundary rule    │                                          │
    │   └────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │   ┌──────────────────┐                                          │
    │   │ Filter line      │  Backtrack until accepted                 │
    │   │ search           │  SOC if Maratos effect                    │
    │   │                  │  If fails → RESTORATION PHASE             │
    │   └────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │   ┌──────────────────┐                                          │
    │   │ Update mu        │  Monotone: decrease after subproblem     │
    │   │                  │  Adaptive: recompute each iteration       │
    │   └────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │   ┌──────────────────┐                                          │
    │   │ Update iterate   │  x ← x + alpha_p * d_x                  │
    │   │                  │  lambda ← lambda + alpha_d * d_lambda    │
    │   │                  │  z ← z + alpha_d * d_z                   │
    │   └────────┬─────────┘                                          │
    │            │                                                     │
    │            └─────────────────────────────────────► (loop back)   │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Quick Reference Card

```
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    IPOPT QUICK REFERENCE                        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                 ║
    ║  Algorithm: Primal-dual interior point with filter line search  ║
    ║  Globalization: Filter method (bi-objective: phi, theta)        ║
    ║  Hessian:   Exact (recommended) or L-BFGS                      ║
    ║  Linear solver: MUMPS (default), MA27/57/86/97, Pardiso        ║
    ║  License:   Eclipse Public License (free, open source)          ║
    ║                                                                 ║
    ║  Key strengths:                                                 ║
    ║    - Large-scale sparse NLP                                     ║
    ║    - Many inequality constraints                                ║
    ║    - Free and open source                                       ║
    ║    - Robust with exact Hessians                                 ║
    ║                                                                 ║
    ║  Key weaknesses:                                                ║
    ║    - Poor warm starting (vs. SNOPT)                             ║
    ║    - L-BFGS mode needs many more iterations                     ║
    ║    - No nonderivative line search                               ║
    ║                                                                 ║
    ║  Essential options to try first:                                ║
    ║    mu_strategy = "adaptive"                                     ║
    ║    limited_memory_max_history = 15  (if using L-BFGS)          ║
    ║    nlp_scaling_method = "none"  (if well-scaled)               ║
    ║    linear_solver = "ma57"  (if HSL available)                   ║
    ║                                                                 ║
    ║  Common exit codes:                                             ║
    ║    0 = Optimal Solution Found                                   ║
    ║    1 = Solved to Acceptable Level                               ║
    ║    2 = Infeasible Problem Detected                              ║
    ║   -1 = Maximum Iterations Exceeded                              ║
    ║   -2 = Restoration Failed                                       ║
    ║   -3 = Error in Step Computation                                ║
    ║                                                                 ║
    ╚══════════════════════════════════════════════════════════════════╝
```

---

## Appendix C: Decision Flowchart — Which Solver Should I Use?

```
    Start
      │
      ▼
    Does the problem have general constraints
    (equality or inequality, not just bounds)?
      │
      ├── NO ──► Does it have bound constraints?
      │            │
      │            ├── NO ──► L-BFGS or Adam (unconstrained)
      │            │           - L-BFGS: smooth, deterministic, < 1M vars
      │            │           - Adam: stochastic, noisy, > 1M vars
      │            │
      │            └── YES ─► L-BFGS-B (smooth, deterministic)
      │                       Adam + projection (stochastic, ML)
      │
      └── YES ─► How many variables?
                   │
                   ├── < 2000 ──► SLSQP (quick prototype)
                   │               or IPOPT/SNOPT (production)
                   │
                   ├── 2k-50k ──► SNOPT (if license available)
                   │               IPOPT (if free/open source needed)
                   │
                   └── > 50k ───► IPOPT (with sparse linear solver)
                                  │
                                  ▼
                                Are function evaluations expensive?
                                  │
                                  ├── YES ──► SNOPT preferred
                                  │           (fewer total evals,
                                  │            better warm start)
                                  │
                                  └── NO ───► IPOPT preferred
                                              (especially with exact
                                               Hessian available)
```
