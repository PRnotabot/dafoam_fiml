# SNOPT: A Comprehensive Technical Guide

## The Complete Inner Workings of a Sparse SQP Optimizer

> *Based on the foundational work of Philip E. Gill (UCSD), Walter Murray (Stanford), and Michael A. Saunders (Stanford).*
> *Primary references: [Gill, Murray, Saunders (2005) SIAM Review](https://epubs.siam.org/doi/10.1137/S0036144504446096) and [SNOPT 7.7 User's Guide](https://ccom.ucsd.edu/~optimizers/static/pdfs/snopt7-7.pdf).*

---

## Table of Contents

1. [What SNOPT Solves](#1-what-snopt-solves)
2. [The Big Picture: SQP as a Strategy](#2-the-big-picture-sqp-as-a-strategy)
3. [Problem Reformulation: Slacks and Structure](#3-problem-reformulation-slacks-and-structure)
4. [The Major Iteration: Anatomy of One SQP Step](#4-the-major-iteration-anatomy-of-one-sqp-step)
5. [The QP Subproblem](#5-the-qp-subproblem)
6. [Variable Classification: Basic, Superbasic, Nonbasic](#6-variable-classification-basic-superbasic-nonbasic)
7. [The Reduced Hessian and BFGS Updates](#7-the-reduced-hessian-and-bfgs-updates)
8. [The Augmented Lagrangian Merit Function](#8-the-augmented-lagrangian-merit-function)
9. [Line Search Procedure](#9-line-search-procedure)
10. [Gradient Handling: User-Supplied vs. Internal Estimation](#10-gradient-handling-user-supplied-vs-internal-estimation)
11. [Gradient Verification: The Verify Level System](#11-gradient-verification-the-verify-level-system)
12. [Scaling: How and Why It Matters](#12-scaling-how-and-why-it-matters)
13. [Elastic Mode and Infeasibility Handling](#13-elastic-mode-and-infeasibility-handling)
14. [Convergence Criteria](#14-convergence-criteria)
15. [The Minor Iteration: Solving the QP](#15-the-minor-iteration-solving-the-qp)
16. [Warm Starting and Basis Information](#16-warm-starting-and-basis-information)
17. [Key Options and Their Mathematical Effects](#17-key-options-and-their-mathematical-effects)
18. [Deep Dive: What Happens When You Scale All Gradients?](#18-deep-dive-what-happens-when-you-scale-all-gradients)
19. [Deep Dive: Internal Gradient Computation](#19-deep-dive-internal-gradient-computation)
20. [Deep Dive: Gradient Discrepancy and Its Consequences](#20-deep-dive-gradient-discrepancy-and-its-consequences)
21. [SNOPT in the Context of FIML Optimization](#21-snopt-in-the-context-of-fiml-optimization)
22. [Diagnostic Reading: Understanding SNOPT Output](#22-diagnostic-reading-understanding-snopt-output)
23. [Common Pitfalls and Troubleshooting](#23-common-pitfalls-and-troubleshooting)
24. [References](#24-references)

---

## 1. What SNOPT Solves

SNOPT (Sparse Nonlinear OPTimizer) solves the general nonlinear programming problem:

```
                    minimize    f(x)
                       x

                    subject to  c_E(x) = 0            (nonlinear equalities)
                                c_I(x) >= 0            (nonlinear inequalities)
                                A_L x  >= b_L           (linear constraints)
                                l <= x <= u             (bound constraints)
```

where:
- `x in R^n` is the vector of design variables
- `f: R^n -> R` is a smooth (at least twice continuously differentiable) objective function
- `c_E, c_I` are smooth nonlinear constraint functions
- `A_L` is a sparse matrix of linear constraint coefficients

**Key assumptions:**
1. All functions are **smooth** (C^2 continuous). SNOPT is *not* designed for non-smooth, noisy, or discontinuous problems.
2. **First derivatives** (gradients) are available. Second derivatives are assumed unavailable or too expensive.
3. The constraint Jacobian is **sparse**. This is what distinguishes SNOPT from dense solvers like NPSOL.

### Compact Notation

SNOPT internally groups all constraints into a single vector function `F(x)`:

```
         F(x) = [ f(x)  ]     (objective, row iObj)
                [ c(x)  ]     (nonlinear constraints)
                [ A_L x ]     (linear constraints)
```

The problem becomes:

```
         minimize    F_iObj(x) + f_add
            x
         subject to  l_F <= F(x) <= u_F
                      l   <=  x   <= u
```

where `l_F, u_F` encode equality constraints (l_F_i = u_F_i) and inequality constraints (one-sided or two-sided bounds).

---

## 2. The Big Picture: SQP as a Strategy

Sequential Quadratic Programming is a strategy for solving NLPs by solving a **sequence of simpler QP subproblems**. The intuition is analogous to Newton's method for unconstrained optimization, but extended to handle constraints.

### The SQP Philosophy

```
         ┌─────────────────────────────────────────────────────────┐
         │                    MAJOR ITERATION k                    │
         │                                                         │
         │  1. At current point x_k with multipliers pi_k:        │
         │     - Evaluate f(x_k), c(x_k)                          │
         │     - Evaluate (or estimate) gradients: nabla f, J      │
         │                                                         │
         │  2. Form QP subproblem:                                 │
         │     - Quadratic model of Lagrangian                     │
         │     - Linearized constraints                            │
         │                                                         │
         │  3. Solve QP -> search direction p_k                    │
         │     (this involves MINOR iterations)                    │
         │                                                         │
         │  4. Line search along p_k:                              │
         │     - Reduce augmented Lagrangian merit function         │
         │     - Find step length alpha_k                          │
         │                                                         │
         │  5. Update: x_{k+1} = x_k + alpha_k * p_k              │
         │             pi_{k+1} from QP solution                   │
         │             H_{k+1} via BFGS update                     │
         │                                                         │
         │  6. Check convergence -> if not, goto step 1            │
         └─────────────────────────────────────────────────────────┘
```

**Why QP subproblems?** A QP is the simplest optimization problem that captures both:
- Curvature of the objective (via the Hessian approximation)
- Constraint geometry (via linearization)

The QP solution provides both the search direction *and* updated Lagrange multiplier estimates.

---

## 3. Problem Reformulation: Slacks and Structure

### Slack Variable Introduction

SNOPT converts all inequality constraints to equalities by introducing slack variables. For a constraint:

```
         l_i <= c_i(x) <= u_i
```

SNOPT introduces slack `s_i` and writes:

```
         c_i(x) - s_i = 0,     l_i <= s_i <= u_i
```

The full reformulated problem with `m` constraints and `n` variables now has `n + m` variables `(x, s)`:

```
         minimize    f(x)
          x, s
         subject to  F(x) - s = 0          (m equality constraints)
                      l_x <= x <= u_x       (n variable bounds)
                      l_s <= s <= u_s       (m slack bounds)
```

### Why Slacks?

This reformulation is not merely cosmetic. It allows SNOPT to:
1. Treat all constraints uniformly as equalities
2. Apply the simplex-like machinery of basis partitioning
3. Classify variables (including slacks) as basic/superbasic/nonbasic
4. Handle one-sided and two-sided constraints identically

### The Jacobian Structure

Define the extended variable vector `y = (x, s)^T`. The constraint Jacobian of `F(x) - s = 0` is:

```
         A(y) = [ J(x) | -I ]
```

where `J(x) = dF/dx` is the `m x n` Jacobian of the original constraint functions and `I` is the `m x m` identity. This structure is exploited throughout SNOPT's linear algebra.

---

## 4. The Major Iteration: Anatomy of One SQP Step

Each major iteration performs the following operations in detail:

### Step 4.1: Function and Gradient Evaluation

At the current iterate `x_k`:

```
         Evaluate:   f_k = f(x_k)                          (scalar objective)
                     c_k = c(x_k)                          (constraint vector)
                     g_k = nabla f(x_k)                    (n-vector, objective gradient)
                     J_k = nabla c(x_k)                    (m x n matrix, constraint Jacobian)
```

Depending on the **Derivative level** option, some or all gradient components may be estimated by finite differences (see [Section 10](#10-gradient-handling-user-supplied-vs-internal-estimation)).

### Step 4.2: Form the QP Subproblem

The QP objective approximates the Lagrangian `L(x, pi) = f(x) - pi^T c(x)`:

```
         minimize    g_k^T (x - x_k) + 1/2 (x - x_k)^T H_k (x - x_k)
          x, s

         subject to  J_k x - s = -c_k + J_k x_k     (linearized constraints)
                     l <= (x, s)^T <= u                (bounds)
```

Here `H_k` is a positive-definite approximation to the Hessian of the Lagrangian `nabla^2_{xx} L(x_k, pi_k)`, maintained via BFGS updates.

### Step 4.3: Solve the QP (Minor Iterations)

The QP is solved by SQOPT (SNOPT's internal QP solver) using a reduced-Hessian active-set method. This involves multiple **minor iterations** where variables move between basic, superbasic, and nonbasic status.

The QP solution gives:
- `x_hat_k`: the QP minimizer (defines search direction `p_k = x_hat_k - x_k`)
- `pi_hat_k`: QP multipliers (updated Lagrange multiplier estimates)

### Step 4.4: Line Search

Find `alpha_k in (0, 1]` such that the merit function is sufficiently decreased:

```
         M(x_k + alpha_k * p_k, s_k + alpha_k * delta_s_k, pi_hat_k)
           < M(x_k, s_k, pi_k) - sigma * alpha_k * Delta_k
```

where `Delta_k` is the predicted decrease and `sigma` is a small positive constant.

### Step 4.5: Updates

```
         x_{k+1}  = x_k + alpha_k * p_k
         pi_{k+1} = pi_hat_k                (from QP solution)
         H_{k+1}  = BFGS_update(H_k, ...)   (quasi-Newton Hessian update)
```

### Step 4.6: Convergence Check

Test the KKT conditions to the specified tolerances (see [Section 14](#14-convergence-criteria)).

---

## 5. The QP Subproblem

### Full QP Formulation

At major iteration `k`, the QP subproblem is:

```
         minimize    q_k(d) = g_k^T d + 1/2 d^T H_k d
            d

         subject to  J_k (x_k + d) - s = 0
                      l_x <= x_k + d <= u_x
                      l_s <=    s    <= u_s
```

where `d = x - x_k` is the step. Equivalently, with `A_k = [J_k | -I]`:

```
         minimize    g_k^T d + 1/2 d^T H_k d
          d, s

         subject to  A_k [d; s]^T = b_k
                      l <= [x_k + d; s]^T <= u
```

where `b_k = J_k x_k - c_k` is the right-hand side from linearization.

### What `H_k` Approximates

`H_k` is a positive-definite approximation to:

```
         nabla^2_{xx} L(x_k, pi_k) = nabla^2 f(x_k) - sum_i (pi_k)_i * nabla^2 c_i(x_k)
```

The key insight: `H_k` approximates the *Lagrangian Hessian*, not just the objective Hessian. This is critical for capturing the curvature introduced by constraints.

### The Role of `H_k` in Practice

```
         ┌──────────────────────────────────────────────────┐
         │  If H_k is exact:     Quadratic convergence      │
         │  If H_k is BFGS:     Superlinear convergence     │
         │  If H_k = I:          Steepest descent (slow!)    │
         │  If H_k is poor:     May converge to wrong point  │
         └──────────────────────────────────────────────────┘
```

---

## 6. Variable Classification: Basic, Superbasic, Nonbasic

This classification is one of SNOPT's most distinctive features, borrowed from the simplex method for LP and extended to NLP.

### The Partition

At any point, the `n + m` extended variables `y = (x, s)^T` are partitioned into three sets:

```
         y = (x_B, x_S, x_N)

         B x_B + S x_S + N x_N = b

         where:
           B: m x m nonsingular "basis matrix"  (columns for basic variables)
           S: m x n_S matrix                    (columns for superbasic variables)
           N: m x n_N matrix                    (columns for nonbasic variables)
```

### Variable Types

| Type | Count | Status | Role |
|------|-------|--------|------|
| **Basic** (`x_B`) | `m` | Between their bounds | Determined by constraints; eliminated via basis factorization |
| **Superbasic** (`x_S`) | `n_S` | Between their bounds | Free to move; the "true" degrees of freedom |
| **Nonbasic** (`x_N`) | `n_N` | At one of their bounds | Fixed; candidates for activation via pricing |

### The Degrees of Freedom

```
         n_S = (degrees of freedom at the solution)
             = (number of variables between bounds) - m

         For LP:  n_S = 0 at optimum (vertex solution)
         For NLP: n_S >= 0 (measures "distance from a vertex")
```

**SNOPT works most efficiently when `n_S` is small** (a few hundred). This is because it maintains a dense `n_S x n_S` reduced Hessian matrix. Problems with thousands of superbasic variables are still solvable but slower.

### Geometric Intuition

```
         Feasible Region (2D example with 4 constraints)

              Nonbasic         ┌───────────────────┐
              variables        │                   │
              → at bounds      │     Superbasic    │
              → on edges       │     variables     │
                               │     → interior    │
                     ●─────────│──●                │
                     │         │                   │
                     │         └───────────────────┘
                     │
              Basic variables = determined by active constraints
```

### Reduced Gradient

The reduced gradient for each variable `y_j` is:

```
         d_j = g_j - a_j^T pi
```

where `g_j` is the objective gradient component, `a_j` is the j-th column of the constraint Jacobian, and `pi` is the dual variable vector.

**Optimality conditions** in terms of reduced gradients:

```
         d_j <= 0    if y_j is nonbasic at its lower bound
         d_j >= 0    if y_j is nonbasic at its upper bound
         d_j = 0     for superbasic variables (approximately)
         d_j = 0     for basic variables (by definition, since B^T pi = g_B)
```

### Pricing

**Pricing** is the process of selecting a nonbasic variable with a favorable reduced gradient to become superbasic. This is analogous to the simplex pivot selection. SNOPT uses either:
- **Partial pricing**: examines a subset of nonbasic variables per iteration (default for large problems)
- **Complete pricing**: examines all nonbasic variables

---

## 7. The Reduced Hessian and BFGS Updates

### The Reduced-Hessian Approach

Rather than maintaining the full `n x n` Hessian approximation, SNOPT works with the **reduced Hessian** in the space of superbasic variables.

Define `Z` as an `n x n_S` matrix whose columns span the null space of the active constraint Jacobian. Then:

```
         Reduced Hessian:  H_R = Z^T H_k Z    (n_S x n_S matrix)
```

SNOPT stores `H_R` in Cholesky-factored form:

```
         H_R = R^T R
```

where `R` is an `n_S x n_S` upper triangular matrix.

### Storage Implications

```
         Full Hessian:    O(n^2) storage   → impractical for n > 10,000
         Reduced Hessian: O(n_S^2) storage → practical when n_S << n
```

This is why SNOPT excels on problems where `n` is large but the number of superbasic variables at the solution is moderate.

### BFGS Update

After each major iteration, the reduced Hessian is updated using the BFGS (Broyden-Fletcher-Goldfarb-Shanno) formula. Define:

```
         s_k = x_{k+1} - x_k             (step in x-space)
         y_k = nabla_x L_{k+1} - nabla_x L_k   (change in Lagrangian gradient)
```

where `nabla_x L_k = g_k - J_k^T pi_k`.

The BFGS update to `H_k` is:

```
                           H_k s_k s_k^T H_k     y_k y_k^T
         H_{k+1} = H_k - ─────────────────── + ──────────
                            s_k^T H_k s_k        y_k^T s_k
```

In practice, SNOPT applies this update in the reduced space (to `R`) using rank-1 updates to the Cholesky factor.

### The Curvature Condition

The BFGS update requires:

```
         y_k^T s_k > 0    (positive curvature condition)
```

If this fails (indicating the Lagrangian is not convex along the step), SNOPT uses a **damped BFGS update** (Powell's modification):

```
         y_k_damped = theta * y_k + (1 - theta) * H_k s_k

         where theta = min(1, 0.8 * s_k^T H_k s_k / (s_k^T H_k s_k - y_k^T s_k))
```

This ensures `y_k_damped^T s_k > 0` always holds, maintaining positive definiteness of `H_k`.

### Full Memory vs. Limited Memory

| Mode | Storage | Convergence | When Used |
|------|---------|-------------|-----------|
| **Full Memory** | `n_S(n_S+1)/2` | Q-superlinear | `n_S <= 75` (default threshold) |
| **Limited Memory** | `O(n_S * r)` where `r` = num updates | Linear | `n_S > 75` |

**Full memory** stores and updates the complete `R` factor. The **Hessian frequency** option (default 999999) controls periodic resets to identity.

**Limited memory** accumulates BFGS update pairs `{s_i, y_i}` and discards the oldest after `r` updates (controlled by **Hessian updates**, default 10). This is an L-BFGS-like strategy adapted for the reduced space.

### Hessian Recovery from SNOPT

In pyOptSparse, the Hessian can be extracted from the workspace arrays:

```
         lvlHes = iw[71]         # 0=LM, 1=FM, 2=Exact
         lU     = iw[390] - 1    # pointer to U vector
         lenU   = iw[391]        # length of U
         Uvec   = rw[lU:lU+lenU]
         nnH    = iw[23]         # size of Hessian

         # Reconstruct: H = U^T U (upper triangular storage)
         U = zeros(nnH, nnH)
         U[triu_indices(nnH)] = Uvec
         H = U.T @ U
```

---

## 8. The Augmented Lagrangian Merit Function

### Why a Merit Function?

The QP subproblem might suggest a step that decreases the objective but increases constraint violation, or vice versa. The merit function provides a **single scalar measure** that balances objective improvement against constraint satisfaction, ensuring global convergence.

### The Formulation

SNOPT uses a smooth augmented Lagrangian merit function:

```
         M(x, s, pi) = f(x) - pi^T (c(x) - s_N) + 1/2 (c(x) - s_N)^T D (c(x) - s_N)
```

where:
- `pi` is the current Lagrange multiplier estimate
- `s_N` are the nonlinear slack values
- `D = diag(rho_1, rho_2, ..., rho_m)` is a diagonal matrix of **penalty parameters**
- Each `rho_i >= 0` is associated with the i-th constraint

### Intuition

```
         M(x,s,pi) =  f(x)                        ← objective value
                     - pi^T r(x)                   ← Lagrangian term (first-order)
                     + 1/2 r(x)^T D r(x)           ← penalty term (second-order)

         where r(x) = c(x) - s_N  is the constraint residual
```

- At a feasible point: `r(x) = 0`, so `M = f(x)` (merit = objective)
- Away from feasibility: the quadratic penalty term penalizes constraint violation
- The `pi^T r` term provides a "bridge" so that the merit function is smooth and its gradient is continuous

### Penalty Parameter Updates

The penalty parameters `rho_i` are updated to ensure that the search direction is a descent direction for `M`. SNOPT increases `rho_i` when:

1. The QP predicts an increase in `|r_i|` (constraint `i` is getting worse)
2. The line search fails to find sufficient decrease

The update rule ensures:

```
         rho_i >= 2 |pi_hat_i|
```

where `pi_hat_i` is the QP multiplier estimate. This guarantees that the penalty overwhelms the Lagrangian term, making the search direction a descent direction for `M`.

### Comparison with Other Merit Functions

```
         ┌──────────────────────┬────────────────────┬──────────────────┐
         │ Merit Function       │ Smoothness         │ Maratos Effect   │
         ├──────────────────────┼────────────────────┼──────────────────┤
         │ L1 penalty           │ Nonsmooth          │ Yes (severe)     │
         │ Augmented Lagrangian │ Smooth (C^1)       │ Mild             │
         │ Filter method        │ N/A (set-based)    │ No               │
         └──────────────────────┴────────────────────┴──────────────────┘
```

SNOPT's augmented Lagrangian merit function is smooth, which allows the use of gradient-based line searches and avoids the issues with nondifferentiable penalty functions.

---

## 9. Line Search Procedure

### Purpose

After the QP provides a search direction `p_k = x_hat_k - x_k`, the line search finds a step length `alpha_k in (0, 1]` satisfying sufficient decrease in the merit function.

### Two Line Search Modes

#### 1. Derivative Line Search (Default)

Uses both function values and directional derivatives of the merit function:

```
         phi(alpha) = M(x_k + alpha * p_k, ...)

         phi'(alpha) = nabla M^T p_k   (directional derivative)
```

The line search uses **safeguarded cubic interpolation** based on function values and derivatives at the current and previous trial points. It seeks `alpha` satisfying the **Wolfe conditions**:

```
         Sufficient decrease:   phi(alpha) <= phi(0) + c_1 * alpha * phi'(0)
         Curvature condition:   |phi'(alpha)| <= c_2 * |phi'(0)|
```

where `c_1` is a small constant (typically 1e-4) and `c_2` is the **Linesearch tolerance** (default 0.9).

#### 2. Nonderivative Line Search

Uses only function values (no directional derivative). Employs **safeguarded quadratic interpolation**. This is useful when:
- Derivatives are expensive and function evaluations are cheap
- You want to avoid derivative computations during the line search

**Important:** When using `Nonderivative linesearch`, the user must skip gradient computation when SNOPT sets `mode = 0` in the callback. Otherwise, the expected savings are not realized.

### Line Search Tolerance

The **Linesearch tolerance** `eta in [0, 1]` controls how accurately the line search is performed:

```
         eta close to 0:  Very accurate line search (more function evals, fewer major iters)
         eta close to 1:  Loose line search (fewer function evals, more major iters)
         Default: 0.9    (moderately loose, good for expensive functions)
```

For problems with very cheap function evaluations, setting `eta = 0.1` can reduce the total number of major iterations.

### Step Length Bounds

The step `alpha` is bounded:

```
         0 < alpha <= 1    (for the initial QP step)
```

A full step `alpha = 1` is always tried first. If the merit function is not sufficiently decreased, `alpha` is reduced by interpolation. Near the solution, `alpha = 1` should be accepted (full Newton step), and this is indeed what happens with a good Hessian approximation.

---

## 10. Gradient Handling: User-Supplied vs. Internal Estimation

### The Derivative Level Option

The **Derivative level** option tells SNOPT which gradients the user provides:

| Level | Objective Gradient | Constraint Jacobian | Description |
|-------|--------------------|---------------------|-------------|
| 3 | User provides | User provides | All derivatives known (default, recommended) |
| 2 | FD estimated | User provides | Only Jacobian known |
| 1 | User provides | FD estimated | Only objective gradient known |
| 0 | FD estimated | FD estimated | No derivatives known |

### How SNOPT Computes Finite Differences Internally

When SNOPT must estimate missing gradient components, it uses the following procedure:

#### Forward Differences

For each variable `x_j` with an unknown derivative:

```
         dF_i/dx_j ≈ (F_i(x + h_j e_j) - F_i(x)) / h_j
```

where `e_j` is the j-th unit vector and the perturbation `h_j` is:

```
         h_j = h_1 * (1 + |x_j|)
```

The parameter `h_1` is the **Difference interval** (default `5.5e-7 ≈ eps^(1/2)` where `eps` is machine epsilon).

#### Central Differences

SNOPT switches to central differences when:
1. The line search produces a very small step (suggesting the forward difference is inaccurate)
2. The current point is close to optimal

```
         dF_i/dx_j ≈ (F_i(x + h_j e_j) - F_i(x - h_j e_j)) / (2 h_j)
```

with perturbation:

```
         h_j = h_2 * (1 + |x_j|)
```

where `h_2` is the **Central difference interval** (default `6.7e-5 ≈ eps^(1/3)`).

#### Cost of Finite Differences

```
         Forward differences: n additional function evaluations per gradient
         Central differences: 2n additional function evaluations per gradient
```

For an optimization problem with `n` design variables and using `Derivative level 0`, each major iteration requires approximately `n + 1` function evaluations (1 for the function value, `n` for forward differences). This makes FD gradients prohibitively expensive for large `n`.

### The Function Precision Parameter

The **Function precision** `eps_R` (default `3.0e-13`) tells SNOPT how accurately the functions can be computed. This is critical for finite difference step sizing:

```
         Optimal FD step:  h ≈ sqrt(eps_R)     (forward differences)
         Optimal CD step:  h ≈ cbrt(eps_R)     (central differences)
```

**Why this matters:** If your functions are computed by an iterative solver (CFD, FEM), the precision may be much worse than machine epsilon. For example:
- Tight CFD solver: `eps_R ≈ 1e-10` → optimal FD step ≈ `1e-5`
- Loose CFD solver: `eps_R ≈ 1e-6`  → optimal FD step ≈ `1e-3`

Setting an incorrect `Function precision` leads to either:
- **Too small**: FD step falls below function noise → garbage gradients
- **Too large**: FD step is unnecessarily large → truncation error

```
                   Total FD Error as a function of step size h

         Error
          |  ╲                                    ╱
          |   ╲  Truncation error               ╱  (grows with h)
          |    ╲  O(h) or O(h^2)              ╱
          |     ╲                           ╱
          |      ╲         ┌──────┐      ╱
          |       ╲        │Optimal│   ╱
          |        ╲───────│  h    │──╱  Cancellation error
          |                └──────┘     O(eps_R / h)
          |
          └──────────────────────────────────────── h
                          h* ≈ sqrt(eps_R)
```

### Derivative Level and the User Callback

In the pyOptSparse/SNOPT interface, the user function callback receives a `mode` parameter:

```python
def userfg(mode, nnJac, x, fobj, gobj, fcon, gcon, nState):
    if mode == 0:
        # Compute function values ONLY (no gradients needed)
        # Used during nonderivative linesearch
        fobj = ...
        fcon = ...
    elif mode == 1:
        # Compute gradients ONLY
        gobj = ...
        gcon = ...
    elif mode == 2:
        # Compute BOTH functions and gradients
        fobj, fcon = ...
        gobj, gcon = ...

    # Return mode = -1 to indicate function undefined at x
    # Return mode = -2 to request termination
    return mode, fobj, gobj, fcon, gcon
```

### Selective Gradient Provision

SNOPT allows a powerful hybrid: provide *some* gradient components analytically and let SNOPT estimate the rest by FD. Before calling the user function, SNOPT initializes all gradient array elements to a sentinel value. At `Derivative level 0`, any element that the user does not overwrite is estimated by finite differences.

This is particularly useful when:
- Some derivatives are analytic (e.g., geometric constraints)
- Others come from adjoint solvers that may not be available for all objectives/constraints

---

## 11. Gradient Verification: The Verify Level System

### Purpose

Gradient verification is SNOPT's built-in mechanism to detect errors in user-supplied derivatives. Incorrect gradients are one of the most common causes of optimization failure.

### Verify Levels

| Level | What is Checked | Cost | When to Use |
|-------|-----------------|------|-------------|
| -1 | Nothing | 0 calls | Production runs with trusted gradients |
| 0 | Cheap test: 2 funcon + 3 funobj calls | Minimal | Quick sanity check (default) |
| 1 | Each objective gradient component | `O(n)` | Debugging objective gradient |
| 2 | Each Jacobian column | `O(n)` | Debugging constraint Jacobian |
| 3 | Both levels 1 and 2 | `O(n)` | New function development |

### How Verification Works

At the first feasible point (satisfying bounds and linear constraints), SNOPT:

1. Evaluates the user-supplied gradient `g_user`
2. Computes a finite-difference estimate `g_FD` using the **Difference interval**
3. Compares component-by-component

For each component `j`, the comparison yields:

```
         Relative error:  err_j = |g_user_j - g_FD_j| / max(|g_user_j|, |g_FD_j|, 1)
```

Each component is labeled:

| Label | Meaning | Typical `err_j` |
|-------|---------|-----------------|
| **OK** | Gradients agree | `< 1e-6` |
| **Bad?** | Possible error | `1e-6 to 1e-2` |
| **BAD** | Definite error | `> 1e-2` |
| **Constant** | Both are zero or near-zero | N/A |

### Interpreting Verify Level Output

A typical verify level 3 output looks like:

```
         Column       j    Comp       dj           Finite diff    Difference   RelError   OK?
             1        1    1.234e+00   1.235e+00    1.000e-03      8.1e-04      OK
             2        2    0.000e+00   2.310e-08    2.310e-08      2.3e-08      OK
             3        3    5.678e+01   5.672e+01   -6.000e-02      1.1e-03      Bad?
             4        4    9.012e+00  -9.012e+00   -1.802e+01      2.0e+00      BAD
```

### Common Causes of Gradient Discrepancies

```
    ┌─────────────────────────────────┬────────────────────────────────────────┐
    │ Discrepancy Pattern             │ Likely Cause                           │
    ├─────────────────────────────────┼────────────────────────────────────────┤
    │ All gradients wrong by factor   │ Objective/constraint scaling error     │
    │ Sign flip on some components    │ Constraint sign convention mismatch    │
    │ Some components zero, should    │ Missing dependency in chain rule       │
    │ not be                          │                                        │
    │ Random-looking errors           │ Uninitialized memory / stale cache     │
    │ Gradients "Bad?" but close      │ Function noise / tight FD step         │
    │ Gradients exact for some vars,  │ Partial derivative implementation      │
    │ wrong for others                │ error in specific variable group       │
    │ Wrong magnitude, right sign     │ Missing Jacobian transpose vs.         │
    │                                 │ non-transpose confusion                │
    └─────────────────────────────────┴────────────────────────────────────────┘
```

### Verification with `Verify level -1` in Production

Setting `Verify level -1` (as commonly done in DaFoam/FIML scripts) skips all gradient checking. This is appropriate only when:
1. Gradients have been validated during development
2. The adjoint solver is trusted
3. You want to avoid the extra function evaluations

**Risk:** If gradients become incorrect (e.g., due to code changes or solver parameter changes), the optimization will silently fail or converge to a wrong point.

---

## 12. Scaling: How and Why It Matters

### The Scaling Problem

Poor scaling causes the Hessian condition number to grow, degrading convergence:

```
         Condition number:  kappa(H) = lambda_max / lambda_min

         kappa(H) ~ 10^4:  Fine, converges normally
         kappa(H) ~ 10^8:  Slow progress, SQP algorithm struggles
         kappa(H) ~ 10^12: Algorithm effectively fails
```

### Scale Options

| Option | Description |
|--------|-------------|
| 0 | No scaling. Use when problem is naturally well-scaled (all coefficients < 100) |
| 1 | Scale linear constraints and variables using an iterative procedure |
| 2 | Additionally scale nonlinear constraints (requires good starting point) |

### The Scaling Procedure

SNOPT's iterative scaling procedure targets a Jacobian where all nonzero elements are approximately 1.0:

1. Compute column scale factors `c_j` and row scale factors `r_i`
2. Apply: `A_scaled = diag(r) * A * diag(c)`
3. Iterate until the ratio `rho_j = max|a_ij| / min|a_ij|` in each column stabilizes

The **Scale tolerance** (default 0.9) controls when iteration stops: another pass executes only if `max_j rho_j` decreases by more than a factor of `(1 - tolerance)`.

### Implicit Scaling by the User

Even without SNOPT's internal scaling, the user affects scaling through:

1. **Choice of units**: Lengths in meters vs. millimeters → 1000x difference
2. **Objective normalization**: `f(x)/f_0` vs. raw `f(x)`
3. **Constraint normalization**: `c(x)/c_ref` vs. raw `c(x)`

### Effect of Scaling on Convergence

```
    Well-scaled problem                   Poorly-scaled problem

    Contours of Lagrangian:               Contours of Lagrangian:

         ┌──────────┐                          ┌──────────────────────┐
         │ ╭──────╮  │                          │╭────────────────────╮│
         │ │╭────╮│  │                          ││╭──────────────────╮││
         │ ││ ●  ││  │   ← Near-circular       │││       ●         │││ ← Elongated
         │ │╰────╯│  │     contours             ││╰──────────────────╯││   ellipses
         │ ╰──────╯  │     → fast convergence   │╰────────────────────╯│   → slow
         └──────────┘                          └──────────────────────┘   convergence
```

---

## 13. Elastic Mode and Infeasibility Handling

### When Does Elastic Mode Activate?

Elastic mode is triggered when:

1. A QP subproblem is **infeasible** (the linearized constraints cannot be satisfied simultaneously within bounds)
2. The QP dual variables become **excessively large** (indicating near-infeasibility)

### The Elastic Reformulation

In elastic mode, SNOPT relaxes the nonlinear constraints by introducing elastic variables `v, w >= 0`:

```
         minimize    f(x) + gamma * e^T (v + w)
            x, v, w

         subject to  c(x) - v + w = 0     (relaxed nonlinear constraints)
                     v, w >= 0             (elastic variables)
                     l <= x <= u           (original bounds)
```

where:
- `gamma` is the **Elastic weight** (default `1e4`)
- `e` is the vector of ones
- `v_i > 0` means constraint `i` is violated below its lower bound
- `w_i > 0` means constraint `i` is violated above its upper bound

### Interpretation

```
         gamma large:  Strongly penalizes infeasibility → tries to be feasible
         gamma small:  Accepts infeasibility → finds minimum of objective ignoring constraints
```

If the original problem is **truly infeasible**, elastic mode minimizes a weighted combination of the objective and total constraint violation. The final solution with `v, w != 0` indicates which constraints cannot be satisfied.

### Penalty Parameter Growth

If elastic mode fails to achieve feasibility, SNOPT increases `gamma`:

```
         gamma_{new} = omega * 10^r * (1 + ||g(x)||_2)
```

where `omega` is the elastic weight and `r` counts unsuccessful attempts. This aggressive growth ensures that SNOPT ultimately either finds a feasible point or correctly identifies infeasibility.

---

## 14. Convergence Criteria

SNOPT declares convergence when the KKT (Karush-Kuhn-Tucker) conditions are satisfied to specified tolerances.

### The KKT Conditions

At an optimal point `(x*, pi*)`:

```
         1. Stationarity:    nabla f(x*) - J(x*)^T pi* = 0    (gradient of Lagrangian = 0)
         2. Primal feasibility: c(x*) = 0, l <= x* <= u       (constraints satisfied)
         3. Dual feasibility:   pi* >= 0 for inequality constraints
         4. Complementarity:    pi*_i * c_i(x*) = 0           (active constraint or zero multiplier)
```

### SNOPT's Convergence Tests

#### Major Feasibility Tolerance (`eps_feas`, default `1e-6`)

Tests constraint satisfaction:

```
         max_i |violation_i| / max(1, ||x||) <= eps_feas
```

This is a **relative** test: constraints are satisfied relative to the size of the variables.

#### Major Optimality Tolerance (`eps_opt`, default `1e-6`)

Tests reduced gradient (complementarity) conditions:

```
         max_j |Comp_j| / max(1, ||pi||) <= eps_opt
```

where `Comp_j` is the complementarity measure for variable `j`:

```
         Comp_j = d_j * min(x_j - l_j, u_j - x_j)   if d_j is the reduced gradient
```

This measures how far the KKT complementarity conditions are from being satisfied.

#### Both Must Be Satisfied Simultaneously

SNOPT only declares **Finished successfully** (inform = 0 or 1) when *both* feasibility and optimality tests pass.

### Exit Conditions Summary

```
    ┌──────────┬─────────────────────────────────────────────────────────────┐
    │ Inform   │ Meaning                                                     │
    ├──────────┼─────────────────────────────────────────────────────────────┤
    │ 0 or 1   │ Optimality conditions satisfied to tolerance                │
    │ 2        │ Feasible point found (feasibility problem only)             │
    │ 3        │ Requested accuracy not achievable                           │
    │ 4        │ Weak QP minimizer (KKT approx satisfied but not tightly)    │
    │ 10-15    │ Problem appears infeasible                                  │
    │ 20-22    │ Problem appears unbounded                                   │
    │ 31       │ Iteration limit reached                                     │
    │ 32       │ Major iteration limit reached                               │
    │ 33       │ Superbasics limit too small                                  │
    │ 41       │ Current point cannot be improved (stalled)                   │
    │ 42-44    │ Numerical difficulties (singular basis, ill-conditioning)    │
    │ 51-55    │ Incorrect derivatives detected                               │
    └──────────┴─────────────────────────────────────────────────────────────┘
```

---

## 15. The Minor Iteration: Solving the QP

### SQOPT: SNOPT's Internal QP Solver

The QP subproblem is solved by SQOPT, which uses a **primal simplex method** enhanced for quadratic objectives.

### Minor Iteration Flow

```
         ┌─────────────────────────────────────────────────────────┐
         │                    MINOR ITERATION                      │
         │                                                         │
         │  1. Compute reduced gradient g_S = Z^T g_reduced        │
         │                                                         │
         │  2. If ||g_S|| < tol: check optimality                  │
         │     - Price nonbasic variables (compute d_j)             │
         │     - If all d_j optimal: QP solved → exit               │
         │     - Else: promote a nonbasic variable to superbasic    │
         │                                                         │
         │  3. Compute search direction in superbasic space:       │
         │     p_S = -R^{-1} R^{-T} g_S     (using Cholesky R)    │
         │                                                         │
         │  4. Compute basic variable changes via basis solve:     │
         │     p_B = -B^{-1} S p_S                                 │
         │                                                         │
         │  5. Ratio test: find max step before a variable         │
         │     hits a bound                                         │
         │                                                         │
         │  6. If a variable hits bound:                           │
         │     - Basic → nonbasic: basis exchange (simplex pivot)  │
         │     - Superbasic → nonbasic: remove from S              │
         │                                                         │
         │  7. Update basis factorization B = L U                  │
         │                                                         │
         │  8. Update Cholesky factor R                            │
         │                                                         │
         │  9. Go to step 1                                        │
         └─────────────────────────────────────────────────────────┘
```

### The Basis Factorization

The basis matrix `B` is factored using **LUSOL** (a sparse LU factorization package by Saunders):

```
         P B Q = L U
```

where `P, Q` are permutation matrices, `L` is lower triangular, and `U` is upper triangular. LUSOL is specifically designed for:
- Sparse matrices with column-oriented storage
- Efficient updates when a single column of `B` changes (rank-1 update)
- Numerical stability via threshold pivoting

### When Minor Iterations Are Needed

Even for problems with only equality constraints, minor iterations may occur:
1. **Temporary bounds**: At the first QP, all variables start as nonbasic (fixed at initial values). Minor iterations free them.
2. **Iterative refinement**: If the system `B x_B = b - S x_S - N x_N` is ill-conditioned, refinement iterations (counted as minor iterations) improve accuracy.
3. **Basis changes**: Variables hitting bounds trigger simplex-like pivots.

### Proximal Point Modification

To handle degenerate or poorly conditioned QPs, SNOPT can add a proximal point term:

```
         minimize  g^T d + 1/2 d^T H d + 1/(2 mu) ||d||^2
```

The **Proximal iterations limit** (default 10000) controls how many proximal-point iterations are allowed.

---

## 16. Warm Starting and Basis Information

### The Warm Start Mechanism

SNOPT can be warm-started using information from a previous solve:

```
         Warm start requires:
           - xs: variable + slack values from previous solution
           - hs: basis status vector (B/S/N classification)
```

### Basis Status Vector `hs`

For each variable/slack `y_j`, `hs[j]` encodes:

| Value | Meaning |
|-------|---------|
| 0 | Nonbasic at lower bound |
| 1 | Nonbasic at upper bound |
| 2 | Superbasic |
| 3 | Basic |

### When to Warm Start

Warm starting is effective when:
- Solving a sequence of related problems (e.g., continuation methods)
- Restarting after a time limit
- Parametric optimization studies

The basis information dramatically reduces the work in the first major iteration by avoiding the need to determine the variable classification from scratch.

### Crash Options

When warm starting is not available, SNOPT uses a **Crash** procedure to find an initial basis:

| Option | Strategy |
|--------|----------|
| 0 | Slack basis: `B = I` (all slacks are basic) |
| 1 | One-phase: seek triangular basis from all rows/columns |
| 2 | Two-phase: first linear equations, then nonlinear |
| 3 | Three-phase: linear equalities, linear inequalities, nonlinear (default) |

The **Crash tolerance** (default 0.1) controls which elements are considered for the initial basis; elements smaller than `tolerance * max_element` in a column are ignored.

---

## 17. Key Options and Their Mathematical Effects

### Comprehensive Options Reference

#### Derivative-Related Options

| Option | Default | Mathematical Effect |
|--------|---------|-------------------|
| `Derivative level` | 3 | Determines which gradient components are provided vs. estimated by FD |
| `Difference interval` | 5.5e-7 | Step `h_1` for forward FD: `h = h_1(1+|x_j|)` |
| `Central difference interval` | 6.7e-5 | Step `h_2` for central FD: `h = h_2(1+|x_j|)` |
| `Function precision` | 3.0e-13 | Relative accuracy of function values; affects FD step selection |
| `Verify level` | 0 | Gradient checking: -1=none, 0=cheap, 1=obj, 2=con, 3=both |

#### Convergence Options

| Option | Default | Mathematical Effect |
|--------|---------|-------------------|
| `Major feasibility tolerance` | 1e-6 | `max|violation_i|/max(1,||x||) <= tol` |
| `Major optimality tolerance` | 1e-6 | `max|Comp_j|/max(1,||pi||) <= tol` |
| `Minor feasibility tolerance` | 1e-6 | Feasibility of variables/bounds in QP |

#### Algorithm Options

| Option | Default | Mathematical Effect |
|--------|---------|-------------------|
| `Hessian full memory` | if n_S <= 75 | Store complete reduced Hessian `R^T R` |
| `Hessian limited memory` | if n_S > 75 | L-BFGS style updates, reset every `r` updates |
| `Hessian updates` | 10 | Limited-memory: number of BFGS pairs before reset |
| `Hessian frequency` | 999999 | Full-memory: reset to identity after this many updates |
| `Linesearch tolerance` | 0.9 | Accuracy of line search (0=exact, 1=loose) |
| `Elastic weight` | 1e4 | Initial penalty `gamma` for elastic mode |
| `Superbasics limit` | n+1 | Max number of superbasic variables |

#### Iteration Limits

| Option | Default | Mathematical Effect |
|--------|---------|-------------------|
| `Major iterations limit` | 1000 | Maximum major iterations |
| `Minor iterations limit` | 500 | Maximum minor iterations per QP |
| `Iterations limit` | 10000 | Total minor iterations across all QPs |

#### Scaling and Feasibility Options

| Option | Default | Mathematical Effect |
|--------|---------|-------------------|
| `Scale option` | 0 | 0=none, 1=linear, 2=linear+nonlinear |
| `Scale tolerance` | 0.9 | Convergence criterion for iterative scaling |
| `Crash option` | 3 | Initial basis strategy (0-3) |

---

## 18. Deep Dive: What Happens When You Scale All Gradients?

This section addresses the question: **If all input gradients are multiplied by a constant factor `alpha`, what happens?**

### Case 1: Objective Gradient Scaled by `alpha`

If you provide `alpha * nabla f(x)` instead of `nabla f(x)`:

```
         Effect on QP subproblem:
           g_k → alpha * g_k

         QP objective becomes:
           q(d) = alpha * g_k^T d + 1/2 d^T H_k d
```

The QP solution changes because the balance between the linear term (gradient) and quadratic term (Hessian) shifts:

```
         QP solution:  d* = -H_k^{-1} (alpha * g_k)  = alpha * d_original*

         (in unconstrained case; with constraints, the effect is more complex)
```

**Consequences:**
1. **Search direction changes**: The step direction is scaled, potentially pointing away from the true minimum
2. **Hessian updates corrupt**: The BFGS update uses `y_k = nabla L_{k+1} - nabla L_k`. If `nabla f` is scaled but the true curvature isn't, the secant equation `H_{k+1} s_k = y_k` becomes inconsistent
3. **Merit function confused**: The augmented Lagrangian `M = alpha*f - pi^T c + 1/2 c^T D c` has the wrong balance between objective and constraint terms
4. **Convergence test affected**: The optimality condition `||nabla f - J^T pi|| <= tol` is tested with the scaled gradient, so the optimizer may declare convergence at a non-optimal point

### Case 2: Constraint Jacobian Scaled by `alpha`

If you provide `alpha * J(x)` instead of `J(x)`:

```
         Linearized constraints become:
           alpha * J_k (x - x_k) + c_k ≈ 0
           → J_k (x - x_k) ≈ -c_k / alpha     (effectively wrong linearization)
```

**Consequences:**
1. **QP constraints are wrong**: The linearization no longer matches the actual constraint surface
2. **Multiplier estimates distorted**: `pi_hat` from the QP satisfies `alpha * J^T pi_hat = ...`, so `pi_hat = pi_true / alpha`
3. **Basis factorization affected**: The basis matrix `B` has scaled columns, affecting LUSOL's pivoting
4. **Step direction deviates**: The search direction no longer follows the constraint surface correctly

### Case 3: All Gradients (Objective + Jacobian) Scaled by `alpha`

```
         Optimality condition (original):  nabla f = J^T pi
         With scaling:                     alpha * nabla f = alpha * J^T pi'
         Simplifies to:                    nabla f = J^T pi'
```

If both are scaled by the **same** factor, the optimality condition structure is preserved. However:

1. **BFGS update is still corrupted**: `y_k = alpha * (nabla L_{k+1} - nabla L_k)` but `s_k` is unchanged, so the Hessian converges to `alpha * H_true` instead of `H_true`
2. **QP solution changes**: `d* = -(alpha * H)^{-1} (alpha * g) = H^{-1} g / 1 = d_original* / alpha` ... wait, this depends on the Hessian history
3. **Convergence point is correct** (same KKT point) but **convergence rate degrades** because the Hessian approximation has the wrong scale
4. **Penalty parameters adjust**: SNOPT will increase penalty parameters to compensate, but this may take many iterations

### Summary: Gradient Scaling Effects

```
    ┌──────────────────────┬────────────┬────────────┬────────────────────┐
    │ What's Scaled        │ Correct    │ Convergence│ Hessian            │
    │                      │ Solution?  │ Speed      │ Approximation      │
    ├──────────────────────┼────────────┼────────────┼────────────────────┤
    │ Obj gradient by α    │ NO         │ Degraded   │ Corrupted          │
    │ Jacobian by α        │ NO         │ Degraded   │ Corrupted          │
    │ Both by same α       │ YES (same  │ Degraded   │ Converges to α*H   │
    │                      │ KKT point) │            │ (wrong scale)      │
    │ Both by α, and α→1   │ YES        │ Degrades   │ Recovers slowly    │
    │ as optimization      │            │ then       │                    │
    │ converges            │            │ recovers   │                    │
    └──────────────────────┴────────────┴────────────┴────────────────────┘
```

### Practical Implication for Adjoint-Based Optimization

In DaFoam/OpenMDAO workflows, the adjoint solver returns `df/dx` and `dc/dx`. If the adjoint solve is not fully converged, the returned gradients may be **uniformly attenuated** (effectively scaled by a factor `< 1`). This leads to:
- Slow convergence (small steps because the gradient magnitude is underestimated)
- Inaccurate Hessian approximation
- Possible convergence to a slightly wrong point

**Recommendation:** Always ensure adjoint solver convergence to a tolerance tighter than the optimization tolerances.

---

## 19. Deep Dive: Internal Gradient Computation

### When Does SNOPT Compute Gradients Internally?

SNOPT computes gradients by finite differences in these situations:

1. **`Derivative level` < 3**: Missing gradient components are estimated
2. **Gradient verification** (`Verify level` >= 0): FD estimates are computed for comparison
3. **Automatic switching to central differences**: Near the solution or after small line search steps

### The Complete Internal FD Procedure

```
         For each variable x_j with missing derivative:

         1. Determine perturbation:
            h_j = h_1 * max(1, |x_j|)           (forward differences)
            h_j = h_2 * max(1, |x_j|)           (central differences)

         2. Respect bounds:
            If x_j + h_j > u_j:  use h_j = -(h_j)   (backward difference)

         3. Compute perturbed function:
            F_plus = F(x + h_j * e_j)

         4. Estimate gradient:
            Forward:   dF/dx_j ≈ (F_plus - F_0) / h_j
            Central:   dF/dx_j ≈ (F_plus - F_minus) / (2*h_j)

         5. Apply to all constraint functions simultaneously:
            One perturbation of x_j gives ALL dF_i/dx_j for i=1,...,m
```

### Cost Analysis

For `n` design variables and `m` constraints with `Derivative level 0`:

```
         Forward differences:
           Extra function evals per major iteration: n
           Total per major iteration: 1 (base) + n (FD) = n + 1

         Central differences:
           Extra function evals per major iteration: 2n
           Total per major iteration: 1 (base) + 2n = 2n + 1

         With user-supplied gradients (Derivative level 3):
           Extra function evals: 0 (just 1 function + 1 gradient eval)
```

For a typical DaFoam case with `n ~ 200` design variables where each CFD solve takes minutes, the difference is dramatic:
- User gradients (adjoint): ~2 CFD solves per major iteration
- FD gradients: ~201 CFD solves per major iteration → **100x slower**

### Interaction with Nonderivative Linesearch

When `Nonderivative linesearch` is set, SNOPT separates function evaluation from gradient evaluation:
- During the line search: only `mode = 0` calls (functions only)
- At the accepted point: `mode = 2` call (functions + gradients)

This reduces the total number of gradient evaluations but requires the user to properly handle the `mode` flag.

### When Central Differences are Triggered

SNOPT automatically switches from forward to central differences when:

```
         1. Line search step alpha_k < 0.01 * alpha_{max}
            → Suggests gradient inaccuracy is causing poor search directions

         2. Optimality measure is small:
            max|d_j| < 10 * eps_opt
            → Need higher accuracy near the solution

         3. Difference interval h gives: |g_FD - g_user| > sqrt(h) * |g_user|
            → Forward differences are unreliable
```

---

## 20. Deep Dive: Gradient Discrepancy and Its Consequences

### Types of Gradient Errors

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    TAXONOMY OF GRADIENT ERRORS                      │
    ├──────────────────────┬──────────────────────────────────────────────┤
    │ Systematic bias      │ Gradients consistently too large/small       │
    │                      │ E.g., adjoint not converged, missing term    │
    ├──────────────────────┼──────────────────────────────────────────────┤
    │ Random noise         │ Gradient errors vary randomly                │
    │                      │ E.g., noisy function evals, iterative solver │
    ├──────────────────────┼──────────────────────────────────────────────┤
    │ Structured error     │ Some components correct, others wrong        │
    │                      │ E.g., missing coupling term in multiphysics  │
    ├──────────────────────┼──────────────────────────────────────────────┤
    │ Catastrophic error   │ Completely wrong gradients (wrong sign, etc) │
    │                      │ E.g., bug in adjoint code                    │
    └──────────────────────┴──────────────────────────────────────────────┘
```

### What SNOPT Does When Gradients Are Wrong

#### During Optimization (No Verification)

If `Verify level = -1` and gradients are incorrect:

1. **QP subproblem is formed with wrong data**: The linearized constraints don't match the actual constraint surface, and the quadratic model of the Lagrangian has wrong curvature information

2. **Search direction is wrong**: The QP solution `p_k` does not point toward the true optimum

3. **Line search may still succeed**: Even with a wrong direction, the merit function may decrease along `p_k` (especially if the error is small). SNOPT doesn't know the direction is wrong.

4. **BFGS update compounds the error**: The Hessian update uses `y_k = nabla L_{k+1} - nabla L_k`. With wrong gradients:
   ```
            y_k_wrong = y_k_true + (error_{k+1} - error_k)

            H_{k+1} incorporates the gradient error into the curvature model
   ```
   Over many iterations, the Hessian approximation drifts away from the true Lagrangian Hessian.

5. **Convergence behavior**:
   - **Small errors** (< 1e-4 relative): Optimization may converge to approximately correct point, but convergence rate drops from superlinear to linear
   - **Moderate errors** (1e-4 to 1e-1): Optimization converges to a wrong point or oscillates
   - **Large errors** (> 1e-1): Optimization diverges, hits iteration limit, or exits with `inform = 41` ("current point cannot be improved")

#### During Verification

If `Verify level >= 1`, SNOPT compares user gradients with FD estimates:

```
         For each component j:
           g_user_j   (user supplied)
           g_FD_j     (finite difference estimate)

           discrepancy = |g_user_j - g_FD_j|
           rel_error   = discrepancy / max(|g_user_j|, |g_FD_j|, 1)
```

**Actions based on discrepancy:**

| Situation | SNOPT Response |
|-----------|----------------|
| All components OK | Prints confirmation, proceeds with optimization |
| Some components "Bad?" | Prints warning, proceeds anyway (user should investigate) |
| Components marked "BAD" | Prints error, may exit with `inform = 51, 52, or 55` |
| Verification at `Verify level 0` fails | Prints summary warning, proceeds |

### The Subtle Danger: Consistent But Wrong Gradients

The most insidious case is when gradients are consistently wrong in a way that FD checking doesn't catch:

```
         Example: Adjoint gives df/dx = -2.345
                  FD estimate gives   df/dx = -2.340
                  True gradient is    df/dx = -2.345

         → Verify level says "OK!" because user ≈ FD
         → But if there's a systematic bias that is smooth, both the
           analytical and FD gradients can agree while being wrong
           (e.g., both computed from a not-fully-converged CFD solution)
```

### Gradient Error and the BFGS Update (Detailed)

The BFGS secant condition requires:

```
         H_{k+1} s_k = y_k
```

With gradient errors `e_k = g_k^{error} - g_k^{true}`:

```
         y_k^{observed} = (g_{k+1} + e_{k+1} - J_{k+1}^T pi_{k+1})
                        - (g_k + e_k - J_k^T pi_k)
                        = y_k^{true} + (e_{k+1} - e_k)
```

If the error is **constant** (`e_{k+1} = e_k`): `y_k^{observed} = y_k^{true}` → BFGS is unaffected!

If the error **varies** (`e_{k+1} ≠ e_k`): the difference `e_{k+1} - e_k` corrupts the curvature information, and the Hessian approximation degrades.

**Implication:** A constant gradient bias is less damaging to BFGS convergence than a variable (noise-like) gradient error. However, even constant bias corrupts the search direction and convergence point through the QP subproblem.

### Diagnostic: How to Detect Gradient Issues During Optimization

Even with `Verify level = -1`, you can detect gradient problems from the SNOPT output:

```
    Symptom                          │ Likely Gradient Issue
    ─────────────────────────────────┼──────────────────────────────────
    Very small steps (alpha << 1)    │ Directional derivative doesn't
    every iteration                  │ match merit function decrease
                                     │
    Oscillating objective            │ Gradient direction inconsistent
                                     │ with actual function changes
                                     │
    "Current point cannot be         │ Search direction is not a
    improved" (inform 41)            │ descent direction (wrong gradient)
                                     │
    Many rejected line search steps  │ Predicted decrease not realized
                                     │ (gradient-function mismatch)
                                     │
    Major optimality not decreasing  │ Reduced gradient doesn't converge
    even as feasibility improves     │ to zero (wrong gradient of L)
```

---

## 21. SNOPT in the Context of FIML Optimization

### FIML Problem Structure

Field Inversion and Machine Learning (FIML) optimizations, as implemented in DaFoam, have a specific structure that interacts with SNOPT in important ways.

A typical FIML problem:

```
         minimize    J(beta) = sum_i (u_i^{CFD}(beta) - u_i^{exp})^2
           beta

         subject to  R(u, beta) = 0     (CFD governing equations, implicit)
                     l <= beta <= u      (bounds on inversion field)
```

where `beta` is a field of correction factors (potentially hundreds or thousands of values), `u` is the flow field, and `R = 0` represents the discretized RANS equations.

### SNOPT Behavior in FIML

#### High-Dimensional Design Space

FIML problems often have `n ~ 100-5000` design variables (one per mesh cell or cluster). This means:
- The number of superbasic variables `n_S` can be large
- Limited-memory Hessian is typically used (automatic if `n_S > 75`)
- Each BFGS update pair `(s_k, y_k)` captures limited curvature information

#### Typical SNOPT Configuration for FIML

```python
# From DaFoam FIML tutorials
prob.driver.opt_settings = {
    "Major feasibility tolerance": 1.0e-6,
    "Major optimality tolerance": 1.0e-6,
    "Minor feasibility tolerance": 1.0e-6,
    "Verify level": -1,            # Skip verification (adjoint trusted)
    "Function precision": 1.0e-6,  # CFD solver accuracy
    "Major iterations limit": 50,  # Limited budget
    "Nonderivative linesearch": None,  # Use nonderiv LS
    "Print file": "opt_SNOPT_print.txt",
    "Summary file": "opt_SNOPT_summary.txt",
}
```

#### Why `Nonderivative linesearch`?

In FIML/DaFoam, each function evaluation requires a full CFD solve. During the line search, only function values are needed (not gradients). Using a nonderivative line search avoids unnecessary adjoint solves during backtracking.

#### Why `Verify level -1`?

1. Each verification requires `~n` additional CFD solves
2. The adjoint solver has been validated separately
3. For `n = 500` design variables, verification would cost 500 extra CFD solves

#### Why `Function precision 1e-6`?

The CFD solver (OpenFOAM via DaFoam) iterates until residuals drop below a threshold. The function values (drag, lift, pressure) are accurate to approximately this level. Setting `Function precision` accordingly ensures SNOPT doesn't try to distinguish between function values that are within the noise floor.

### Convergence Patterns in FIML

```
         Iteration │ Objective  │ Feasibility │ Optimality │ Step  │ Penalty
         ──────────┼────────────┼─────────────┼────────────┼───────┼────────
             0     │ 1.23e+02   │ 0.00e+00    │ 5.67e+01   │  ---  │ 0.0e+00
             1     │ 8.91e+01   │ 0.00e+00    │ 3.45e+01   │ 1.0   │ 0.0e+00
             2     │ 5.23e+01   │ 0.00e+00    │ 1.89e+01   │ 1.0   │ 0.0e+00
            ...    │   ...      │    ...       │   ...      │  ...  │  ...
            20     │ 1.05e+00   │ 0.00e+00    │ 4.56e-02   │ 0.3   │ 0.0e+00
            30     │ 9.87e-01   │ 0.00e+00    │ 1.23e-03   │ 0.1   │ 0.0e+00
            40     │ 9.85e-01   │ 0.00e+00    │ 8.76e-05   │ 0.05  │ 0.0e+00
            50     │ 9.85e-01   │ 0.00e+00    │ 2.34e-06   │ 0.02  │ 0.0e+00
```

**Typical pattern:**
- Fast initial decrease (large steps, limited-memory Hessian improving)
- Slowing convergence as optimality approaches tolerance
- Step lengths decrease near the solution (limited-memory Hessian has limited curvature info)

**No constraints case:** Feasibility stays at 0 (unconstrained or only bounds). Penalty parameters stay at 0. The problem reduces to bound-constrained optimization.

---

## 22. Diagnostic Reading: Understanding SNOPT Output

### The SNOPT Print File

The print file (`opt_SNOPT_print.txt`) contains detailed information about each major iteration. Here's how to read the key columns:

```
         Major Minors    Step   nCon  Feasible  Optimal   MeritFunction  nS  Penalty
           0     12              1  0.0E+00  5.7E+01   1.2345E+02       5  0.0E+00
           1      4    1.0E+00   2  0.0E+00  3.5E+01   8.9100E+01       7  0.0E+00
           2      2    1.0E+00   3  1.2E-07  1.9E+01   5.2300E+01       8  0.0E+00
```

| Column | Meaning |
|--------|---------|
| `Major` | Major iteration number |
| `Minors` | Number of QP iterations to solve the subproblem |
| `Step` | Line search step length `alpha_k` (1.0 = full step) |
| `nCon` | Cumulative constraint function evaluations |
| `Feasible` | Maximum constraint violation (primal infeasibility) |
| `Optimal` | Maximum complementarity violation (dual infeasibility) |
| `MeritFunction` | Current merit function value |
| `nS` | Number of superbasic variables |
| `Penalty` | `||D||` (norm of penalty parameters) |

### What to Watch For

```
    ┌─────────────────────────────────────────────────────────────────┐
    │ HEALTHY OPTIMIZATION                                            │
    │                                                                 │
    │ - Step ≈ 1.0 (full Newton steps accepted)                      │
    │ - Feasible decreasing steadily                                  │
    │ - Optimal decreasing steadily                                   │
    │ - MeritFunction decreasing monotonically                        │
    │ - nS stabilizes after a few iterations                         │
    │ - Minors stays small (1-5)                                      │
    │ - Penalty stays at 0 (or stabilizes)                           │
    ├─────────────────────────────────────────────────────────────────┤
    │ UNHEALTHY OPTIMIZATION                                          │
    │                                                                 │
    │ - Step << 1 (e.g., 1e-4): gradient or function issue           │
    │ - Feasible oscillating: nonlinear constraints poorly handled    │
    │ - Optimal not decreasing: wrong gradients or bad Hessian        │
    │ - MeritFunction increasing: penalty params growing              │
    │ - nS growing unboundedly: problem has many DOF, slow progress   │
    │ - Minors very large (100+): QP difficult, possible degeneracy   │
    │ - Penalty growing rapidly: approaching infeasibility            │
    └─────────────────────────────────────────────────────────────────┘
```

### The Summary File

The summary file (`opt_SNOPT_summary.txt`) provides a condensed version:

```
         SNOPTC EXIT   0 -- finished successfully
         SNOPTC INFO   1 -- optimality conditions satisfied

         Problem name               myProblem
         No. of iterations              43   Objective value     9.8500E-01
         No. of major iterations        43   Linear objective    0.0000E+00
         Penalty parameter          0.000E+00  Nonlinear objective 9.850E-01
         No. of calls to funobj        89   No. of calls to funcon    89
         No. of superbasics             8   No. of basic nonlinears    0
         No. of degenerate steps        0   Percentage                0.00
```

Key diagnostics:
- `No. of calls to funobj/funcon`: Total function evaluations (includes line search)
- `No. of superbasics`: Degrees of freedom at solution
- `No. of degenerate steps`: Cycling indicator (should be 0 or very small)

---

## 23. Common Pitfalls and Troubleshooting

### Problem: "Current point cannot be improved" (Inform 41)

**Causes:**
1. Incorrect gradients (most common)
2. Function noise larger than `Function precision`
3. Poorly scaled problem
4. Starting point in a flat region

**Diagnosis:**
```
         1. Run with Verify level 3 → check for BAD gradients
         2. Increase Function precision if functions are noisy
         3. Scale variables/constraints to have similar magnitudes
         4. Try a different starting point
```

### Problem: Very Small Steps Every Iteration

**Causes:**
1. Directional derivative doesn't agree with actual function decrease
2. Function precision set too tight
3. Gradient noise

**Diagnosis:**
```
         Check the print file for:
           Step = 1.0E-06  or similar
           → Merit function decrease much less than predicted

         Fix: Set "Function precision" to match actual function accuracy
              (e.g., 1e-6 for CFD with loose convergence)
```

### Problem: Many Minor Iterations

**Causes:**
1. Poorly conditioned basis matrix `B`
2. Many active bound constraints changing
3. Degenerate QP (multiple variables at bounds)

**Diagnosis:**
```
         If Minors > 100 consistently:
           → Consider Scale option 1 or 2
           → Increase Superbasics limit if nS is near the limit
           → Check for redundant constraints
```

### Problem: Optimization Converges to Infeasible Point

**Causes:**
1. Problem is truly infeasible
2. Constraints too tight for the function precision
3. Elastic mode activated and never recovered

**Diagnosis:**
```
         Check for:
           Penalty >> 0 in print file → elastic mode active
           Inform 13 or 14 → SNOPT confirmed infeasibility
           Feasible tolerance > constraint violation → try tighter tolerance
```

### Problem: Superbasics Limit Reached (Inform 33)

**Causes:**
1. Problem has more degrees of freedom than expected
2. `Superbasics limit` set too low

**Fix:**
```
         Increase Superbasics limit:
           prob.driver.opt_settings["Superbasics limit"] = 2 * n_vars
```

---

## 24. References

### Primary Sources

1. **Gill, P.E., Murray, W., Saunders, M.A.** (2005). "SNOPT: An SQP Algorithm for Large-Scale Constrained Optimization." *SIAM Review*, 47(1), 99-131. [DOI: 10.1137/S0036144504446096](https://epubs.siam.org/doi/10.1137/S0036144504446096)

2. **Gill, P.E., Murray, W., Saunders, M.A.** (2002). "SNOPT: An SQP Algorithm for Large-Scale Constrained Optimization." *SIAM Journal on Optimization*, 12(4), 979-1006. [DOI: 10.1137/S1052623499350013](https://epubs.siam.org/doi/10.1137/S1052623499350013)

3. **SNOPT 7.7 User's Guide.** [PDF](https://ccom.ucsd.edu/~optimizers/static/pdfs/snopt7-7.pdf)

4. **SNOPT 7 User's Guide.** [PDF](https://web.stanford.edu/group/SOL/guides/sndoc7.pdf)

### Software Documentation

5. **pyOptSparse Documentation: SNOPT.** [Link](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/optimizers/SNOPT.html)

6. **GAMS SNOPT Documentation.** [Link](https://www.gams.com/latest/docs/S_SNOPT.html)

7. **AIMMS SNOPT Description.** [Link](https://documentation.aimms.com/user-guide/aimms-ide/Solvers/SNOPT/SNOPT_Description_of_SNOPT_Algorithm.html)

8. **Stanford SOL SNOPT Page.** [Link](https://web.stanford.edu/group/SOL/software/snoptHelp/About_SNOPT.htm)

### Background Theory

9. **Nocedal, J., Wright, S.J.** (2006). *Numerical Optimization*, 2nd Edition. Springer. (Chapters 12, 18 on SQP methods)

10. **Gill, P.E., Murray, W., Wright, M.H.** (1981). *Practical Optimization*. Academic Press.

11. **Powell, M.J.D.** (1978). "A fast algorithm for nonlinearly constrained optimization calculations." *Numerical Analysis*, Lecture Notes in Mathematics 630, Springer.

---

## Appendix A: SNOPT Algorithm Flowchart

```
    ┌──────────────────┐
    │   Problem Setup   │
    │  (bounds, x_0)   │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Crash / Warm     │──── Determine initial basis B, S, N partition
    │  Start            │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                     MAJOR ITERATION LOOP                         │
    │                                                                  │
    │   ┌──────────────────┐                                          │
    │   │ Evaluate f, c    │                                          │
    │   │ Evaluate/estimate│                                          │
    │   │ nabla f, J       │                                          │
    │   └────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │   ┌──────────────────┐                                          │
    │   │ Form QP          │  q(d) = g^T d + 1/2 d^T H d             │
    │   │ subproblem       │  s.t. J d = -(c - J x_k), bounds        │
    │   └────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │   ┌──────────────────────────────────────┐                      │
    │   │        MINOR ITERATION LOOP          │                      │
    │   │                                      │                      │
    │   │  Solve for p_S (superbasic step)     │                      │
    │   │  Solve for p_B (basic step via B)    │                      │
    │   │  Ratio test → bound hits             │                      │
    │   │  Update basis partition B, S, N       │                      │
    │   │  Update R (Cholesky of reduced H)     │                      │
    │   │  Price nonbasic variables             │                      │
    │   │                                      │                      │
    │   └────────┬─────────────────────────────┘                      │
    │            │  QP solution: p_k, pi_hat_k                        │
    │            ▼                                                     │
    │   ┌──────────────────┐                                          │
    │   │  Line Search      │  Find alpha: M(x+alpha*p) < M(x)       │
    │   │  (derivative or   │  using cubic/quadratic interpolation    │
    │   │   nonderivative)  │                                          │
    │   └────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │   ┌──────────────────┐                                          │
    │   │ Update x, pi, H  │  x ← x + alpha*p                       │
    │   │ BFGS update to R │  H ← BFGS(H, s_k, y_k)                 │
    │   └────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │   ┌──────────────────┐     YES                                  │
    │   │ Converged?       │──────────► EXIT (inform 0 or 1)          │
    │   │ (KKT conditions) │                                          │
    │   └────────┬─────────┘                                          │
    │            │ NO                                                   │
    │            │                                                     │
    │            └─────────────────────────────────► (loop back)       │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Quick Reference Card

```
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    SNOPT QUICK REFERENCE                        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                 ║
    ║  Algorithm: SQP with reduced-Hessian QP solver                 ║
    ║  Merit fn:  Augmented Lagrangian (smooth)                      ║
    ║  Hessian:   BFGS (full or limited memory)                      ║
    ║  QP solver: SQOPT (active-set simplex + reduced Hessian)       ║
    ║  Basis:     LUSOL (sparse LU by Saunders)                      ║
    ║                                                                 ║
    ║  Key tolerances:                                                ║
    ║    Major feas tol → constraint satisfaction                     ║
    ║    Major opt tol  → KKT optimality                             ║
    ║    Function prec  → tells SNOPT about function noise           ║
    ║    Difference int → FD step size control                       ║
    ║                                                                 ║
    ║  Sweet spot: Large n, sparse constraints, small n_S            ║
    ║  Weakness:   Non-smooth functions, very large n_S              ║
    ║                                                                 ║
    ║  Golden rule: ALWAYS verify gradients during development       ║
    ║               (Verify level 3), then disable in production     ║
    ║               (Verify level -1)                                ║
    ║                                                                 ║
    ╚══════════════════════════════════════════════════════════════════╝
```

---

*Guide compiled from SNOPT source code (v7.7), official documentation, published papers by Gill, Murray & Saunders, and practical experience with pyOptSparse/DaFoam/OpenMDAO integration.*
