---
date: 2026-02-09
topic: critical-analysis-geko-moe-ensemble
parent: 2026-02-09-geko-moe-ensemble-fiml-brainstorm.md
---

# Critical Analysis: Is GEKO+MoE+Ensemble the Right Long-Term Bet?

## Purpose

This document interrogates the brainstorm's core assumptions honestly. The question is not "can we build this?" but "does this actually solve the problems that matter in turbulence modeling, and is there something fundamentally better?"

---

## 1. The Hierarchy of Turbulence Modeling Failures

Before evaluating any architecture, we must be clear about what actually fails in RANS and why. Not all failures are equal — some are fixable by better calibration, others require fundamentally different mathematics.

### Level 1: Calibration Failures (Fixable by GEKO-style tuning)

These are cases where the *form* of the model is correct but the *coefficients* are suboptimal for a given flow:
- Default SST over-predicts eddy viscosity in mild APG → flow separates too late
- Default k-epsilon under-predicts near-wall heat transfer in impinging jets
- Spreading rate mismatch in free shear layers

**GEKO directly addresses Level 1.** Its tunable coefficients span exactly this failure space. ML-augmented GEKO (our proposal) would automate and spatially-vary this calibration. This is valuable and publishable, but it is *calibration optimization*, not *model improvement*.

### Level 2: Structural Failures of the Boussinesq Hypothesis (NOT fixable by any EVM)

The Boussinesq hypothesis assumes Reynolds stress is proportional to mean strain rate:

```
tau_ij = -2 * nu_t * S_ij + (2/3) * k * delta_ij
```

This is fundamentally wrong when:
- **Reynolds stress anisotropy matters** — secondary flows in ducts, wing-body junctions, turbomachinery passages. The stress tensor has 6 independent components; Boussinesq collapses them to 1 scalar (nu_t). No amount of coefficient tuning recovers the lost 5 degrees of freedom.
- **Normal stress effects drive the flow** — impinging flows, stagnation points, flows with strong rotation
- **Turbulence is far from local equilibrium** — rapid distortion (shock-BL interaction), massive separation where production ≠ dissipation locally

**GEKO cannot fix Level 2.** Its CCORNER adds a single nonlinear term for secondary flows, but this is a patch — it cannot represent arbitrary anisotropy. No scalar eddy viscosity model can.

### Level 3: Scale Resolution Failures (NOT fixable by any RANS model)

These require resolving turbulent structures:
- Laminar-turbulent transition (partially addressed by gamma-Re_theta)
- Large-scale unsteady separation (DES/DDES territory)
- Acoustic noise sources
- Turbulent mixing in reacting flows

**Neither GEKO nor any RANS model addresses Level 3.** This requires scale-resolving simulation (LES, DES, DNS).

### Where Our Proposal Sits

| Failure Level | Our GEKO+MoE+Ensemble | Honest Assessment |
|--------------|----------------------|-------------------|
| L1: Calibration | Directly targets this | Strong fit |
| L2: Boussinesq | Cannot address fundamentally | **Ceiling exists** |
| L3: Scale resolution | Out of scope | Not a goal |

**The honest conclusion:** Our proposal is excellent for Level 1 problems and will produce genuine improvements for many industrial flows where calibration is the bottleneck. But it has a **hard ceiling** — it cannot overcome the Boussinesq limitation no matter how sophisticated the MoE or ensemble strategy.

---

## 2. What Could Be Better Than GEKO for the Long Term?

### Option A: Tensor Basis Neural Networks (TBNN)

Instead of learning scalar corrections to eddy viscosity, learn corrections to the full Reynolds stress tensor using the Pope (1975) tensor basis decomposition:

```
tau_ij = Σ_n g_n(invariants) * T_ij^(n)
```

where `T_ij^(n)` are the 10 tensor basis functions formed from the strain rate and rotation rate tensors, and `g_n` are scalar coefficients learned by a neural network from the 5 scalar invariants.

**Strengths:**
- Breaks through the Boussinesq ceiling — can represent arbitrary anisotropy
- Galilean invariance guaranteed by construction
- Recent work (2024-2025) shows TBNN corrections improve secondary flows in ducts, anisotropy prediction in channels, and wake mixing in turbine cascades
- TBNN-LSTM extension handles unsteady flows

**Weaknesses:**
- 10 scalar functions to learn instead of 1 (beta) — much harder training
- Stability is worse (6 stress components can each go wrong)
- Interpretability is lost — you can't distill `g_3(I_1,...,I_5) * T_ij^(3)` into something a turbulence modeler reads at a glance
- No existing implementation in DAFoam's FIML pipeline

**Verdict:** Higher ceiling but much harder path. This is the 5-10 year vision.

### Option B: ML-Augmented EARSM (Explicit Algebraic Reynolds Stress Model)

EARSM occupies the middle ground between linear EVM and full Reynolds Stress Model. It algebraically derives nonlinear stress-strain relationships from the RSM transport equations under local equilibrium assumptions:

```
tau_ij = -2 * nu_t * (beta_1 * S_ij + beta_2 * (S_ik*W_kj - W_ik*S_kj) + ...)
```

The beta coefficients are rational functions of strain and rotation invariants.

Recent work (2024-2025):
- NN-augmented EARSM where neural networks compute the beta coefficients — "much better results than standard EARSM" for channel flow and flat-plate BL
- Data-driven EARSM derivation via symbolic regression (SpaRTA) — tailor-made models for turbine wake mixing
- AutoTurb (2025): LLM-based automatic discovery of algebraic corrections to linear stress models

**Strengths:**
- Captures key anisotropy effects (secondary flows, curvature, rotation) that GEKO fundamentally cannot
- Still algebraic — no extra PDEs to solve (unlike full RSM)
- The beta coefficients in EARSM are scalar functions, compatible with the same FIML training pipeline
- More physically grounded than pure TBNN (derived from RSM theory, not purely data-driven)

**Weaknesses:**
- Local equilibrium assumption limits accuracy in highly non-equilibrium flows
- More complex implementation than GEKO
- Fewer tunable knobs than GEKO (less obvious MoE alignment)

**Verdict:** This is the strongest "next step beyond GEKO" candidate. It breaks the Boussinesq ceiling while remaining tractable for coupled adjoint training.

### Option C: GEKO + Nonlinear Extension (GEKO++)

Evolve GEKO itself beyond the Boussinesq hypothesis by adding ML-learned nonlinear stress-strain terms:

```
tau_ij = -2 * nu_t(GEKO coeffs) * S_ij + NN(features) * (S_ik*W_kj - W_ik*S_kj) + ...
```

Keep GEKO's linear EVM as the backbone, but add tensor basis corrections only where needed (flagged by MoE gating or ensemble disagreement).

**Strengths:**
- Preserves GEKO's interpretability and stability for regions where linear EVM suffices
- Adds anisotropy capability only where data indicates it's needed
- Graceful degradation: if NN fails, falls back to standard GEKO
- Compatible with our MoE architecture — some experts handle linear corrections (CSEP, CMIX), others handle nonlinear corrections

**Weaknesses:**
- Theoretical consistency is questionable — mixing GEKO's blending functions with additional nonlinear terms may have unforeseen interactions
- Implementation complexity is high
- Validation would require extensive testing

**Verdict:** Attractive hybrid but risky. Would need careful theoretical justification.

### Recommended Long-Term Trajectory

```
Phase     Base Model         Correction Type         Ceiling
──────    ──────────         ───────────────         ───────
Now       k-omega SST        Scalar beta (FIML)      Level 1
Year 1    GEKO               Spatially-varying        Level 1 (better)
                             GEKO coefficients
Year 2-3  GEKO + EARSM       Scalar + nonlinear       Level 1 + partial L2
                             stress corrections
Year 5+   EARSM/TBNN         Full tensor basis        Level 2
                             corrections
```

---

## 3. Does Our MoE+Ensemble Architecture Address the Real Challenges?

### Challenge 1: Generalization Across Flow Types

**The claim:** MoE with expert specialization enables compositional generalization — each expert learns one flow mechanism, and new flows are handled by composing experts.

**Critical assessment:** This is *partially* correct but overstated.

*What MoE actually solves:* It prevents the "one expert's correction is another flow's poison" problem. If Expert 1 learns APG separation correction and Expert 2 learns free shear mixing, a flow with both APG and free shear regions can use Expert 1 in the BL and Expert 2 in the wake. The monolithic NN would compromise between both.

*What MoE does NOT solve:* It doesn't help with flows whose physics is **outside all experts' training distribution**. If none of the K experts has seen shock-BL interaction, the ensemble will disagree (good for OOD detection) but cannot produce a useful correction. MoE provides **interpolation within the convex hull of training physics**, not true extrapolation.

*What's needed additionally:*
- A rich and diverse training case set covering the major flow physics categories
- A fall-back mechanism (which our ensemble OOD detection provides)
- Possibly meta-learning (Point 5 from SOTA doc) for rapid adaptation when new physics is encountered

**Grade: B+** — Genuine improvement over monolithic NN, but not a complete solution.

### Challenge 2: Physical Realizability and Stability

**The claim:** GEKO's bounded coefficients + ensemble mean smoothing provide stability.

**Critical assessment:** This is the **weakest part** of our proposal.

GEKO coefficient bounds prevent individual coefficients from going to absurd values, but:
- *Spatially discontinuous* coefficient fields can still destabilize the solver (sharp gradients in CSEP between adjacent cells)
- Ensemble mean can mask instability (one member diverges, mean looks fine but is meaningless)
- MoE gating transitions between experts can create artificial gradients
- None of this provides a *guarantee* of stability — only the Lyapunov approach (Point 2) does that

*What's needed additionally:*
- Smoothness regularization on the spatially-varying coefficient fields (TV or Sobolev penalty)
- Per-ensemble-member convergence checks (reject diverged members before averaging)
- Gradient clipping on coefficient spatial gradients in the gating output

**Grade: C+** — Reasonable heuristics but no guarantee. This is a known gap.

### Challenge 3: Uncertainty Quantification

**The claim:** Ensemble spread provides calibrated UQ and OOD detection.

**Critical assessment:** This is **well-supported by literature but has a known flaw**.

The 2025 Novello et al. finding is critical: Deep Ensembles are *overconfident* out-of-distribution. The ensemble members were all trained on similar data, so they converge to similar wrong answers OOD — the spread collapses precisely when it should be large.

*Mitigations we propose:*
- Anchored regularization (pull toward default coefficients in low-data regions) — this helps but doesn't fully solve it
- Training case bagging for diversity — moderate help
- Architecture variation — moderate help

*What's actually needed:*
- A **separate OOD detector** that does not rely on ensemble spread alone. Options:
  - Input feature density estimation (flag cells whose features are far from training distribution)
  - Gaussian Process gating (provides principled uncertainty on gating weights)
  - Mahalanobis distance in feature space
- Or combine ensemble with GP (ensemble of GPs, or GP on ensemble outputs)

**Grade: B-** — The ensemble UQ story needs a standalone OOD detector to be credible for safety-critical applications.

### Challenge 4: Interpretability and Certifiability

**The claim:** GEKO coefficient corrections are interpretable; symbolic distillation per expert is tractable.

**Critical assessment:** This is the **strongest part** of the proposal and a genuine competitive advantage.

- "CSEP = 2.5 in the APG region" is meaningful to any turbulence modeler
- Each expert outputs 1-3 coefficient corrections from 3-4 features — ideal for PySR
- The gating weights themselves visualize which physics dominates where
- Symbolic distillation produces something publishable and implementable in any CFD solver

*Potential issues:*
- If experts don't cleanly specialize (gating collapse or uniform routing), interpretability degrades
- If corrections are small perturbations around defaults, SR may not find meaningful expressions

**Grade: A-** — This is where GEKO+MoE truly shines relative to alternatives.

### Challenge 5: Training Data Scarcity

**The claim:** Not directly addressed in our proposal.

**Critical assessment:** This is a **significant gap** that will bite us in practice.

To train K=3 experts on diverse physics, we need:
- Multiple APG/separation cases (different geometries, Re ranges) for Expert 1
- Multiple free shear/mixing cases for Expert 2
- Multiple near-wall/heat transfer cases for Expert 3
- Held-out cases from each category for validation

This requires ~10-20 distinct DNS/LES datasets with volumetric fields. Available:
- Ramp/bump/backward-facing step (separation) — ~3-5 cases in public domain
- Channel flows at various Re (near-wall) — abundant
- Mixing layers and jets — ~3-5 cases
- Periodic hills — well-studied, ~2-3 cases
- Airfoils (transition + separation) — some available from CFL3D validation
- Complex 3D geometries — **almost no public DNS data**

*What's needed:* Combine with Point 3 from SOTA doc — train from surface experimental data, not just volumetric DNS. This massively expands the usable data pool.

**Grade: C** — Cannot ignore this; data scarcity will limit ambitions regardless of architecture quality.

### Challenge 6: Industrial Scalability

**The claim:** GEKO is already industrial; MoE is cheap per-cell; ensemble is parallelizable.

**Critical assessment:** Mostly correct.

- GEKO solves 2 PDEs (like SST) — no extra cost from the base model
- MoE forward pass per cell: K small NNs + softmax gating ~ microseconds. For a 1M cell mesh with 1000 iterations, this adds ~seconds total. Negligible vs. PDE solve.
- Ensemble: M=5 runs are embarrassingly parallel. Wall time = 1 run if you have 5 cores.
- Training: M * N_cases * N_iterations adjoint solves. This is the bottleneck. For M=5, N=10, and 50 optimization iterations → 2500 primal+adjoint solves. At ~5 min each on 4 cores → ~200 hours. Parallelizable to ~40 hours with 5 concurrent jobs.

**Grade: A-** — Scalable for research; industrial deployment requires the symbolic distillation step (no NN inference, just a formula).

---

## 4. Alignment Assessment Summary

```
                                    Our Proposal's
Turbulence Challenge                Alignment           Gap
────────────────────                ─────────           ───
Calibration optimization            ★★★★★  Excellent   —
Generalization across flows         ★★★★☆  Good        Need diverse training data
Physical interpretability           ★★★★★  Excellent   —
Uncertainty quantification          ★★★☆☆  Adequate    OOD detection needs work
Solver stability                    ★★☆☆☆  Weak        No guarantees
Reynolds stress anisotropy          ★☆☆☆☆  Not addressed  Boussinesq ceiling
Training data efficiency            ★★☆☆☆  Weak        Need multi-modal data (Pt 3)
Industrial deployability            ★★★★☆  Good        After symbolic distillation
```

**The bottom line:** Our proposal is *strongly aligned* with the practical, near-term needs of the turbulence modeling community (better calibration, interpretability, UQ). It is *not aligned* with the fundamental theoretical frontier (anisotropy, realizability). This is fine for a 2-3 year research program that produces practical tools and publications, but we should be honest that it has a ceiling.

---

## 5. Refined Project Vision

### The Honest Framing

"We propose a practical framework for **automated, interpretable, uncertainty-aware RANS calibration** that generalizes across flow types by combining GEKO's physically-modular coefficient space with mixture-of-experts compositional learning and ensemble uncertainty quantification."

Note: **calibration**, not **modeling**. We are making the best possible linear EVM, not transcending the EVM paradigm. This is still highly valuable — the vast majority of industrial CFD uses linear EVMs, and most RANS failures in practice *are* calibration failures.

### What We Should Explicitly NOT Claim

- "Our model captures Reynolds stress anisotropy" — it doesn't
- "Our model generalizes to any flow" — it generalizes within the convex hull of training physics
- "Our ensemble provides reliable uncertainty everywhere" — overconfident OOD without additional safeguards
- "GEKO is the final answer" — it's the right base model for now, with a planned evolution path

### What We CAN Claim If It Works

- First end-to-end adjoint-trained MoE turbulence correction
- First framework aligning MoE expert specialization with GEKO's physics-modular coefficients
- Demonstrated compositional generalization across multiple flow types
- Interpretable corrections distillable to symbolic expressions per expert
- Calibrated uncertainty with OOD-safe fallback to uncorrected GEKO
- Open-source implementation in DAFoam

### The Long-Term Extension Path

The architecture we build is **deliberately model-agnostic**. The MoE+Ensemble+Adjoint training pipeline works regardless of what the experts output. In the near term, experts output δCSEP, δCMIX, δCNW (scalar GEKO corrections). In the long term:

```
Year 1-2: Expert outputs scalar GEKO coefficient corrections (Level 1)
Year 3-4: Expert outputs EARSM beta coefficients (Level 1 + partial Level 2)
Year 5+:  Expert outputs tensor basis coefficients g_n (Level 2)
```

The gating network, ensemble strategy, adjoint training loop, and symbolic distillation pipeline are all reusable. Only the expert architectures and the base model change. **This is the strongest argument for the current proposal: it builds reusable infrastructure that scales to harder problems.**

---

## 6. Open Questions Requiring Resolution

1. **Is the GEKO paper (AIAA J. 2025) sufficient to implement the model from scratch?** If proprietary details are withheld, we may need to use a simplified GEKO variant or work from the NASA TMR (Turbulence Modeling Resource) documentation.

2. **Should we integrate the multi-modal data direction (Point 3) from the start?** The data scarcity problem is severe enough that ignoring it may render the architecture untestable. Surface-Cp-only training should be a Phase 1 capability, not a future extension.

3. **Do we need a standalone OOD detector alongside the ensemble?** The literature is clear that ensembles alone are insufficient. A feature-space density estimator (e.g., kernel density or isolation forest on the 11 FIML features) is cheap and would significantly strengthen the UQ story.

4. **What validation cases are available and sufficient?** We need to define a concrete benchmark suite *before* designing the training pipeline. Candidates:
   - NASA Turbulence Modeling Resource cases (flat plate, bump, hump, 2D hill, axisymmetric jet)
   - Ramp cases from DAFoam tutorials
   - McConkey et al. curated RANS-DNS dataset
   - Stanford DNS data archive

---

## References

- Menter, F.R. et al. (2025). "Generalized k-omega (GEKO) Two-Equation Turbulence Model." [AIAA Journal](https://arc.aiaa.org/doi/10.2514/1.J065393).
- arXiv:2502.11218 (2025). "Bayesian Optimization of the GEKO Turbulence Model." [arXiv](https://arxiv.org/abs/2502.11218).
- Cherroud, S. et al. (2025). "Space-dependent aggregation of stochastic data-driven turbulence models." [arXiv](https://arxiv.org/abs/2306.16996).
- Novello et al. (2025). "Quantifying Out-of-Training Uncertainty of Neural-Network based Turbulence Closures." [arXiv](https://arxiv.org/abs/2508.16891).
- Obaldía et al. (2025). "Physics-guided Bayesian NNs for zonal corrections and UQ." [arXiv](https://arxiv.org/abs/2511.14534).
- Patel et al. (2024). "On the Generalization Capability of FIML." [Aerospace](https://www.mdpi.com/2226-4310/11/7/592).
- Revisiting TBNN for Reynolds stress modeling (2024). [arXiv](https://arxiv.org/abs/2403.11746).
- NN-augmented EARSM in 2D flow (2024). [Springer](https://link.springer.com/chapter/10.1007/978-3-031-69035-8_2).
- AutoTurb: LLM-based algebraic turbulence model discovery (2025). [AIP](https://pubs.aip.org/aip/pof/article-abstract/37/1/015211/3331579/).
- Data-driven EARSM for turbine wakes (2025). [ASME](https://asmedigitalcollection.asme.org/turbomachinery/article-abstract/147/6/061007/1207846/).
- Duraisamy, K. (2021). "Perspectives on ML-augmented RANS and LES." [Phys. Rev. Fluids](https://arxiv.org/abs/2009.10675).
