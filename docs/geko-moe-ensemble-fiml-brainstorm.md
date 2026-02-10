---
date: 2026-02-09
topic: geko-moe-ensemble-generalizable-fiml
---

# Generalizable FIML via GEKO + Mixture-of-Experts + Ensemble UQ

## What We're Exploring

A research architecture that combines three ideas from the SOTA document (primarily Points 1 and 7, with GEKO as the base model) to build a **generalizable, uncertainty-aware, data-driven turbulence correction framework**:

1. **GEKO as the base RANS model** — its physically-interpretable tunable coefficients (CSEP, CNW, CMIX, CJET, CCURV, CCORNER) create a natural "knob space" that maps directly to distinct flow physics, making it far more amenable to ML augmentation than fixed-coefficient models like SA or SST.

2. **Mixture-of-Experts (MoE) correction architecture** (Point 1) — instead of a monolithic NN mapping features → beta, use K expert sub-networks each specializing in a different flow mechanism, with a gating network that routes based on local flow features.

3. **Ensemble UQ** (Point 7) — train M ensembles of the MoE model to provide calibrated uncertainty quantification, out-of-distribution detection, and improved mean predictions.

The key thesis: **GEKO's modular coefficient structure provides a physics-informed prior for MoE expert specialization**, creating a natural alignment between the model's tunable knobs and the experts' specialization domains.

---

## Literature Foundation

### GEKO: The Ideal ML-Augmentable Base Model

The Generalized k-omega (GEKO) model (Menter 2019; Menter & Lechner 2021; full AIAA Journal paper 2025) is a two-equation RANS model designed with explicitly tunable, physically-interpretable coefficients:

| Coefficient | Physical Domain | Effect | Default |
|-------------|----------------|--------|---------|
| **CSEP** | Boundary-layer separation | Controls eddy viscosity in APG regions; higher → more separation sensitivity | 1.75 |
| **CNW** | Near-wall behavior | Affects inner boundary layer; higher → higher wall shear stress/heat transfer | 0.5 |
| **CMIX** | Free shear mixing | Controls spreading rates of free shear layers; boundary-layer shielded | correlated with CSEP |
| **CJET** | Jet flows | Adjusts jet spreading rate independently of mixing layers | — |
| **CCURV** | Streamline curvature | Curvature correction for rotating/curved flows | — |
| **CCORNER** | Corner flows | Nonlinear stress-strain for secondary flows in corners/junctions | — |

**Critical property:** These coefficients are *independent* — adjusting CSEP does not affect the calibration controlled by CNW, thanks to blending functions that deactivate mixing/jet coefficients inside boundary layers and vice versa. This orthogonality is precisely what makes GEKO uniquely suited for compositional ML augmentation.

**Recent ML+GEKO work:**
- Bayesian optimization of CSEP/CNW for converging-diverging channel (2025, arXiv:2502.11218): Achieved improved reattachment prediction over default GEKO using sparse DNS data
- ML-assisted GEKO optimization for turbocharger compressor (2022): Neural network regression establishing correlations between flow features and optimal GEKO parameters
- ANSYS TechCon 2023: Adjoint optimization + ML for GEKO tuning in gas turbine combustion

**Key limitation:** GEKO is currently proprietary to ANSYS. No open-source implementation exists in OpenFOAM. Implementing it in DAFoam would require writing a new `DAkOmegaGEKO` turbulence model class from the published equations.

### MoE for Turbulence: The Cherroud et al. Breakthrough

The most relevant prior work is Cherroud et al. (2025), "Space-dependent aggregation of stochastic data-driven turbulence models" (J. Comput. Phys., arXiv:2306.16996), which independently arrived at essentially the same MoE idea:

- **Expert models:** Trained via Bayesian symbolic identification (SBL-SpaRTA) for specific flow classes (channels, jets, boundary layers, separated flows)
- **Gating function:** Exponentially weighted average based on local flow features: `g_k = exp(-||delta_k(x) - delta_bar(x)||^2 / (2*sigma_w^2))`
- **Flow features:** 11 local features including Q-criterion, turbulence intensity, pressure gradients, strain/rotation ratios (remarkably similar to DAFoam's 11 features)
- **Uncertainty:** Combines inter-model variability (structural) with intra-model parametric uncertainty via Polynomial Chaos expansion
- **Result:** XMA achieves more accurate predictions than baseline AND individual expert models on unseen flows, with reliable uncertainty estimates

**Differences from our proposed approach:**
- Cherroud uses *symbolic* expert corrections (interpretable but limited expressiveness)
- Their experts correct Reynolds stress anisotropy, not scalar beta multipliers
- They train experts *separately* then aggregate; we propose *jointly* training MoE end-to-end via adjoint
- They don't use GEKO — their base model is standard k-omega SST

### Ensemble UQ for Turbulence: State of the Art

Recent work (2024-2025) on uncertainty quantification for data-driven RANS:

- **Deep Ensembles vs. Gaussian Processes:** Novello et al. (arXiv:2508.16891, 2025) compared Deep Ensembles, MC Dropout, SVI, and GP methods. **Key finding: Deep Ensembles and SVI are overconfident in out-of-training regions**, while GP methods give more robust OOD uncertainty. This is critical for deployment safety.

- **Physics-guided Bayesian NNs:** Obaldía et al. (arXiv:2511.14534, 2025) propose tensor-based BNNs with flow-regime classifiers that isolate shear-dominated regions — effectively a physics-guided ensemble approach.

- **Space-dependent stochastic aggregation** (Cherroud 2025, discussed above): Combines ensemble and MoE naturally — each expert is stochastic, and the aggregation provides both structural and parametric uncertainty.

### Generalization: The Core Challenge

The generalization problem is well-documented:
- Patel et al. (2024, Aerospace 11(7):592): "On the Generalization Capability of a Data-Driven Turbulence Model by Field Inversion and Machine Learning" — augmented model trained on 2D separated airfoil flows gives **poor predictive capability for NASA wall-mounted hump** (different flow class). They propose sensor-based localization.
- Duraisamy (2021, Phys. Rev. Fluids): Model-consistent training with physics-informed priors is necessary but not sufficient; generalization requires careful characterization of underlying assumptions.
- Wu et al. (2025): Conditioned field inversion (FI-CND) with binary gating (attached=off, separated=on) — a special case of MoE with K=2 hard experts.

---

## The Proposed Architecture: GEKO-MoE-Ensemble

### Core Concept

```
                         ┌─────────────────────────┐
                         │   GEKO Base Model        │
                         │   k-omega with tunable   │
                         │   CSEP, CNW, CMIX, ...   │
                         └────────────┬────────────┘
                                      │
                           Flow features (11+)
                                      │
                         ┌────────────▼────────────┐
                         │    Gating Network        │
                         │    g(features) → w_k     │
                         │    (softmax weights)     │
                         └────────────┬────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  Expert 1: APG   │  │  Expert 2: Free  │  │  Expert 3: Near  │
    │  Separation      │  │  Shear/Mixing    │  │  Wall/Heat Xfer  │
    │  (maps to CSEP)  │  │  (maps to CMIX)  │  │  (maps to CNW)   │
    │                  │  │                  │  │                  │
    │  Output: δCSEP   │  │  Output: δCMIX   │  │  Output: δCNW    │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                     │
             ▼                     ▼                     ▼
    CSEP_local = CSEP_0 + Σ w_k * δCSEP_k    (and similarly for CMIX, CNW)
                                      │
                         ┌────────────▼────────────┐
                         │   GEKO Solver with       │
                         │   Spatially-Varying      │
                         │   Coefficients           │
                         └─────────────────────────┘
```

### Why GEKO Coefficients Instead of Beta Multipliers?

The standard FIML approach learns `beta * Production`. The GEKO approach learns **spatially-varying GEKO coefficients**. This is fundamentally different and better:

1. **Physical interpretability:** `CSEP=2.5 at cell X` means "this region needs stronger separation sensitivity" — directly interpretable by turbulence modelers. `beta=0.7 at cell X` means nothing on its own.

2. **Natural bounds:** GEKO coefficients have physically meaningful ranges (e.g., CSEP ∈ [0.3, 3.0]) that bound the correction automatically. No need for ad-hoc `outputUpperBound`/`outputLowerBound`.

3. **Orthogonal corrections:** Because GEKO's blending functions deactivate irrelevant coefficients in each region, each expert's output naturally applies only where it matters. Expert 1 (APG/separation) can aggressively tune CSEP without affecting free shear behavior, because GEKO's internal blending handles the isolation.

4. **MoE-GEKO alignment:** Each MoE expert maps to a specific GEKO coefficient (or small subset), creating a natural 1:1 correspondence between expert specialization and physical mechanism. This is the key insight — GEKO was designed to be modular; MoE was designed to be modular; combining them creates a doubly-modular architecture.

5. **Symbolic distillation compatibility:** Because each expert outputs a single coefficient correction (e.g., δCSEP as a function of 3-4 features), symbolic regression can easily distill each expert into a simple formula. The final product is a modified GEKO model with interpretable, closed-form, spatially-varying coefficients.

### Two Possible Correction Modes

**Mode A: GEKO Coefficient Correction (Preferred)**
- Each expert outputs corrections to GEKO coefficients: `CSEP(x) = CSEP_default + NN_sep(features(x))`
- Advantages: interpretable, bounded, orthogonal, distillable
- Requires: implementing GEKO in DAFoam with support for spatially-varying coefficients

**Mode B: Hybrid Beta + GEKO**
- Use GEKO as the base model (with globally-optimized coefficients)
- Apply the standard beta multiplier approach on top: `beta(x) * P_k`
- The MoE learns the beta correction, but starting from a better-calibrated base
- Advantages: simpler implementation, leverages existing FIML infrastructure
- Disadvantage: loses the interpretability advantage of coefficient corrections

**Recommendation:** Start with Mode B (low-hanging fruit), then progress to Mode A as the GEKO implementation matures.

### Ensemble Strategy

Train M = 5-10 instances of the MoE model with diversity from:

1. **Random initialization** of all NN weights (standard deep ensemble diversity)
2. **Training case subsets** (bagging): each ensemble member trained on 80% of available cases
3. **Architecture variation**: slight differences in hidden layer sizes or number of experts
4. **Coefficient initialization variation**: start each ensemble member's GEKO base from slightly different default coefficients

At prediction time:
- **Mean correction:** `CSEP_mean(x) = (1/M) Σ_m CSEP_m(x)` — better than any single model
- **Uncertainty:** `CSEP_std(x)` — flags regions of high disagreement
- **OOD detection:** where `CSEP_std(x) > threshold`, fall back to default GEKO (no correction)
- **QoI uncertainty:** run M primal solves (parallelizable) to get bounds on CL, CD, etc.

**Addressing the overconfidence problem:** The literature shows Deep Ensembles are overconfident OOD. Mitigation strategies:
- Add an **anchored ensemble** regularization that pulls predictions toward default coefficients in regions far from training data
- Use a **GP-augmented gating network** that provides calibrated uncertainty on the gating weights themselves
- Apply **post-hoc calibration** on held-out validation flows

---

## Why This Approach Is Novel

| Element | Prior Work | Our Contribution |
|---------|-----------|-----------------|
| GEKO + ML | Bayesian optimization of global coefficients (2025) | **Spatially-varying** GEKO coefficients via coupled adjoint-trained MoE |
| MoE for turbulence | Cherroud's symbolic experts + post-hoc aggregation | **End-to-end** adjoint-trained MoE with physics-aligned expert-coefficient mapping |
| FIML correction | Monolithic NN → beta multiplier | **Structured** correction via GEKO coefficient space |
| Ensemble UQ for RANS | Standard deep ensembles (overconfident OOD) | **Anchored** ensembles with GP-augmented gating for calibrated UQ |
| Generalization | Train on everything / sensor-based localization | **Compositional** generalization via expert specialization + GEKO modularity |

No one has combined all three: GEKO as base model + MoE with expert-coefficient alignment + ensemble UQ with OOD-aware fallback.

---

## Key Decisions

### Decision 1: GEKO Implementation Strategy
**Options:**
- **(A) Implement GEKO from scratch in DAFoam** — Write `DAkOmegaGEKO.C/H` following the published equations and the existing `DAkOmegaSST` pattern. Major effort (~2-3 months) but gives full control and AD compatibility.
- **(B) Start with k-omega SST + beta, later add GEKO** — Use existing SST infrastructure to prove the MoE+Ensemble concept, then add GEKO as a follow-on. Lower risk, faster to first results.

**Recommendation:** Option B first, then A. The MoE+Ensemble architecture is model-agnostic; proving it on SST first de-risks the GEKO implementation.

### Decision 2: Expert-to-Physics Mapping
**Options:**
- **(A) Soft alignment** — Experts are free to learn any correction; we only encourage specialization via load-balancing loss and diverse training cases.
- **(B) Hard alignment** — Expert 1 is architecturally constrained to only output CSEP corrections, Expert 2 only CMIX, etc.
- **(C) Semi-hard alignment** — Experts output corrections to *subsets* of GEKO coefficients (e.g., Expert 1: {CSEP, CCURV}, Expert 2: {CMIX, CJET}, Expert 3: {CNW, CCORNER}).

**Recommendation:** Option C. Hard alignment is too restrictive (some flow regions need multiple coefficient adjustments); soft alignment may not achieve the desired specialization. Semi-hard provides structure while maintaining flexibility.

### Decision 3: Gating Architecture
**Options:**
- **(A) Feature-based softmax gating** — Standard MoE: gating NN takes the same 11 flow features as input, outputs softmax weights. Simple, proven.
- **(B) Gaussian Process gating** — Use a GP classifier for gating, providing calibrated uncertainty on the routing itself. More expensive, but better UQ.
- **(C) Physics-informed gating** — Use specific features known to indicate flow regime: pressure gradient for separation, Q-criterion for vortical, wall distance for near-wall. Reduces gating NN complexity.

**Recommendation:** Start with A, evolve to C. Physics-informed gating with just 2-3 features per expert makes the gating interpretable and less prone to overfitting.

### Decision 4: Number of Experts (K)
Start with K=3, aligned to the three primary GEKO coefficient groups:
1. **APG/Separation expert** (CSEP domain)
2. **Free shear/mixing expert** (CMIX/CJET domain)
3. **Near-wall/heat transfer expert** (CNW domain)

Can later add K=4 (curvature) and K=5 (corner flows) as training data for those regimes becomes available. The MoE architecture allows adding experts without retraining existing ones.

### Decision 5: Ensemble Size (M)
M=5 ensemble members is the practical sweet spot:
- Literature shows diminishing returns beyond M=5 for deep ensembles
- 5 primal+adjoint solves per training iteration is computationally feasible
- 5 prediction runs (parallelizable) provide reasonable uncertainty bounds
- Budget: if 1 FIML training costs T hours, ensemble training costs ~5T (parallelizable to ~T wall time with 5x resources)

---

## Research Roadmap

### Phase 0: Groundwork (Month 1-2)
- Implement GEKO equations in DAFoam as `DAkOmegaGEKO` (based on Menter 2019 paper + AIAA 2025 paper)
- Validate against published GEKO results (flat plate, backward-facing step, periodic hills)
- Verify CoDiPack AD works through the new model

### Phase 1: MoE on SST (Month 2-4)
- Extend `DARegression::compute()` to support K=3 sub-networks + gating layer
- Partition `regressionPar` into K expert blocks + 1 gating block
- Train on diverse cases: ramp (separation), jet mixing, wall-bounded channel
- Demonstrate expert specialization via gating weight visualization
- Compare against monolithic NN baseline

### Phase 2: Ensemble MoE (Month 4-6)
- Implement M=5 ensemble training at Python/OpenMDAO level
- Add anchored regularization toward beta=1 (no correction)
- Evaluate uncertainty calibration on held-out flows
- Demonstrate OOD detection: train on ramps+channels, test on airfoil
- Publish first paper: "Compositional Generalization and Uncertainty Quantification for FIML via Mixture-of-Experts Ensembles"

### Phase 3: GEKO + Coefficient MoE (Month 6-10)
- Switch base model from SST to GEKO
- Modify experts to output coefficient corrections (δCSEP, δCMIX, δCNW)
- Implement spatially-varying GEKO coefficients
- Train on expanded case set (10+ configurations)
- Demonstrate that GEKO+MoE generalizes better than SST+MoE

### Phase 4: Distillation & Deployment (Month 10-14)
- Apply symbolic regression to each expert (should be tractable: 3-4 features → 1 coefficient per expert)
- Produce closed-form GEKO coefficient expressions
- Validate distilled model on unseen complex geometry
- Publish second paper: "Interpretable, Generalizable Turbulence Model from GEKO-based FIML with Symbolic Distillation"

---

## Open Questions

1. **GEKO equations availability:** The full GEKO formulation is in the 2025 AIAA Journal paper (Menter). Is this sufficient to implement, or are there proprietary details not in the publication?

2. **CoDiPack through GEKO:** GEKO has blending functions with switches (F_blend for CMIX/CJET). Will CoDiPack handle the non-smooth transitions cleanly for reverse-mode AD?

3. **Training data diversity:** For K=3 experts to meaningfully specialize, we need training cases that exercise each physical mechanism distinctly. What cases are available?
   - Separation: ramp, bump, airfoil stall
   - Free shear: planar mixing layer, round jet
   - Near-wall: channel flow, flat plate heat transfer

4. **Computational cost:** Training M=5 ensembles of K=3 expert MoE models on N=10 cases = 50 coupled primal+adjoint solves per optimization iteration. Is this tractable?

5. **Comparison baseline:** What should we compare against?
   - Standard SA FIML (monolithic NN)
   - Standard SST FIML
   - Default GEKO with globally-optimized coefficients (Bayesian optimization)
   - Cherroud et al.'s XMA approach (if reproducible)

---

## Key References

1. Menter, F.R. (2019). "Development of a Generalized k-omega Two-Equation Turbulence Model." NASA TM.
2. Menter, F.R. & Lechner, R. (2021). "Best Practice: GEKO Turbulence Model in ANSYS CFD." ANSYS Technical Paper.
3. Menter, F.R. et al. (2025). "Generalized k-omega (GEKO) Two-Equation Turbulence Model." AIAA Journal.
4. Cherroud, S. et al. (2025). "Space-dependent aggregation of stochastic data-driven turbulence models." J. Comput. Phys.
5. arXiv:2502.11218 (2025). "Bayesian Optimization of the GEKO Turbulence Model for Predicting Flow Separation."
6. Novello et al. (2025). "Quantifying Out-of-Training Uncertainty of Neural-Network based Turbulence Closures." arXiv:2508.16891.
7. Patel et al. (2024). "On the Generalization Capability of a Data-Driven Turbulence Model by FIML." Aerospace 11(7):592.
8. Parish, E.J. & Duraisamy, K. (2016). "A paradigm for data-driven predictive modeling using field inversion and machine learning." J. Comput. Phys.
9. Wu, J. et al. (2025). "Conditioned field inversion and symbolic regression." AIAA Journal.
10. Duraisamy, K. (2021). "Perspectives on ML-augmented RANS and LES models of turbulence." Phys. Rev. Fluids.
11. Obaldía et al. (2025). "Physics-guided Bayesian NNs for zonal corrections and UQ in separated flows." arXiv:2511.14534.

---

## Next Steps

-> `/workflows:plan` for detailed implementation plan of Phase 0 and Phase 1
