---
date: 2026-02-10
topic: critical-analysis-geko-moe-ensemble-deepened
parent: 2026-02-09-critical-analysis-geko-moe-ensemble.md
deepened: true
---

# Deepened Critical Analysis: GEKO+MoE+Ensemble for Generalizable FIML

## Enhancement Summary

**Deepened on:** 2026-02-10
**Sections enhanced:** 6 (all major sections)
**Research sources used:** 15+ papers (2021-2025), 2 open datasets, OOD detection survey

### Key Improvements Over Original Analysis
1. Concrete validation dataset benchmark suite identified (McConkey: 29 cases, AeroFlowData: high-Re)
2. Regularization strategy for spatially-varying coefficients grounded in literature (TV, Sobolev, PLDR)
3. OOD detection upgraded from vague "feature density" to concrete Mahalanobis + class-conditional Gaussian approach
4. EARSM path made concrete via SpaRTA methodology — tensor polynomial library + sparse regression is directly compatible with MoE expert outputs
5. Data scarcity mitigation elevated from "future work" to "Phase 1 requirement"

---

## 1. The Hierarchy of Turbulence Modeling Failures

*[Original content preserved — see parent document]*

### Research Deepening: Quantifying the Levels

The distinction between Level 1 (calibration) and Level 2 (Boussinesq) is not binary — there is a spectrum. Recent literature clarifies this:

**Level 1.5: "Soft" Boussinesq failures addressable by nonlinear EVM extensions**

Some flows that appear to require full Reynolds stress modeling can actually be partially captured by adding 1-2 nonlinear terms to the constitutive relation. The [SpaRTA framework](https://arxiv.org/abs/1905.07510) (Schmelzer et al. 2019) demonstrated that sparse symbolic regression can discover which tensor polynomial terms matter most:

```
tau_ij = -2*nu_t*S_ij                          [Standard Boussinesq, Level 1]
       + c_1 * (S_ik*W_kj - W_ik*S_kj)        [Rotation-strain coupling, Level 1.5]
       + c_2 * (S_ik*S_kj - (1/3)*S_mn*S_nm*δ_ij)  [Normal stress, Level 1.5]
```

SpaRTA builds from a library of candidate tensor basis functions and uses elastic net regularization to select the sparsest model that fits DNS data. **Key finding:** for many industrially-relevant separated flows, only 1-2 additional nonlinear terms are needed beyond Boussinesq.

**Implication for our project:** The "hard ceiling" at Level 2 is softer than initially stated. Our GEKO+MoE architecture can be extended to Level 1.5 by having some experts output nonlinear tensor coefficients (c_1, c_2 above) in addition to scalar GEKO coefficient corrections. This doesn't require full TBNN — just 2 additional scalar outputs per expert.

**Updated failure hierarchy:**

| Level | Failure Type | Our Proposal | With EARSM Extension |
|-------|-------------|-------------|---------------------|
| L1 | Calibration (wrong coefficients) | Directly targets | Directly targets |
| L1.5 | Mild anisotropy (rotation-strain, normal stress) | Not addressed | **Addressable** |
| L2 | Full anisotropy (6 DOF Reynolds stress) | Cannot address | Partially |
| L3 | Scale resolution | Out of scope | Out of scope |

---

## 2. What Could Be Better Than GEKO for the Long Term?

*[Original content preserved — see parent document]*

### Research Deepening: The SpaRTA-MoE Bridge

The most significant finding from the literature review is that **SpaRTA and MoE are naturally complementary**:

**SpaRTA workflow** (4 steps):
1. Build a library of candidate tensor basis functions: {T_ij^(1), T_ij^(2), ..., T_ij^(10)} × {scalar features}
2. For each flow class, use sparse regression (elastic net) to select which terms matter
3. Infer coefficients for selected terms
4. Cross-validate across flow cases

**MoE-SpaRTA fusion** (our novel contribution):
- Each MoE expert learns the SpaRTA coefficients for its flow class
- Expert 1 (APG separation): discovers it needs T^(1) and T^(3) with coefficients c_1(features), c_3(features)
- Expert 2 (free shear): discovers it needs T^(2) and T^(5) with different coefficients
- Expert 3 (near-wall): discovers it only needs linear correction (no nonlinear terms needed)
- The gating network routes cells to the appropriate expert based on local flow features

**Why this works:** SpaRTA's sparsity promotion means each expert will select only 1-3 tensor terms from the full library of 10. Different experts select different terms for different physics. This is exactly what MoE is designed for — different experts having different architectures for different subproblems.

**Practical implication:** Instead of the Year 3-4 timeline for EARSM integration, we can plan for a "SpaRTA-lite" extension as early as Year 2 by adding 2 nonlinear coefficient outputs to each MoE expert.

### Research Deepening: AutoTurb and LLM-Assisted Model Discovery

[AutoTurb (2025)](https://pubs.aip.org/aip/pof/article-abstract/37/1/015211/3331579/) uses LLMs to discover algebraic corrections to linear stress models. While the approach is novel, there's a critical limitation: LLMs propose candidate expressions based on pattern matching in their training data, not physics. The discovered models need extensive validation.

**Relevance to our work:** AutoTurb-style LLM-assisted discovery could be used as a *screening step* before SpaRTA — the LLM proposes candidate library functions, SpaRTA selects and calibrates. This is a speculative extension, not a core dependency.

### Updated Long-Term Trajectory

```
Phase     Base Model         Correction Type              Ceiling
──────    ──────────         ───────────────              ───────
Now       k-omega SST        Scalar beta (FIML)           Level 1
Year 1    GEKO               Spatially-varying GEKO       Level 1 (better)
                             coefficients via MoE
Year 2    GEKO + SpaRTA-lite MoE experts output scalar    Level 1 + Level 1.5
                             + 2 nonlinear coefficients
Year 3-4  EARSM base         Full tensor polynomial       Level 1.5 + partial L2
                             MoE experts
Year 5+   TBNN base          Full tensor basis MoE        Level 2
```

---

## 3. Does Our MoE+Ensemble Architecture Address the Real Challenges?

### Challenge 1: Generalization — Research Deepening

**Conditioned Field Inversion (FI-CND)** — [Wu et al. 2025, AIAA Journal](https://doi.org/10.2514/1.J064416) — provides direct evidence for the MoE concept:
- Binary conditioning: separate corrections for attached vs. separated regions
- Result: significant improvement in generalization to unseen geometries
- **Limitation:** binary (K=2) hard gating is crude; our soft MoE with K=3-5 experts is a principled extension

**Space-dependent aggregation** — [Cherroud et al. 2025](https://arxiv.org/abs/2306.16996) — validates the full MoE idea:
- Gaussian-weighted gating based on 11 flow features
- Expert corrections from Bayesian symbolic identification
- **Key result:** XMA outperforms both baseline and individual experts on unseen flows
- **Key limitation:** experts trained separately, not end-to-end; symbolic (limited expressiveness)

**What our approach adds beyond both:**
1. End-to-end adjoint training (physics-consistent, vs. Cherroud's decoupled approach)
2. Soft K-expert routing (richer than Wu's binary gating)
3. GEKO-aligned expert specialization (interpretable, vs. generic correction fields)
4. Ensemble UQ on top of MoE (neither Wu nor Cherroud provides uncertainty)

**Revised Grade: A-** — Stronger than originally assessed when grounded in the FI-CND and XMA literature. The MoE concept is validated; our contribution is the end-to-end + GEKO alignment + UQ layer.

### Challenge 2: Stability — Research Deepening

The literature on regularization for spatially-varying RANS corrections is more mature than initially appreciated:

**Three established regularization strategies** ([Patel et al. 2023, J. Comput. Phys.](https://www.sciencedirect.com/science/article/pii/S0021999123004990)):

1. **Total Variation (TV):** Penalizes `||∇α||_1` — promotes piecewise-constant fields. Good for sharp transitions (e.g., separation point) but can produce staircasing artifacts.

2. **Sobolev (H1) Gradient:** Replaces `∇_α J` with the solution of `-∇²g + g = ∇_α J`. Produces inherently smooth corrections. **This is the recommended default** — directly compatible with DAFoam's adjoint pipeline (add one Poisson solve per optimization step).

3. **Piecewise Linear Dimensionality Reduction (PLDR):** Represent the correction field as `α(x) = Σ_r β_r * φ_r(x)` where φ_r are basis functions (e.g., radial basis functions, POD modes). Reduces degrees of freedom from N_cells to R << N_cells. **Key advantage:** eliminates the near-wall spikiness that TV and Sobolev struggle with.

**Practical recommendation for our architecture:**
- Apply Sobolev smoothing to each expert's output field (cheap, one Poisson solve per expert per iteration)
- Apply PLDR for the gating network outputs specifically (gating weights should be smooth to prevent artificial gradients between expert domains)
- Keep per-ensemble-member convergence monitoring (reject diverged members)

**Additional finding:** [Foures et al. 2014 / Patel 2023] showed that spatial smoothing also acts as implicit regularization against overfitting in field inversion — it prevents the correction from fitting noise in the reference data.

**Revised Grade: B** — Upgraded from C+. The Sobolev + PLDR approach is well-established, cheap to implement, and directly addresses the spatial discontinuity concern. Still no formal guarantee (no Lyapunov), but a much stronger practical strategy.

### Challenge 3: Uncertainty Quantification — Research Deepening

**Concrete OOD detection strategy** (grounded in [ACM CSUR 2025 survey](https://arxiv.org/abs/2409.11884)):

The most practical approach for our 11-dimensional feature space:

1. **Mahalanobis distance in feature space:** Compute the class-conditional mean μ_k and covariance Σ_k of the 11 FIML features for each training case class k. At prediction time, for each cell:
   ```
   d_M(x) = min_k sqrt((f(x) - μ_k)^T * Σ_k^{-1} * (f(x) - μ_k))
   ```
   If `d_M(x) > threshold`, flag as OOD.

   **Cost:** Negligible — one matrix-vector product per cell per evaluation. Pre-compute Σ_k^{-1} offline.

2. **Class-conditional Gaussian (GEM-style):** Model the feature distribution of each training case as a multivariate Gaussian. This provides a *probability* of each cell belonging to the training distribution, not just a distance.

3. **MoE gating entropy as secondary OOD signal:** When the gating network assigns roughly equal weight to all experts (high entropy), the network is uncertain about which physics applies. This is a natural, free OOD indicator:
   ```
   H(g) = -Σ_k g_k * log(g_k)
   High H → uncertain routing → possible OOD
   ```

**Combined OOD detection pipeline:**
```
OOD_score(x) = w_1 * d_M(x)           [feature space distance]
             + w_2 * H(gating(x))      [gating entropy]
             + w_3 * σ_ensemble(x)      [ensemble spread]

If OOD_score > threshold → fall back to uncorrected GEKO
```

This three-signal approach is significantly more robust than ensemble-only UQ. Each signal catches different failure modes:
- Mahalanobis catches extrapolation in feature space
- Gating entropy catches ambiguous physics
- Ensemble spread catches model disagreement

**Revised Grade: B+** — Upgraded from B-. The three-signal OOD detection pipeline is concrete, cheap, and well-grounded in literature.

### Challenge 5: Training Data Scarcity — Research Deepening

**Available benchmark datasets** (concrete inventory):

| Dataset | Cases | Flow Types | Data | Access |
|---------|-------|-----------|------|--------|
| [McConkey et al. 2021](https://www.nature.com/articles/s41597-021-01034-2) | 29 per model (×4 models) | Periodic hills, square duct, parametric bumps, converging-diverging channel, curved BFS | 895,640 points with RANS features + DNS/LES labels | Kaggle, free |
| [AeroFlowData 2025](https://www.nature.com/articles/s41597-025-05846-4) | Multiple configs | Civil aircraft (airfoils, wings), supersonic/hypersonic | DES, DNS, wind tunnel experimental | [aeroflowdata.nwpu.edu.cn](https://aeroflowdata.nwpu.edu.cn/), free registration |
| [Johns Hopkins TDB](https://turbulence.pha.jhu.edu/) | 10+ | Channel, isotropic, MHD, boundary layer | Full DNS fields | API access, free |
| NASA TMR | ~20 | Flat plate, bump, hump, airfoil, jet | RANS validation + some experimental | [turbmodels.larc.nasa.gov](https://turbmodels.larc.nasa.gov/), free |
| DAFoam tutorials | 2 | Ramp (steady + unsteady) | Pre-configured FIML cases | Already available |

**Assessment:** McConkey alone provides 29 cases with matched RANS+DNS — sufficient for K=3 experts with held-out validation. Combined with AeroFlowData's high-Re cases and JHTDB, we have **50+ cases** spanning separation, ducts, bumps, channels, jets, and airfoils.

**The real gap is format, not quantity:** Each dataset uses different formats, meshes, and coordinate systems. A significant practical effort is needed to:
1. Re-mesh each case in OpenFOAM format
2. Interpolate DNS/LES data onto RANS meshes
3. Compute the 11 FIML features from RANS solutions
4. Validate baseline RANS vs. reference data

**Recommendation:** Budget 1-2 months of Phase 0 specifically for dataset preparation. McConkey is the easiest starting point — they already provide OpenFOAM input files.

**Revised Grade: B** — Upgraded from C. Data exists in sufficient quantity and diversity. The bottleneck is reformatting and meshing, not availability.

---

## 4. Alignment Assessment Summary (Revised)

```
                                    Original    Revised     What Changed
Turbulence Challenge                Grade       Grade
────────────────────                ─────       ─────       ────────────
Calibration optimization            ★★★★★       ★★★★★       —
Generalization across flows         ★★★★☆       ★★★★½       FI-CND + XMA validate MoE
Physical interpretability           ★★★★★       ★★★★★       —
Uncertainty quantification          ★★★☆☆       ★★★★☆       3-signal OOD pipeline
Solver stability                    ★★☆☆☆       ★★★☆☆       Sobolev + PLDR regularization
Reynolds stress anisotropy          ★☆☆☆☆       ★★½☆☆       SpaRTA-lite extension (L1.5)
Training data efficiency            ★★☆☆☆       ★★★☆☆       McConkey + AeroFlowData
Industrial deployability            ★★★★☆       ★★★★☆       —
```

**Net change:** The proposal is stronger than the original analysis suggested, particularly in UQ (concrete OOD pipeline), stability (Sobolev+PLDR), and the Boussinesq ceiling (SpaRTA-lite extension reaches Level 1.5). The two remaining weak points — full anisotropy and formal stability guarantees — are inherent to the linear EVM paradigm and will only be resolved in Year 3+ with the EARSM/TBNN transition.

---

## 5. Refined Project Vision (Deepened)

### The Vision Statement (Updated)

"We propose a practical framework for **automated, interpretable, uncertainty-aware RANS calibration and mild nonlinear correction** that generalizes across flow types by combining GEKO's physically-modular coefficient space with mixture-of-experts compositional learning, ensemble uncertainty quantification with three-signal OOD detection, and a planned extension path to tensor-polynomial corrections via SpaRTA-MoE fusion."

**Change from original:** Added "and mild nonlinear correction" and the SpaRTA extension path. This honestly represents what the architecture can achieve (Level 1 + Level 1.5) without overclaiming Level 2 capability.

### What We CAN Claim If It Works (Updated)

- First end-to-end adjoint-trained MoE turbulence correction framework
- First alignment of MoE expert specialization with GEKO's physics-modular coefficients
- Demonstrated compositional generalization across 5+ flow types from McConkey benchmark
- Three-signal OOD detection (Mahalanobis + gating entropy + ensemble spread)
- Sobolev-regularized spatially-varying coefficients for solver stability
- Interpretable corrections distillable to symbolic expressions per expert
- Open-source implementation in DAFoam with planned SpaRTA-lite extension to nonlinear corrections

### Critical Path Dependencies

```
[Dataset Prep] ────────────────────┐
  McConkey reformatting (Month 1)  │
  AeroFlowData subset (Month 2)   │
                                   ├──► [MoE Training on SST]
[MoE Architecture in DARegression] │      (Month 3-5)
  K sub-networks + gating (Month 2)│
  Sobolev smoothing (Month 2)     ─┘
                                          │
                          ┌───────────────┤
                          ▼               ▼
                   [Ensemble UQ]    [GEKO Implementation]
                   M=5 training      DAkOmegaGEKO.C
                   OOD pipeline      CoDiPack validation
                   (Month 5-7)       (Month 4-7)
                          │               │
                          └───────┬───────┘
                                  ▼
                           [GEKO+MoE+Ensemble]
                           Full pipeline (Month 7-9)
                                  │
                          ┌───────┴───────┐
                          ▼               ▼
                   [Symbolic         [SpaRTA-lite]
                    Distillation]     Nonlinear ext.
                   (Month 9-11)      (Month 10-14)
```

---

## 6. Open Questions (Updated with Research Answers)

### Q1: Is the GEKO paper sufficient to implement from scratch?
**Partially answered.** The 2025 AIAA Journal paper provides the equations. The [ANSYS best practices document](https://www.semanticscholar.org/paper/Best-Practice-:-Generalized-k-Two-Equation-Model-in-Menter-Lechner/3fb36ad80d8a5059c5a3b9034b9dab8cefa30156) provides coefficient ranges and blending function details. Together they should be sufficient. **Risk:** some implementation-specific details (numerical limiters, wall function interaction) may require experimentation.

### Q2: Should we integrate multi-modal data from the start?
**Yes.** McConkey provides volumetric DNS data but only for simple geometries. AeroFlowData provides high-Re experimental data but surface-only. To train on both, surface-Cp training (DAFunctionVariance mode: surface) must work from Phase 1. **Action:** verify DAFoam's surface variance mode works with McConkey-style cases in Month 1.

### Q3: Do we need a standalone OOD detector?
**Yes — now concretely specified.** The three-signal pipeline (Mahalanobis + gating entropy + ensemble spread) is cheap and well-grounded. **Action:** implement Mahalanobis distance computation on the 11 FIML features as a post-processing diagnostic in Month 5, before it's needed for the ensemble pipeline.

### Q4: What validation cases are sufficient?
**Answered.** McConkey benchmark (29 cases) + DAFoam ramp tutorials provide the core. Proposed split:

| Purpose | Cases | Source |
|---------|-------|--------|
| Expert 1 training (APG/sep) | Periodic hills ×3 Re, converging-diverging channel ×3 Re, curved BFS | McConkey |
| Expert 2 training (mixing) | To be sourced from JHTDB or AeroFlowData | TBD |
| Expert 3 training (near-wall) | Square duct ×3 Re, parametric bumps ×5 | McConkey |
| Held-out validation | 2 periodic hills configs, 1 bump config | McConkey |
| OOD test | DAFoam ramp (different flow class entirely) | DAFoam tutorials |

**Gap remaining:** Free shear/mixing training data from McConkey is limited. Need to source jet or mixing layer DNS from JHTDB or AeroFlowData.

### Q5 (New): What is the minimum viable experiment?
The smallest experiment that tests the core thesis (MoE generalizes better than monolithic NN):
- **Base model:** k-omega SST (already in DAFoam)
- **Training:** 6 McConkey cases (3 periodic hills + 3 square ducts)
- **Architecture:** K=2 experts (separation + near-wall) + softmax gating
- **Baseline:** Monolithic NN with same total parameter count
- **Test:** Held-out periodic hill config + unseen bump
- **Success metric:** MoE error on held-out < monolithic NN error, AND gating weights show meaningful spatial structure

This can be done in ~2 months using existing DAFoam infrastructure. If it fails, the entire proposal needs rethinking. If it succeeds, it justifies the full GEKO+Ensemble+SpaRTA roadmap.

---

## References (Expanded)

### Core Architecture
- Menter, F.R. et al. (2025). "Generalized k-omega (GEKO) Two-Equation Turbulence Model." [AIAA Journal](https://arc.aiaa.org/doi/10.2514/1.J065393)
- Cherroud, S. et al. (2025). "Space-dependent aggregation of stochastic data-driven turbulence models." [J. Comput. Phys. / arXiv](https://arxiv.org/abs/2306.16996)
- Wu, J. et al. (2025). "Conditioned Field Inversion and Symbolic Regression." [AIAA Journal](https://doi.org/10.2514/1.J064416)

### EARSM and Nonlinear Extensions
- Schmelzer, M. et al. (2020). "Discovery of Algebraic Reynolds-Stress Models Using Sparse Symbolic Regression (SpaRTA)." [Flow Turb. Combust. / arXiv](https://arxiv.org/abs/1905.07510)
- Cherroud, S. et al. (2022). "Sparse Bayesian Learning of Explicit Algebraic Reynolds-Stress models." [Int. J. Heat Fluid Flow](https://www.sciencedirect.com/science/article/abs/pii/S0142727X22001151)
- NN-augmented EARSM (2024). [Springer](https://link.springer.com/chapter/10.1007/978-3-031-69035-8_2)
- AutoTurb (2025). [Physics of Fluids](https://pubs.aip.org/aip/pof/article-abstract/37/1/015211/3331579/)
- Data-driven EARSM for turbine wakes (2025). [ASME J. Turbomach.](https://asmedigitalcollection.asme.org/turbomachinery/article-abstract/147/6/061007/1207846/)

### Uncertainty and OOD Detection
- Novello et al. (2025). "Quantifying Out-of-Training Uncertainty of NN Turbulence Closures." [arXiv](https://arxiv.org/abs/2508.16891)
- OOD Detection Survey (2025). [ACM Computing Surveys](https://arxiv.org/abs/2409.11884)
- Obaldía et al. (2025). "Physics-guided Bayesian NNs for zonal corrections." [arXiv](https://arxiv.org/abs/2511.14534)

### Regularization and Stability
- Patel et al. (2023). "Dimensionality reduction for regularization of sparse data-driven RANS." [J. Comput. Phys.](https://www.sciencedirect.com/science/article/pii/S0021999123004990)
- Foures et al. / Patel (2022). "Efficient assimilation of sparse data into RANS via discrete adjoint." [J. Comput. Phys.](https://www.sciencedirect.com/science/article/pii/S0021999122007306)
- turbo-RANS Bayesian GEKO optimization (2025). [arXiv](https://arxiv.org/abs/2502.11218)

### Datasets
- McConkey et al. (2021). "A curated dataset for data-driven turbulence modelling." [Scientific Data](https://www.nature.com/articles/s41597-021-01034-2) — Kaggle: 10.34740/kaggle/dsv/2637500
- AeroFlowData (2025). "High-Reynolds-Number Turbulence Database." [Scientific Data](https://www.nature.com/articles/s41597-025-05846-4) — [aeroflowdata.nwpu.edu.cn](https://aeroflowdata.nwpu.edu.cn/)
- Johns Hopkins Turbulence Database. [turbulence.pha.jhu.edu](https://turbulence.pha.jhu.edu/)

### Foundational
- Parish & Duraisamy (2016). "A paradigm for data-driven predictive modeling using FIML." J. Comput. Phys.
- Duraisamy (2021). "Perspectives on ML-augmented RANS and LES." [Phys. Rev. Fluids / arXiv](https://arxiv.org/abs/2009.10675)
- Patel et al. (2024). "On the Generalization Capability of FIML." [Aerospace](https://www.mdpi.com/2226-4310/11/7/592)
