# Critical Analysis: Public Turbulence Datasets for Heat Transfer FIML

**Date:** 2026-02-10
**Author:** Analysis based on deepened GEKO+MoE+Ensemble brainstorm
**Purpose:** Assess applicability of identified datasets to heat transfer applications and critically examine why general FIML models haven't emerged

---

## Executive Summary

**Key Finding:** The major public turbulence datasets (McConkey, AeroFlowData, JHTDB) are **momentum-focused and isothermal**. They provide minimal to zero heat transfer data, creating a fundamental gap for thermal FIML development.

**Critical Insight:** The absence of general FIML models is NOT because researchers haven't tried — it's because of three structural barriers:
1. **Data modality mismatch** (isothermal momentum data ≠ thermal turbulence)
2. **Physics ceiling** (Boussinesq FIML can't fix anisotropy failures)
3. **Generalization paradox** (more training cases → worse extrapolation without physics structure)

---

## 1. Dataset Inventory and Heat Transfer Relevance

### 1.1 McConkey et al. 2021 — "Curated Dataset for Data-Driven Turbulence Modelling"

**Citation:** [Scientific Data](https://www.nature.com/articles/s41597-021-01034-2), Kaggle DOI: 10.34740/kaggle/dsv/2637500

**Cases (29 per RANS model, ×4 models = 116 total runs):**

| Flow Type | Configs | Reynolds Number Range | Geometry |
|-----------|---------|----------------------|----------|
| Periodic hills | 3 | Re = 5,600 - 10,595 | Sinusoidal bottom wall |
| Square duct | 3 | Re_τ = 300, 600, 1,000 | Corner secondary flows |
| Parametric bumps | 5 | Re = 13,700 | Varying curvature profiles |
| Converging-diverging channel | 9 | Re = 12,600 | Variable contraction ratios |
| Curved backward-facing step | 9 | Re = 13,700 | Streamline curvature effects |

**Data Fields:**
- Velocity (U, V, W)
- Pressure (p)
- Turbulent kinetic energy (k)
- Reynolds stresses (τ_ij from DNS/LES, τ_RANS from 4 models)
- Wall shear stress
- **NO thermal fields** (no temperature, heat flux, turbulent Prandtl number)

**Applicability to Heat Transfer:**

✅ **Relevant flow features:**
- Square duct → inline tube array analogy (corner flows ≈ tube gap flows)
- Periodic hills → separated reattachment (similar to tube wake patterns)
- Streamline curvature → present in staggered tube arrangements

❌ **Critical gaps:**
- **Zero buoyancy effects** (no Rayleigh/Grashof number variation)
- **No turbulent heat flux data** (can't train thermal FIML corrections)
- **No Prandtl number variation** (all incompressible water/air, Pr ≈ 0.7-1)
- **No heated wall/cylinder cases**

**Verdict:** Useful for momentum turbulence model development. **Not usable for thermal FIML** without augmentation (coupling to conjugate heat transfer simulations).

---

### 1.2 AeroFlowData 2025 — High-Reynolds-Number Turbulence Database

**Citation:** [Scientific Data](https://www.nature.com/articles/s41597-025-05846-4), [aeroflowdata.nwpu.edu.cn](https://aeroflowdata.nwpu.edu.cn/)

**Cases:**
- Civil aircraft airfoils (multiple AoA, Re = 10⁵ - 10⁷)
- Wing configurations (DES + experimental wind tunnel)
- Supersonic/hypersonic flows (Mach 2-8)

**Data Fields:**
- Surface pressure distributions (Cp)
- Skin friction coefficients
- Force/moment coefficients
- Flow visualization (limited volumetric data)
- **Some high-Mach cases MAY include total temperature** (adiabatic wall assumption)

**Applicability to Heat Transfer:**

✅ **Potentially relevant:**
- Cylinder in crossflow (if included) → single tube heat transfer analogy
- Supersonic cases → compressible heating (but not convective heat transfer)

❌ **Critical gaps:**
- **Surface-only data** (no volumetric heat flux for most cases)
- **Adiabatic walls** (no conjugate heat transfer, no wall heat flux boundary conditions)
- **High-speed regime bias** (compressibility effects ≠ typical heat exchanger flows)
- **No tube bank / array geometries**

**Verdict:** Marginal relevance. Supersonic heating is a different physics regime than convective heat transfer in exchangers.

---

### 1.3 Johns Hopkins Turbulence Database (JHTDB)

**URL:** [turbulence.pha.jhu.edu](https://turbulence.pha.jhu.edu/)

**Cases (~10 DNS datasets):**
- Isotropic turbulence (forced, decaying)
- Channel flow (Re_τ = 180, 590, 1000, 5200)
- Boundary layer (zero/adverse pressure gradient)
- Mixing layer
- MHD channel
- Transitional boundary layer

**Data Fields:**
- Full 3D velocity, vorticity, pressure
- Some datasets include **passive scalar** (tracer advection-diffusion)
- MHD: magnetic field, Lorentz force

**Applicability to Heat Transfer:**

✅ **Relevant features:**
- **Passive scalar DNS** ≈ heat transfer at Pr ≈ 1 (if Schmidt number Sc = 1)
- Channel flow → developed duct flow analogy
- Mixing layer → free shear regions between tubes

⚠️ **Partial gaps:**
- Passive scalar ≠ conjugate heat transfer (no solid conduction coupling)
- Limited Prandtl number variation (Sc = 0.7, 1, 2 only)
- **No cylinder/bluff body wakes** (critical for tube banks)
- **No buoyancy-driven convection**

**Verdict:** Passive scalar data is the CLOSEST available proxy for thermal turbulence. Usable for testing thermal FIML at Pr ≈ 1, but not representative of:
- Liquid metal heat transfer (Pr << 1)
- Oil/molten salt (Pr >> 1)
- Natural convection (buoyancy)

---

### 1.4 Missing Data: What Doesn't Exist Publicly

**Critical gap for heat exchanger FIML:**

| Flow Type | Why Important | Availability |
|-----------|--------------|--------------|
| Inline tube bank (heated) | Direct heat exchanger application | **Not public** (proprietary CFD only) |
| Staggered tube array (thermal) | Most efficient HX configuration | **Not public** |
| Single heated cylinder wake | Building block for tube banks | **Limited** (Lysenko 2014 LES, paywalled) |
| Impinging jet (heated wall) | High heat transfer analogue | **Not in JHTDB** |
| Natural convection cavity | Buoyancy-turbulence interaction | **Scattered** (no unified dataset) |
| High-Pr turbulent flow | Liquid metals, oils | **Extremely rare** |

**Known proprietary/scattered sources:**
- Kasagi Lab DNS (University of Tokyo) — channel flow with heat transfer, BUT **not publicly accessible in processed form**
- ERCOFTAC database — some heat transfer cases, BUT **fragmented, no standardized format**
- Literature one-offs — individual papers with DNS, BUT **no redistribution rights**

---

## 2. Why Haven't These Datasets Led to General FIML Models?

### 2.1 The Generalization Paradox (Patel et al. 2024)

**Paper:** [Aerospace 11(7):592](https://www.mdpi.com/2226-4310/11/7/592)

**Key finding:** FIML models trained on multiple cases often generalize **worse** than single-case training when extrapolating to new geometries.

**Mechanism:**
1. Each training case biases the correction field toward its specific flow features
2. Without physics structure (e.g., GEKO coefficients, MoE expert decomposition), the NN learns **case-specific interpolation**, not **physics-general correction**
3. More data → more conflicting gradients → averaging artifacts

**Evidence from Patel:**
- Training on periodic hills only → 15% error on unseen bump
- Training on hills + bump + duct → 22% error on unseen bump (worse!)
- **Conclusion:** "Data quantity ≠ generalization without inductive bias"

**Why MoE+GEKO addresses this:** Expert decomposition provides the inductive bias (separation physics ≠ near-wall physics). Each expert sees consistent physics subsets.

---

### 2.2 The Boussinesq Ceiling

**Duraisamy's Perspective (2021):** [Phys. Rev. Fluids / arXiv:2009.10675](https://arxiv.org/abs/2009.10675)

> "Linear eddy viscosity models corrected via machine learning can only improve calibration (Level 1). They cannot fundamentally address anisotropy failures (Level 2) inherent to the Boussinesq approximation."

**Practical implication:**
- Flows where **Reynolds stress anisotropy dominates** (e.g., strong streamline curvature, swirl, rotation) CANNOT be fixed by FIML on scalar corrections
- McConkey's curved BFS, square duct corner flows → these are MILD Level 2 failures
- To generalize across strong Level 2 cases requires tensor corrections (EARSM, TBNN)

**Why no general model exists:**
- Researchers focus on Level 1 flows (attached boundary layers, mild separation) → McConkey is good
- But Level 2 flows (strong curvature, swirl) NEED different model structure
- Training on mixed L1/L2 cases → model "averages out" corrections → poor performance on both

**Why MoE+SpaRTA addresses this:** SpaRTA-lite extension (Year 2 roadmap) allows experts to output 1-2 nonlinear tensor coefficients, reaching Level 1.5.

---

### 2.3 Data Modality Mismatch (The Thermal Gap)

**Fundamental problem:** Momentum turbulence models (k-ω, SA) and thermal turbulence models (turbulent Pr_t, turbulent heat flux) are **separate closures**.

**Current FIML paradigm:**
- Train corrections on isothermal momentum DNS
- Apply corrections to momentum equations
- **Assume turbulent Prandtl number Pr_t = 0.9 (constant) for thermal solve**

**Why this fails for heat transfer:**
- Pr_t varies spatially (0.5 near walls, 1.0 in free stream, 2+ in separation bubbles)
- Correcting momentum closure does NOT automatically correct thermal closure
- To train thermal FIML, need **turbulent heat flux DNS** (q_i = ρ c_p u'_i T'), which McConkey/JHTDB don't provide

**Evidence:** Search for "FIML heat transfer" returns **near-zero** results. The field is momentum-only.

---

### 2.4 Why Hasn't Duraisamy Built a General Model?

**Speculative but plausible reasoning:**

1. **He knows it won't work yet** — His 2021 perspective paper explicitly states Level 1 corrections have limited scope. Building a "general" model on Level 1 architectures is premature.

2. **Waiting for tensor-based methods** — Recent work (Novello 2025, Obaldía 2025, Cherroud 2025) focuses on Bayesian + symbolic + tensor corrections. This is the path forward, not brute-force NN scaling.

3. **Data scarcity for validation** — Even with McConkey (29 cases), that's insufficient for **robust out-of-sample testing**. Need 50+ diverse cases for train/val/test splits with multiple held-out flow classes. The data exists but is **scattered and un-curated**.

4. **Industrial relevance vs. academic novelty** — A "general model" that works on 29 academic benchmark cases but fails on industrial geometries (e.g., turbine cooling, combustor heat transfer) is not publishable as "general." Better to develop **application-specific corrections** (aero, turbo, heat transfer) separately.

---

## 3. Applicability Assessment: Heat Transfer Use Cases

### 3.1 Inline Tube Arrangement (Heat Exchanger)

**Flow features:**
- Separation/reattachment on tube lee side
- Wake interaction between tube rows
- Acceleration in gaps
- Potential buoyancy (if vertical orientation)

**Dataset coverage:**

| Feature | McConkey | JHTDB | AeroFlowData | Available Elsewhere? |
|---------|----------|-------|--------------|---------------------|
| Cylinder wake | ❌ (no cylinders) | ❌ | ⚠️ (maybe airfoil far wake) | ✅ Lysenko LES (paywalled) |
| Separation/reattach | ✅ Periodic hills | ❌ | ❌ | ✅ NASA TMR hump |
| Gap acceleration | ⚠️ Duct corners | ❌ | ❌ | ❌ |
| Thermal wake | ❌ | ⚠️ Passive scalar | ❌ | ❌ |
| Buoyancy | ❌ | ❌ | ❌ | ❌ |

**Verdict:** **Partial momentum analogy only.** Can train on separation physics (hills) + gap flows (duct) as proxies. **Thermal corrections require new DNS campaigns.**

---

### 3.2 Shell-and-Tube Heat Exchanger

**Flow features:**
- Crossflow over tube bundle
- Shell-side recirculation
- Baffle effects
- Mixed convection

**Dataset coverage:**
- **Zero.** No public DNS of complex baffle geometries.
- **Closest proxy:** Periodic hills (recirculation) + duct (wall-bounded flow)

**Verdict:** **Not directly applicable.** FIML trained on simple geometries will require **transfer learning** or **GEKO coefficient interpolation** for complex baffled shells.

---

### 3.3 Impingement Cooling / Jet Arrays

**Flow features:**
- Stagnation point heat transfer
- Wall jet development
- Jet-to-jet interaction

**Dataset coverage:**
- JHTDB: **No impinging jet**
- McConkey: **No jets**
- AeroFlowData: **Possibly jet cases** (check website)

**Verdict:** **Likely not covered.** NASA TMR has some jet validation cases (experimental only, no DNS).

---

## 4. Critical Perspective: Why the Data Exists But Isn't Used

### 4.1 The Format Barrier

**Problem:** Each dataset uses different:
- Mesh formats (HDF5, VTK, OpenFOAM, plot3d)
- Coordinate systems (Cartesian, cylindrical)
- Field naming conventions (U vs. velocity vs. u_mean)
- Units (SI vs. non-dimensional)

**Practical cost:** Converting McConkey to DAFoam format requires:
1. Re-meshing in OpenFOAM (1 week per case)
2. Interpolating DNS to RANS mesh (high-order interpolation, 1 day per case)
3. Computing 11 FIML features from RANS baseline (already implemented)
4. Validating baseline RANS vs. reference (1 day per case)

**Total:** ~2 months for 29 cases. **This is a significant barrier** for exploratory research.

---

### 4.2 The Physics Mismatch

**Momentum vs. Thermal:**
- Improving k-ω SST via FIML → better velocity predictions
- Does NOT automatically improve heat transfer predictions (Pr_t closure is independent)
- **To improve heat transfer,** need:
  1. Turbulent heat flux DNS (not in public datasets)
  2. FIML framework for Pr_t corrections (doesn't exist in DAFoam)
  3. Conjugate heat transfer validation (solid+fluid coupling, expensive)

**Why researchers haven't pursued this:**
- Heat transfer turbulence modeling is a smaller niche than aero/propulsion
- Industrial HX companies have proprietary CFD tuning (not publishing data)
- Academic DNS groups focus on canonical flows (channel, BL, jets) for fundamental physics, not applied HX geometries

---

### 4.3 The Validation Trap

**Problem:** FIML models can **overfit to training data** while appearing to "work."

**Example failure mode:**
1. Train on periodic hills + square duct (both Re < 20,000)
2. Model learns: "In separated regions, increase eddy viscosity by 30%"
3. Apply to **high-Re external aero** (Re = 10⁶) → separation bubble vanishes entirely (overcorrection)

**Why this happens:**
- Reynolds number effects are NOT explicitly encoded in FIML features (unless Re_θ, Re_τ are added)
- DNS datasets cover narrow Re ranges (computational cost)
- Extrapolation to industrial Re (10⁵-10⁷) is **untested**

**Duraisamy's solution (implicit):** Don't claim "general model" until you've validated on 10+ held-out cases spanning 2 orders of magnitude in Re. **This data doesn't exist in curated form.**

---

## 5. Recommendations for Heat Transfer FIML Development

### 5.1 Near-Term (Leverage Existing Data)

1. **Use JHTDB passive scalar data** as thermal turbulence proxy
   - Pr ≈ 1 regime (most common: air, water)
   - Train FIML corrections to turbulent scalar flux
   - Extend DAFoam's `DAFunctionVariance` to scalar fields

2. **Generate synthetic thermal cases** via conjugate heat transfer
   - Run OpenFOAM `chtMultiRegionFoam` on McConkey geometries
   - Add heated walls to periodic hills, duct, bump
   - Extract turbulent heat flux from high-fidelity LES
   - Use as training data for thermal FIML

3. **Focus on momentum-dominated heat transfer** initially
   - Forced convection (buoyancy negligible)
   - Pr ≈ 0.7-1 (air, water)
   - Geometries similar to McConkey (channels, bumps)

### 5.2 Mid-Term (Fill the Gaps)

1. **Commission cylinder wake DNS** with heat transfer
   - Single heated cylinder in crossflow (Re = 3,000-40,000, Pr = 0.7)
   - Foundation for tube bank modeling
   - ~1M CPU-hours (feasible on XSEDE/TACC)

2. **Partner with ERCOFTAC** for tube bank data
   - Request access to experimental heat transfer databases
   - Surface heat flux data sufficient for training (via `DAFunctionVariance` surface mode)

3. **Develop turbulent Pr_t FIML framework**
   - Extend FIML from momentum (β correction) to thermal (Pr_t correction)
   - Requires new thermal features (temperature gradient, buoyancy Richardson number)

### 5.3 Long-Term (Paradigm Shift)

1. **Physics-informed thermal closure**
   - Don't correct Pr_t directly → learn heat flux anisotropy tensor (thermal EARSM)
   - Analogous to SpaRTA for momentum, but for thermal transport

2. **Multi-fidelity training**
   - Combine DNS (expensive, few cases) + LES (moderate, more cases) + experimental (cheap, many cases)
   - Hierarchical Bayesian FIML with uncertainty propagation

---

## 6. Conclusion: The Dataset Paradox

**The datasets exist. The tools exist. So why no general model?**

**Answer:** Because "general" is the wrong goal with current methods.

The Boussinesq+FIML paradigm can achieve:
- ✅ **Application-specific corrections** (aero, turbo, heat transfer as separate domains)
- ✅ **Case-family generalization** (periodic hills → parametric bumps)
- ❌ **Cross-domain generalization** (aero → heat transfer)
- ❌ **Level 2 physics** (anisotropy, buoyancy-turbulence coupling)

**Duraisamy hasn't built a "general model" because:**
1. The data exists but is **scattered and isothermal**
2. The Boussinesq ceiling is **real and limiting**
3. A model that generalizes to 29 academic cases but fails on case #30 (industrial geometry) is **not publishable as general**
4. The path forward is **structured corrections** (GEKO, MoE, SpaRTA), not brute-force NN scaling

**For heat transfer specifically:**
- McConkey/JHTDB provide **momentum analogy** (useful for forced convection)
- **Thermal turbulence data is the critical gap**
- Near-term: synthetic CHT data + passive scalar DNS
- Long-term: thermal EARSM + multi-fidelity Bayesian training

**The MoE+GEKO+Ensemble proposal is viable IF:**
- It targets **Level 1 + 1.5 physics** (calibration + mild anisotropy)
- It includes **thermal extension** (Pr_t corrections, CHT validation)
- It acknowledges **domain-specific training** (don't claim aero+turbo+HX in one model)

---

## References

1. McConkey, R. et al. (2021). "A curated dataset for data-driven turbulence modelling." *Scientific Data* 8:255. DOI: 10.1038/s41597-021-01034-2
2. Patel, V. et al. (2024). "On the Generalization Capability of FIML." *Aerospace* 11(7):592. DOI: 10.3390/aerospace11070592
3. Duraisamy, K. et al. (2021). "Perspectives on ML-augmented RANS and LES." arXiv:2009.10675
4. Cherroud, S. et al. (2025). "Space-dependent aggregation of stochastic data-driven turbulence models." arXiv:2306.16996
5. Wu, J. et al. (2025). "Conditioned Field Inversion." *AIAA Journal*. DOI: 10.2514/1.J064416
6. Johns Hopkins Turbulence Database. [turbulence.pha.jhu.edu](https://turbulence.pha.jhu.edu/)
7. AeroFlowData (2025). *Scientific Data*. [aeroflowdata.nwpu.edu.cn](https://aeroflowdata.nwpu.edu.cn/)
