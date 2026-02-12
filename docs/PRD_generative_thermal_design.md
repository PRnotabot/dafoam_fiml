# Product Requirements Document: NuForge

## Generative Thermal Design Engine for the AI Cooling Era

**Date:** 2026-02-12
**Version:** 0.1 (Draft)
**Working product name:** NuForge (Nu = Nusselt number, the dimensionless heat transfer coefficient)

---

## 1. Vision

**One-liner:** NuForge generates physics-optimal, AM-ready cooling geometries in hours --- with turbulence accuracy that incumbents cannot match.

**The thesis:** The thermal management industry is experiencing a once-in-a-generation crisis. AI chip power densities have gone from 30 W/cm^2 to 150+ W/cm^2 in five years, heading toward 350+ W/cm^2. Rack power has exploded from 7 kW to 130+ kW. Conventional cooling designs --- designed manually, simulated with inaccurate turbulence models, iterated over months --- cannot keep up.

NuForge is a generative thermal design engine that combines three capabilities no competitor integrates:

1. **Adjoint-based topology optimization** --- treats every mesh cell as a design variable, discovering geometries humans cannot conceive. 100-1000x more design freedom than parametric tools.
2. **FIML-corrected turbulence models** --- learns corrections from high-fidelity data to close the 20-30% accuracy gap of standard RANS in internal cooling flows (microchannels, impingement jets, pin fins).
3. **Neural operator surrogates** --- provides seconds-speed thermal predictions for design exploration before launching full-fidelity optimization.

The result: cold plates and heat sinks that are 20-50% better than manually designed alternatives, with physics accuracy validated against high-fidelity data, delivered as AM-ready geometry files in hours instead of months.

---

## 2. The Problem

### 2.1 The Thermal Crisis Is Real and Worsening

| GPU Generation | TDP | Rack Power | Heat Flux | Year |
|---------------|-----|------------|-----------|------|
| NVIDIA V100 | 300W | ~10 kW | ~30 W/cm^2 | 2017 |
| NVIDIA A100 | 400W | ~20 kW | ~40 W/cm^2 | 2020 |
| NVIDIA H100 SXM | 700W | ~40 kW | ~85 W/cm^2 | 2022 |
| NVIDIA GB200 NVL72 | 1200W/GPU | 120-132 kW | ~150 W/cm^2 | 2024 |
| NVIDIA Blackwell Ultra / Rubin | TBD | 250-900 kW | 350+ W/cm^2 | 2026-27 |

- Goldman Sachs projects **165% increase in global data center power demand by 2030**.
- NVIDIA Blackwell GB200 NVL72 racks **mandate liquid cooling** --- air cooling is physically impossible above ~50 kW/rack.
- Boyd Corporation has shipped **5 million cold plates** to hyperscalers, but demand is outpacing supply.
- **Overheating issues have been publicly reported** for NVIDIA Blackwell deployments, frustrating customers (SemiWiki).

### 2.2 The Design Process Is Broken

Today's cold plate design workflow at companies like Boyd, Wakefield Thermal, and ATS:

```
Engineer's intuition → Manual CAD → CFD simulation → Iterate manually → Prototype → Test → Ship

Timeline: 3-6+ months
Optimization: None (parametric tweaks at best)
Turbulence accuracy: Standard k-omega SST (20-30% error for impingement/microchannels)
Design freedom: ~5-10 parameters (channel width, depth, spacing, fin count)
```

**No major cold plate manufacturer uses adjoint-based optimization or topology optimization in their production design workflow.**

### 2.3 Existing Tools Don't Solve It

| Tool | Topology Opt? | Adjoint? | Turbulence | Manufacturing Constraints | Price |
|------|--------------|----------|------------|--------------------------|-------|
| ANSYS Icepak/Fluent | No (shape opt only) | Shape only | Standard RANS | No | $25-100K+/yr |
| Siemens STAR-CCM+ | Yes (level-set) | Yes | Standard RANS | Limited | $50-150K+/yr |
| Siemens Flotherm | No | No | Simplified | No | ~$20-50K/yr |
| Cadence Celsius | No | No | Standard | No | Enterprise |
| Diabatix ColdStream | Yes | Yes | Standard RANS | CNC, AM, die cast | EUR 4K/mo |
| ToffeeX | Yes | Yes | Standard RANS | AM, CNC, etching | Custom |
| nTopology | Geometry only | No (ext. CFD) | N/A | AM-focused | ~$10-20K/yr |
| **NuForge** | **Yes** | **Yes** | **FIML-corrected** | **AM-native** | **See below** |

### 2.4 The Gap NuForge Fills

**No existing tool combines topology optimization with accurate turbulence modeling for internal cooling flows.**

Standard RANS turbulence models (k-omega SST, k-epsilon) are 20-30% off for the flow physics that dominate cold plate performance:
- **Impingement jets** (stagnation point heat transfer overpredicted by ~30%)
- **Microchannel flows** (entrance effects, secondary flows poorly captured)
- **Pin fin arrays** (wake mixing and horseshoe vortex interaction)
- **Separated/reattaching flows** (recirculation zones in manifold transitions)

This means that **even "optimized" designs from Diabatix and ToffeeX are optimized with respect to wrong physics**. They find the best design according to an inaccurate model. NuForge finds the best design according to a corrected, data-validated model.

---

## 3. Target Users

### 3.1 Primary: Thermal Design Engineers at Cold Plate / Heat Sink Companies

**Profile:** Mechanical/thermal engineers at companies like Boyd, JetCool, Fabric8Labs, Wakefield Thermal, ATS, Aavid, and Tier-2 thermal solution providers. They design cooling components for OEM customers (NVIDIA, AMD, Google, Microsoft, Meta).

**Pain points:**
- 3-6+ month design cycles when customers need answers in weeks.
- Limited design exploration --- test 5-10 variants manually, not 10,000.
- Uncertain simulation accuracy for novel geometries (microchannels, impingement, TPMS).
- Pressure to meet ever-increasing heat flux targets with each GPU generation.
- No automated path from optimization result to AM-ready geometry.

**Budget:** $50K-200K/year for design tools. Willing to pay for results (better designs faster).

### 3.2 Secondary: Hardware Thermal Teams at Hyperscalers and OEMs

**Profile:** Thermal architects at Google, Microsoft, Meta, Amazon, NVIDIA, AMD, Intel who define cooling requirements and evaluate/select thermal solutions from suppliers.

**Pain points:**
- Need to evaluate 10+ cooling concepts per GPU generation.
- Want to move from "specify requirements → wait for supplier design" to "generate optimal design → send to supplier for manufacturing."
- Need accurate thermal predictions for non-uniform heat maps (chiplets, HBM stacks, interposers).
- Increasing interest in bringing thermal design in-house for competitive advantage.

**Budget:** $200K-1M+/year for design tools. Willing to pay premium for accuracy and speed.

### 3.3 Tertiary: AM Thermal Component Manufacturers

**Profile:** Companies like Fabric8Labs (ECAM copper), Alloy Enterprises, ADDMAN Group that manufacture AM cooling components but don't have advanced design optimization tools.

**Pain points:**
- Customers bring unoptimized designs that don't exploit AM's geometric freedom.
- Want to offer "design + manufacture" bundles (higher margin, stickier customers).
- Need designs that are optimized for their specific AM process (min feature size, overhang angles, material properties).

**Budget:** $30K-100K/year for software. Would prefer revenue-share or per-design pricing.

---

## 4. Product Description

### 4.1 What NuForge Is

NuForge is a **generative thermal design engine** that takes a thermal problem specification (heat sources, flow boundary conditions, design space, manufacturing method) and automatically produces an optimized cooling geometry with validated thermal performance predictions.

### 4.2 Core Workflow

```
                    ┌─────────────────────────────────────────────┐
                    │             NuForge Workflow                 │
                    └─────────────────────────────────────────────┘

     Step 1                 Step 2                 Step 3                 Step 4
  ┌──────────┐         ┌──────────────┐       ┌───────────────┐     ┌───────────────┐
  │  DEFINE  │         │   EXPLORE    │       │   OPTIMIZE    │     │   DELIVER     │
  │          │         │              │       │               │     │               │
  │ - Heat   │  ──►    │ Neural Op    │  ──►  │ Adjoint-based │ ──► │ - AM-ready    │
  │   sources│         │ surrogate    │       │ topology opt  │     │   STL/STEP    │
  │ - Flow   │         │ (~seconds)   │       │ with FIML     │     │ - Performance │
  │   BCs    │         │              │       │ turbulence    │     │   report      │
  │ - Design │         │ Explore 1000s│       │ (~hours)      │     │ - Validation  │
  │   space  │         │ of concepts  │       │               │     │   against LES │
  │ - Mfg    │         │ interactively│       │ Generate      │     │ - MFG specs   │
  │   method │         │              │       │ optimal geom  │     │               │
  └──────────┘         └──────────────┘       └───────────────┘     └───────────────┘
```

**Step 1 --- Define (5 minutes):**
User uploads or defines:
- Chip/component heat map (uniform, non-uniform, transient)
- Coolant type and flow conditions (inlet temperature, pressure, flow rate)
- Design envelope (bounding box, keep-out zones, inlet/outlet locations)
- Manufacturing method (LPBF aluminum, ECAM copper, CNC milled, brazed)
- Objectives and constraints (min thermal resistance, max pressure drop, max temperature uniformity, volume)

**Step 2 --- Explore (seconds to minutes):**
A pre-trained neural operator (DeepONet or FNO architecture) provides instant thermal field predictions for parametric variations (flow rate sweeps, inlet configuration, design space sizing). This gives the engineer rapid intuition before committing to a full optimization run. The surrogate is trained on NuForge's database of previous optimizations.

**Step 3 --- Optimize (2-8 hours on cloud GPU):**
The adjoint-based topology optimizer runs on cloud infrastructure:
- **Forward solve:** OpenFOAM-based conjugate heat transfer (DAHeatTransferFoam / DATopoChtFoam)
- **Turbulence model:** FIML-corrected RANS (k-omega SST or SA with learned beta corrections for internal cooling flows)
- **Adjoint solve:** Discrete adjoint via CoDiPack (reverse-mode AD) computes gradients of objective w.r.t. every cell's material state
- **Topology update:** Density-based or level-set update with manufacturing constraints (min feature size, overhang angles, connectivity)
- **Iteration:** 50-200 optimization iterations until convergence

**Step 4 --- Deliver (automatic):**
- Optimized geometry exported as **STL** (AM-ready) and **STEP** (CAD-compatible)
- Performance report: thermal resistance, pressure drop, temperature uniformity, Nusselt number distribution
- Validation: automated comparison against high-fidelity LES (if reference data available) or FIML uncertainty bounds
- Manufacturing specification: recommended print orientation, support strategy, post-processing requirements

### 4.3 What Makes NuForge Different

#### Differentiator 1: FIML-Corrected Turbulence (The Physics Moat)

This is NuForge's deepest moat --- a capability that requires the founder's specific PhD expertise and cannot be easily replicated.

**The problem:** All competitors (Diabatix, ToffeeX, Siemens STAR-CCM+) optimize using standard RANS turbulence models. For internal cooling flows, these models produce 20-30% errors in heat transfer predictions:

| Flow Type | k-omega SST Error | FIML-Corrected Error | Source |
|-----------|-------------------|----------------------|--------|
| Impingement jets (stagnation) | +25-35% | < 5% | Duraisamy et al. |
| Microchannel entrance | -15-25% | < 8% | Parish & Duraisamy |
| Pin fin array wake mixing | +20-30% | < 10% | Singh et al. |
| Separated flow in manifold | +30-50% | < 12% | DAFoam FIML validation |

NuForge pre-trains FIML correction models for each internal cooling flow class using LES/DNS reference data. These corrections are embedded in the optimization loop via DAFoam's coupled FIML training infrastructure. The optimizer sees accurate physics, so it finds genuinely optimal geometries --- not geometries that are "optimal" according to wrong physics.

**Why competitors can't easily replicate this:**
- Requires discrete adjoint through the turbulence model (DAFoam's core infrastructure)
- Requires a library of high-fidelity reference data for internal cooling flows (LES/DNS databases)
- Requires the FIML pipeline (11 Galilean-invariant features, coupled NN training, symbolic distillation)
- The founder's PhD is literally in this exact intersection: aerospace heat transfer + shape optimization

#### Differentiator 2: Open-Core Trust and Ecosystem

Every competitor (Diabatix, ToffeeX, Siemens, ANSYS) is fully proprietary. Users cannot inspect the solver, validate the physics, or extend the platform.

NuForge's open-core model:
- **Open:** DAFoam/OpenFOAM solver core, basic optimization examples, FIML training pipeline
- **Paid:** Cloud optimization runs, pre-trained FIML models, neural operator surrogates, manufacturing constraint library, enterprise features (SSO, team collaboration, API access), design-as-a-service

This builds trust with engineering teams who need to validate and certify their tools. It also creates a community that contributes training data, validation cases, and feature development.

#### Differentiator 3: Neural Operator Speed Layer

No competitor offers instant thermal predictions. Every design evaluation requires a full CFD solve (minutes to hours). NuForge's neural operator surrogates, trained on the growing database of optimized designs, provide:
- **Seconds-speed** thermal field predictions for initial design exploration
- **Trade-off curves** (thermal resistance vs. pressure drop Pareto front) generated interactively
- **Sensitivity maps** showing which regions of the design space have the most impact
- **Transfer learning:** Surrogates trained on one chip geometry transfer to similar geometries with minimal fine-tuning

#### Differentiator 4: AM-Native Output

NuForge doesn't just produce "nice CFD pictures" --- it produces files that go straight to the 3D printer:
- **STL export** with controlled mesh resolution for AM slicing
- **STEP export** for CNC/traditional manufacturing
- **Process parameter recommendations** for the specific AM method (laser power, scan speed for LPBF; current density for ECAM)
- **Support structure optimization** (minimize supports while maintaining print quality)
- **Material-aware optimization** (copper vs. aluminum property libraries with temperature-dependent properties --- a gap Diabatix explicitly cannot handle)

---

## 5. Core Features (MVP → V1 → V2)

### 5.1 MVP (Months 1-6): Design-as-a-Service + Core Engine

The MVP is not a self-serve platform. It is the founder running NuForge's optimization engine for paying customers, with a lightweight web interface for problem definition and results delivery.

| Feature | Description | Priority |
|---------|-------------|----------|
| **Conjugate HT topology opt** | Adjoint-based density method on DAFoam/OpenFOAM for steady CHT | Must-have |
| **Standard RANS** | k-omega SST with wall functions, validated for internal flows | Must-have |
| **Web problem definition** | Simple form: upload heat map, set BCs, define design space | Must-have |
| **STL export** | AM-ready geometry from optimized density field (marching cubes) | Must-have |
| **Performance report** | Auto-generated: thermal resistance, pressure drop, T_max, T_uniformity | Must-have |
| **2-3 FIML-corrected models** | Pre-trained corrections for impingement and microchannel flows | Nice-to-have |
| **Cloud compute backend** | GPU-accelerated adjoint solves on AWS/GCP | Must-have |

**Target:** 5-10 paying design-as-a-service customers. Charge $5K-15K per design project.

### 5.2 V1 (Months 7-12): Self-Serve Platform

| Feature | Description |
|---------|-------------|
| **Self-serve web UI** | Full problem definition, optimization monitoring, results download |
| **FIML model library** | Pre-trained corrections for 5+ internal cooling flow classes |
| **Manufacturing constraints** | LPBF (min feature, overhang, connectivity), CNC (tool access), ECAM |
| **Multi-objective** | Pareto front generation (thermal resistance vs. pressure drop) |
| **STEP export** | Smooth B-rep geometry via OpenCASCADE |
| **Neural operator v1** | Trained on MVP design database; provides instant thermal estimates |
| **API access** | RESTful API for programmatic optimization runs |
| **Validation dashboard** | Compare NuForge predictions against uploaded experimental/LES data |

**Target:** 20-50 platform users. Subscription pricing: $2K-5K/month.

### 5.3 V2 (Months 13-24): Intelligence Layer

| Feature | Description |
|---------|-------------|
| **Neural operator design explorer** | Interactive real-time design space exploration with instant thermal predictions |
| **Automated FIML training** | Users upload their own LES/experimental data; NuForge trains custom corrections |
| **Transient optimization** | Time-dependent thermal loads (chip power cycling, fast-charge profiles) |
| **Two-phase cooling** | Boiling/condensation in cold plates (emerging for 350+ W/cm^2 applications) |
| **Design similarity search** | "Find a design like this but for a different chip geometry" |
| **Symbolic model extraction** | Distill learned FIML corrections into publishable algebraic expressions |
| **White-label API** | Thermal solution companies embed NuForge in their own design workflows |

**Target:** 100+ users. Enterprise tier: $50K+/year.

---

## 6. Technical Architecture

### 6.1 Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Web Frontend                          │
│              (React / Next.js / Three.js)                │
│     Problem definition, 3D visualization, monitoring     │
├─────────────────────────────────────────────────────────┤
│                    API Layer                              │
│                (FastAPI / Python)                         │
│       Job management, user auth, billing, webhooks       │
├─────────────────────────────────────────────────────────┤
│                 Intelligence Layer                        │
│          Neural operator inference (PyTorch/JAX)          │
│          FIML model management and inference              │
│          Design database and similarity search            │
├─────────────────────────────────────────────────────────┤
│                 Optimization Engine                       │
│                                                           │
│   ┌─────────────────────────────────────────────────┐    │
│   │  DAFoam / OpenFOAM Core (C++ with CoDiPack AD)  │    │
│   │                                                   │    │
│   │  ┌────────────┐  ┌────────────┐  ┌────────────┐ │    │
│   │  │ Primal     │  │ Adjoint    │  │ Topology   │ │    │
│   │  │ CHT Solve  │  │ Solve      │  │ Update     │ │    │
│   │  │ (OpenFOAM) │  │ (PETSc)    │  │ (MMA/GCMMA)│ │    │
│   │  └────────────┘  └────────────┘  └────────────┘ │    │
│   │                                                   │    │
│   │  ┌────────────┐  ┌────────────┐                  │    │
│   │  │ FIML       │  │ Mesh &     │                  │    │
│   │  │ Regression │  │ Geometry   │                  │    │
│   │  │ (NN/Beta)  │  │ Export     │                  │    │
│   │  └────────────┘  └────────────┘                  │    │
│   └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│                 Cloud Infrastructure                      │
│           GPU compute (AWS p4d/p5, GCP A3)               │
│           Object storage (S3/GCS for designs)             │
│           Job queue (Celery/Redis)                        │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Solver Core

- **Primal solver:** OpenFOAM-based conjugate heat transfer (derived from DAHeatTransferFoam / DATopoChtFoam)
- **Turbulence:** k-omega SST with optional FIML beta correction via DARegression
- **Adjoint:** Discrete adjoint via CoDiPack reverse-mode AD (existing DAFoam ADR infrastructure)
- **Linear solver:** PETSc KSP with GPU-accelerated preconditioners (Priority 1 from ECP strategy doc)
- **Optimizer:** Method of Moving Asymptotes (MMA) or GCMMA via IPOPT/pyOptSparse
- **Topology method:** Density-based with Brinkman penalization (fluid/solid interpolation), SIMP-like continuation
- **Manufacturing filter:** Density filter + Heaviside projection + overhang constraint for AM

### 6.3 FIML Pipeline

Pre-trained correction models for internal cooling flow classes:

| Flow Class | Training Data Source | Features Used | Expected Accuracy Gain |
|-----------|---------------------|---------------|----------------------|
| Impingement jets | LES database (published) | VoS, PoD, ReWall, pGradStream | ~25% error → ~5% |
| Microchannel entrance | DNS (flat plate analog) | chiSA, ReWall, PSoSS | ~20% error → ~8% |
| Pin fin arrays | LES (in-line/staggered) | SCurv, UOrth, KoU2 | ~25% error → ~10% |
| Manifold separation | LES (sudden expansion) | VoS, PoD, pGradStream | ~35% error → ~12% |
| TPMS structures | LES (gyroid/diamond cells) | All 11 features | ~30% error → ~15% |

Each correction model is trained using DAFoam's coupled FIML pipeline: neural network weights are design variables, the variance between RANS prediction and LES reference data is the objective, and the adjoint provides gradients.

### 6.4 Neural Operator Surrogates

Architecture: DeepONet (branch: chip heat map encoding; trunk: spatial coordinates → thermal field)

Training data: Accumulated from customer optimization runs (with permission) + synthetic parametric studies.

Capabilities:
- Input: Heat map (2D image) + flow conditions (scalar) + design envelope (geometry)
- Output: Temperature field, pressure field, velocity field at inference speed (~0.1 seconds)
- Accuracy target: < 10% error vs. full CFD for the design space exploration phase (not used for final validation)
- Transfer: Fine-tune on new chip geometries with 50-100 CFD evaluations

---

## 7. Competitive Positioning

### 7.1 Positioning Statement

> **For** thermal design engineers at cooling solution companies and hyperscaler hardware teams **who** need to design high-performance cold plates and heat sinks for AI accelerators, **NuForge is** a generative thermal design engine **that** automatically produces AM-ready, physics-optimal cooling geometries with FIML-corrected turbulence accuracy. **Unlike** Diabatix ColdStream and ANSYS Icepak, **NuForge** combines topology optimization with data-driven turbulence corrections and neural operator surrogates, delivering designs that are 20-50% better than manually designed alternatives with validated physics accuracy.

### 7.2 Competitive Moat Depth

| Moat | Depth | Time to replicate |
|------|-------|-------------------|
| FIML-corrected turbulence in optimization loop | Very Deep | 2-3 years (requires adjoint + FIML expertise + training data) |
| DAFoam discrete adjoint infrastructure | Deep | 3-5 years (10+ years of development) |
| Open-core community + trust | Medium | 1-2 years (but hard to switch once adopted) |
| Neural operator design database | Growing | Compounds with each customer design (network effects) |
| AM process co-optimization | Medium | 1 year (domain expertise + process models) |

### 7.3 Direct Competitor Comparison

| Dimension | NuForge | Diabatix | ToffeeX | Siemens STAR-CCM+ |
|-----------|---------|----------|---------|--------------------|
| **Turbulence accuracy** | FIML-corrected RANS | Standard RANS | Standard RANS | Standard RANS |
| **Topology optimization** | Adjoint + density | Adjoint + density | Physics-driven generative | Adjoint + level-set |
| **Real-time exploration** | Neural operator surrogates | No | No | No |
| **Temp-dependent properties** | Yes | **No** | Unknown | Yes |
| **Open-source core** | Yes (DAFoam/OpenFOAM) | No | No | No |
| **STEP export** | V1 | No (STL only) | Unknown | Yes |
| **Price (target)** | $2-5K/mo | EUR 4K/mo | Custom | $50-150K/yr |
| **Beachhead** | AI chip cooling | General thermal | Aerospace/auto | General CFD |
| **Funding** | Bootstrapped | $2.4M | $7.3M | Siemens AG |

---

## 8. Go-to-Market Strategy

### 8.1 Phase 1: Design-as-a-Service (Months 1-6)

**Motion:** The founder personally runs NuForge's optimization engine for customers, delivering optimized cold plate designs as a consulting engagement.

**Why service-led:**
- Generates revenue from day 1 (no "build for 12 months then hope people come").
- Each customer engagement validates the product and generates training data for neural operators.
- Direct contact with thermal engineers reveals the real pain points and workflow requirements.
- Builds case studies and reference designs for marketing.

**Target customers:**
1. **Tier-2 thermal solution providers** (not Boyd --- they're too large and slow to adopt). Target companies with 20-200 employees who design custom cold plates and need a competitive edge.
2. **AM thermal manufacturers** (Fabric8Labs, Alloy Enterprises, ADDMAN) who want to offer "optimized design + AM manufacturing" bundles.
3. **Startup hardware companies** building custom AI accelerators who don't have large thermal engineering teams.

**Pricing:** $5K-15K per design project (depending on complexity). This is 10-50x cheaper than the fully-loaded cost of a thermal engineer spending 3 months on manual design.

**Acquisition channels:**
- Direct outreach via LinkedIn to thermal engineering leads at target companies.
- Conference presence: Hot Chips, OCP Summit, Thermal Live, Semi-Therm.
- Technical content: Blog posts comparing FIML-corrected vs. standard RANS for impingement jets, microchannels. Publish validation results.
- Open-source community: DAFoam tutorials for thermal topology optimization attract engineers who then convert to paid services.

### 8.2 Phase 2: Self-Serve Platform (Months 7-12)

**Motion:** Launch ColdStream-competing web platform with self-serve optimization runs.

**Pricing tiers:**

| Tier | Price | Includes |
|------|-------|---------|
| **Starter** | $2,000/mo | 5 optimization runs/month, standard RANS, STL export |
| **Professional** | $4,000/mo | 15 runs/month, FIML-corrected turbulence, STEP + STL, neural operator explorer |
| **Enterprise** | Custom ($50K+/yr) | Unlimited runs, custom FIML training, API access, SSO, dedicated support |
| **Open-source** | Free | DAFoam core + tutorials (self-hosted, no cloud, no FIML models) |

### 8.3 Phase 3: Platform + Ecosystem (Months 13-24)

**Motion:** Expand verticals (EV battery cooling, power electronics, industrial heat exchangers) and build ecosystem.

- **White-label API:** Thermal solution companies embed NuForge in their own design workflows.
- **Marketplace:** Pre-trained FIML models for specific flow classes. Community contributes and monetizes.
- **Training data partnerships:** Partner with national labs and universities who have LES/DNS databases.

---

## 9. Business Model

### 9.1 Revenue Streams

| Stream | MVP | V1 | V2 |
|--------|-----|----|----|
| Design-as-a-service | $5-15K/project | $5-15K/project | Premium only |
| Platform subscription | --- | $2-5K/mo | $2-50K+/mo |
| Cloud compute | --- | Usage-based | Usage-based |
| Enterprise licenses | --- | --- | $50K+/yr |
| White-label API | --- | --- | Revenue share |

### 9.2 Unit Economics (Target at Scale)

- **Cloud compute cost per optimization run:** ~$50-200 (4-8 hours on GPU instance)
- **Platform subscription margin:** 70-80% (software + cloud markup)
- **Design-as-a-service margin:** 60-70% (founder's time is the main cost initially)
- **CAC target:** < $5K per self-serve customer (content-led acquisition)
- **LTV/CAC target:** > 5x (12+ month retention, $24-60K LTV)

### 9.3 Revenue Projections (Conservative)

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| DaaS customers | 10-15 | 20-30 | 30-50 |
| Platform subscribers | 0 | 20-50 | 100-200 |
| DaaS revenue | $75-150K | $150-300K | $200-400K |
| Platform revenue | $0 | $480K-1.5M | $2.4-6M |
| **Total ARR** | **$75-150K** | **$630K-1.8M** | **$2.6-6.4M** |

---

## 10. Key Metrics

### 10.1 Product Metrics

| Metric | Definition | MVP Target | V1 Target |
|--------|-----------|------------|-----------|
| **Design improvement** | Thermal resistance reduction vs. customer's baseline design | > 20% | > 25% |
| **Physics accuracy** | Prediction error vs. experimental/LES data | < 15% (std RANS) | < 8% (FIML) |
| **Time-to-design** | Wall-clock from problem definition to AM-ready geometry | < 48 hours | < 8 hours |
| **Optimization convergence** | Number of adjoint iterations to reach < 1% change | < 200 | < 150 |

### 10.2 Business Metrics

| Metric | Definition | Year 1 Target |
|--------|-----------|---------------|
| **Revenue** | Total ARR | $75-150K |
| **Customers** | Paying DaaS + platform customers | 10-15 |
| **Retention** | % customers who reorder/renew | > 80% |
| **NPS** | Net Promoter Score | > 50 |
| **FIML models trained** | Number of validated correction models | 3-5 |
| **Design database size** | Optimized designs in the neural operator training set | 200-500 |

---

## 11. Risks and Mitigations

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| **Incumbents add topology opt** | High | Low (2-3 years) | FIML is the deeper moat; incumbents won't replicate data-driven turbulence correction quickly |
| **Diabatix/ToffeeX raise large rounds** | Medium | Medium | Open-core community + FIML accuracy advantage. Compete on physics, not funding. |
| **nTopology adds CFD** | Medium | High (acquired Cloudfluid) | nTop is geometry-first, not physics-first. Their CFD will be for validation, not optimization. |
| **FIML accuracy doesn't generalize** | High | Medium | Start with narrow flow classes (impingement, microchannels) where FIML is well-validated. Expand carefully. |
| **Cloud compute costs too high** | Medium | Low | GPU costs declining 30%/year. Optimize solver for GPU (PETSc GPU). Use neural operators to reduce CFD evaluations. |
| **Slow customer adoption** | Medium | Medium | DaaS model proves value before asking customers to adopt platform. Free open-source tier reduces adoption friction. |
| **Single-founder risk** | High | Medium | Hire first engineer within 6 months of first revenue. Document everything. Open-source core ensures continuity. |
| **AM market slower than expected** | Low | Low | Tool also works for CNC/traditional manufacturing. AM is a differentiator, not a requirement. |

---

## 12. Technical Roadmap

### Phase 1: Foundation (Months 1-3)
- [ ] Port DAFoam's DATopoChtFoam to a standalone optimization service
- [ ] Build minimal web UI for problem definition (heat map upload, BCs, design space)
- [ ] Implement STL export from optimized density field (marching cubes + smoothing)
- [ ] Set up cloud compute pipeline (Docker + AWS Batch or GCP Cloud Run)
- [ ] Validate on 3 canonical cases: impingement cold plate, microchannel heat sink, pin fin array

### Phase 2: FIML Integration (Months 3-6)
- [ ] Train FIML correction models for impingement jets using published LES data
- [ ] Train FIML correction models for microchannel flows using DNS data
- [ ] Integrate FIML corrections into the topology optimization loop (coupled training not required --- pre-trained corrections applied as fixed beta fields)
- [ ] Validate FIML-corrected optimization vs. standard RANS optimization on benchmark cases
- [ ] Publish validation results (blog + preprint)
- [ ] Acquire first 5 DaaS customers

### Phase 3: Platform (Months 6-12)
- [ ] Build self-serve web platform (React + FastAPI + job queue)
- [ ] Train neural operator surrogate on design database from Phase 2
- [ ] Add STEP export via OpenCASCADE
- [ ] Implement manufacturing constraint library (LPBF, CNC, ECAM)
- [ ] Add multi-objective optimization (Pareto front: thermal resistance vs. pressure drop)
- [ ] Launch subscription pricing
- [ ] Acquire 20-30 total customers

### Phase 4: Intelligence (Months 12-24)
- [ ] Interactive neural operator design explorer (real-time thermal field visualization)
- [ ] Automated FIML training pipeline (users upload their own reference data)
- [ ] Expand to EV battery thermal management vertical
- [ ] Expand to power electronics cooling vertical
- [ ] Symbolic distillation of FIML corrections for publication and certification
- [ ] White-label API for thermal solution companies

---

## 13. Why This Founder, Why Now

### Founder-Product Fit

| Dimension | Alignment |
|-----------|-----------|
| **PhD in aerospace heat transfer shape optimization** | Literally the core technology (adjoint-based thermal optimization) |
| **Experience with DAFoam/OpenFOAM adjoint framework** | Built the open-source engine this product is based on |
| **FIML pipeline expertise** | Implemented the NN training, feature computation, symbolic regression pipeline |
| **BS Mechanics + MS Energy Science** | Deep physical intuition for thermal-fluid systems across energy applications |
| **AI/ML awareness** | Neural operators, generative design, and differentiable simulation trends |
| **Domain interest alignment** | Heat exchangers, EV cooling, data centers, chip design, H2, CCS --- all thermal-intensive |

### Why Now

1. **AI chip heat flux crisis:** NVIDIA GB200 mandates liquid cooling. Every hyperscaler is scrambling for better cold plates. The market is desperate for solutions *right now*.

2. **AM maturity for thermal:** Fabric8Labs' ECAM copper cold plates are in production. The manufacturing side is ready for optimized designs --- but the design tools haven't caught up.

3. **FIML maturity:** DAFoam's FIML pipeline is production-quality. The coupled training, 11 features, symbolic regression --- all developed over the past 5+ years. This couldn't have been built 3 years ago.

4. **Neural operator readiness:** DeepONet and FNO have moved from research to practical tools (NVIDIA PhysicsNeMo, JAX-CFD). The surrogate layer is now buildable.

5. **Open-source momentum:** DAFoam has a growing community. OpenFOAM is the most widely used open-source CFD code. Building on this ecosystem provides immediate credibility and distribution.

6. **Competitors are small and unfunded:** Diabatix ($2.4M) and ToffeeX ($7.3M) are small. Neither has FIML. Neither has neural operators. The window is open.

---

## Appendix A: Market Size References

- [Goldman Sachs --- AI Power Demand 165% by 2030](https://www.goldmansachs.com/insights/articles/ai-to-drive-165-increase-in-data-center-power-demand-by-2030)
- [Grand View Research --- Data Center Cooling Market $56B by 2030](https://www.grandviewresearch.com/industry-analysis/data-center-cooling-market)
- [Precedence Research --- Liquid Cooling $25.8B by 2035](https://www.precedenceresearch.com/data-center-liquid-cooling-market)
- [Boyd --- 5 Million Cold Plates Delivered](https://www.boydcorp.com/about-boyd/resources/news-and-events/boyd-delivered-5-millionth-liquid-cold-plate-for-ai-cooling.html)
- [Fabric8Labs --- 48% Better Thermal-Hydraulic Performance](https://www.fabric8labs.com/ai-accelerator-cooling-bottleneck/)
- [McKinsey --- AI Power: Expanding Data Center Capacity](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/ai-power-expanding-data-center-capacity-to-meet-growing-demand)

## Appendix B: Academic References for Thermal-Fluid Topology Optimization

- [Alexandersen 2023 --- Topology Optimization of Fluid Flow (MATLAB Tutorial)](https://link.springer.com/article/10.1007/s00158-022-03420-9)
- [Lundgren et al. 2024 --- Large-Scale 3D Multiphysics Topology Optimization](https://www.tandfonline.com/doi/full/10.1080/0305215X.2024.2389281)
- [Kong et al. 2025 --- Adjoint Lattice Boltzmann Multiscale Topology Optimization](https://pubs.aip.org/aip/pof/article/37/8/087208/3359551/)
- [TOFLUX 2025 --- Differentiable Topology Optimization Framework (JAX)](https://arxiv.org/html/2508.17564v1)
- [Narrow-Band Topology Optimization 2025 (Large-Scale Thermal-Fluid)](https://arxiv.org/html/2508.04261)
- [ML Manufacturing Constraints for Rollbonded Cooling Plates 2025](https://link.springer.com/article/10.1007/s00158-025-04192-8)

## Appendix C: Competitor References

- [Diabatix ColdStream](https://www.diabatix.com)
- [Diabatix Pricing](https://www.diabatix.com/pricing)
- [Diabatix Topology Optimization Technology](https://www.diabatix.com/technology/topology-optimization)
- [ToffeeX](https://toffeex.com/)
- [ToffeeX Series A ($7.3M)](https://toffeex.com/news/toffeeam-secures-5-million-in-series-a-funding/)
- [nTopology Series D ($133M)](https://www.ntop.com/resources/blog/ntopology-secures-65m-in-series-d-funding/)
- [ANSYS Icepak 2025 R2](https://www.ansys.com/blog/whats-new-ansys-icepak-2025-r2)
- [Siemens STAR-CCM+ Topology Optimization](https://blogs.sw.siemens.com/simcenter/topology-optimization-cfd-creating-designs-like-nature/)
- [Cadence Celsius Studio](https://www.cadence.com/en_US/home/tools/system-analysis/thermal-solutions/celsius-studio.html)
