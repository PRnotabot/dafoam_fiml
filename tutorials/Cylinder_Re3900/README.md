# Field Inversion and Machine Learning: Cylinder at Re = 3900

## Problem Description

Steady RANS (Spalart-Allmaras) simulation of flow around a circular cylinder at Re = 3900.
The baseline SA model poorly predicts separation, wake recovery, and base pressure.
Field Inversion and Machine Learning (FIML) corrects the SA production term by introducing
a spatially-varying multiplier beta(x):

```
Production = Cb1 * S_tilda * nuTilda * beta(x)
```

where beta = 1.0 recovers the standard SA model.

## Available Experimental Data

Three distinct types of data constrain the inversion, each requiring a different
DAFunctionVariance mode:

| Data Type | Quantity | Location | DAFoam Mode | Reference File |
|-----------|----------|----------|-------------|----------------|
| Surface Cp | p (scalar) | Cylinder wall (28 points, 0-180 deg) | `surface` | `pData` (boundary field) |
| Wake centerline Ux | U (vector, index [0]) | y = 0, x/D = 1..10 behind cylinder | `probePoint` | `UData` (volume field) |
| Wake profiles Ux, Uy | U (vector, indices [0,1]) | Vertical cuts at x/D = 1.06, 1.54, 2.02 | `probePoint` | `UData` (volume field) |

### Data Convention

All reference data must use the same non-dimensionalization as the CFD simulation.
The flow parameters are:

```
Re = 3900, D = 1.0, nu = 1e-5
U0 = Re * nu / D = 0.039 m/s
qInf = 0.5 * rho * U0^2 = 7.605e-4 Pa  (rho = 1.0)
```

**Important**: The `CpRef.json` currently stores values computed with U0 = 3.9 (i.e., nu = 1e-3).
If your CFD uses nu = 1e-5 / U0 = 0.039, you must rescale the dimensional pressure:

```
p_sim = Cp * qInf_sim           (qInf_sim = 0.5 * 0.039^2 = 7.605e-4)
U_sim = (U_exp / U_exp_ref) * U0_sim
```

Cp itself is non-dimensional and does not need rescaling. Only the dimensional p and U
values stored in pData / UData must match your simulation's unit system.

---

## Reference Data Files

DAFunctionVariance reads reference data from OpenFOAM field files named `<varName>Data`
in the `0/` (or time) directory. The naming convention is strict:

| Function varName | Reference File Name | OpenFOAM Class | What Gets Read |
|------------------|---------------------|----------------|----------------|
| `p` | `pData` | `volScalarField` | Boundary values on `cylinder` patch (surface mode) |
| `U` | `UData` | `volVectorField` | Cell values at probe locations (probePoint mode) |
| `betaFINuTilda` | `betaFINuTildaData` | `volScalarField` | All cell values (field mode) |

### pData — Surface Pressure Reference

Used by `CpVar` in **surface mode**. DAFoam reads `pData.boundaryField()[patchI][faceI]`
for every face on the `cylinder` patch.

Since experimental data gives Cp at 28 angular positions, the `createRefField.py` script
interpolates these to all mesh faces on the cylinder patch using cubic interpolation.

The `internalField` value is never read in surface mode — only the `cylinder` boundary matters.

```
// pData structure (surface mode)
FoamFile { class volScalarField; object pData; }
dimensions      [0 2 -2 0 0 0 0];  // kinematic pressure [m^2/s^2]
internalField   uniform 0;          // not used in surface mode
boundaryField
{
    cylinder
    {
        type    fixedValue;
        value   nonuniform List<scalar> N_FACES ( ... );  // interpolated from CpRef
    }
    inlet  { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    // ... other patches: zeroGradient or appropriate type
}
```

The `createRefField.py` script generates this file by:
1. Initializing DAFoam to access the mesh
2. Getting face centers on the cylinder patch
3. Computing theta = atan2(y, x) for each face
4. Interpolating Cp(theta) from experimental data
5. Converting: p_face = Cp_face * qInf
6. Writing the boundary field

### UData — Wake Velocity Reference

Used by `UWakeCenterline` and `UWakeProfiles` in **probePoint mode**. DAFoam reads
`UData[cellI]` only for cells containing the probe coordinates. All other cell values
are ignored.

For probePoint mode, the reference values are read from cells that contain the probe
points (found via `myFindCell`). The rest of the volume field is never queried.

```
// UData structure (probePoint mode)
FoamFile { class volVectorField; object UData; }
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);    // default; only probe cells matter
boundaryField
{
    cylinder { type noSlip; value uniform (0 0 0); }
    inlet    { type fixedValue; value uniform (U0 0 0); }
    outlet   { type zeroGradient; }
    // ... other patches
}
```

**How to populate UData**: A script must:
1. Initialize DAFoam to get the mesh
2. For each probe coordinate, call `myFindCell` to get the cell index
3. Set `UData[cellI] = (Ux_exp, Uy_exp, 0)` for that cell
4. Write the field

Multiple variance functions (centerline + profiles) can share the same `UData` file.
Each function specifies its own `probePointCoords` and reads only those cells.

**Probe point placement**: Points must be strictly **inside** the mesh volume, not on
boundaries. For wake data this is natural (points are in the flow field). Ensure z-coordinate
matches the mesh mid-plane (e.g., z = 0.05 for a 2D mesh spanning z = [0, 0.1]).

### betaFINuTildaData — Regularization Reference

Used by `betaVar` in **field mode**. A uniform value of 1.0 means "penalize deviation
from the unmodified SA model."

```
// betaFINuTildaData structure
FoamFile { class volScalarField; object betaFINuTildaData; }
dimensions      [0 0 0 0 0 0 0];
internalField   uniform 1.0;
boundaryField
{
    cylinder { type zeroGradient; }
    inlet    { type zeroGradient; }
    outlet   { type zeroGradient; }
    // ... all patches: zeroGradient
}
```

**If this file is absent**: DAFunctionVariance sets `isRefData_ = 0` and the function
silently returns 0.0. This means **no regularization** — the optimizer can push beta to
extreme values that overfit sparse data but are non-physical and useless for ML training.
Always include this file.

---

## Objective Function Design

The total objective is a weighted sum of data-mismatch terms plus regularization:

```
J = CpVar + UWakeCenterline + UWakeProfiles + betaVar
```

Each term is computed as:

```
term = scale * (1 / nRefPoints) * sum_i ( var[i] - varData[i] )^2
```

The `scale` parameter in each function's dict controls its relative weight.

### Function Definitions

```python
"function": {
    # --- DATA TERMS ---

    # 1. Surface Cp on cylinder wall
    "CpVar": {
        "type": "variance",
        "source": "patchToFace",
        "patches": ["cylinder"],
        "scale": 1.0,                    # see Scaling section
        "mode": "surface",
        "varName": "p",
        "varType": "scalar",
        "timeDependentRefData": False,
    },

    # 2. Ux on wake centerline (y=0, behind cylinder)
    "UWakeCenterline": {
        "type": "variance",
        "source": "allCells",            # required for probePoint
        "scale": 1.0,                    # see Scaling section
        "mode": "probePoint",
        "probePointCoords": centerline_coords,  # [[x1,0,z], [x2,0,z], ...]
        "varName": "U",
        "varType": "vector",
        "indices": [0],                  # Ux only
        "timeDependentRefData": False,
    },

    # 3. Ux, Uy at vertical wake profiles
    "UWakeProfiles": {
        "type": "variance",
        "source": "allCells",
        "scale": 1.0,                    # see Scaling section
        "mode": "probePoint",
        "probePointCoords": profile_coords,  # all profile points combined
        "varName": "U",
        "varType": "vector",
        "indices": [0, 1],              # Ux and Uy
        "timeDependentRefData": False,
    },

    # --- REGULARIZATION ---

    # 4. Beta smoothness / proximity to SA baseline
    "betaVar": {
        "type": "variance",
        "source": "allCells",
        "scale": 0.01,                  # see Scaling section
        "mode": "field",
        "varName": "betaFINuTilda",
        "varType": "scalar",
        "timeDependentRefData": False,
    },
},
```

### Scaling Strategy

Each variance term computes a mean-squared-error (divided by nRefPoints), so the raw
magnitude depends on the physical units. The `scale` parameter normalizes and weights
each term.

**Step 1: Estimate initial magnitudes** by running `python runScript_FI.py -task run_model`
once with all scales set to 1.0. This prints each function's value before any optimization.

**Step 2: Set scales** to normalize each data term to O(1):

```
scale_CpVar      = 1.0 / CpVar_initial
scale_Ucenterline = 1.0 / UWakeCenterline_initial
scale_Uprofiles   = 1.0 / UWakeProfiles_initial
```

**Step 3: Set regularization weight** relative to the now-normalized data terms.
Typical range: 0.001 to 0.1. Start with 0.01.

```
scale_betaVar = 0.01
```

**Interpretation**:
- `scale_betaVar = 0.1`: Strong regularization. Beta stays close to 1.0. Smoother field,
  better for ML, but may not fully match data.
- `scale_betaVar = 0.001`: Weak regularization. Beta can deviate significantly. Better data
  match but potentially noisy/non-physical beta field.

**Quick estimates** (for U0 = 0.039, qInf = 7.6e-4):
- CpVar: If baseline SA has ~30% Cp error in separation, raw MSE ~ (0.3 * qInf)^2 ~ 5e-8.
  Scale ~ 1/5e-8 = 2e7.
- UWakeCenterline: If ~20% error in wake recovery, raw MSE ~ (0.2 * U0)^2 ~ 6e-5.
  Scale ~ 1/6e-5 = 1.7e4.
- UWakeProfiles: Similar magnitude to centerline.

These are rough — always calibrate with `run_model`.

### OpenMDAO Objective Wiring

```python
class Top(Multipoint):
    def setup(self):
        ...
        self.add_subsystem(
            "obj",
            om.ExecComp("val = CpErr + UcErr + UpErr + betaReg"),
        )

    def configure(self):
        ...
        self.connect("scenario1.aero_post.CpVar", "obj.CpErr")
        self.connect("scenario1.aero_post.UWakeCenterline", "obj.UcErr")
        self.connect("scenario1.aero_post.UWakeProfiles", "obj.UpErr")
        self.connect("scenario1.aero_post.betaVar", "obj.betaReg")
        self.add_objective("obj.val", scaler=1.0)
```

The `scaler=1.0` on the objective is fine because individual term scaling is handled
by the `scale` parameter in each function dict.

---

## Probe Coordinate Format

Probe coordinates are stored in JSON files and loaded in the runScript:

```json
{
    "centerlineCoords": [
        [1.0, 0.0, 0.05],
        [1.5, 0.0, 0.05],
        [2.0, 0.0, 0.05],
        ...
    ],
    "profileCoords": [
        [1.06, -1.0, 0.05],
        [1.06, -0.8, 0.05],
        ...
        [1.06,  1.0, 0.05],
        [1.54, -1.0, 0.05],
        ...
    ]
}
```

Requirements:
- All coordinates must be strictly inside the mesh volume (not on boundaries)
- z-coordinate must match the mesh mid-plane for 2D cases
- Points outside the mesh domain are silently ignored (reduce nRefPoints)

---

## File Structure

```
Cylinder_Re3900/
├── CpRef.json                  # Experimental Cp data (28 points, 0-180 deg)
├── probePointCoords.json       # Surface probe coordinates (for Cp)
├── wakeData.json               # Wake velocity data (user provides)
│                                #   centerlineCoords, profileCoords,
│                                #   Ux_centerline, Ux_profiles, Uy_profiles
├── README.md                   # This file
└── steady_SA/
    └── train/
        ├── runScript_FI.py     # Field inversion optimization script
        ├── createRefField.py   # Generate pData and UData from experimental JSON
        ├── preProcessing.sh    # Setup script (copy 0_orig -> 0, run createRefField)
        ├── c1/                 # Case directory
        │   ├── 0_orig/
        │   │   ├── U
        │   │   ├── p
        │   │   ├── nuTilda
        │   │   ├── nut
        │   │   ├── betaFINuTilda        # uniform 1.0
        │   │   └── betaFINuTildaData    # uniform 1.0 (regularization ref)
        │   │   # pData and UData are generated by createRefField.py into 0/
        │   ├── constant/
        │   │   ├── polyMesh/            # Mesh (user provides)
        │   │   ├── transportProperties
        │   │   └── turbulenceProperties # SA model
        │   └── system/
        │       ├── controlDict
        │       ├── fvSchemes
        │       ├── fvSolution
        │       └── decomposeParDict
        └── tf_training/
            └── trainModel.py           # Post-FI MLP training
```

---

## Workflow

### Step 0: Prepare experimental data files

Ensure `CpRef.json` and `wakeData.json` use consistent units matching the simulation.

### Step 1: Set up the OpenFOAM case

Place mesh in `c1/constant/polyMesh/`, boundary conditions in `c1/0_orig/`,
and solver settings in `c1/system/`. Verify baseline SA converges:

```bash
cd c1 && cp -r 0_orig 0
mpirun -np 4 DASimpleFoam -parallel
```

### Step 2: Generate reference data fields

```bash
cd train
python createRefField.py
```

This creates `c1/0/pData` and `c1/0/UData` from experimental JSON files.

### Step 3: Run field inversion

```bash
mpirun -np 4 python runScript_FI.py -optimizer IPOPT -task run_model
# Check initial function values, adjust scales, then:
mpirun -np 4 python runScript_FI.py -optimizer IPOPT -task run_driver
```

### Step 4: Inspect results

- Check `opt_IPOPT.txt` for convergence
- Visualize the inverted beta field in ParaView
- Compare surface Cp and wake profiles against experimental data

### Step 5: Train ML model

```bash
cd tf_training
python trainModel.py
```

---

## Design Decisions and Rationale

### Why surface mode for Cp (not probePoint)?

Cp is a wall quantity. Surface mode compares `p.boundaryField()[patchI][faceI]` — the actual
wall pressure. ProbePoint would compare `p[cellI]` at the cell center adjacent to the wall,
which differs from the wall value by a boundary layer pressure gradient. The 28 experimental
points at ~6 degree spacing are dense enough for reliable cubic interpolation to all mesh faces.

### Why probePoint mode for wake velocity?

Wake measurements are interior field data at specific spatial locations — exactly what
probePoint mode is designed for. The probe coordinates must be inside the mesh volume
(not on boundaries). This is analogous to OpenFOAM's `probes` function object but with
adjoint-compatible gradient computation.

### Why a single UData file serves both wake functions?

Multiple variance functions with `varName: "U"` all read the same `UData` file. Each function
specifies its own `probePointCoords` and only reads the cells containing those points.
The reference values at probe cells must be set correctly for all probe locations in one
pass by `createRefField.py`.

### Why betaFINuTildaData must exist

Without this file, the `betaVar` function returns 0 (no regularization). With N_cells
free parameters and only ~100 surface/probe constraints, the optimizer produces extreme,
oscillatory beta values that overfit the data and are useless for ML generalization.
The file with uniform 1.0 encodes the prior: "stay close to standard SA unless data
demands otherwise."
