#!/usr/bin/env python
"""
Create reference pressure field (pRef) on cylinder surface from experimental Cp data.
Reads CpRef.json and interpolates to mesh face centers on the cylinder patch.
"""

import json
import numpy as np
from scipy.interpolate import interp1d
from mpi4py import MPI
from dafoam import PYDAFOAM

gcomm = MPI.COMM_WORLD

# Flow conditions
Re = 3900.0
D = 1.0
nu = 1.0e-5
U0 = Re * nu / D
rho = 1.0
qInf = 0.5 * rho * U0 ** 2

# Load experimental Cp data
with open("../CpRef.json") as f:
    data = json.load(f)

theta_exp = np.array(data["theta"])  # angle in degrees from stagnation point
Cp_exp = np.array(data["Cp"])

# Create interpolation function
Cp_interp = interp1d(theta_exp, Cp_exp, kind="cubic", fill_value="extrapolate")

# Initialize solver to get mesh info
daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0, 0]},
        "useWallFunction": True,
    },
    "function": {},
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)

# Get face centers on cylinder patch
patchName = "cylinder"
faceCenters = DASolver.getSurfaceCoordinates(patchName)

# Compute theta for each face center (angle from stagnation point)
# Assuming cylinder axis is in z-direction, flow in x-direction
x = faceCenters[:, 0]
y = faceCenters[:, 1]
theta = np.degrees(np.arctan2(y, x))  # angle in degrees

# Interpolate Cp to face centers
Cp_faces = Cp_interp(theta)

# Convert Cp to pressure: p = Cp * qInf (assuming p_inf = 0)
p_faces = Cp_faces * qInf

# Write pRef field
DASolver.writeSurfaceField(patchName, "pRef", p_faces)

if gcomm.rank == 0:
    print(f"Created pRef field on {patchName} patch")
    print(f"Theta range: [{theta.min():.1f}, {theta.max():.1f}] degrees")
    print(f"Cp range: [{Cp_faces.min():.3f}, {Cp_faces.max():.3f}]")
    print(f"p range: [{p_faces.min():.6f}, {p_faces.max():.6f}]")
