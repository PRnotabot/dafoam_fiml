#!/usr/bin/env python
"""
Compute turbulence features for the field-inverted periodic hill case.

After field inversion produces the optimal betaFINuTilda field, this script
runs a single primal solve with the regression model enabled in feature-writing
mode. This writes the Galilean-invariant turbulence features (PoD, VoS, etc.)
to the solution directory for use in symbolic regression.

Prerequisites:
    1. Field inversion completed (runScript_FI.py)
    2. The optimized betaFINuTilda field exists in the latest time directory

Usage:
    mpirun --oversubscribe -np 4 python runScript_features.py
"""

import os
import json
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder
from mphys.scenario_aerodynamic import ScenarioAerodynamic


# =============================================================================
# Input Parameters
# =============================================================================
U0 = 0.028
nuTilda0 = 1e-4
nCells = 3500
dp0 = 6.634074021107811e-06

with open("probePointCoords.json") as f:
    probePointCoords = json.load(f)

# Features to compute for symbolic regression
featureNames = ["PoD", "VoS", "PSoSS", "SCurv"]

daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1e5,
    "fvSource": {
        "gradP": {
            "type": "uniformPressureGradient",
            "value": dp0,
            "direction": [1.0, 0.0, 0.0],
        },
    },
    # Enable regression model in feature-writing mode
    # The model is active but uses default output (beta=1.0) since
    # we just need it to compute and write feature fields
    "regressionModel": {
        "active": True,
        "model": {
            "modelType": "neuralNetwork",
            "inputNames": featureNames,
            "outputName": "betaFINuTilda",
            "hiddenLayerNeurons": [5],
            "inputShift": [0.0] * len(featureNames),
            "inputScale": [1.0] * len(featureNames),
            "outputShift": 1.0,
            "outputScale": 1.0,
            "activationFunction": "tanh",
            "printInputInfo": True,
            "defaultOutputValue": 1.0,
            "outputUpperBound": 1e1,
            "outputLowerBound": -1e1,
            "writeFeatures": True,
        },
    },
    "function": {
        "UProbeVar": {
            "type": "variance",
            "source": "allCells",
            "scale": 1.0,
            "mode": "probePoint",
            "probePointCoords": probePointCoords["probePointCoords"],
            "varName": "U",
            "varType": "vector",
            "indices": [0, 1],
            "timeDependentRefData": False,
        },
        "betaVar": {
            "type": "variance",
            "source": "allCells",
            "scale": 1.0,
            "mode": "field",
            "varName": "betaFINuTilda",
            "varType": "scalar",
            "timeDependentRefData": False,
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-8,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
    },
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "inputInfo": {
        "beta": {
            "type": "field",
            "fieldName": "betaFINuTilda",
            "fieldType": "scalar",
            "distributed": False,
            "components": ["solver", "function"],
        },
    },
}


class Top(Multipoint):
    def setup(self):
        dafoam_builder = DAFoamBuilder(options=daOptions, mesh_options=None, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        self.mphys_add_scenario("scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder))
        self.add_subsystem("obj", om.ExecComp("val=error + 0.001 * regulation"))

    def configure(self):
        self.dvs.add_output("beta", val=np.ones(nCells), distributed=False)
        self.connect("beta", "scenario1.beta")
        self.add_design_var("beta", lower=-5.0, upper=10.0, scaler=1.0)
        self.connect("scenario1.aero_post.UProbeVar", "obj.error")
        self.connect("scenario1.aero_post.betaVar", "obj.regulation")
        self.add_objective("obj.val", scaler=1.0)


prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")

# Just run the primal once to compute and write features
print("Running primal solve with feature computation...")
prob.run_model()
print("Done. Feature fields written to the latest time directory.")
print(f"Features computed: {featureNames}")
