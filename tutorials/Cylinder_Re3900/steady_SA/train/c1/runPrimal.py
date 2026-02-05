#!/usr/bin/env python
"""
Run primal solver for cylinder at Re=3900.
Can run with baseline SA or augmented model.
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-augmented", help="use augmented model", type=bool, default=False)
args = parser.parse_args()

gcomm = MPI.COMM_WORLD

# Flow conditions
Re = 3900.0
D = 1.0
nu = 1.0e-5
U0 = Re * nu / D

daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1e8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0, 0]},
        "useWallFunction": True,
    },
    "regressionModel": {
        "active": args.augmented,
        "model": {
            "writeFeatures": True,
            "modelType": "externalTensorFlow",
            "inputNames": ["PoD", "VoS", "chiSA", "PSoSS"],
            "outputName": "betaFINuTilda",
            "hiddenLayerNeurons": [20, 20],
            "inputShift": [0.0, 0.0, 0.0, 0.0],
            "inputScale": [1.0, 1.0, 1.0, 1.0],
            "outputShift": 1.0,
            "outputScale": 1.0,
            "activationFunction": "tanh",
            "printInputInfo": True,
            "defaultOutputValue": 1.0,
            "outputUpperBound": 1e1,
            "outputLowerBound": -1e1,
            "writeFeatures": True,
        }
    },
    "tensorflow": {
        "active": args.augmented,
        "model": {
            "predictBatchSize": 10000,
            "nInputs": 4,
        },
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["cylinder"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["cylinder"],
            "directionMode": "fixedDirection",
            "direction": [0.0, 1.0, 0.0],
            "scale": 1.0,
        },
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
