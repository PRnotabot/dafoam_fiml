#!/usr/bin/env python
"""
DAFoam field inversion for periodic hill using experimental data.

This script performs field inversion to find the optimal spatially-varying
correction factor betaFINuTilda for the Spalart-Allmaras turbulence model,
using experimental velocity measurements from Rapp & Manhart (2011) at
Re_H = 5600 as reference data.

Key differences from the baseline PeriodicHill tutorial:
    - Uses probePoint variance mode (sparse experimental measurement locations)
      instead of field mode (full-field k-epsilon reference)
    - Reference data comes from PIV/LDA experimental measurements
    - Probe points are defined at experimental measurement stations

Prerequisites:
    1. Run preProcessing.sh to copy initial conditions and generate UData
    2. Ensure OpenFOAM environment is loaded

Usage:
    mpirun --oversubscribe -np 4 python runScript_FI.py -optimizer IPOPT
    mpirun --oversubscribe -np 4 python runScript_FI.py -task run_model
    mpirun --oversubscribe -np 4 python runScript_FI.py -task check_totals

After field inversion, the optimized betaFINuTilda field and turbulence features
are available for symbolic regression (see sr_training/).
"""

# =============================================================================
# Imports
# =============================================================================
import os
import argparse
import json
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from mphys.scenario_aerodynamic import ScenarioAerodynamic


parser = argparse.ArgumentParser()
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
parser.add_argument("-task", help="type of run to do", type=str, default="run_driver")
args = parser.parse_args()

# =============================================================================
# Input Parameters
# =============================================================================
U0 = 0.028  # Bulk velocity at crest [m/s]
nuTilda0 = 1e-4  # Initial nuTilda
nCells = 3500  # Number of mesh cells
dp0 = 6.634074021107811e-06  # Streamwise pressure gradient

# Load probe point coordinates from experimental data
with open("probePointCoords.json") as f:
    probePointCoords = json.load(f)

nProbes = len(probePointCoords["probePointCoords"])
print(f"Loaded {nProbes} experimental probe points")

# =============================================================================
# DAFoam Options
# =============================================================================
daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1e5,
    # Uniform pressure gradient to drive the periodic channel flow
    "fvSource": {
        "gradP": {
            "type": "uniformPressureGradient",
            "value": dp0,
            "direction": [1.0, 0.0, 0.0],
        },
    },
    # Enable feature computation for later use in symbolic regression
    "regressionModel": {
        "active": False,
    },
    "function": {
        # Primary objective: match experimental velocity at probe locations
        # Uses probePoint mode for sparse experimental measurements
        "UProbeVar": {
            "type": "variance",
            "source": "allCells",
            "scale": 1.0,
            "mode": "probePoint",
            "probePointCoords": probePointCoords["probePointCoords"],
            "varName": "U",
            "varType": "vector",
            "indices": [0, 1],  # Match both Ux and Uy components
            "timeDependentRefData": False,
        },
        # Regularization: penalize deviation of beta from 1.0
        # This prevents overfitting and keeps beta physically reasonable
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
    # Design variable: betaFINuTilda field (one value per cell)
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


# =============================================================================
# OpenMDAO Problem Setup
# =============================================================================
class Top(Multipoint):
    def setup(self):
        dafoam_builder = DAFoamBuilder(options=daOptions, mesh_options=None, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        self.mphys_add_scenario("scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder))

        # Composite objective: velocity error + regularization
        # The regularization weight controls the trade-off:
        # - Higher weight -> beta stays closer to 1.0 (less correction)
        # - Lower weight -> beta adjusts more freely to match data
        self.add_subsystem("obj", om.ExecComp("val=error + 0.001 * regulation"))

    def configure(self):
        self.dvs.add_output("beta", val=np.ones(nCells), distributed=False)
        self.connect("beta", "scenario1.beta")

        # Beta bounds: [-5, 10] allows both reduction and amplification
        # of the SA production term
        self.add_design_var("beta", lower=-5.0, upper=10.0, scaler=1.0)

        # Connect objectives
        self.connect("scenario1.aero_post.UProbeVar", "obj.error")
        self.connect("scenario1.aero_post.betaVar", "obj.regulation")
        self.add_objective("obj.val", scaler=1.0)


# =============================================================================
# Optimization Setup
# =============================================================================
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys.html")

prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = args.optimizer

if args.optimizer == "SNOPT":
    prob.driver.opt_settings = {
        "Major feasibility tolerance": 1.0e-6,
        "Major optimality tolerance": 1.0e-6,
        "Minor feasibility tolerance": 1.0e-6,
        "Verify level": -1,
        "Function precision": 1.0e-6,
        "Major iterations limit": 50,
        "Linesearch tolerance": 0.999,
        "Hessian updates": 50,
        "Nonderivative linesearch": None,
        "Print file": "opt_SNOPT_print.txt",
        "Summary file": "opt_SNOPT_summary.txt",
    }
elif args.optimizer == "IPOPT":
    prob.driver.opt_settings = {
        "tol": 1.0e-5,
        "constr_viol_tol": 1.0e-5,
        "max_iter": 50,
        "print_level": 5,
        "output_file": "opt_IPOPT.txt",
        "mu_strategy": "adaptive",
        "limited_memory_max_history": 10,
        "nlp_scaling_method": "none",
        "alpha_for_y": "full",
        "recalc_y": "yes",
    }
elif args.optimizer == "SLSQP":
    prob.driver.opt_settings = {
        "ACC": 1.0e-5,
        "MAXIT": 100,
        "IFILE": "opt_SLSQP.txt",
    }
else:
    print("optimizer arg not valid!")
    exit(1)

prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]
prob.driver.options["print_opt_prob"] = True
prob.driver.hist_file = "OptView.hst"

if args.task == "run_driver":
    prob.run_driver()
elif args.task == "run_model":
    prob.run_model()
elif args.task == "compute_totals":
    prob.run_model()
    totals = prob.compute_totals()
    if MPI.COMM_WORLD.rank == 0:
        print(totals)
elif args.task == "check_totals":
    prob.run_model()
    prob.check_totals(compact_print=False, step=1e-3, form="central", step_calc="abs")
else:
    print("task arg not found!")
    exit(1)
