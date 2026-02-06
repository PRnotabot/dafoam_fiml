#!/usr/bin/env python
"""
Stage 2: Neural Network Compression via Coupled Knowledge Distillation.

This script compresses a trained FIML neural network (from Stage 1) into a
smaller, more interpretable network while maintaining physics-consistency
through DAFoam's coupled adjoint optimization.

Steps:
    1. Load Stage 1 trained parameters from designVariable.json
    2. Rank input features by first-layer weight saliency
    3. Select top-K features and configure a smaller NN
    4. Initialize student weights from teacher via knowledge distillation
    5. Run coupled DAFoam optimization to refine compressed weights
    6. Save compressed model to designVariable_compressed.json

Usage:
    mpirun --oversubscribe -np 4 python runCompression.py
    mpirun --oversubscribe -np 4 python runCompression.py -n_features 3 -hidden 5
    mpirun --oversubscribe -np 4 python runCompression.py -task run_model
"""

import argparse
import numpy as np
import json
import copy
import os
import sys
from mpi4py import MPI

import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder
from mphys.scenario_aerodynamic import ScenarioAerodynamic

from distillation_utils import (
    compute_feature_importance,
    initialize_student_weights,
    load_trained_parameters,
    save_parameters,
    compute_n_parameters,
)

np.set_printoptions(precision=8, threshold=10000)

# =============================================================================
# Argument parsing
# =============================================================================
parser = argparse.ArgumentParser(description="Stage 2: Compress FIML neural network")
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
parser.add_argument("-task", help="type of run to do", type=str, default="run_driver")
parser.add_argument("-n_features", help="number of input features to keep", type=int, default=3)
parser.add_argument("-hidden", help="neurons in single hidden layer", type=int, default=5)
parser.add_argument("-max_iter", help="max optimizer iterations", type=int, default=30)
parser.add_argument("-teacher_dir", help="path to Stage 1 results", type=str, default="..")
args = parser.parse_args()

comm = MPI.COMM_WORLD

# =============================================================================
# Stage 1 configuration (must match runScript.py)
# =============================================================================
ALL_INPUT_NAMES = ["PoD", "VoS", "chiSA", "PSoSS"]
TEACHER_HIDDEN = [20, 20]
TEACHER_N_INPUTS = len(ALL_INPUT_NAMES)

# =============================================================================
# Load teacher and compute feature importance
# =============================================================================
teacher_json = os.path.join(args.teacher_dir, "designVariable.json")
if not os.path.exists(teacher_json):
    if comm.rank == 0:
        print(f"ERROR: Teacher parameters not found at {teacher_json}")
        print("Run Stage 1 (runScript.py) first, or specify -teacher_dir.")
    sys.exit(1)

teacher_params = load_trained_parameters(teacher_json)

# Rank features
rankings = compute_feature_importance(teacher_params, TEACHER_N_INPUTS, TEACHER_HIDDEN)
top_features = rankings[: args.n_features]
top_features_sorted = np.sort(top_features)  # preserve original ordering for consistency

selected_names = [ALL_INPUT_NAMES[i] for i in top_features_sorted]
dropped_names = [ALL_INPUT_NAMES[i] for i in range(TEACHER_N_INPUTS) if i not in top_features_sorted]

if comm.rank == 0:
    print("=" * 60)
    print("Stage 2: Neural Network Compression")
    print("=" * 60)
    print(f"\nTeacher architecture: {TEACHER_N_INPUTS} inputs -> {TEACHER_HIDDEN} -> 1 output")
    print(f"Teacher parameters: {len(teacher_params)}")
    print(f"\nFeature importance ranking:")
    for rank, idx in enumerate(rankings):
        marker = " <-- KEPT" if idx in top_features_sorted else " (dropped)"
        print(f"  {rank + 1}. {ALL_INPUT_NAMES[idx]} (index {idx}){marker}")
    print(f"\nStudent architecture: {args.n_features} inputs ({selected_names}) -> [{args.hidden}] -> 1 output")
    student_n_params = compute_n_parameters(args.n_features, [args.hidden])
    print(f"Student parameters: {student_n_params}")
    print(f"Compression ratio: {len(teacher_params) / student_n_params:.1f}x")

# =============================================================================
# Initialize student weights from teacher
# =============================================================================
student_hidden = [args.hidden]
student_init = initialize_student_weights(
    teacher_params, TEACHER_N_INPUTS, TEACHER_HIDDEN, student_hidden, top_features_sorted
)

if comm.rank == 0:
    print(f"\nStudent initialized with {len(student_init)} parameters from teacher")

# =============================================================================
# DAFoam configuration (same cases as runScript.py)
# =============================================================================
cases = ["c1", "c2"]
U0 = [10.0, 20.0]
CDData = np.array([0.1683, 0.7101])

probePointCoordsPath = os.path.join(args.teacher_dir, "probePointCoords.json")
with open(probePointCoordsPath) as f:
    probePointCoords = json.load(f)

daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1e3,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0[0], 0, 0]},
        "useWallFunction": True,
    },
    "regressionModel": {
        "active": True,
        "model1": {
            "modelType": "neuralNetwork",
            "inputNames": selected_names,
            "outputName": "betaFINuTilda",
            "hiddenLayerNeurons": student_hidden,
            "inputShift": [0.0] * args.n_features,
            "inputScale": [1.0] * args.n_features,
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
        "pVar": {
            "type": "variance",
            "source": "patchToFace",
            "patches": ["bot"],
            "scale": 1.0,
            "mode": "surface",
            "varName": "p",
            "varType": "scalar",
            "timeDependentRefData": False,
        },
        "UFieldVar": {
            "type": "variance",
            "source": "boxToCell",
            "min": [-10.0, -10.0, -10.0],
            "max": [10.0, 10.0, 10.0],
            "scale": 0.1,
            "mode": "field",
            "varName": "U",
            "varType": "vector",
            "indices": [0, 1],
            "timeDependentRefData": False,
        },
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
        "CDError": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["bot"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
            "calcRefVar": True,
            "ref": [0.0],
        },
        "betaNuTildaVar": {
            "type": "variance",
            "source": "allCells",
            "scale": 0.01,
            "mode": "field",
            "varName": "betaFINuTilda",
            "varType": "scalar",
            "timeDependentRefData": False,
        },
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["bot"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-6,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
    },
    "normalizeStates": {
        "U": U0[0],
        "p": U0[0] * U0[0] / 2.0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "inputInfo": {
        "model1": {
            "type": "regressionPar",
            "components": ["solver", "function"],
        },
    },
}


# =============================================================================
# OpenMDAO problem setup
# =============================================================================
class Top(Multipoint):
    def setup(self):
        builders = {}
        for idxI, case in enumerate(cases):
            options = copy.deepcopy(daOptions)
            options["primalBC"]["U0"]["value"] = [U0[idxI], 0.0, 0.0]
            options["function"]["CDError"]["ref"] = [float(CDData[idxI])]
            # Use the same case run directories as the teacher
            run_dir = os.path.join(args.teacher_dir, case)
            builders[case] = DAFoamBuilder(
                options=options, mesh_options=None, scenario="aerodynamic", run_directory=run_dir
            )
            builders[case].initialize(self.comm)

        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        self.scenarios = {}
        for case in cases:
            self.scenarios[case] = self.mphys_add_scenario(case, ScenarioAerodynamic(aero_builder=builders[case]))

        self.add_subsystem("obj", om.ExecComp("value=c1+c2"))

    def configure(self):
        nParameters1 = self.c1.coupling.solver.DASolver.getNRegressionParameters("model1")
        # Initialize with distilled weights from teacher
        self.dvs.add_output("parameter1", val=student_init[:nParameters1])

        for case in cases:
            self.connect("parameter1", "%s.model1" % case)
            self.connect("%s.aero_post.pVar" % case, "obj.%s" % case)

        self.add_design_var("parameter1", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_objective("obj.value", scaler=1.0)


prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_compressed.html")

# Optimizer setup
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = args.optimizer

if args.optimizer == "SNOPT":
    prob.driver.opt_settings = {
        "Major feasibility tolerance": 1.0e-5,
        "Major optimality tolerance": 1.0e-5,
        "Minor feasibility tolerance": 1.0e-5,
        "Verify level": -1,
        "Function precision": 1.0e-5,
        "Major iterations limit": args.max_iter,
        "Nonderivative linesearch": None,
        "Print file": "opt_SNOPT_print.txt",
        "Summary file": "opt_SNOPT_summary.txt",
    }
elif args.optimizer == "IPOPT":
    prob.driver.opt_settings = {
        "tol": 1.0e-5,
        "constr_viol_tol": 1.0e-5,
        "max_iter": args.max_iter,
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
        "MAXIT": args.max_iter,
        "IFILE": "opt_SLSQP.txt",
    }
else:
    print("optimizer arg not valid!")
    sys.exit(1)

prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]
prob.driver.options["print_opt_prob"] = True
prob.driver.hist_file = "OptView_compressed.hst"

# =============================================================================
# Run
# =============================================================================
if args.task == "run_driver":
    prob.run_driver()
    opt_dv = {
        "parameter1": prob.get_val("parameter1").tolist(),
    }
    save_parameters(np.array(opt_dv["parameter1"]), "designVariable_compressed.json")
    if comm.rank == 0:
        print("\n" + "=" * 60)
        print("Compression complete!")
        print(f"Compressed parameters saved to designVariable_compressed.json")
        print(f"Input features: {selected_names}")
        print(f"Architecture: {args.n_features} -> {student_hidden} -> 1")
        print("=" * 60)
elif args.task == "run_model":
    prob.run_model()
elif args.task == "compute_totals":
    prob.run_model()
    totals = prob.compute_totals()
    if comm.rank == 0:
        print(totals)
elif args.task == "check_totals":
    prob.run_model()
    prob.check_totals(compact_print=False, step=1e-3, form="central", step_calc="abs")
else:
    print("task arg not found!")
    sys.exit(1)
