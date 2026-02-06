#!/usr/bin/env python
"""
Symbolic Distillation Pipeline for FIML Turbulence Modeling.

Orchestrates three stages to convert a black-box neural network turbulence
correction into an interpretable algebraic equation:

    Stage 1: Coupled NN training (runScript.py) — trains a large NN via
             adjoint-based optimization with physics constraints.
    Stage 2: Coupled compression (runCompression.py) — prunes features and
             compresses the NN while maintaining physics-consistency.
    Stage 3: Symbolic regression (PySR) — discovers an algebraic equation
             that approximates the compressed NN.

Usage:
    # Full pipeline (requires DAFoam environment for Stages 1-2)
    mpirun --oversubscribe -np 4 python runPipeline.py

    # Skip Stage 1 if already trained
    mpirun --oversubscribe -np 4 python runPipeline.py -skip_stage1

    # Skip Stages 1 & 2, only run symbolic regression
    python runPipeline.py -skip_stage1 -skip_stage2

    # Customize compression
    mpirun --oversubscribe -np 4 python runPipeline.py -n_features 2 -hidden 4
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np

from distillation_utils import (
    load_trained_parameters,
    evaluate_nn,
    compute_feature_importance,
    compute_n_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Symbolic Distillation Pipeline for FIML")
    parser.add_argument("-skip_stage1", action="store_true", help="skip Stage 1 (coupled NN training)")
    parser.add_argument("-skip_stage2", action="store_true", help="skip Stage 2 (compression)")
    parser.add_argument("-n_features", type=int, default=3, help="number of features to keep (Stage 2)")
    parser.add_argument("-hidden", type=int, default=5, help="neurons in compressed hidden layer (Stage 2)")
    parser.add_argument("-optimizer", type=str, default="IPOPT", help="optimizer for Stages 1 & 2")
    parser.add_argument("-max_iter_s2", type=int, default=30, help="max iterations for Stage 2")
    parser.add_argument("-np", type=int, default=4, dest="nprocs", help="number of MPI processes")
    parser.add_argument("-sr_iterations", type=int, default=100, help="PySR iterations (Stage 3)")
    parser.add_argument("-sr_maxsize", type=int, default=25, help="PySR max equation complexity")
    parser.add_argument("-output_dir", type=str, default="results", help="output directory for Stage 3")
    return parser.parse_args()


def run_stage1(args):
    """Stage 1: Coupled NN training via runScript.py."""
    print("\n" + "=" * 60)
    print("STAGE 1: Coupled Neural Network Training")
    print("=" * 60)

    teacher_json = os.path.join("..", "designVariable.json")
    if os.path.exists(teacher_json):
        print(f"Found existing {teacher_json}, skipping Stage 1.")
        print("Delete this file to re-run Stage 1.")
        return True

    cmd = [
        "mpirun", "--oversubscribe", "-np", str(args.nprocs),
        "python", os.path.join("..", "runScript.py"),
        "-optimizer", args.optimizer,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print("ERROR: Stage 1 failed!")
        return False

    print("Stage 1 complete.")
    return True


def run_stage2(args):
    """Stage 2: Coupled compression via runCompression.py."""
    print("\n" + "=" * 60)
    print("STAGE 2: Neural Network Compression")
    print("=" * 60)

    compressed_json = "designVariable_compressed.json"
    if os.path.exists(compressed_json):
        print(f"Found existing {compressed_json}, skipping Stage 2.")
        print("Delete this file to re-run Stage 2.")
        return True

    cmd = [
        "mpirun", "--oversubscribe", "-np", str(args.nprocs),
        "python", "runCompression.py",
        "-optimizer", args.optimizer,
        "-n_features", str(args.n_features),
        "-hidden", str(args.hidden),
        "-max_iter", str(args.max_iter_s2),
    ]
    print(f"Running: {' '.join(cmd)}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(cmd, cwd=script_dir)
    if result.returncode != 0:
        print("ERROR: Stage 2 failed!")
        return False

    print("Stage 2 complete.")
    return True


def run_stage3(args):
    """
    Stage 3: Symbolic regression on compressed NN.

    Instead of requiring field inversion data, we evaluate the compressed NN
    on a grid of input values and fit PySR to that mapping. This is valid
    because the compressed NN IS the function we want to approximate.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: Symbolic Regression")
    print("=" * 60)

    # Load compressed parameters
    compressed_json = "designVariable_compressed.json"
    if not os.path.exists(compressed_json):
        print(f"ERROR: {compressed_json} not found. Run Stage 2 first.")
        return False

    compressed_params = load_trained_parameters(compressed_json)

    # Determine which features were kept by re-running feature importance
    teacher_json = os.path.join("..", "designVariable.json")
    all_input_names = ["PoD", "VoS", "chiSA", "PSoSS"]
    teacher_params = load_trained_parameters(teacher_json)
    rankings = compute_feature_importance(teacher_params, 4, [20, 20])
    top_features = np.sort(rankings[: args.n_features])
    selected_names = [all_input_names[i] for i in top_features]
    student_hidden = [args.hidden]

    print(f"Compressed NN: {selected_names} -> {student_hidden} -> 1")
    print(f"Parameters: {len(compressed_params)}")

    # Generate training data by evaluating the compressed NN on a grid
    n_samples = 10000
    np.random.seed(42)

    # Sample inputs from reasonable ranges for each feature
    # These ranges are based on typical SA turbulence model feature values
    feature_ranges = {
        "PoD": (0.0, 5.0),
        "VoS": (0.0, 2.0),
        "chiSA": (-1.0, 10.0),
        "PSoSS": (0.0, 3.0),
    }

    X = np.zeros((n_samples, args.n_features))
    for i, name in enumerate(selected_names):
        lo, hi = feature_ranges[name]
        X[:, i] = np.random.uniform(lo, hi, n_samples)

    y = evaluate_nn(compressed_params, args.n_features, student_hidden, X)

    print(f"\nGenerated {n_samples} samples from compressed NN")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")

    # Run PySR
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("\nERROR: PySR not installed. Install with: pip install pysr")
        print("Saving NN evaluation data to nn_data.npz for manual SR training.")
        os.makedirs(args.output_dir, exist_ok=True)
        np.savez(
            os.path.join(args.output_dir, "nn_data.npz"),
            X=X,
            y=y,
            feature_names=selected_names,
        )
        return False

    os.makedirs(args.output_dir, exist_ok=True)

    # Split into train/val
    n_val = int(n_samples * 0.2)
    indices = np.random.permutation(n_samples)
    X_train, X_val = X[indices[n_val:]], X[indices[:n_val]]
    y_train, y_val = y[indices[n_val:]], y[indices[:n_val]]

    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "tanh", "sqrt", "square", "abs"],
        constraints={"/": (5, 3), "exp": 3, "log": 3, "sqrt": 3},
        nested_constraints={"exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0}},
        niterations=args.sr_iterations,
        populations=30,
        population_size=50,
        maxsize=args.sr_maxsize,
        maxdepth=6,
        variable_names=selected_names,
        loss="loss(prediction, target) = (prediction - target)^2",
        parsimony=0.0032,
        weight_optimize=0.001,
        adaptive_parsimony_scaling=20.0,
        progress=True,
        verbosity=1,
        random_state=0,
        deterministic=True,
        procs=0,
        warm_start=False,
        batching=False,
    )

    print(f"\nRunning PySR with {args.sr_iterations} iterations...")
    model.fit(X_train, y_train)

    # Report results
    best_eq = model.get_best()
    print(f"\nBest equation: {best_eq['sympy_format']}")
    print(f"Complexity: {best_eq['complexity']}")
    print(f"Loss (MSE): {best_eq['loss']:.6e}")

    # Validate
    lambda_func = best_eq["lambda_format"]
    y_pred_val = lambda_func(*[X_val[:, i] for i in range(X_val.shape[1])])
    ss_res = np.sum((y_val - y_pred_val) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    print(f"Validation R²: {r2:.4f}")

    # Save results
    report = {
        "selected_features": selected_names,
        "compressed_architecture": {"n_inputs": args.n_features, "hidden": student_hidden, "n_outputs": 1},
        "best_equation": str(best_eq["sympy_format"]),
        "complexity": int(best_eq["complexity"]),
        "loss": float(best_eq["loss"]),
        "validation_r2": float(r2),
    }
    report_path = os.path.join(args.output_dir, "distillation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save equation as Python function
    eq_path = os.path.join(args.output_dir, "equation.py")
    with open(eq_path, "w") as f:
        f.write('"""Auto-generated beta_FI equation from symbolic distillation."""\n\n')
        f.write("import numpy as np\n\n\n")
        f.write(f"def beta_fi({', '.join(selected_names)}):\n")
        f.write(f'    """Correction factor: beta_FI = {best_eq["sympy_format"]}"""\n')
        f.write(f"    return {best_eq['sympy_format']}\n")

    print(f"\nResults saved to {args.output_dir}/")
    print(f"  Report: {report_path}")
    print(f"  Equation: {eq_path}")

    # Print Pareto front
    equations = model.equations_
    if equations is not None and len(equations) > 0:
        print("\nPareto front (top 5):")
        sorted_eqs = equations.sort_values("complexity")
        for i, (idx, row) in enumerate(sorted_eqs.head(5).iterrows()):
            print(f"  {i + 1}. Complexity {row['complexity']}: {row['sympy_format']} (loss={row['loss']:.6e})")

    return True


def main():
    args = parse_args()

    print("=" * 60)
    print("Symbolic Distillation Pipeline for FIML")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Features to keep: {args.n_features}")
    print(f"  Compressed hidden layer: [{args.hidden}]")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  SR iterations: {args.sr_iterations}")

    # Stage 1
    if not args.skip_stage1:
        if not run_stage1(args):
            sys.exit(1)
    else:
        print("\nSkipping Stage 1 (--skip_stage1)")
        if not os.path.exists(os.path.join("..", "designVariable.json")):
            print("ERROR: Stage 1 output (../designVariable.json) not found!")
            sys.exit(1)

    # Stage 2
    if not args.skip_stage2:
        if not run_stage2(args):
            sys.exit(1)
    else:
        print("\nSkipping Stage 2 (--skip_stage2)")
        if not os.path.exists("designVariable_compressed.json"):
            print("ERROR: Stage 2 output (designVariable_compressed.json) not found!")
            sys.exit(1)

    # Stage 3 (runs on rank 0 only, PySR uses Julia parallelism)
    from mpi4py import MPI

    if MPI.COMM_WORLD.rank == 0:
        success = run_stage3(args)
    else:
        success = True

    success = MPI.COMM_WORLD.bcast(success, root=0)

    if MPI.COMM_WORLD.rank == 0:
        print("\n" + "=" * 60)
        if success:
            print("Pipeline complete!")
        else:
            print("Pipeline finished with warnings (see above).")
        print("=" * 60)


if __name__ == "__main__":
    main()
