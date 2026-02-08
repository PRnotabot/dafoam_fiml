#!/usr/bin/env python
"""
Symbolic Regression for FIML Periodic Hill (Experimental Data).

After field inversion produces an optimal betaFINuTilda field, this script
discovers an interpretable algebraic equation beta_FI = f(features) using
PySR symbolic regression.

The workflow:
    1. Run field inversion (runScript_FI.py) to obtain optimized beta field
    2. Run a primal solve with regression features enabled to compute
       turbulence features at each cell
    3. This script reads beta and features, then runs PySR to find an
       algebraic expression

For the periodic hill at Re=5600, the Spalart-Allmaras correction factor
beta_FI multiplies the production term in the SA equation. The SR-discovered
equation provides an interpretable, generalizable turbulence model correction.

Prerequisites:
    - Field inversion completed (runScript_FI.py -task run_driver)
    - Feature fields computed (runScript_features.py or primal with writeFeatures)
    - PySR installed: pip install pysr

Usage:
    python runSR.py
    python runSR.py -niterations 200 -maxsize 30
    python runSR.py -features PoD VoS PSoSS SCurv
"""

import argparse
import os
import sys
import re
import json
import numpy as np

# =============================================================================
# Argument Parsing
# =============================================================================
parser = argparse.ArgumentParser(description="Symbolic regression for FIML beta correction")
parser.add_argument("-niterations", type=int, default=100, help="PySR iterations (default: 100)")
parser.add_argument("-populations", type=int, default=30, help="Populations in genetic algorithm (default: 30)")
parser.add_argument("-maxsize", type=int, default=25, help="Max equation complexity (default: 25)")
parser.add_argument("-maxdepth", type=int, default=6, help="Max expression tree depth (default: 6)")
parser.add_argument("-val_split", type=float, default=0.2, help="Validation fraction (default: 0.2)")
parser.add_argument("-seed", type=int, default=0, help="Random seed (default: 0)")
parser.add_argument("-output_dir", type=str, default="results", help="Output directory (default: results)")
parser.add_argument("-data_dir", type=str, default="..", help="Directory with FI results (default: ..)")
parser.add_argument(
    "-features",
    nargs="+",
    default=["PoD", "VoS", "PSoSS", "SCurv"],
    help="Input features to use (default: PoD VoS PSoSS SCurv)",
)
parser.add_argument("-turbo", action="store_true", help="Use turbo mode")
args = parser.parse_args()


# =============================================================================
# OpenFOAM Field Reader
# =============================================================================
def read_scalar_field(filepath):
    """Read an OpenFOAM volScalarField and return values as numpy array."""
    with open(filepath, "r") as f:
        content = f.read()

    # Check for uniform field
    uniform_match = re.search(r"internalField\s+uniform\s+([\d.eE+-]+)", content)
    if uniform_match:
        val = float(uniform_match.group(1))
        # Need nCells - try to detect from other context
        return None, val

    # Non-uniform field
    match = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(", content)
    if not match:
        raise ValueError(f"Cannot parse scalar field from {filepath}")

    nCells = int(match.group(1))
    data_start = match.end()
    data_end = content.find(")", data_start)
    data_str = content[data_start:data_end]

    values = []
    for line in data_str.strip().split("\n"):
        line = line.strip()
        if line:
            values.append(float(line))

    return np.array(values), None


def read_vector_field(filepath):
    """Read an OpenFOAM volVectorField and return values as numpy array."""
    with open(filepath, "r") as f:
        content = f.read()

    match = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(", content)
    if not match:
        raise ValueError(f"Cannot parse vector field from {filepath}")

    nCells = int(match.group(1))
    data_start = match.end()
    data_end = content.find(")", data_start)
    data_str = content[data_start:data_end]

    values = []
    for line in data_str.strip().split("\n"):
        line = line.strip().strip("()")
        if line:
            coords = line.split()
            if len(coords) == 3:
                values.append([float(c) for c in coords])

    return np.array(values)


def find_latest_time_dir(case_dir):
    """Find the latest time directory in an OpenFOAM case."""
    time_dirs = []
    for d in os.listdir(case_dir):
        try:
            t = float(d)
            if os.path.isdir(os.path.join(case_dir, d)):
                time_dirs.append((t, d))
        except ValueError:
            continue
    if not time_dirs:
        return None
    time_dirs.sort(key=lambda x: x[0])
    return time_dirs[-1][1]


# =============================================================================
# Data Loading
# =============================================================================
def load_fi_data(data_dir, feature_names):
    """
    Load field inversion results: beta field and turbulence features.

    Parameters
    ----------
    data_dir : str
        Path to the field inversion case directory
    feature_names : list of str
        Names of turbulence features to load

    Returns
    -------
    tuple
        (features, beta, nCells) where features is (nCells, nFeatures)
        and beta is (nCells,)
    """
    # Find latest time directory with results
    time_dir = find_latest_time_dir(data_dir)
    if time_dir is None:
        raise FileNotFoundError(f"No time directories found in {data_dir}")

    time_path = os.path.join(data_dir, time_dir)
    print(f"Reading fields from {time_path}")

    # Read beta field
    beta_path = os.path.join(time_path, "betaFINuTilda")
    if not os.path.exists(beta_path):
        # Try reading from the optimization result
        beta_path = os.path.join(data_dir, "0", "betaFINuTilda")
    if not os.path.exists(beta_path):
        raise FileNotFoundError(f"betaFINuTilda not found in {time_path} or {data_dir}/0")

    beta_arr, beta_uniform = read_scalar_field(beta_path)
    if beta_arr is None:
        raise ValueError("betaFINuTilda is uniform - field inversion has not been run")
    print(f"  betaFINuTilda: min={beta_arr.min():.4f}, max={beta_arr.max():.4f}, mean={beta_arr.mean():.4f}")

    nCells = len(beta_arr)

    # Read feature fields
    features = np.zeros((nCells, len(feature_names)))
    for i, fname in enumerate(feature_names):
        fpath = os.path.join(time_path, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Feature field '{fname}' not found at {fpath}. "
                f"Ensure regression features are written (writeFeatures: True in regressionModel)."
            )
        farr, funiform = read_scalar_field(fpath)
        if farr is None:
            farr = np.full(nCells, funiform)
        features[:, i] = farr
        print(f"  {fname}: min={farr.min():.4f}, max={farr.max():.4f}, mean={farr.mean():.4f}")

    return features, beta_arr, nCells


# =============================================================================
# Symbolic Regression
# =============================================================================
def create_pysr_model(args, feature_names):
    """Create and configure PySR regressor."""
    from pysr import PySRRegressor

    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "tanh", "sqrt", "square", "abs"],
        constraints={
            "/": (5, 3),
            "exp": 3,
            "log": 3,
            "sqrt": 3,
        },
        nested_constraints={
            "exp": {"exp": 0, "log": 0},
            "log": {"exp": 0, "log": 0},
        },
        niterations=args.niterations,
        populations=args.populations,
        population_size=50,
        maxsize=args.maxsize,
        maxdepth=args.maxdepth,
        variable_names=feature_names,
        loss="loss(prediction, target) = (prediction - target)^2",
        parsimony=0.0032,
        weight_optimize=0.001,
        adaptive_parsimony_scaling=20.0,
        progress=True,
        verbosity=1,
        random_state=args.seed,
        deterministic=True,
        procs=0,
        turbo=args.turbo,
        warm_start=False,
        batching=False,
    )
    return model


def export_results(model, output_dir, feature_names, X_train, y_train, X_val, y_val):
    """Export discovered equation to multiple formats."""
    os.makedirs(output_dir, exist_ok=True)

    best_eq = model.get_best()
    sympy_expr = best_eq["sympy_format"]

    # SymPy expression
    with open(os.path.join(output_dir, "equation_sympy.txt"), "w") as f:
        f.write(str(sympy_expr))

    # LaTeX
    try:
        from sympy import latex

        latex_expr = latex(sympy_expr)
    except Exception:
        latex_expr = str(sympy_expr)

    with open(os.path.join(output_dir, "equation.tex"), "w") as f:
        f.write(f"% Discovered equation for beta_FI({', '.join(feature_names)})\n")
        f.write(f"% Complexity: {best_eq['complexity']}\n")
        f.write(f"% Loss: {best_eq['loss']:.6e}\n")
        f.write(r"\begin{equation}" + "\n")
        f.write(r"    \beta_{\mathrm{FI}} = " + latex_expr + "\n")
        f.write(r"\end{equation}" + "\n")

    # Python function
    with open(os.path.join(output_dir, "equation.py"), "w") as f:
        f.write('"""Auto-generated beta_FI equation from symbolic regression."""\n\n')
        f.write("import numpy as np\n\n\n")
        f.write(f"def beta_fi({', '.join(feature_names)}):\n")
        f.write("    # Discovered algebraic expression for SA correction factor\n")
        expr_str = str(sympy_expr)
        for func in ["exp", "log", "sqrt", "tanh", "abs", "Abs"]:
            if f"np.{func}" not in expr_str:
                np_func = f"np.abs" if func == "Abs" else f"np.{func}"
                expr_str = expr_str.replace(f"{func}(", f"{np_func}(")
        f.write(f"    return {expr_str}\n")

    # C++ for DAFoam integration
    with open(os.path.join(output_dir, "equation.cpp"), "w") as f:
        f.write("// Auto-generated beta_FI equation for DAFoam integration\n")
        f.write(f"// Complexity: {best_eq['complexity']}\n")
        f.write(f"// Loss: {best_eq['loss']:.6e}\n")
        f.write(f"// Features: {', '.join(feature_names)}\n\n")
        cpp_expr = str(sympy_expr)
        cpp_expr = cpp_expr.replace("abs(", "mag(").replace("Abs(", "mag(")
        f.write(f"scalar betaFI = {cpp_expr};\n")

    # Validation metrics
    lambda_func = best_eq["lambda_format"]

    def compute_metrics(X, y):
        y_pred = np.asarray(lambda_func(*[X[:, i] for i in range(X.shape[1])])).flatten()
        mse = np.mean((y - y_pred) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "r2": float(r2)}

    report = {
        "best_equation": str(sympy_expr),
        "complexity": int(best_eq["complexity"]),
        "loss": float(best_eq["loss"]),
        "features": feature_names,
        "training_metrics": compute_metrics(X_train, y_train),
        "validation_metrics": compute_metrics(X_val, y_val),
    }

    # Pareto front
    equations = model.equations_
    if equations is not None:
        report["pareto_front"] = []
        for _, row in equations.iterrows():
            report["pareto_front"].append(
                {
                    "equation": str(row["sympy_format"]),
                    "complexity": int(row["complexity"]),
                    "loss": float(row["loss"]),
                }
            )

    with open(os.path.join(output_dir, "sr_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report


# =============================================================================
# Main
# =============================================================================
def main():
    np.random.seed(args.seed)
    feature_names = args.features

    print("=" * 60)
    print("Symbolic Regression for Periodic Hill FIML (Experimental)")
    print("=" * 60)
    print(f"Features: {feature_names}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load field inversion data
    print(f"\nLoading field inversion data from {args.data_dir}...")
    try:
        features, beta, nCells = load_fi_data(args.data_dir, feature_names)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo generate the required data:")
        print("  1. Run field inversion:")
        print("     mpirun --oversubscribe -np 4 python runScript_FI.py")
        print("  2. Run primal with feature writing enabled:")
        print("     mpirun --oversubscribe -np 4 python runScript_FI.py -task run_model")
        sys.exit(1)

    print(f"\nLoaded {nCells} cells with {len(feature_names)} features")

    # Filter out cells where beta is very close to 1.0 (no correction needed)
    # This focuses SR on the physically interesting cells
    delta_beta = np.abs(beta - 1.0)
    active_mask = delta_beta > 0.01
    n_active = np.sum(active_mask)
    print(f"Active cells (|beta - 1| > 0.01): {n_active} / {nCells} ({100*n_active/nCells:.1f}%)")

    X = features[active_mask]
    y = beta[active_mask]

    # Train/validation split
    n_samples = len(X)
    n_val = int(n_samples * args.val_split)
    indices = np.random.permutation(n_samples)
    X_train = X[indices[n_val:]]
    y_train = y[indices[n_val:]]
    X_val = X[indices[:n_val]]
    y_val = y[indices[:n_val]]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Feature statistics
    print("\nFeature statistics (active cells):")
    for i, name in enumerate(feature_names):
        print(f"  {name}: [{X[:, i].min():.4f}, {X[:, i].max():.4f}], mean={X[:, i].mean():.4f}")
    print(f"  beta: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}")

    # Save data for potential reuse
    np.savez(
        os.path.join(args.output_dir, "sr_data.npz"),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
    )

    # Run PySR
    print(f"\nConfiguring PySR (iterations={args.niterations}, maxsize={args.maxsize})...")
    try:
        model = create_pysr_model(args, feature_names)
    except ImportError:
        print("\nERROR: PySR not installed. Install with: pip install pysr")
        print("Data saved to results/sr_data.npz for later use.")
        sys.exit(1)

    print("Starting symbolic regression...\n" + "-" * 60)
    model.fit(X_train, y_train)
    print("-" * 60)

    # Results
    best_eq = model.get_best()
    print(f"\nBest equation: {best_eq['sympy_format']}")
    print(f"Complexity: {best_eq['complexity']}")
    print(f"Loss (MSE): {best_eq['loss']:.6e}")

    # Export
    report = export_results(model, args.output_dir, feature_names, X_train, y_train, X_val, y_val)

    print(f"\nTraining  R² = {report['training_metrics']['r2']:.4f}")
    print(f"Validation R² = {report['validation_metrics']['r2']:.4f}")

    # Pareto front summary
    if "pareto_front" in report:
        print(f"\nPareto Front (top 5 by complexity):")
        for eq in sorted(report["pareto_front"], key=lambda e: e["complexity"])[:5]:
            print(f"  C={eq['complexity']:2d}  Loss={eq['loss']:.4e}  {eq['equation']}")

    # Save model
    import pickle

    with open(os.path.join(args.output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(f"\nAll results saved to {os.path.abspath(args.output_dir)}/")
    print("  equation.py    - Python callable")
    print("  equation.tex   - LaTeX format")
    print("  equation.cpp   - C++ for DAFoam")
    print("  sr_report.json - Full report with metrics")


if __name__ == "__main__":
    main()
