#!/usr/bin/env python
"""
PySR Symbolic Regression Training for FIML Turbulence Modeling.

This script discovers interpretable algebraic equations for the SA turbulence
model correction factor beta_FI using symbolic regression on field inversion data.

Usage:
    mpirun -np 1 python trainModel_SR.py [options]

Note: PySR uses Julia internally for parallel processing, so MPI parallelism
      is handled differently than the TensorFlow training. Run with -np 1.

Prerequisites:
    - Run field inversion first: mpirun -np 4 python ../runScript_FI.py -index 0
    - Run field inversion first: mpirun -np 4 python ../runScript_FI.py -index 1
    - This generates c1_data/ and c2_data/ directories with feature fields
"""

import argparse
import numpy as np
import os
import sys
from mpi4py import MPI

# Add parent directory to path for pyofm import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyofm import PYOFM

# Import local utilities
from sr_utils import (
    export_equation_formats,
    validate_equation,
    plot_pareto_front,
    plot_prediction_comparison,
    save_equation_report,
    save_model,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PySR symbolic regression for FIML beta correction factor"
    )
    parser.add_argument(
        "-niterations",
        type=int,
        default=100,
        help="Number of PySR iterations (default: 100)",
    )
    parser.add_argument(
        "-populations",
        type=int,
        default=30,
        help="Number of populations in genetic algorithm (default: 30)",
    )
    parser.add_argument(
        "-maxsize",
        type=int,
        default=25,
        help="Maximum complexity of equations (default: 25)",
    )
    parser.add_argument(
        "-maxdepth",
        type=int,
        default=6,
        help="Maximum depth of expression tree (default: 6)",
    )
    parser.add_argument(
        "-val_split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)",
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "-output_dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "-turbo",
        action="store_true",
        help="Use turbo mode for faster convergence (less exploration)",
    )
    return parser.parse_args()


def load_field_inversion_data(nCells: int = 5000):
    """
    Load field inversion training data from c1_data and c2_data directories.

    Parameters
    ----------
    nCells : int
        Number of cells per case (default: 5000)

    Returns
    -------
    tuple
        (inputs, outputs) arrays where inputs has shape (n_samples, 4) and
        outputs has shape (n_samples,)
    """
    comm = MPI.COMM_WORLD
    ofm = PYOFM(comm=comm)

    cases = ["c1_data", "c2_data"]
    features = ["PoD", "VoS", "chiSA", "PSoSS"]

    inputs = None
    outputs = None

    for case in cases:
        case_path = os.path.join("..", case)
        if not os.path.exists(case_path):
            raise FileNotFoundError(
                f"Case directory {case_path} not found. "
                f"Run field inversion first: mpirun -np 4 python ../runScript_FI.py"
            )

        input_data = []
        # Read input features
        for feature in features:
            field = np.zeros(nCells)
            ofm.readField(feature, "volScalarField", case_path, field)
            input_data.append(field)
        input_data = np.asarray(input_data).transpose()

        # Read output (beta correction factor)
        output_data = np.zeros(nCells)
        ofm.readField("betaFINuTilda", "volScalarField", case_path, output_data)

        # Concatenate data from multiple cases
        if inputs is None:
            inputs = input_data.copy()
            outputs = output_data.copy()
        else:
            inputs = np.concatenate((inputs, input_data), axis=0)
            outputs = np.concatenate((outputs, output_data), axis=0)

    return inputs, outputs


def create_pysr_model(args):
    """
    Create and configure a PySR regressor model.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments

    Returns
    -------
    PySRRegressor
        Configured PySR model
    """
    from pysr import PySRRegressor

    # Define binary and unary operators
    binary_operators = ["+", "-", "*", "/"]
    unary_operators = ["exp", "log", "tanh", "sqrt", "square", "abs"]

    # Complexity constraints to prevent overly complex expressions
    constraints = {
        "/": (5, 3),      # Numerator max complexity 5, denominator max 3
        "exp": 3,         # Argument max complexity 3
        "log": 3,
        "sqrt": 3,
    }

    # Prevent nested transcendental functions
    nested_constraints = {
        "exp": {"exp": 0, "log": 0},
        "log": {"exp": 0, "log": 0},
    }

    model = PySRRegressor(
        # Search space
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        constraints=constraints,
        nested_constraints=nested_constraints,

        # Search parameters
        niterations=args.niterations,
        populations=args.populations,
        population_size=50,
        maxsize=args.maxsize,
        maxdepth=args.maxdepth,

        # Variable names for interpretability
        variable_names=["PoD", "VoS", "chiSA", "PSoSS"],

        # Loss function
        loss="loss(prediction, target) = (prediction - target)^2",

        # Optimization settings
        parsimony=0.0032,  # Penalty for complexity
        weight_optimize=0.001,  # Frequency of constant optimization
        adaptive_parsimony_scaling=20.0,

        # Output settings
        select_k_features=None,  # Use all features
        progress=True,
        verbosity=1,

        # Reproducibility
        random_state=args.seed,
        deterministic=True,
        procs=0,  # Use Julia's built-in parallelism

        # Turbo mode for faster convergence
        turbo=args.turbo,

        # Extra settings for stability
        warm_start=False,
        batching=False,  # Use full dataset for stability
    )

    return model


def main():
    """Main training function."""
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("PySR Symbolic Regression for FIML Beta Correction Factor")
    print("=" * 60)

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\nLoading field inversion data...")
    try:
        inputs, outputs = load_field_inversion_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease run field inversion first:")
        print("  cd ..")
        print("  mpirun -np 4 python runScript_FI.py -index 0")
        print("  mpirun -np 4 python runScript_FI.py -index 1")
        sys.exit(1)

    print(f"Loaded {inputs.shape[0]} samples with {inputs.shape[1]} features")
    print(f"Input features: PoD, VoS, chiSA, PSoSS")
    print(f"Output: betaFINuTilda")

    # Print data statistics
    print("\nData statistics:")
    feature_names = ["PoD", "VoS", "chiSA", "PSoSS"]
    for i, name in enumerate(feature_names):
        print(f"  {name}: min={inputs[:, i].min():.4f}, max={inputs[:, i].max():.4f}, "
              f"mean={inputs[:, i].mean():.4f}, std={inputs[:, i].std():.4f}")
    print(f"  betaFI: min={outputs.min():.4f}, max={outputs.max():.4f}, "
          f"mean={outputs.mean():.4f}, std={outputs.std():.4f}")

    # Split data into training and validation sets
    n_samples = inputs.shape[0]
    n_val = int(n_samples * args.val_split)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train = inputs[train_indices]
    y_train = outputs[train_indices]
    X_val = inputs[val_indices]
    y_val = outputs[val_indices]

    print(f"\nTraining samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    # Create and fit PySR model
    print("\nConfiguring PySR model...")
    print(f"  Max iterations: {args.niterations}")
    print(f"  Populations: {args.populations}")
    print(f"  Max complexity: {args.maxsize}")
    print(f"  Max depth: {args.maxdepth}")
    print(f"  Turbo mode: {args.turbo}")

    model = create_pysr_model(args)

    print("\nStarting symbolic regression...")
    print("(This may take several minutes depending on settings)")
    print("-" * 60)

    model.fit(X_train, y_train)

    print("-" * 60)
    print("\nTraining complete!")

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    best_eq = model.get_best()
    print(f"\nBest equation found:")
    print(f"  {best_eq['sympy_format']}")
    print(f"\nComplexity: {best_eq['complexity']}")
    print(f"Training loss (MSE): {best_eq['loss']:.6e}")

    # Validate on training and validation sets
    lambda_func = best_eq["lambda_format"]

    print("\nValidation metrics:")
    train_metrics = validate_equation(lambda_func, X_train, y_train)
    val_metrics = validate_equation(lambda_func, X_val, y_val)

    print("\n  Training set:")
    print(f"    MSE: {train_metrics['mse']:.6e}")
    print(f"    RMSE: {train_metrics['rmse']:.6f}")
    print(f"    R²: {train_metrics['r2']:.4f}")
    print(f"    Max error: {train_metrics['max_error']:.6f}")
    print(f"    Correlation: {train_metrics['correlation']:.4f}")

    print("\n  Validation set:")
    print(f"    MSE: {val_metrics['mse']:.6e}")
    print(f"    RMSE: {val_metrics['rmse']:.6f}")
    print(f"    R²: {val_metrics['r2']:.4f}")
    print(f"    Max error: {val_metrics['max_error']:.6f}")
    print(f"    Correlation: {val_metrics['correlation']:.4f}")

    # Export equation to multiple formats
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)

    export_paths = export_equation_formats(model, output_dir)
    print("\nEquation exported to:")
    for fmt, path in export_paths.items():
        print(f"  {fmt}: {path}")

    # Save comprehensive report
    report_path = save_equation_report(
        model, output_dir,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val
    )

    # Save model for reuse
    model_path = os.path.join(output_dir, "model.pkl")
    save_model(model, model_path)

    # Generate plots
    print("\nGenerating plots...")
    try:
        plot_pareto_front(
            model,
            output_path=os.path.join(output_dir, "pareto_front.png")
        )

        y_pred_val = lambda_func(*[X_val[:, i] for i in range(X_val.shape[1])])
        plot_prediction_comparison(
            y_val, y_pred_val,
            output_path=os.path.join(output_dir, "predictions.png"),
            title="Validation Set: Predicted vs True beta_FI"
        )
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    # Print Pareto front summary
    print("\n" + "=" * 60)
    print("PARETO FRONT (Top 5 by complexity)")
    print("=" * 60)

    equations = model.equations_
    if equations is not None and len(equations) > 0:
        sorted_eqs = equations.sort_values("complexity")
        for i, (idx, row) in enumerate(sorted_eqs.head(5).iterrows()):
            print(f"\n{i+1}. Complexity: {row['complexity']}, Loss: {row['loss']:.6e}")
            print(f"   {row['sympy_format']}")

    print("\n" + "=" * 60)
    print(f"All results saved to: {os.path.abspath(output_dir)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
