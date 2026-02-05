"""
Utility functions for PySR symbolic regression training.

Provides export, validation, and visualization capabilities for
discovered equations in the FIML turbulence modeling workflow.
"""

import numpy as np
import json
import os
from typing import Callable, Dict, List, Optional, Tuple, Any


def export_equation_formats(
    model,
    output_dir: str,
    variable_names: List[str] = None,
    prefix: str = "equation_best"
) -> Dict[str, str]:
    """
    Export the best equation to multiple formats.

    Parameters
    ----------
    model : PySRRegressor
        Fitted PySR model
    output_dir : str
        Directory to save outputs
    variable_names : List[str], optional
        Variable names for the equation
    prefix : str
        Prefix for output files

    Returns
    -------
    Dict[str, str]
        Dictionary with format names as keys and file paths as values
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs = {}

    # Get the best equation
    best_eq = model.get_best()

    # 1. SymPy expression
    sympy_expr = best_eq["sympy_format"]
    sympy_path = os.path.join(output_dir, f"{prefix}_sympy.txt")
    with open(sympy_path, "w") as f:
        f.write(str(sympy_expr))
    outputs["sympy"] = sympy_path

    # 2. LaTeX format
    try:
        from sympy import latex
        latex_expr = latex(sympy_expr)
    except Exception:
        latex_expr = str(sympy_expr)

    latex_path = os.path.join(output_dir, f"{prefix}.tex")
    with open(latex_path, "w") as f:
        f.write("% PySR discovered equation for beta_FI(PoD, VoS, chiSA, PSoSS)\n")
        f.write("% Complexity: {}\n".format(best_eq["complexity"]))
        f.write("% Loss: {:.6e}\n".format(best_eq["loss"]))
        f.write(r"\begin{equation}" + "\n")
        f.write(r"    \beta_{\mathrm{FI}} = " + latex_expr + "\n")
        f.write(r"\end{equation}" + "\n")
    outputs["latex"] = latex_path

    # 3. Python callable
    if variable_names is None:
        variable_names = ["PoD", "VoS", "chiSA", "PSoSS"]

    py_path = os.path.join(output_dir, f"{prefix}.py")
    with open(py_path, "w") as f:
        f.write('"""\n')
        f.write("Auto-generated beta_FI equation from PySR symbolic regression.\n")
        f.write(f"Complexity: {best_eq['complexity']}\n")
        f.write(f"Loss: {best_eq['loss']:.6e}\n")
        f.write('"""\n\n')
        f.write("import numpy as np\n\n\n")
        f.write(f"def beta_fi({', '.join(variable_names)}):\n")
        f.write('    """\n')
        f.write("    Compute the field inversion correction factor beta_FI.\n\n")
        f.write("    Parameters\n")
        f.write("    ----------\n")
        for var in variable_names:
            f.write(f"    {var} : float or np.ndarray\n")
            f.write(f"        Input feature {var}\n")
        f.write("\n    Returns\n")
        f.write("    -------\n")
        f.write("    float or np.ndarray\n")
        f.write("        The correction factor beta_FI\n")
        f.write('    """\n')
        # Convert sympy expression to numpy-compatible string
        expr_str = _sympy_to_numpy_str(sympy_expr, variable_names)
        f.write(f"    return {expr_str}\n")
    outputs["python"] = py_path

    # 4. C++ format for DAFoam integration
    cpp_path = os.path.join(output_dir, f"{prefix}.cpp")
    with open(cpp_path, "w") as f:
        f.write("// Auto-generated beta_FI equation from PySR symbolic regression\n")
        f.write(f"// Complexity: {best_eq['complexity']}\n")
        f.write(f"// Loss: {best_eq['loss']:.6e}\n\n")
        f.write("// For integration into DAFoam regression model\n")
        cpp_expr = _sympy_to_cpp_str(sympy_expr, variable_names)
        f.write(f"scalar betaFI = {cpp_expr};\n")
    outputs["cpp"] = cpp_path

    return outputs


def _sympy_to_numpy_str(expr, variable_names: List[str]) -> str:
    """Convert a SymPy expression to a NumPy-compatible string."""
    expr_str = str(expr)
    # Replace common functions with numpy equivalents
    replacements = {
        "exp": "np.exp",
        "log": "np.log",
        "sqrt": "np.sqrt",
        "tanh": "np.tanh",
        "abs": "np.abs",
        "Abs": "np.abs",
        "sin": "np.sin",
        "cos": "np.cos",
    }
    for old, new in replacements.items():
        # Only replace if not already prefixed with np.
        if f"np.{old}" not in expr_str:
            expr_str = expr_str.replace(f"{old}(", f"{new}(")
    return expr_str


def _sympy_to_cpp_str(expr, variable_names: List[str]) -> str:
    """Convert a SymPy expression to C++ compatible string."""
    expr_str = str(expr)
    # Replace Python-specific with C++ equivalents
    replacements = {
        "**": "pow",  # Will need special handling
        "abs": "mag",
        "Abs": "mag",
    }
    # Handle power operator
    import re
    # This is a simplified conversion; complex expressions may need manual review
    expr_str = re.sub(r'(\w+)\*\*(\d+(?:\.\d+)?)', r'pow(\1, \2)', expr_str)
    expr_str = expr_str.replace("abs(", "mag(")
    expr_str = expr_str.replace("Abs(", "mag(")
    return expr_str


def validate_equation(
    func: Callable,
    X: np.ndarray,
    y: np.ndarray,
    variable_names: List[str] = None
) -> Dict[str, float]:
    """
    Compute validation metrics for a discovered equation.

    Parameters
    ----------
    func : Callable
        Function that takes X columns as arguments and returns predictions
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        True target values of shape (n_samples,)
    variable_names : List[str], optional
        Names of variables (for reporting)

    Returns
    -------
    Dict[str, float]
        Dictionary containing MSE, RMSE, R², max_error, correlation
    """
    # Get predictions
    if X.ndim == 1:
        y_pred = func(X)
    else:
        # Unpack columns as arguments
        y_pred = func(*[X[:, i] for i in range(X.shape[1])])

    y_pred = np.asarray(y_pred).flatten()
    y = np.asarray(y).flatten()

    # Compute metrics
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    max_error = np.max(np.abs(y - y_pred))
    correlation = np.corrcoef(y, y_pred)[0, 1] if len(y) > 1 else 1.0

    # Mean Absolute Error
    mae = np.mean(np.abs(y - y_pred))

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "max_error": float(max_error),
        "mae": float(mae),
        "correlation": float(correlation),
        "n_samples": int(len(y)),
    }


def plot_pareto_front(
    model,
    output_path: str = None,
    title: str = "Pareto Front: Complexity vs Accuracy"
) -> None:
    """
    Plot the Pareto front of discovered equations.

    Parameters
    ----------
    model : PySRRegressor
        Fitted PySR model
    output_path : str, optional
        Path to save the plot. If None, displays interactively.
    title : str
        Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping Pareto front plot")
        return

    equations = model.equations_
    if equations is None or len(equations) == 0:
        print("No equations found in model")
        return

    complexities = equations["complexity"].values
    losses = equations["loss"].values

    # Identify Pareto-optimal points
    pareto_mask = _get_pareto_mask(complexities, losses)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all equations
    ax.scatter(
        complexities[~pareto_mask],
        losses[~pareto_mask],
        alpha=0.4,
        label="Dominated",
        color="gray",
        s=30
    )

    # Plot Pareto front
    ax.scatter(
        complexities[pareto_mask],
        losses[pareto_mask],
        alpha=0.9,
        label="Pareto Optimal",
        color="blue",
        s=60,
        edgecolors="black"
    )

    # Connect Pareto points
    pareto_complexities = complexities[pareto_mask]
    pareto_losses = losses[pareto_mask]
    sorted_idx = np.argsort(pareto_complexities)
    ax.plot(
        pareto_complexities[sorted_idx],
        pareto_losses[sorted_idx],
        "b--",
        alpha=0.5
    )

    # Mark the best equation
    best_idx = model.equations_["loss"].idxmin()
    ax.scatter(
        [complexities[best_idx]],
        [losses[best_idx]],
        s=150,
        marker="*",
        color="red",
        label="Best (lowest loss)",
        zorder=5
    )

    ax.set_xlabel("Complexity", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Pareto front plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def _get_pareto_mask(complexities: np.ndarray, losses: np.ndarray) -> np.ndarray:
    """Identify Pareto-optimal points (minimize both complexity and loss)."""
    n = len(complexities)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # j dominates i if j is better or equal in both and strictly better in one
                if (complexities[j] <= complexities[i] and losses[j] <= losses[i] and
                    (complexities[j] < complexities[i] or losses[j] < losses[i])):
                    pareto_mask[i] = False
                    break

    return pareto_mask


def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = None,
    title: str = "Prediction vs Truth"
) -> None:
    """
    Create a parity plot comparing predictions to true values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    output_path : str, optional
        Path to save the plot. If None, displays interactively.
    title : str
        Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping prediction plot")
        return

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = 0.1 * (max_val - min_val)
    line_range = [min_val - margin, max_val + margin]
    ax.plot(line_range, line_range, "r--", linewidth=2, label="Perfect prediction")

    # Compute and display R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    ax.text(
        0.05, 0.95,
        f"R² = {r2:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    ax.set_xlabel(r"True $\beta_{FI}$", fontsize=12)
    ax.set_ylabel(r"Predicted $\beta_{FI}$", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Prediction comparison plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def save_equation_report(
    model,
    output_dir: str,
    X_train: np.ndarray = None,
    y_train: np.ndarray = None,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    variable_names: List[str] = None
) -> str:
    """
    Save a comprehensive report of the symbolic regression results.

    Parameters
    ----------
    model : PySRRegressor
        Fitted PySR model
    output_dir : str
        Directory to save the report
    X_train : np.ndarray, optional
        Training features for validation
    y_train : np.ndarray, optional
        Training targets for validation
    X_val : np.ndarray, optional
        Validation features
    y_val : np.ndarray, optional
        Validation targets
    variable_names : List[str], optional
        Names of input variables

    Returns
    -------
    str
        Path to the saved report
    """
    os.makedirs(output_dir, exist_ok=True)

    if variable_names is None:
        variable_names = ["PoD", "VoS", "chiSA", "PSoSS"]

    report = {
        "variable_names": variable_names,
        "best_equation": {},
        "pareto_front": [],
        "training_metrics": None,
        "validation_metrics": None,
    }

    # Best equation info
    best_eq = model.get_best()
    report["best_equation"] = {
        "sympy": str(best_eq["sympy_format"]),
        "complexity": int(best_eq["complexity"]),
        "loss": float(best_eq["loss"]),
    }

    # All Pareto-optimal equations
    equations = model.equations_
    if equations is not None:
        complexities = equations["complexity"].values
        losses = equations["loss"].values
        pareto_mask = _get_pareto_mask(complexities, losses)

        for idx in np.where(pareto_mask)[0]:
            report["pareto_front"].append({
                "sympy": str(equations.iloc[idx]["sympy_format"]),
                "complexity": int(equations.iloc[idx]["complexity"]),
                "loss": float(equations.iloc[idx]["loss"]),
            })

    # Compute metrics if data provided
    lambda_func = model.get_best()["lambda_format"]

    if X_train is not None and y_train is not None:
        y_pred_train = lambda_func(*[X_train[:, i] for i in range(X_train.shape[1])])
        report["training_metrics"] = validate_equation(
            lambda_func, X_train, y_train, variable_names
        )

    if X_val is not None and y_val is not None:
        y_pred_val = lambda_func(*[X_val[:, i] for i in range(X_val.shape[1])])
        report["validation_metrics"] = validate_equation(
            lambda_func, X_val, y_val, variable_names
        )

    # Save JSON report
    report_path = os.path.join(output_dir, "sr_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save Pareto front separately
    pareto_path = os.path.join(output_dir, "pareto_front.json")
    with open(pareto_path, "w") as f:
        json.dump(report["pareto_front"], f, indent=2)

    print(f"Report saved to {report_path}")
    return report_path


def save_model(model, output_path: str) -> None:
    """
    Save the PySR model to a pickle file.

    Parameters
    ----------
    model : PySRRegressor
        Fitted PySR model
    output_path : str
        Path to save the model
    """
    import pickle
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {output_path}")


def load_model(input_path: str):
    """
    Load a PySR model from a pickle file.

    Parameters
    ----------
    input_path : str
        Path to the saved model

    Returns
    -------
    PySRRegressor
        Loaded model
    """
    import pickle
    with open(input_path, "rb") as f:
        return pickle.load(f)
