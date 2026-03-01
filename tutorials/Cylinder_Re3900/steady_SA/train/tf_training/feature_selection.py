#!/usr/bin/env python
"""
Feature selection for FIML training.

Ranks all 11 DAFoam features by predictive relevance for the beta field using
multiple complementary methods (Spearman correlation, mutual information,
variance threshold, redundancy detection) and produces diagnostic plots.

Usage:
    mpirun -np 1 python feature_selection.py --case c1_data --nCells 25000
    mpirun -np 1 python feature_selection.py --case c1_data c2_data --nCells 25000 --beta_threshold 0.05
    mpirun -np 1 python feature_selection.py --case c1_data --nCells 25000 --top_k 5
"""

import argparse
import os
import numpy as np
from mpi4py import MPI
from pyofm import PYOFM
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# All features available in DARegression.C
ALL_FEATURES = [
    "VoS", "PoD", "chiSA", "pGradStream", "PSoSS",
    "SCurv", "UOrth", "KoU2", "ReWall", "CoP", "TauoK",
]

SA_FEATURES = ["VoS", "PoD", "chiSA", "pGradStream", "PSoSS", "SCurv", "UOrth"]
SST_FEATURES = ["VoS", "PSoSS", "SCurv", "UOrth", "KoU2", "ReWall", "CoP", "TauoK"]


def load_data(cases, nCells, feature_names, beta_name="betaFINuTilda"):
    """Load feature and beta fields from OpenFOAM case directories."""
    ofm = PYOFM(comm=MPI.COMM_WORLD)
    all_inputs = []
    all_outputs = []

    for case in cases:
        if not os.path.exists(case):
            raise FileNotFoundError(
                f"Data directory '{case}' not found. Run field inversion first."
            )
        feats = []
        for fname in feature_names:
            field = np.zeros(nCells)
            ofm.readField(fname, "volScalarField", case, field)
            feats.append(field)
        all_inputs.append(np.column_stack(feats))

        beta = np.zeros(nCells)
        ofm.readField(beta_name, "volScalarField", case, beta)
        all_outputs.append(beta)

    inputs = np.vstack(all_inputs)
    outputs = np.concatenate(all_outputs)
    return inputs, outputs


def compute_rankings(inputs, outputs, feature_names):
    """Compute feature rankings using multiple methods.

    Returns a dict with per-method scores and a combined ranking.
    """
    n_features = len(feature_names)
    results = {
        "features": feature_names,
        "spearman_abs": np.zeros(n_features),
        "pearson_abs": np.zeros(n_features),
        "mi_score": np.zeros(n_features),
        "variance": np.zeros(n_features),
    }

    # Spearman (monotonic) and Pearson (linear) correlations
    for i in range(n_features):
        rho_s, _ = spearmanr(inputs[:, i], outputs)
        rho_p, _ = pearsonr(inputs[:, i], outputs)
        results["spearman_abs"][i] = abs(rho_s)
        results["pearson_abs"][i] = abs(rho_p)
        results["variance"][i] = np.std(inputs[:, i])

    # Mutual information (nonlinear dependence)
    results["mi_score"] = mutual_info_regression(
        inputs, outputs, n_neighbors=5, random_state=42
    )

    # Rank each method (1 = best)
    def rank_descending(arr):
        return np.argsort(np.argsort(-arr)) + 1

    r_spearman = rank_descending(results["spearman_abs"])
    r_pearson = rank_descending(results["pearson_abs"])
    r_mi = rank_descending(results["mi_score"])

    # Combined rank: average of all three
    results["combined_score"] = (r_spearman + r_pearson + r_mi) / 3.0
    results["combined_rank"] = rank_descending(-results["combined_score"])
    # Lower combined_score is better, so we rank ascending
    results["combined_rank"] = np.argsort(np.argsort(results["combined_score"])) + 1

    return results


def compute_correlation_matrix(inputs, feature_names):
    """Pairwise Pearson correlation matrix to detect redundant features."""
    return np.corrcoef(inputs, rowvar=False)


def detect_redundant_pairs(corr_matrix, feature_names, threshold=0.90):
    """Find pairs of features with |correlation| > threshold."""
    n = len(feature_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > threshold:
                pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
    return pairs


def compute_permutation_importance(inputs, outputs, feature_names, n_repeats=10):
    """Estimate feature importance by permutation (model-free baseline using MSE
    of a constant predictor vs permuted feature contribution).

    Uses a simple k-NN regressor to avoid TensorFlow dependency.
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        inputs, outputs, test_size=0.2, random_state=42
    )

    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(X_train, y_train)
    baseline_mse = np.mean((y_test - knn.predict(X_test)) ** 2)

    importances = np.zeros(len(feature_names))
    for i in range(len(feature_names)):
        mse_drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, i] = np.random.permutation(X_perm[:, i])
            perm_mse = np.mean((y_test - knn.predict(X_perm)) ** 2)
            mse_drops.append(perm_mse - baseline_mse)
        importances[i] = np.mean(mse_drops)

    return importances, baseline_mse


def plot_feature_rankings(results, output_dir):
    """Bar chart of feature rankings by each method."""
    features = results["features"]
    n = len(features)
    x = np.arange(n)

    # Sort by combined rank
    order = np.argsort(results["combined_rank"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = [
        ("spearman_abs", "|Spearman rho|"),
        ("pearson_abs", "|Pearson r|"),
        ("mi_score", "Mutual Information"),
    ]

    for ax, (key, title) in zip(axes, methods):
        vals = results[key][order]
        labels = [features[i] for i in order]
        ax.barh(x, vals, color="steelblue", edgecolor="white")
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.set_xlabel(title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Feature Rankings (top = most predictive)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_rankings.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_correlation_matrix(corr, feature_names, output_dir):
    """Heatmap of pairwise feature correlations."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    n = len(feature_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(feature_names, fontsize=9)

    for i in range(n):
        for j in range(n):
            color = "white" if abs(corr[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_correlation_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_feature_distributions(inputs, feature_names, output_dir):
    """Histograms of each feature's distribution."""
    n = len(feature_names)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    for i, fname in enumerate(feature_names):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.hist(inputs[:, i], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_title(fname, fontsize=10)
        ax.set_ylabel("Count")
        std = np.std(inputs[:, i])
        ax.text(0.95, 0.95, f"std={std:.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_feature_vs_beta(inputs, outputs, feature_names, output_dir, max_points=5000):
    """Scatter plots of each feature vs beta."""
    n = len(feature_names)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    # Subsample for plotting speed
    if len(outputs) > max_points:
        idx = np.random.choice(len(outputs), max_points, replace=False)
    else:
        idx = np.arange(len(outputs))

    for i, fname in enumerate(feature_names):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.scatter(inputs[idx, i], outputs[idx], s=1, alpha=0.3, color="steelblue")
        ax.set_xlabel(fname)
        ax.set_ylabel("beta")
        rho, _ = spearmanr(inputs[:, i], outputs)
        ax.set_title(f"{fname} (rho={rho:.3f})", fontsize=9)
        ax.grid(alpha=0.3)

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Feature vs Beta (Spearman rho shown)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_vs_beta.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def print_report(results, corr_matrix, redundant_pairs, perm_importance,
                 dead_features, output_dir):
    """Print and save a summary report."""
    features = results["features"]
    order = np.argsort(results["combined_rank"])

    lines = []
    lines.append("=" * 75)
    lines.append("FEATURE SELECTION REPORT")
    lines.append("=" * 75)
    lines.append("")

    # Rankings table
    header = f"{'Rank':<5} {'Feature':<15} {'|Spearman|':>11} {'|Pearson|':>11} {'MI':>10} {'Std':>10} {'Perm Imp':>10}"
    lines.append(header)
    lines.append("-" * 75)
    for rank_pos, idx in enumerate(order, 1):
        lines.append(
            f"{rank_pos:<5} {features[idx]:<15} "
            f"{results['spearman_abs'][idx]:>11.4f} "
            f"{results['pearson_abs'][idx]:>11.4f} "
            f"{results['mi_score'][idx]:>10.4f} "
            f"{results['variance'][idx]:>10.4f} "
            f"{perm_importance[idx]:>10.6f}"
        )

    # Warnings
    lines.append("")
    if dead_features:
        lines.append("DEAD FEATURES (std < 0.01, remove these):")
        for f in dead_features:
            lines.append(f"  - {f}")
    else:
        lines.append("No dead features detected.")

    lines.append("")
    if redundant_pairs:
        lines.append("REDUNDANT PAIRS (|corr| > 0.90, consider removing one):")
        for f1, f2, corr in redundant_pairs:
            lines.append(f"  - {f1} <-> {f2}: r = {corr:.3f}")
    else:
        lines.append("No highly redundant feature pairs detected.")

    report = "\n".join(lines)
    print(report)

    path = os.path.join(output_dir, "feature_selection_report.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {path}")


def select_features(results, perm_importance, dead_features, redundant_pairs, top_k):
    """Select the best features based on combined ranking, removing dead/redundant ones."""
    features = results["features"]
    order = np.argsort(results["combined_rank"])

    selected = []
    excluded_redundant = set()

    for idx in order:
        fname = features[idx]
        if fname in dead_features:
            continue
        if fname in excluded_redundant:
            continue
        selected.append(fname)
        # If this feature is part of a redundant pair, exclude the worse partner
        for f1, f2, _ in redundant_pairs:
            if fname == f1:
                excluded_redundant.add(f2)
            elif fname == f2:
                excluded_redundant.add(f1)
        if len(selected) >= top_k:
            break

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Feature selection for FIML beta prediction"
    )
    parser.add_argument("--case", nargs="+", required=True,
                        help="Case data directories (e.g., c1_data c2_data)")
    parser.add_argument("--nCells", type=int, required=True,
                        help="Number of cells per case")
    parser.add_argument("--beta_name", default="betaFINuTilda",
                        help="Name of beta field (default: betaFINuTilda)")
    parser.add_argument("--beta_threshold", type=float, default=0.0,
                        help="Filter cells where |beta-1| > threshold (0 = no filter)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of features to recommend (default: 5)")
    parser.add_argument("--turb_model", choices=["SA", "SST"], default="SA",
                        help="Turbulence model determines available features")
    parser.add_argument("--output_dir", default="feature_analysis",
                        help="Directory for output plots and reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Select candidate features based on turbulence model
    if args.turb_model == "SA":
        feature_names = SA_FEATURES
    else:
        feature_names = SST_FEATURES

    print(f"Turbulence model: {args.turb_model}")
    print(f"Candidate features ({len(feature_names)}): {feature_names}")
    print(f"Cases: {args.case}")
    print(f"Cells per case: {args.nCells}")

    # Load data
    print("\nLoading data...")
    inputs, outputs = load_data(args.case, args.nCells, feature_names, args.beta_name)
    print(f"  Total samples: {len(outputs)}")
    print(f"  Beta range: [{outputs.min():.4f}, {outputs.max():.4f}]")

    # Optional filtering
    if args.beta_threshold > 0:
        mask = np.abs(outputs - 1.0) > args.beta_threshold
        inputs = inputs[mask]
        outputs = outputs[mask]
        print(f"  After filtering (|beta-1| > {args.beta_threshold}): {len(outputs)} samples")

    # Feature statistics
    print("\nFeature statistics:")
    for i, fname in enumerate(feature_names):
        col = inputs[:, i]
        print(f"  {fname:<15} min={col.min():.4f}  max={col.max():.4f}  "
              f"mean={col.mean():.4f}  std={col.std():.4f}")

    # Dead feature detection
    dead_features = [
        feature_names[i] for i in range(len(feature_names))
        if np.std(inputs[:, i]) < 0.01
    ]

    # Compute rankings
    print("\nComputing feature rankings...")
    results = compute_rankings(inputs, outputs, feature_names)

    # Correlation matrix
    print("Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(inputs, feature_names)
    redundant_pairs = detect_redundant_pairs(corr_matrix, feature_names, threshold=0.90)

    # Permutation importance
    print("Computing permutation importance (this may take a minute)...")
    perm_imp, baseline_mse = compute_permutation_importance(
        inputs, outputs, feature_names, n_repeats=5
    )
    print(f"  Baseline k-NN MSE: {baseline_mse:.6f}")

    # Report
    print_report(results, corr_matrix, redundant_pairs, perm_imp,
                 dead_features, args.output_dir)

    # Recommendation
    selected = select_features(
        results, perm_imp, dead_features, redundant_pairs, args.top_k
    )
    print(f"\nRECOMMENDED FEATURES (top {args.top_k}): {selected}")
    print("Use these in trainModel_enhanced.py with --features " + ",".join(selected))

    # Plots
    print("\nGenerating plots...")
    plot_feature_rankings(results, args.output_dir)
    plot_correlation_matrix(corr_matrix, feature_names, args.output_dir)
    plot_feature_distributions(inputs, feature_names, args.output_dir)
    plot_feature_vs_beta(inputs, outputs, feature_names, args.output_dir)

    # Save selected features to file for downstream use
    sel_path = os.path.join(args.output_dir, "selected_features.txt")
    with open(sel_path, "w") as f:
        for feat in selected:
            f.write(feat + "\n")
    print(f"Selected features written to {sel_path}")


if __name__ == "__main__":
    main()
