#!/usr/bin/env python
"""
Training monitoring and visualization for FIML.

Provides plotting functions called by trainModel_enhanced.py after training.
Can also regenerate plots standalone from saved results.

Usage (standalone):
    python training_monitor.py --results_dir training_results
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Core plotting functions
# ---------------------------------------------------------------------------

def plot_training_curves(history, output_dir):
    """Loss curves with overfitting detection and best-epoch marker."""
    h = history.history if hasattr(history, "history") else history
    epochs = range(1, len(h["loss"]) + 1)
    best_epoch = int(np.argmin(h["val_loss"])) + 1

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(epochs, h["loss"], "b-", lw=1.2, label="Train")
    ax.semilogy(epochs, h["val_loss"], "r-", lw=1.2, label="Validation")
    ax.axvline(best_epoch, color="green", ls="--", alpha=0.5,
               label=f"Best epoch: {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Overfitting warning: final val >> final train
    if len(h["val_loss"]) > 50:
        if np.mean(h["val_loss"][-20:]) > 2 * np.mean(h["loss"][-20:]):
            ax.set_title("Training Curves  [OVERFITTING DETECTED]", color="red")
        else:
            ax.set_title("Training Curves")
    else:
        ax.set_title("Training Curves")

    plt.tight_layout()
    _save(fig, output_dir, "training_curves.png")


def plot_parity(y_true, y_pred, output_dir, label="filtered"):
    """Predicted vs true scatter with MSE/R^2 annotation."""
    mse, r2, max_err = _compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    idx = _subsample(len(y_true), 8000)
    ax.scatter(y_true[idx], y_pred[idx], s=1, alpha=0.3, c="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.text(0.05, 0.95,
            f"MSE={mse:.6f}\nR²={r2:.4f}\nMax err={max_err:.4f}\nN={len(y_true)}",
            transform=ax.transAxes, va="top", fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
    ax.set_xlabel("Beta (truth)")
    ax.set_ylabel("Beta (predicted)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"Parity ({label})")
    plt.tight_layout()
    _save(fig, output_dir, f"parity_{label}.png")
    return {"mse": mse, "r2": r2, "max_error": max_err}


def plot_beta_distribution(y_true, y_pred, output_dir):
    """Overlay histograms: FI truth vs NN prediction."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(y_true.min(), y_pred.min()),
                       max(y_true.max(), y_pred.max()), 50)
    ax.hist(y_true, bins=bins, alpha=0.5, density=True, label="FI truth", color="steelblue")
    ax.hist(y_pred, bins=bins, alpha=0.5, density=True, label="NN predicted", color="coral")
    ax.set_xlabel("Beta")
    ax.set_ylabel("Density")
    ax.set_title("Beta Distribution Comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, output_dir, "beta_distribution.png")


def plot_residual_analysis(y_true, y_pred, output_dir):
    """Three-panel: residuals vs true, residual histogram, abs-error CDF."""
    resid = y_pred - y_true
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: residual vs true
    ax = axes[0]
    idx = _subsample(len(y_true), 5000)
    ax.scatter(y_true[idx], resid[idx], s=1, alpha=0.3, c="steelblue")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("True beta")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs True")
    ax.grid(alpha=0.3)

    # Panel 2: residual histogram
    ax = axes[1]
    ax.hist(resid, bins=50, color="steelblue", edgecolor="white", density=True)
    ax.axvline(0, color="red", ls="--")
    ax.set_xlabel("Residual")
    ax.set_title(f"Residual Dist (mean={resid.mean():.5f})")
    ax.grid(alpha=0.3)

    # Panel 3: absolute error CDF with percentile markers
    ax = axes[2]
    sorted_err = np.sort(np.abs(resid))
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.plot(sorted_err, cdf, c="steelblue", lw=1.5)
    for p in [90, 95, 99]:
        v = np.percentile(np.abs(resid), p)
        ax.axhline(p / 100, color="gray", ls=":", alpha=0.4)
        ax.text(v, p / 100, f" P{p}={v:.4f}", fontsize=7)
    ax.set_xlabel("|Error|")
    ax.set_ylabel("CDF")
    ax.set_title("Absolute Error CDF")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, "residual_analysis.png")


def plot_region_breakdown(y_true, y_pred, output_dir):
    """MSE and R^2 broken down by beta-deviation regions."""
    regions = [
        ("Freestream (|b-1|<0.02)",   np.abs(y_true - 1.0) < 0.02),
        ("Mild (0.02-0.1)",           (np.abs(y_true - 1.0) >= 0.02) & (np.abs(y_true - 1.0) < 0.1)),
        ("Moderate (0.1-0.5)",        (np.abs(y_true - 1.0) >= 0.1) & (np.abs(y_true - 1.0) < 0.5)),
        ("Strong (>0.5)",             np.abs(y_true - 1.0) >= 0.5),
    ]

    names, mses, r2s, counts = [], [], [], []
    for name, mask in regions:
        n = mask.sum()
        if n < 2:
            continue
        mse, r2, _ = _compute_metrics(y_true[mask], y_pred[mask])
        names.append(name)
        mses.append(mse)
        r2s.append(r2)
        counts.append(n)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))

    # MSE per region
    axes[0].barh(x, mses, color="steelblue")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel("MSE")
    axes[0].invert_yaxis()
    for i, v in enumerate(mses):
        axes[0].text(v, i, f" {v:.6f}", va="center", fontsize=8)

    # R^2 per region
    colors = ["green" if r > 0.8 else "orange" if r > 0.5 else "red" for r in r2s]
    axes[1].barh(x, r2s, color=colors)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlabel("R²")
    axes[1].invert_yaxis()
    for i, v in enumerate(r2s):
        axes[1].text(max(v, 0), i, f" {v:.3f}", va="center", fontsize=8)

    plt.suptitle("Performance by Region")
    plt.tight_layout()
    _save(fig, output_dir, "region_breakdown.png")


# ---------------------------------------------------------------------------
# Orchestrator: generate everything + save metrics
# ---------------------------------------------------------------------------

def generate_all_plots(history, y_true_all, y_pred_all, filter_mask, output_dir):
    """Generate all monitoring plots and return a metrics dict.

    Args:
        history: TF History object or dict with 'loss'/'val_loss' keys.
        y_true_all: Beta truth for ALL cells (unfiltered).
        y_pred_all: Beta prediction for ALL cells.
        filter_mask: Boolean mask identifying correction-region cells.
        output_dir: Where to save plots and metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    y_true_f = y_true_all[filter_mask]
    y_pred_f = y_pred_all[filter_mask]

    print("\nGenerating monitoring plots...")
    plot_training_curves(history, output_dir)
    m_filt = plot_parity(y_true_f, y_pred_f, output_dir, label="filtered")
    plot_parity(y_true_all, y_pred_all, output_dir, label="all_cells")
    plot_beta_distribution(y_true_f, y_pred_f, output_dir)
    plot_residual_analysis(y_true_f, y_pred_f, output_dir)
    plot_region_breakdown(y_true_all, y_pred_all, output_dir)

    # Aggregate metrics
    fs_mask = np.abs(y_true_all - 1.0) < 0.02
    h = history.history if hasattr(history, "history") else history
    metrics = {
        "mse_filtered": m_filt["mse"],
        "r2_filtered": m_filt["r2"],
        "max_error_filtered": m_filt["max_error"],
        "mse_all_cells": float(np.mean((y_true_all - y_pred_all) ** 2)),
        "mse_freestream": float(np.mean((y_true_all[fs_mask] - y_pred_all[fs_mask]) ** 2)) if fs_mask.any() else 0.0,
        "n_total": len(y_true_all),
        "n_filtered": int(filter_mask.sum()),
        "best_val_epoch": int(np.argmin(h["val_loss"])) + 1,
    }

    # Save metrics JSON
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in metrics.items()}, f, indent=2)
    print(f"  Metrics saved to {path}")

    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true, y_pred):
    """Return (MSE, R^2, max_abs_error)."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return mse, r2, float(np.max(np.abs(y_true - y_pred)))


def _subsample(n, max_pts):
    """Random index subsample for scatter plots."""
    if n > max_pts:
        return np.random.choice(n, max_pts, replace=False)
    return np.arange(n)


def _save(fig, output_dir, filename):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Standalone: regenerate plots from saved data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate FIML training plots")
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    data = np.load(os.path.join(args.results_dir, "training_data.npz"))
    with open(os.path.join(args.results_dir, "training_history.json")) as f:
        history = json.load(f)

    generate_all_plots(
        history, data["y_true_all"], data["y_pred_all"],
        data["filter_mask"], args.results_dir,
    )
