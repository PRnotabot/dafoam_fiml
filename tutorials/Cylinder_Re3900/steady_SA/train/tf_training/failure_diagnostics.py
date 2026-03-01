#!/usr/bin/env python
"""
Systematic failure mode diagnostics for FIML training.

Checks for the most common failure modes documented in the FIML pipeline and
prints actionable recommendations. Designed to be called after training
(from trainModel_enhanced.py) or standalone.

Each check returns a (severity, message) tuple:
    severity: "OK", "WARN", "FAIL"

Usage (standalone):
    python failure_diagnostics.py --results_dir training_results
"""

import os
import json
import numpy as np


# ---------------------------------------------------------------------------
# Individual diagnostic checks
# ---------------------------------------------------------------------------

def check_target_imbalance(y_true, threshold=0.05):
    """Check if beta target distribution is heavily skewed toward 1.0."""
    frac_boring = np.mean(np.abs(y_true - 1.0) < threshold)
    if frac_boring > 0.80:
        return "FAIL", (
            f"Target imbalance: {frac_boring*100:.0f}% of cells have |beta-1| < {threshold}. "
            f"Training without filtering will learn 'always output 1.0'. "
            f"FIX: Use spatial filtering (--beta_threshold {threshold}) or sample weighting."
        )
    if frac_boring > 0.60:
        return "WARN", (
            f"Mild target imbalance: {frac_boring*100:.0f}% of cells near beta=1. "
            f"Consider filtering or weighting."
        )
    return "OK", f"Target balance acceptable ({frac_boring*100:.0f}% near beta=1)."


def check_dead_features(inputs, feature_names, std_threshold=0.01):
    """Detect features with near-zero variance (no discriminative power)."""
    dead = []
    for i, fname in enumerate(feature_names):
        std = np.std(inputs[:, i])
        if std < std_threshold:
            dead.append((fname, std))

    if dead:
        names = ", ".join(f"{n} (std={s:.5f})" for n, s in dead)
        return "FAIL", (
            f"Dead features detected: {names}. "
            f"These carry no information. FIX: Remove them from the feature list."
        )
    return "OK", "All features have meaningful variance."


def check_overfitting(history):
    """Detect overfitting from training/validation loss divergence."""
    h = history.history if hasattr(history, "history") else history
    n = len(h["val_loss"])
    if n < 40:
        return "OK", "Too few epochs to assess overfitting."

    # Compare last 20% of epochs
    tail = max(1, n // 5)
    train_tail = np.mean(h["loss"][-tail:])
    val_tail = np.mean(h["val_loss"][-tail:])

    ratio = val_tail / train_tail if train_tail > 1e-12 else 1.0

    if ratio > 3.0:
        return "FAIL", (
            f"Severe overfitting: val_loss/train_loss = {ratio:.1f}x in final epochs. "
            f"FIX: Reduce network size, add L2 regularization, or increase data."
        )
    if ratio > 1.5:
        return "WARN", (
            f"Mild overfitting: val_loss/train_loss = {ratio:.1f}x. "
            f"Consider adding regularization or using early stopping with lower patience."
        )
    return "OK", f"No overfitting detected (val/train ratio = {ratio:.2f})."


def check_underfitting(history, mse_threshold=0.05):
    """Detect underfitting from high converged loss."""
    h = history.history if hasattr(history, "history") else history
    final_val = h["val_loss"][-1]
    final_train = h["loss"][-1]

    if final_train > mse_threshold:
        return "FAIL", (
            f"Underfitting: train loss = {final_train:.5f} (threshold: {mse_threshold}). "
            f"Model cannot fit training data. "
            f"FIX: Increase network capacity ([20,20] -> [30,30]), add features, "
            f"or check for noisy FI data."
        )
    if final_val > mse_threshold and final_train < mse_threshold:
        return "WARN", (
            f"Val loss high ({final_val:.5f}) but train loss OK ({final_train:.5f}). "
            f"Possible underfitting on unseen regions or overfitting. Check parity plot."
        )
    return "OK", f"Converged losses acceptable (train={final_train:.5f}, val={final_val:.5f})."


def check_noisy_fi_data(inputs, y_true, feature_names, n_neighbors=20):
    """Detect noisy FI data: similar features mapping to very different betas.

    Uses local variance in beta among k-nearest neighbors in feature space.
    High local variance = the FI beta field is noisy/non-smooth.
    """
    from sklearn.neighbors import NearestNeighbors

    # Subsample for speed
    n = len(y_true)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        X, y = inputs[idx], y_true[idx]
    else:
        X, y = inputs, y_true

    # Normalize features for distance computation
    std = np.std(X, axis=0)
    std[std < 1e-10] = 1.0
    X_norm = X / std

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_norm)
    _, indices = nn.kneighbors(X_norm)

    # Local beta variance for each point
    local_vars = np.array([np.var(y[indices[i]]) for i in range(len(y))])
    median_local_var = np.median(local_vars)
    global_var = np.var(y)

    # Noise ratio: if local variance is large relative to global, data is noisy
    noise_ratio = median_local_var / global_var if global_var > 1e-12 else 0.0

    if noise_ratio > 0.5:
        return "FAIL", (
            f"Noisy FI data: median local beta variance = {median_local_var:.5f} "
            f"({noise_ratio*100:.0f}% of global variance). "
            f"Similar features map to very different betas. "
            f"FIX: Increase betaVar.scale in runScript_FI.py and re-run field inversion."
        )
    if noise_ratio > 0.2:
        return "WARN", (
            f"Moderately noisy FI data (local/global variance ratio = {noise_ratio:.2f}). "
            f"May limit achievable accuracy."
        )
    return "OK", f"FI data appears smooth (noise ratio = {noise_ratio:.3f})."


def check_normalization_mismatch(inputs_all, inputs_filtered, feature_names):
    """Warn if filtered data statistics differ significantly from full data.

    If the normalizer is adapted on filtered data but deployed on full domain,
    freestream features will be improperly normalized.
    """
    issues = []
    for i, fname in enumerate(feature_names):
        mean_all = np.mean(inputs_all[:, i])
        mean_filt = np.mean(inputs_filtered[:, i])
        std_all = np.std(inputs_all[:, i])
        std_filt = np.std(inputs_filtered[:, i])

        # Check if mean shift is > 0.5 std
        if std_all > 1e-10:
            mean_shift = abs(mean_all - mean_filt) / std_all
            std_ratio = std_filt / std_all if std_all > 1e-10 else 1.0
            if mean_shift > 0.5 or abs(std_ratio - 1.0) > 0.5:
                issues.append(fname)

    if issues:
        return "WARN", (
            f"Normalization mismatch risk for features: {issues}. "
            f"Filtered data statistics differ significantly from full domain. "
            f"NOTE: trainModel_enhanced.py adapts normalizer on ALL data (correct). "
            f"If you see this, the guide's Section 8 advice to normalize on filtered data is WRONG."
        )
    return "OK", "Feature statistics consistent between filtered and full data."


def check_param_sample_ratio(n_params, n_samples):
    """Check if there are enough samples relative to model parameters."""
    ratio = n_samples / n_params if n_params > 0 else float("inf")
    if ratio < 5:
        return "FAIL", (
            f"Parameter-to-sample ratio dangerously low: {n_samples}/{n_params} = {ratio:.1f}x. "
            f"FIX: Reduce network size or add more training data (more cases)."
        )
    if ratio < 10:
        return "WARN", (
            f"Parameter-to-sample ratio marginal: {ratio:.1f}x (recommend >10x). "
            f"Consider reducing network size."
        )
    return "OK", f"Sample/parameter ratio healthy: {ratio:.1f}x."


def check_extreme_predictions(y_pred, lower=-5.0, upper=10.0):
    """Check for extreme predicted beta values that could crash the CFD solver."""
    n_extreme = np.sum((y_pred < lower) | (y_pred > upper))
    frac = n_extreme / len(y_pred)
    if frac > 0.01:
        return "FAIL", (
            f"{n_extreme} cells ({frac*100:.2f}%) have extreme predictions "
            f"(outside [{lower}, {upper}]). "
            f"FIX: Add output clipping or tighten outputUpperBound/outputLowerBound in DAFoam config."
        )
    if n_extreme > 0:
        return "WARN", (
            f"{n_extreme} cells with extreme predictions. "
            f"Likely fine but monitor solver stability."
        )
    return "OK", f"All predictions within safe bounds [{lower}, {upper}]."


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_diagnostics(inputs_all, inputs_filtered, y_true_all, y_pred_all,
                        filter_mask, history, feature_names, n_params,
                        output_dir=None):
    """Run all diagnostic checks and print a summary report.

    Returns list of (severity, check_name, message) tuples.
    """
    y_true_f = y_true_all[filter_mask]

    checks = [
        ("Target Imbalance",       check_target_imbalance(y_true_all)),
        ("Dead Features",          check_dead_features(inputs_all, feature_names)),
        ("Overfitting",            check_overfitting(history)),
        ("Underfitting",           check_underfitting(history)),
        ("Noisy FI Data",          check_noisy_fi_data(inputs_filtered, y_true_f, feature_names)),
        ("Normalization Mismatch", check_normalization_mismatch(inputs_all, inputs_filtered, feature_names)),
        ("Param/Sample Ratio",     check_param_sample_ratio(n_params, filter_mask.sum())),
        ("Extreme Predictions",    check_extreme_predictions(y_pred_all)),
    ]

    # Print report
    results = []
    print("\n" + "=" * 70)
    print("FAILURE MODE DIAGNOSTICS")
    print("=" * 70)
    for name, (severity, msg) in checks:
        icon = {"OK": "[OK]  ", "WARN": "[WARN]", "FAIL": "[FAIL]"}[severity]
        print(f"\n{icon} {name}")
        print(f"       {msg}")
        results.append((severity, name, msg))

    # Summary
    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    n_warn = sum(1 for s, _, _ in results if s == "WARN")
    print(f"\nSummary: {n_fail} failures, {n_warn} warnings, "
          f"{len(results) - n_fail - n_warn} OK")

    if output_dir:
        path = os.path.join(output_dir, "diagnostics_report.txt")
        with open(path, "w") as f:
            for severity, name, msg in results:
                f.write(f"[{severity}] {name}: {msg}\n")
        print(f"Report saved to {path}")

    return results


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run FIML failure diagnostics")
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    data = np.load(os.path.join(args.results_dir, "training_data.npz"))
    with open(os.path.join(args.results_dir, "training_history.json")) as f:
        history = json.load(f)
    with open(os.path.join(args.results_dir, "config.json")) as f:
        config = json.load(f)

    run_all_diagnostics(
        inputs_all=data["inputs_all"],
        inputs_filtered=data["inputs_filtered"],
        y_true_all=data["y_true_all"],
        y_pred_all=data["y_pred_all"],
        filter_mask=data["filter_mask"],
        history=history,
        feature_names=config["features"],
        n_params=config["n_params"],
        output_dir=args.results_dir,
    )
