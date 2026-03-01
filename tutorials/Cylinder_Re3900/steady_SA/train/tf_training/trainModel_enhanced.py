#!/usr/bin/env python
"""
Enhanced FIML MLP training for cylinder flow.

Fixes errors from cylinder_fiml_ml_training_guide.md and integrates
training_monitor.py and failure_diagnostics.py for systematic analysis.

Key improvements over tf_training/trainModel.py:
  - Spatial filtering + sample weighting (fixes target imbalance)
  - Normalizer adapted on ALL data, not just filtered (fixes deployment mismatch)
  - Per-feature normalization (axis=-1 instead of axis=None)
  - L2 regularization, early stopping, LR scheduling
  - Correct DAFoam normalization export (fixes guide Section 5 formula)
  - Integrated monitoring and failure diagnostics

Usage:
    mpirun -np 1 python trainModel_enhanced.py --case c1_data --nCells 25000
    mpirun -np 1 python trainModel_enhanced.py --case c1_data c2_data --nCells 25000 \
        --features PoD,VoS,chiSA,PSoSS --hidden 20,20 --beta_threshold 0.05

See also:
    feature_selection.py  -- run first to choose features
    training_monitor.py   -- regenerate plots from saved results
    failure_diagnostics.py -- run diagnostics standalone
"""

import argparse
import os
import json
import numpy as np
from mpi4py import MPI
from pyofm import PYOFM
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks

np.random.seed(42)
tf.random.set_seed(42)

# Local imports (same directory)
import training_monitor as monitor
import failure_diagnostics as diag


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cases, nCells, feature_names, beta_name):
    """Read feature fields and beta from OpenFOAM case directories."""
    ofm = PYOFM(comm=MPI.COMM_WORLD)
    all_inputs, all_outputs = [], []

    for case in cases:
        if not os.path.exists(case):
            raise FileNotFoundError(f"'{case}' not found. Run field inversion first.")

        feats = []
        for fname in feature_names:
            field = np.zeros(nCells)
            ofm.readField(fname, "volScalarField", case, field)
            feats.append(field)
        all_inputs.append(np.column_stack(feats))

        beta = np.zeros(nCells)
        ofm.readField(beta_name, "volScalarField", case, beta)
        all_outputs.append(beta)

    return np.vstack(all_inputs), np.concatenate(all_outputs)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(n_features, hidden_layers, activation, l2_reg, inputs_all):
    """Build Keras MLP with per-feature normalization.

    IMPORTANT: Normalizer is adapted on ALL data (not filtered) so that
    deployment on the full CFD domain uses consistent statistics.
    This corrects the guide's Section 8 error.
    """
    # axis=-1 = per-feature normalization (guide uses axis=None which is wrong)
    normalizer = layers.Normalization(input_shape=[n_features], axis=-1)
    normalizer.adapt(inputs_all)

    layer_list = [normalizer]
    for n_units in hidden_layers:
        layer_list.append(layers.Dense(
            units=n_units,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg),
        ))
    layer_list.append(layers.Dense(units=1))  # linear output

    model = tf.keras.Sequential(layer_list)
    return model, normalizer


def export_dafoam_normalization(normalizer, feature_names):
    """Convert TF normalizer params to DAFoam inputShift/inputScale format.

    TF Normalization: (x - mean) / sqrt(var + eps)
    DAFoam applies:   (x + inputShift) * inputScale

    Correct conversion (fixes guide Section 5 error):
        inputShift = -mean
        inputScale = 1 / sqrt(var + eps)

    The guide's Section 5 formula was WRONG:
        inputShift = -mean / sqrt(var)  <-- INCORRECT, produces x/sqrt(v) - mean/var
    """
    mean = normalizer.mean.numpy().flatten()
    var = normalizer.variance.numpy().flatten()
    eps = 1e-7

    input_shift = (-mean).tolist()
    input_scale = (1.0 / np.sqrt(var + eps)).tolist()

    print("\nDAFoam normalization parameters:")
    for i, fname in enumerate(feature_names):
        print(f"  {fname}: inputShift={input_shift[i]:.6f}, inputScale={input_scale[i]:.6f}")

    return input_shift, input_scale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Enhanced FIML MLP training")
    # Data
    parser.add_argument("--case", nargs="+", required=True, help="Case directories")
    parser.add_argument("--nCells", type=int, required=True, help="Cells per case")
    parser.add_argument("--beta_name", default="betaFINuTilda", help="Beta field name")
    parser.add_argument("--features", default="PoD,VoS,chiSA,PSoSS",
                        help="Comma-separated feature names")
    # Filtering
    parser.add_argument("--beta_threshold", type=float, default=0.05,
                        help="Filter cells where |beta-1| > threshold (0=no filter)")
    parser.add_argument("--use_weights", action="store_true", default=True,
                        help="Apply sample weighting by |beta-1|")
    parser.add_argument("--no_weights", dest="use_weights", action="store_false")
    # Architecture
    parser.add_argument("--hidden", default="20,20",
                        help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--activation", default="tanh", choices=["tanh", "relu", "sigmoid"])
    parser.add_argument("--l2_reg", type=float, default=1e-4, help="L2 regularization")
    # Training
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience_stop", type=int, default=80, help="Early stopping patience")
    parser.add_argument("--patience_lr", type=int, default=30, help="LR reduction patience")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation fraction")
    # Output
    parser.add_argument("--output_dir", default="training_results")
    args = parser.parse_args()

    feature_names = [f.strip() for f in args.features.split(",")]
    hidden_layers = [int(x) for x in args.hidden.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    print("Loading data...")
    inputs_all, outputs_all = load_data(
        args.case, args.nCells, feature_names, args.beta_name
    )
    print(f"  {inputs_all.shape[0]} cells, {inputs_all.shape[1]} features")
    print(f"  Beta range: [{outputs_all.min():.4f}, {outputs_all.max():.4f}]")

    # Feature statistics
    for i, fname in enumerate(feature_names):
        col = inputs_all[:, i]
        print(f"  {fname:<15} min={col.min():.4f} max={col.max():.4f} "
              f"mean={col.mean():.4f} std={col.std():.4f}")

    # --- Spatial filtering ---
    if args.beta_threshold > 0:
        filter_mask = np.abs(outputs_all - 1.0) > args.beta_threshold
    else:
        filter_mask = np.ones(len(outputs_all), dtype=bool)

    inputs_filt = inputs_all[filter_mask]
    outputs_filt = outputs_all[filter_mask]
    print(f"\nFiltering (|beta-1| > {args.beta_threshold}): "
          f"{filter_mask.sum()}/{len(filter_mask)} cells kept "
          f"({100*filter_mask.mean():.1f}%)")

    # --- Sample weights ---
    weights = None
    if args.use_weights and args.beta_threshold > 0:
        weights = np.abs(outputs_filt - 1.0)
        weights = weights / weights.mean()  # normalize mean to 1
        weights = np.clip(weights, 0.1, 10.0)
        print(f"  Sample weights: min={weights.min():.2f}, max={weights.max():.2f}")

    # --- Build model ---
    # Normalizer adapted on ALL data for correct deployment behavior
    model, normalizer = build_model(
        len(feature_names), hidden_layers, args.activation, args.l2_reg, inputs_all
    )
    model.summary()

    n_params = model.count_params()
    n_samples = inputs_filt.shape[0]
    print(f"\nParams: {n_params}, Filtered samples: {n_samples}, "
          f"Ratio: {n_samples/n_params:.1f}x")

    # --- Compile ---
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=args.lr),
        loss="mean_squared_error",
    )

    # --- Callbacks ---
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience_stop,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=args.patience_lr, min_lr=1e-6, verbose=1,
        ),
    ]

    # --- Train ---
    print(f"\nTraining: epochs={args.epochs}, batch={args.batch_size}, "
          f"lr={args.lr}, val_split={args.val_split}")
    history = model.fit(
        inputs_filt, outputs_filt,
        sample_weight=weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        callbacks=cb_list,
        verbose=1,
    )

    # --- Predict on ALL cells (for deployment evaluation) ---
    pred_all = model.predict(inputs_all, verbose=0).flatten()
    pred_filt = pred_all[filter_mask]

    # --- Monitoring plots ---
    metrics = monitor.generate_all_plots(
        history, outputs_all, pred_all, filter_mask, args.output_dir
    )

    # --- Failure diagnostics ---
    diag.run_all_diagnostics(
        inputs_all=inputs_all,
        inputs_filtered=inputs_filt,
        y_true_all=outputs_all,
        y_pred_all=pred_all,
        filter_mask=filter_mask,
        history=history,
        feature_names=feature_names,
        n_params=n_params,
        output_dir=args.output_dir,
    )

    # --- DAFoam normalization export ---
    input_shift, input_scale = export_dafoam_normalization(normalizer, feature_names)

    # --- Save everything ---
    # Model
    model.save(os.path.join(args.output_dir, "model"))
    print(f"\nModel saved to {args.output_dir}/model")

    # Training data for standalone re-analysis
    np.savez(
        os.path.join(args.output_dir, "training_data.npz"),
        inputs_all=inputs_all,
        inputs_filtered=inputs_filt,
        y_true_all=outputs_all,
        y_pred_all=pred_all,
        filter_mask=filter_mask,
    )

    # Training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f)

    # Config for reproducibility
    config = {
        "cases": args.case,
        "nCells": args.nCells,
        "features": feature_names,
        "beta_name": args.beta_name,
        "beta_threshold": args.beta_threshold,
        "hidden_layers": hidden_layers,
        "activation": args.activation,
        "l2_reg": args.l2_reg,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_actual": len(history.history["loss"]),
        "n_params": n_params,
        "n_samples_filtered": n_samples,
        "n_samples_total": len(outputs_all),
        "dafoam_inputShift": input_shift,
        "dafoam_inputScale": input_scale,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Write predicted beta field for ParaView
    ofm = PYOFM(comm=MPI.COMM_WORLD)
    ofm.writeField("betaFINuTildaNN", "volScalarField", pred_all[:args.nCells])

    # --- Final summary ---
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"  MSE (filtered):   {metrics['mse_filtered']:.6f}")
    print(f"  R²  (filtered):   {metrics['r2_filtered']:.4f}")
    print(f"  MSE (all cells):  {metrics['mse_all_cells']:.6f}")
    print(f"  MSE (freestream): {metrics['mse_freestream']:.6f}")
    print(f"  Max error:        {metrics['max_error_filtered']:.4f}")
    print(f"  Best val epoch:   {metrics['best_val_epoch']}")
    print(f"\nOutputs in: {args.output_dir}/")
    print(f"  model/              -- saved Keras model")
    print(f"  config.json         -- full config + DAFoam normalization params")
    print(f"  metrics.json        -- evaluation metrics")
    print(f"  training_curves.png -- loss curves")
    print(f"  parity_*.png        -- predicted vs true plots")
    print(f"  residual_analysis.png")
    print(f"  region_breakdown.png")
    print(f"  diagnostics_report.txt")


if __name__ == "__main__":
    main()
