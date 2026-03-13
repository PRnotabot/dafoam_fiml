#!/usr/bin/env python
"""
Train branch MLPs (gate/sign/amplitude) for FIML symbolic distillation.

Trains three tiny, feature-restricted MLPs on field inversion data:
  gate(PoD) x sign(features) x amplitude(features) -> delta_beta

Outputs are consumed by distill_fiml_branch_mlp.py for PySR distillation.
The branch models are frozen after this step so that PySR tuning does not
couple with teacher variance.

Usage:
    cd tf_training
    python train_fiml_branch_mlp.py --run-tag fiml_v1
    python train_fiml_branch_mlp.py --target-source teacher --run-tag fiml_teacher_v1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses

from train_structured_student import (
    DEFAULT_CASES,
    DEFAULT_FEATURES,
    FEATURE_PRESETS,
    compute_binary_metrics,
    compute_regression_metrics,
    convert_value,
    load_dataset,
    load_teacher_predictions,
    make_active_labels,
    resolve_branch_features,
    split_indices,
)
from train_structured_student_staged import (
    build_branch_model,
    select_matrix,
    train_branch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train branch MLPs for FIML symbolic distillation.")
    # Data
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--n-cells", type=int, default=20000)
    parser.add_argument("--raw-field", default="betaFIOmega")
    parser.add_argument("--model-path", default="model")
    parser.add_argument("--target-source", choices=("teacher", "raw"), default="raw")
    parser.add_argument("--baseline-beta", type=float, default=1.0)
    parser.add_argument("--active-threshold", type=float, default=0.01)
    # Splits
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=7)
    # Branch MLP architecture
    parser.add_argument("--feature-preset", choices=tuple(FEATURE_PRESETS.keys()), default="guided")
    parser.add_argument("--gate-features", default=None)
    parser.add_argument("--sign-features", default=None)
    parser.add_argument("--amplitude-features", default=None)
    parser.add_argument("--gate-hidden", type=int, default=4)
    parser.add_argument("--sign-hidden", type=int, default=6)
    parser.add_argument("--amplitude-hidden", type=int, default=8)
    parser.add_argument("--l1", type=float, default=1.0e-5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=120)
    # Output
    parser.add_argument("--run-tag", default="fiml_branch_mlp_v1")
    parser.add_argument("--output-dir", default="fiml_mlp_runs")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.random_state)
    tf.keras.utils.set_random_seed(args.random_state)

    feature_names = tuple(args.features)
    branch_features = resolve_branch_features(args, feature_names)
    output_dir = Path(args.output_dir) / args.run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data --------------------------------------------------- #
    frame = load_dataset(args.cases, feature_names, args.raw_field, args.n_cells)
    frame["delta_beta_raw"] = frame["beta_raw"] - args.baseline_beta

    teacher_model = None
    model_path = Path(args.model_path)
    if model_path.exists():
        teacher_model, teacher_beta = load_teacher_predictions(frame, feature_names, str(model_path))
        frame["beta_teacher"] = teacher_beta
        frame["delta_beta_teacher"] = frame["beta_teacher"] - args.baseline_beta
    elif args.target_source == "teacher":
        raise FileNotFoundError(f"Teacher model not found at {model_path}")

    target_col = "delta_beta_teacher" if args.target_source == "teacher" else "delta_beta_raw"
    frame["delta_beta_target"] = frame[target_col]
    frame["beta_target"] = args.baseline_beta + frame["delta_beta_target"]
    frame["active_mask"] = make_active_labels(frame["delta_beta_target"], args.active_threshold)
    frame["sign_target"] = 0.0
    active = frame["active_mask"] == 1
    frame.loc[active, "sign_target"] = (frame.loc[active, "delta_beta_target"] > 0.0).astype(float)
    frame["amplitude_target"] = np.abs(frame["delta_beta_target"])

    # ---- Split ------------------------------------------------------- #
    train_idx, val_idx, test_idx = split_indices(
        len(frame), args.val_fraction, args.test_fraction, frame["active_mask"].to_numpy(), args.random_state
    )
    frame["split"] = "train"
    frame.loc[val_idx, "split"] = "val"
    frame.loc[test_idx, "split"] = "test"

    train_frame = frame.iloc[train_idx].copy()
    val_frame = frame.iloc[val_idx].copy()

    # ---- Prepare branch data ----------------------------------------- #
    gate_train_X = select_matrix(train_frame, branch_features["gate"])
    gate_val_X = select_matrix(val_frame, branch_features["gate"])

    sign_train_df = train_frame[train_frame["active_mask"] == 1].copy()
    sign_val_df = val_frame[val_frame["active_mask"] == 1].copy()
    sign_train_X = select_matrix(sign_train_df, branch_features["sign"])
    sign_val_X = select_matrix(sign_val_df, branch_features["sign"])

    amp_train_df = train_frame[train_frame["active_mask"] == 1].copy()
    amp_val_df = val_frame[val_frame["active_mask"] == 1].copy()
    amp_train_X = select_matrix(amp_train_df, branch_features["amplitude"])
    amp_val_X = select_matrix(amp_val_df, branch_features["amplitude"])

    n_pos = int(train_frame["active_mask"].sum())
    n_neg = len(train_frame) - n_pos
    pos_weight = 1.0 if n_pos == 0 else float(n_neg / n_pos)
    gate_train_w = np.where(train_frame["active_mask"] > 0, pos_weight, 1.0).astype(np.float32)
    gate_val_w = np.where(val_frame["active_mask"] > 0, pos_weight, 1.0).astype(np.float32)

    amplitude_scale = float(max(amp_train_df["amplitude_target"].mean(), 1.0e-3))

    # ---- Train branches ---------------------------------------------- #
    print("=" * 72)
    print("Training branch MLPs (gate / sign / amplitude)")
    print("=" * 72)
    print(f"  Features       : {list(feature_names)}")
    print(f"  Branch features: { {k: list(v) for k, v in branch_features.items()} }")
    print(f"  Hidden units   : gate={args.gate_hidden} sign={args.sign_hidden} amp={args.amplitude_hidden}")
    print()

    gate_model = build_branch_model(
        len(branch_features["gate"]), args.gate_hidden, "sigmoid", args.l1, gate_train_X, "gate"
    )
    gate_hist = train_branch(
        gate_model,
        gate_train_X,
        train_frame["active_mask"].to_numpy().astype(np.float32),
        gate_val_X,
        val_frame["active_mask"].to_numpy().astype(np.float32),
        gate_train_w,
        gate_val_w,
        args.learning_rate,
        args.epochs,
        args.batch_size,
        args.patience,
        losses.BinaryCrossentropy(),
    )
    print(f"  Gate      : best epoch {np.argmin(gate_hist.history['val_loss']) + 1}, params {gate_model.count_params()}")

    sign_model = build_branch_model(
        len(branch_features["sign"]), args.sign_hidden, "sigmoid", args.l1, sign_train_X, "sign_probability"
    )
    sign_hist = train_branch(
        sign_model,
        sign_train_X,
        sign_train_df["sign_target"].to_numpy().astype(np.float32),
        sign_val_X,
        sign_val_df["sign_target"].to_numpy().astype(np.float32),
        None,
        None,
        args.learning_rate,
        args.epochs,
        args.batch_size,
        args.patience,
        losses.BinaryCrossentropy(),
    )
    print(f"  Sign      : best epoch {np.argmin(sign_hist.history['val_loss']) + 1}, params {sign_model.count_params()}")

    amp_model = build_branch_model(
        len(branch_features["amplitude"]), args.amplitude_hidden, "softplus", args.l1, amp_train_X, "amplitude_scaled"
    )
    amp_hist = train_branch(
        amp_model,
        amp_train_X,
        (amp_train_df["amplitude_target"].to_numpy() / amplitude_scale).astype(np.float32),
        amp_val_X,
        (amp_val_df["amplitude_target"].to_numpy() / amplitude_scale).astype(np.float32),
        None,
        None,
        args.learning_rate,
        args.epochs,
        args.batch_size,
        args.patience,
        losses.MeanSquaredError(),
    )
    print(
        f"  Amplitude : best epoch {np.argmin(amp_hist.history['val_loss']) + 1}, params {amp_model.count_params()}"
    )

    # ---- Full-field predictions -------------------------------------- #
    gate_full_X = select_matrix(frame, branch_features["gate"])
    sign_full_X = select_matrix(frame, branch_features["sign"])
    amp_full_X = select_matrix(frame, branch_features["amplitude"])

    frame["gate_mlp"] = gate_model.predict(gate_full_X, verbose=0).reshape(-1)
    frame["sign_prob_mlp"] = sign_model.predict(sign_full_X, verbose=0).reshape(-1)
    frame["sign_mlp"] = 2.0 * frame["sign_prob_mlp"] - 1.0
    frame["amplitude_mlp"] = amplitude_scale * amp_model.predict(amp_full_X, verbose=0).reshape(-1)
    frame["delta_beta_mlp"] = frame["gate_mlp"] * frame["sign_mlp"] * frame["amplitude_mlp"]
    frame["beta_mlp"] = args.baseline_beta + frame["delta_beta_mlp"]

    mlp_params = gate_model.count_params() + sign_model.count_params() + amp_model.count_params()

    # ---- Save artifacts ---------------------------------------------- #
    gate_model.save(output_dir / "gate_model.keras")
    sign_model.save(output_dir / "sign_model.keras")
    amplitude_model_path = output_dir / "amplitude_model.keras"
    amp_model.save(amplitude_model_path)

    for name, hist in [("gate", gate_hist), ("sign", sign_hist), ("amplitude", amp_hist)]:
        pd.DataFrame(hist.history).to_csv(output_dir / f"{name}_history.csv", index=False)

    frame.to_csv(output_dir / "predictions.csv", index=False)

    # ---- Metrics ----------------------------------------------------- #
    summary = {
        "run_tag": args.run_tag,
        "target_source": args.target_source,
        "features": list(feature_names),
        "feature_preset": args.feature_preset,
        "branch_features": {k: list(v) for k, v in branch_features.items()},
        "hidden_units": {"gate": args.gate_hidden, "sign": args.sign_hidden, "amplitude": args.amplitude_hidden},
        "mlp_param_count": int(mlp_params),
        "gate_param_count": int(gate_model.count_params()),
        "sign_param_count": int(sign_model.count_params()),
        "amplitude_param_count": int(amp_model.count_params()),
        "amplitude_scale": amplitude_scale,
        "baseline_beta": args.baseline_beta,
        "active_threshold": args.active_threshold,
        "rows_total": int(len(frame)),
        "active_fraction": float(frame["active_mask"].mean()),
        "gate_best_epoch": int(np.argmin(gate_hist.history["val_loss"]) + 1),
        "sign_best_epoch": int(np.argmin(sign_hist.history["val_loss"]) + 1),
        "amplitude_best_epoch": int(np.argmin(amp_hist.history["val_loss"]) + 1),
    }
    if teacher_model is not None:
        summary["teacher_param_count"] = int(teacher_model.count_params())

    for split_name in ("train", "val", "test"):
        s = frame[frame["split"] == split_name]
        summary[f"{split_name}_mlp_vs_target"] = compute_regression_metrics(
            s["delta_beta_target"], s["delta_beta_mlp"]
        )
        summary[f"{split_name}_mlp_vs_raw"] = compute_regression_metrics(s["beta_raw"], s["beta_mlp"])
        summary[f"{split_name}_gate"] = compute_binary_metrics(s["active_mask"], s["gate_mlp"])
        a = s[s["active_mask"] == 1]
        if len(a) > 0:
            summary[f"{split_name}_sign"] = compute_binary_metrics(a["sign_target"].astype(int), a["sign_prob_mlp"])
            summary[f"{split_name}_amplitude"] = compute_regression_metrics(a["amplitude_target"], a["amplitude_mlp"])

    (output_dir / "summary.json").write_text(json.dumps(convert_value(summary), indent=2))

    # ---- Print ------------------------------------------------------- #
    print(f"\n  Total params     : {mlp_params}")
    print(f"  Amplitude scale  : {amplitude_scale:.6f}")
    print(f"  Output           : {output_dir}")

    val_t = summary.get("val_mlp_vs_target", {})
    val_g = summary.get("val_gate", {})
    val_s = summary.get("val_sign", {})
    val_a = summary.get("val_amplitude", {})
    print(f"\n  Validation:")
    print(f"    vs target  R2={val_t.get('r2', 'N/A'):.4f}  RMSE={val_t.get('rmse', 'N/A'):.6f}")
    print(f"    Gate       AUC={val_g.get('auc', 'N/A')}  ACC={val_g.get('accuracy', 'N/A')}")
    print(f"    Sign       AUC={val_s.get('auc', 'N/A')}  ACC={val_s.get('accuracy', 'N/A')}")
    print(f"    Amplitude  R2={val_a.get('r2', 'N/A')}  RMSE={val_a.get('rmse', 'N/A')}")
    print(f"\n  Next: python distill_fiml_branch_mlp.py --mlp-run-dir {output_dir}")


if __name__ == "__main__":
    main()
