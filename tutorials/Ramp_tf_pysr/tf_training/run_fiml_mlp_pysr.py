#!/usr/bin/env python
"""
FIML MLP->PySR Pipeline: SR-friendly branch architecture.

Trains tiny branch MLPs (gate/sign/amplitude) on field inversion data,
then distills each branch's MLP output with PySR. The MLP smooths the
raw FI targets, giving PySR cleaner distillation targets than fitting
the noisy/binary FI data directly.

Pipeline:
  1. Load FI data (betaFIOmega + features from c1_data, c2_data)
  2. Train branch MLPs staged (gate, sign, amplitude) on raw FI targets
  3. Distill each branch MLP output with PySR (smooth, continuous targets)
  4. Report distillation gap (SR loss - MLP loss) per branch
  5. Export symbolic equations

Key insight: PySR on MLP outputs >> PySR on raw FI data.
  Raw gate target is binary {0,1} — hard for SR.
  MLP gate output is smooth [0,1] — much easier for SR to approximate.

Usage:
    cd tf_training
    python run_fiml_mlp_pysr.py --target-source raw --run-tag fiml_v1
    python run_fiml_mlp_pysr.py --skip-sr --run-tag mlp_only  # MLP stage only
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
from run_symbolic_distillation import (
    add_shared_pysr_args,
    build_pysr_model,
    resolve_pysr_runtime,
    sample_training_rows,
    sympy_to_numpy_string,
)


def parse_args():
    parser = argparse.ArgumentParser(description="FIML MLP->PySR: train branch MLPs then distill with PySR.")
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
    parser.add_argument("--sign-hidden", type=int, default=8)
    parser.add_argument("--amplitude-hidden", type=int, default=8)
    parser.add_argument("--l1", type=float, default=1.0e-5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=120)
    # PySR distillation
    parser.add_argument("--sr-sample-size", type=int, default=5000)
    parser.add_argument("--sr-active-sample-size", type=int, default=3000)
    parser.add_argument("--skip-sr", action="store_true", help="Train MLPs only, skip PySR distillation.")
    # Output
    parser.add_argument("--run-tag", default="fiml_mlp_pysr_v1")
    parser.add_argument("--output-dir", default="fiml_mlp_pysr_runs")
    add_shared_pysr_args(parser)
    return parser.parse_args()


# ====================================================================== #
# Stage 1: Branch MLP training
# ====================================================================== #


def train_branch_mlps(frame, train_frame, val_frame, branch_features, args):
    """Train gate, sign, amplitude branch MLPs. Returns models and amplitude_scale."""

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

    # Class weights for imbalanced gate
    n_pos = int(train_frame["active_mask"].sum())
    n_neg = len(train_frame) - n_pos
    pos_weight = 1.0 if n_pos == 0 else float(n_neg / n_pos)
    gate_train_w = np.where(train_frame["active_mask"] > 0, pos_weight, 1.0).astype(np.float32)
    gate_val_w = np.where(val_frame["active_mask"] > 0, pos_weight, 1.0).astype(np.float32)

    amplitude_scale = float(max(amp_train_df["amplitude_target"].mean(), 1.0e-3))

    # Gate
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
    print(f"  Gate   : best epoch {np.argmin(gate_hist.history['val_loss']) + 1}, params {gate_model.count_params()}")

    # Sign (active cells only)
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
    print(f"  Sign   : best epoch {np.argmin(sign_hist.history['val_loss']) + 1}, params {sign_model.count_params()}")

    # Amplitude (active cells only, scaled)
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
    print(f"  Amplit  : best epoch {np.argmin(amp_hist.history['val_loss']) + 1}, params {amp_model.count_params()}")

    # Full-field MLP predictions
    gate_full_X = select_matrix(frame, branch_features["gate"])
    sign_full_X = select_matrix(frame, branch_features["sign"])
    amp_full_X = select_matrix(frame, branch_features["amplitude"])

    frame["gate_mlp"] = gate_model.predict(gate_full_X, verbose=0).reshape(-1)
    frame["sign_prob_mlp"] = sign_model.predict(sign_full_X, verbose=0).reshape(-1)
    frame["sign_mlp"] = 2.0 * frame["sign_prob_mlp"] - 1.0
    frame["amplitude_mlp"] = amplitude_scale * amp_model.predict(amp_full_X, verbose=0).reshape(-1)
    frame["delta_beta_mlp"] = frame["gate_mlp"] * frame["sign_mlp"] * frame["amplitude_mlp"]
    frame["beta_mlp"] = args.baseline_beta + frame["delta_beta_mlp"]

    models = {"gate": gate_model, "sign": sign_model, "amplitude": amp_model}
    histories = {"gate": gate_hist, "sign": sign_hist, "amplitude": amp_hist}
    return models, histories, amplitude_scale


# ====================================================================== #
# Stage 2: PySR distillation of MLP outputs
# ====================================================================== #


def distill_branches_with_pysr(frame, train_idx, branch_features, args):
    """Distill each branch MLP output with PySR. Returns SR models."""
    runtime_config = resolve_pysr_runtime(args)

    # Signed gate target for PySR (continuous [-1, +1] instead of [0, 1])
    frame["gate_mlp_signed"] = 2.0 * frame["gate_mlp"] - 1.0

    # Build training data from train split with MLP predictions
    train_df = frame.iloc[train_idx].copy()
    active_train_df = train_df[train_df["active_mask"] == 1].copy()

    gate_sr_data = sample_training_rows(train_df, args.sr_sample_size, args.random_state)
    sign_sr_data = sample_training_rows(active_train_df, args.sr_active_sample_size, args.random_state)
    amp_sr_data = sample_training_rows(active_train_df, args.sr_active_sample_size, args.random_state + 1)

    output_dir = Path(args.output_dir) / args.run_tag
    sr_models = {}

    # Gate PySR
    print(f"  Gate PySR     : {len(gate_sr_data)} samples, features={list(branch_features['gate'])}")
    gate_sr = build_pysr_model(args, branch_features["gate"], output_dir, runtime_config=runtime_config)
    gate_sr.run_id = f"{args.run_tag}_gate"
    gate_sr.fit(
        gate_sr_data[list(branch_features["gate"])].to_numpy(),
        gate_sr_data["gate_mlp_signed"].to_numpy(),
        variable_names=list(branch_features["gate"]),
    )
    sr_models["gate"] = gate_sr

    # Sign PySR
    print(f"  Sign PySR     : {len(sign_sr_data)} samples, features={list(branch_features['sign'])}")
    sign_sr = build_pysr_model(args, branch_features["sign"], output_dir, runtime_config=runtime_config)
    sign_sr.run_id = f"{args.run_tag}_sign"
    sign_sr.fit(
        sign_sr_data[list(branch_features["sign"])].to_numpy(),
        sign_sr_data["sign_mlp"].to_numpy(),
        variable_names=list(branch_features["sign"]),
    )
    sr_models["sign"] = sign_sr

    # Amplitude PySR
    print(f"  Amplitude PySR: {len(amp_sr_data)} samples, features={list(branch_features['amplitude'])}")
    amp_sr = build_pysr_model(args, branch_features["amplitude"], output_dir, runtime_config=runtime_config)
    amp_sr.run_id = f"{args.run_tag}_amp"
    amp_sr.fit(
        amp_sr_data[list(branch_features["amplitude"])].to_numpy(),
        amp_sr_data["amplitude_mlp"].to_numpy(),
        variable_names=list(branch_features["amplitude"]),
    )
    sr_models["amplitude"] = amp_sr

    # SR predictions
    frame["gate_sr_signed"] = gate_sr.predict(frame[list(branch_features["gate"])].to_numpy())
    frame["gate_sr"] = np.clip(0.5 * (1.0 + frame["gate_sr_signed"]), 0.0, 1.0)
    frame["sign_sr"] = np.clip(sign_sr.predict(frame[list(branch_features["sign"])].to_numpy()), -1.0, 1.0)
    frame["amplitude_sr"] = np.maximum(0.0, amp_sr.predict(frame[list(branch_features["amplitude"])].to_numpy()))
    frame["delta_beta_sr"] = frame["gate_sr"] * frame["sign_sr"] * frame["amplitude_sr"]
    frame["beta_sr"] = args.baseline_beta + frame["delta_beta_sr"]

    return sr_models


def export_equations(sr_models, output_dir, branch_features, baseline_beta):
    """Write symbolic equations and pareto fronts to disk."""
    gate_best = sr_models["gate"].get_best()
    sign_best = sr_models["sign"].get_best()
    amp_best = sr_models["amplitude"].get_best()

    gate_raw = str(gate_best["sympy_format"])
    sign_raw = str(sign_best["sympy_format"])
    amp_raw = str(amp_best["sympy_format"])

    gate_expr = f"clip(0.5*(1 + ({gate_raw})), 0, 1)"
    sign_expr = f"clip({sign_raw}, -1, 1)"
    amp_expr = f"max(0, {amp_raw})"
    delta_expr = f"({gate_expr}) * ({sign_expr}) * ({amp_expr})"
    beta_expr = f"{baseline_beta} + {delta_expr}"

    gate_np = sympy_to_numpy_string(gate_best["sympy_format"])
    sign_np = sympy_to_numpy_string(sign_best["sympy_format"])
    amp_np = sympy_to_numpy_string(amp_best["sympy_format"])

    gate_vars = ", ".join(branch_features["gate"])
    sign_vars = ", ".join(branch_features["sign"])
    amp_vars = ", ".join(branch_features["amplitude"])
    all_vars_set = list(branch_features["gate"]) + list(branch_features["sign"]) + list(branch_features["amplitude"])
    all_vars = ", ".join(dict.fromkeys(all_vars_set))  # unique, order-preserving

    python_lines = [
        '"""Auto-generated FIML MLP->PySR symbolic equation."""',
        "",
        "import numpy as np",
        "",
        "",
        f"def gate_raw({gate_vars}):",
        f"    return {gate_np}",
        "",
        "",
        f"def gate({gate_vars}):",
        f"    return np.clip(0.5 * (1.0 + gate_raw({gate_vars})), 0.0, 1.0)",
        "",
        "",
        f"def sign_value({sign_vars}):",
        f"    return np.clip({sign_np}, -1.0, 1.0)",
        "",
        "",
        f"def amplitude({amp_vars}):",
        f"    return np.maximum(0.0, {amp_np})",
        "",
        "",
        f"def delta_beta({all_vars}):",
        f"    return gate({gate_vars}) * sign_value({sign_vars}) * amplitude({amp_vars})",
        "",
        "",
        f"def beta_fiomega({all_vars}):",
        f"    return {baseline_beta} + delta_beta({all_vars})",
        "",
    ]
    (output_dir / "equation.py").write_text("\n".join(python_lines))
    (output_dir / "gate_equation.txt").write_text(gate_expr + "\n")
    (output_dir / "sign_equation.txt").write_text(sign_expr + "\n")
    (output_dir / "amplitude_equation.txt").write_text(amp_expr + "\n")
    (output_dir / "equation_beta.txt").write_text(beta_expr + "\n")

    for name in ("gate", "sign", "amplitude"):
        model = sr_models[name]
        payload = []
        if model.equations_ is not None:
            for _, row in model.equations_.iterrows():
                payload.append({col: convert_value(row[col]) for col in model.equations_.columns})
        (output_dir / f"pareto_{name}.json").write_text(json.dumps(payload, indent=2))

    return {
        "gate_equation": gate_expr,
        "sign_equation": sign_expr,
        "amplitude_equation": amp_expr,
        "beta_equation": beta_expr,
        "gate_best_loss": float(gate_best["loss"]),
        "sign_best_loss": float(sign_best["loss"]),
        "amplitude_best_loss": float(amp_best["loss"]),
        "gate_best_complexity": int(gate_best["complexity"]),
        "sign_best_complexity": int(sign_best["complexity"]),
        "amplitude_best_complexity": int(amp_best["complexity"]),
    }


# ====================================================================== #
# Metrics
# ====================================================================== #


def compute_all_metrics(frame, args, mlp_params, amplitude_scale, histories, teacher_model):
    """Compute MLP, SR, and distillation-gap metrics across splits."""
    summary = {
        "run_tag": args.run_tag,
        "pipeline": "fiml_mlp_pysr",
        "target_source": args.target_source,
        "features": list(args.features),
        "feature_preset": args.feature_preset,
        "branch_features": {k: list(v) for k, v in resolve_branch_features(args, tuple(args.features)).items()},
        "hidden_units": {"gate": args.gate_hidden, "sign": args.sign_hidden, "amplitude": args.amplitude_hidden},
        "mlp_param_count": mlp_params,
        "amplitude_scale": amplitude_scale,
        "rows_total": int(len(frame)),
        "active_fraction": float(frame["active_mask"].mean()),
        "baseline_beta": args.baseline_beta,
        "skip_sr": args.skip_sr,
    }

    for name, hist in histories.items():
        summary[f"{name}_best_epoch"] = int(np.argmin(hist.history["val_loss"]) + 1)

    if teacher_model is not None:
        summary["teacher_param_count"] = int(teacher_model.count_params())
        summary["teacher_vs_raw"] = compute_regression_metrics(frame["beta_raw"], frame["beta_teacher"])

    has_sr = "delta_beta_sr" in frame.columns

    for split_name in ("train", "val", "test"):
        s = frame[frame["split"] == split_name]

        # MLP vs FI target
        summary[f"{split_name}_mlp_vs_target"] = compute_regression_metrics(
            s["delta_beta_target"], s["delta_beta_mlp"]
        )
        summary[f"{split_name}_mlp_vs_raw"] = compute_regression_metrics(s["beta_raw"], s["beta_mlp"])

        # SR vs FI target (if SR was run)
        if has_sr:
            summary[f"{split_name}_sr_vs_target"] = compute_regression_metrics(
                s["delta_beta_target"], s["delta_beta_sr"]
            )
            summary[f"{split_name}_sr_vs_raw"] = compute_regression_metrics(s["beta_raw"], s["beta_sr"])

            # Distillation gap: how well SR reproduces the MLP
            summary[f"{split_name}_distill_gap"] = compute_regression_metrics(
                s["delta_beta_mlp"], s["delta_beta_sr"]
            )

        # Gate classification
        summary[f"{split_name}_gate"] = compute_binary_metrics(s["active_mask"], s["gate_mlp"])

        # Branch metrics on active cells
        a = s[s["active_mask"] == 1]
        if len(a) > 0:
            summary[f"{split_name}_sign"] = compute_binary_metrics(
                a["sign_target"].astype(int), a["sign_prob_mlp"]
            )
            summary[f"{split_name}_amplitude"] = compute_regression_metrics(
                a["amplitude_target"], a["amplitude_mlp"]
            )

    return summary


# ====================================================================== #
# Main
# ====================================================================== #


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

    # ---- Stage 1: Branch MLPs ---------------------------------------- #
    print("=" * 72)
    print("Stage 1: Training branch MLPs (gate / sign / amplitude)")
    print("=" * 72)
    print(f"  Features       : {list(feature_names)}")
    print(f"  Branch features: { {k: list(v) for k, v in branch_features.items()} }")
    print(f"  Hidden units   : gate={args.gate_hidden} sign={args.sign_hidden} amp={args.amplitude_hidden}")
    print()

    mlp_models, histories, amplitude_scale = train_branch_mlps(frame, train_frame, val_frame, branch_features, args)

    mlp_params = sum(m.count_params() for m in mlp_models.values())
    print(f"\n  Total MLP params : {mlp_params}")
    print(f"  Amplitude scale  : {amplitude_scale:.6f}")

    # Save MLP artifacts
    for name, model in mlp_models.items():
        model.save(output_dir / f"{name}_model.keras")
    for name, hist in histories.items():
        pd.DataFrame(hist.history).to_csv(output_dir / f"{name}_history.csv", index=False)

    # ---- Stage 2: PySR distillation ---------------------------------- #
    sr_models = None
    if not args.skip_sr:
        print("\n" + "=" * 72)
        print("Stage 2: PySR distillation of branch MLP outputs")
        print("=" * 72)
        print("  (Distilling smooth MLP outputs, not raw FI targets)")
        print()

        sr_models = distill_branches_with_pysr(frame, train_idx, branch_features, args)
        eq_info = export_equations(sr_models, output_dir, branch_features, args.baseline_beta)
    else:
        print("\n  Skipping PySR (--skip-sr). MLP results only.")
        eq_info = {}

    # ---- Metrics and output ------------------------------------------ #
    summary = compute_all_metrics(frame, args, mlp_params, amplitude_scale, histories, teacher_model)
    summary.update(eq_info)

    frame.to_csv(output_dir / "dataset_full.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(convert_value(summary), indent=2))

    # ---- Print ------------------------------------------------------- #
    print("\n" + "=" * 72)
    print("FIML MLP->PySR pipeline complete")
    print("=" * 72)
    print(f"  Output           : {output_dir}")
    print(f"  MLP params       : {mlp_params}")
    print(f"  Target source    : {args.target_source}")

    val_mlp = summary.get("val_mlp_vs_target", {})
    print(f"\n  MLP validation (vs target):")
    print(f"    R2={val_mlp.get('r2', 'N/A'):.4f}  RMSE={val_mlp.get('rmse', 'N/A'):.6f}")

    val_mlp_raw = summary.get("val_mlp_vs_raw", {})
    print(f"  MLP validation (vs raw beta):")
    print(f"    R2={val_mlp_raw.get('r2', 'N/A'):.4f}  RMSE={val_mlp_raw.get('rmse', 'N/A'):.6f}")

    if not args.skip_sr:
        val_sr = summary.get("val_sr_vs_target", {})
        print(f"\n  SR validation (vs target):")
        print(f"    R2={val_sr.get('r2', 'N/A'):.4f}  RMSE={val_sr.get('rmse', 'N/A'):.6f}")

        val_gap = summary.get("val_distill_gap", {})
        print(f"  Distillation gap (SR vs MLP):")
        print(f"    R2={val_gap.get('r2', 'N/A'):.4f}  RMSE={val_gap.get('rmse', 'N/A'):.6f}")

        print(f"\n  Equations:")
        print(f"    Gate      : {eq_info.get('gate_equation', 'N/A')}")
        print(f"    Sign      : {eq_info.get('sign_equation', 'N/A')}")
        print(f"    Amplitude : {eq_info.get('amplitude_equation', 'N/A')}")

    g = summary.get("val_gate", {})
    s = summary.get("val_sign", {})
    a = summary.get("val_amplitude", {})
    print(f"\n  Branch quality:")
    print(f"    Gate      AUC={g.get('auc', 'N/A')}  ACC={g.get('accuracy', 'N/A')}")
    print(f"    Sign      AUC={s.get('auc', 'N/A')}  ACC={s.get('accuracy', 'N/A')}")
    print(f"    Amplitude R2={a.get('r2', 'N/A')}  RMSE={a.get('rmse', 'N/A')}")


if __name__ == "__main__":
    main()
