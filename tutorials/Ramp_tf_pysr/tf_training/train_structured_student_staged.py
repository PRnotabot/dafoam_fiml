#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, losses, models, optimizers, regularizers

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
    parse_feature_list,
    resolve_branch_features,
    split_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a staged gate-sign-amplitude student for symbolic distillation."
    )
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--n-cells", type=int, default=20000)
    parser.add_argument("--raw-field", default="betaFIOmega")
    parser.add_argument("--model-path", default="model")
    parser.add_argument("--target-source", choices=("teacher", "raw"), default="teacher")
    parser.add_argument("--baseline-beta", type=float, default=1.0)
    parser.add_argument("--active-threshold", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--feature-preset", choices=tuple(FEATURE_PRESETS.keys()), default="guided")
    parser.add_argument("--gate-features", default=None)
    parser.add_argument("--sign-features", default=None)
    parser.add_argument("--amplitude-features", default=None)
    parser.add_argument("--gate-hidden", type=int, default=4)
    parser.add_argument("--sign-hidden", type=int, default=8)
    parser.add_argument("--amplitude-hidden", type=int, default=8)
    parser.add_argument("--l1", type=float, default=1.0e-5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--run-tag", default="structured_student_staged_guided_v1")
    parser.add_argument("--output-dir", default="structured_student_runs")
    return parser.parse_args()


def build_branch_model(input_dim, hidden_units, output_activation, l1_strength, X_train, name):
    regularizer = regularizers.L1(l1_strength) if l1_strength > 0.0 else None
    inputs = layers.Input(shape=(input_dim,), name=f"{name}_input")
    normalizer = layers.Normalization(axis=-1, name=f"{name}_norm")
    normalizer.adapt(X_train)
    x = normalizer(inputs)
    if hidden_units > 0:
        x = layers.Dense(
            hidden_units,
            activation="tanh",
            kernel_regularizer=regularizer,
            name=f"{name}_hidden",
        )(x)
    outputs = layers.Dense(
        1,
        activation=output_activation,
        kernel_regularizer=regularizer,
        name=name,
    )(x)
    return models.Model(inputs=inputs, outputs=outputs, name=f"{name}_model")


def train_branch(model, X_train, y_train, X_val, y_val, sample_weight, val_sample_weight, learning_rate, epochs, batch_size, patience, loss):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        weighted_metrics=[],
    )
    history = model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        validation_data=(X_val, y_val, val_sample_weight),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(5, patience // 2),
                min_lr=1.0e-5,
            ),
        ],
    )
    return history


def select_matrix(frame, columns):
    return frame[list(columns)].to_numpy().astype(np.float32)


def main():
    args = parse_args()
    np.random.seed(args.random_state)
    tf.keras.utils.set_random_seed(args.random_state)

    feature_names = tuple(args.features)
    branch_features = resolve_branch_features(args, feature_names)
    output_dir = Path(args.output_dir) / args.run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

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

    target_column = "delta_beta_teacher" if args.target_source == "teacher" else "delta_beta_raw"
    frame["delta_beta_target"] = frame[target_column]
    frame["beta_target"] = args.baseline_beta + frame["delta_beta_target"]
    frame["active_mask"] = make_active_labels(frame["delta_beta_target"], args.active_threshold)
    frame["sign_target"] = 0.0
    active_rows = frame["active_mask"] == 1
    frame.loc[active_rows, "sign_target"] = (
        frame.loc[active_rows, "delta_beta_target"] > 0.0
    ).astype(float)
    frame["amplitude_target"] = np.abs(frame["delta_beta_target"])

    train_idx, val_idx, test_idx = split_indices(
        len(frame),
        args.val_fraction,
        args.test_fraction,
        frame["active_mask"].to_numpy(),
        args.random_state,
    )
    frame["split"] = "train"
    frame.loc[val_idx, "split"] = "val"
    frame.loc[test_idx, "split"] = "test"

    train_frame = frame.iloc[train_idx].copy()
    val_frame = frame.iloc[val_idx].copy()
    test_frame = frame.iloc[test_idx].copy()

    gate_train_X = select_matrix(train_frame, branch_features["gate"])
    gate_val_X = select_matrix(val_frame, branch_features["gate"])
    gate_full_X = select_matrix(frame, branch_features["gate"])

    sign_train_frame = train_frame[train_frame["active_mask"] == 1].copy()
    sign_val_frame = val_frame[val_frame["active_mask"] == 1].copy()
    sign_full_X = select_matrix(frame, branch_features["sign"])
    sign_train_X = select_matrix(sign_train_frame, branch_features["sign"])
    sign_val_X = select_matrix(sign_val_frame, branch_features["sign"])

    amp_train_frame = train_frame[train_frame["active_mask"] == 1].copy()
    amp_val_frame = val_frame[val_frame["active_mask"] == 1].copy()
    amp_full_X = select_matrix(frame, branch_features["amplitude"])
    amp_train_X = select_matrix(amp_train_frame, branch_features["amplitude"])
    amp_val_X = select_matrix(amp_val_frame, branch_features["amplitude"])

    gate_positive = int(train_frame["active_mask"].sum())
    gate_negative = len(train_frame) - gate_positive
    gate_positive_weight = 1.0 if gate_positive == 0 else float(gate_negative / gate_positive)
    gate_train_weight = np.where(train_frame["active_mask"] > 0, gate_positive_weight, 1.0).astype(np.float32)
    gate_val_weight = np.where(val_frame["active_mask"] > 0, gate_positive_weight, 1.0).astype(np.float32)

    amplitude_scale = float(max(amp_train_frame["amplitude_target"].mean(), 1.0e-3))

    gate_model = build_branch_model(
        input_dim=len(branch_features["gate"]),
        hidden_units=args.gate_hidden,
        output_activation="sigmoid",
        l1_strength=args.l1,
        X_train=gate_train_X,
        name="gate",
    )
    gate_history = train_branch(
        gate_model,
        gate_train_X,
        train_frame["active_mask"].to_numpy().astype(np.float32),
        gate_val_X,
        val_frame["active_mask"].to_numpy().astype(np.float32),
        gate_train_weight,
        gate_val_weight,
        args.learning_rate,
        args.epochs,
        args.batch_size,
        args.patience,
        losses.BinaryCrossentropy(),
    )

    sign_model = build_branch_model(
        input_dim=len(branch_features["sign"]),
        hidden_units=args.sign_hidden,
        output_activation="sigmoid",
        l1_strength=args.l1,
        X_train=sign_train_X,
        name="sign_probability",
    )
    sign_history = train_branch(
        sign_model,
        sign_train_X,
        sign_train_frame["sign_target"].to_numpy().astype(np.float32),
        sign_val_X,
        sign_val_frame["sign_target"].to_numpy().astype(np.float32),
        None,
        None,
        args.learning_rate,
        args.epochs,
        args.batch_size,
        args.patience,
        losses.BinaryCrossentropy(),
    )

    amplitude_model = build_branch_model(
        input_dim=len(branch_features["amplitude"]),
        hidden_units=args.amplitude_hidden,
        output_activation="softplus",
        l1_strength=args.l1,
        X_train=amp_train_X,
        name="amplitude_scaled",
    )
    amp_history = train_branch(
        amplitude_model,
        amp_train_X,
        (amp_train_frame["amplitude_target"].to_numpy() / amplitude_scale).astype(np.float32),
        amp_val_X,
        (amp_val_frame["amplitude_target"].to_numpy() / amplitude_scale).astype(np.float32),
        None,
        None,
        args.learning_rate,
        args.epochs,
        args.batch_size,
        args.patience,
        losses.MeanSquaredError(),
    )

    frame["gate_student"] = gate_model.predict(gate_full_X, verbose=0).reshape(-1)
    frame["sign_probability_student"] = sign_model.predict(sign_full_X, verbose=0).reshape(-1)
    frame["sign_student"] = 2.0 * frame["sign_probability_student"] - 1.0
    frame["amplitude_student"] = amplitude_scale * amplitude_model.predict(amp_full_X, verbose=0).reshape(-1)
    frame["delta_beta_student"] = (
        frame["gate_student"] * frame["sign_student"] * frame["amplitude_student"]
    )
    frame["beta_student"] = args.baseline_beta + frame["delta_beta_student"]

    histories = {
        "gate": pd.DataFrame(gate_history.history),
        "sign": pd.DataFrame(sign_history.history),
        "amplitude": pd.DataFrame(amp_history.history),
    }
    for name, history_frame in histories.items():
        history_frame.to_csv(output_dir / f"{name}_history.csv", index=False)

    gate_model.save(output_dir / "gate_model.keras")
    sign_model.save(output_dir / "sign_model.keras")
    amplitude_model.save(output_dir / "amplitude_model.keras")
    frame.to_csv(output_dir / "dataset_with_student_predictions.csv", index=False)

    summary = {
        "run_tag": args.run_tag,
        "training_mode": "staged",
        "target_source": args.target_source,
        "features": list(feature_names),
        "feature_preset": args.feature_preset,
        "branch_features": {name: list(values) for name, values in branch_features.items()},
        "hidden_units": {
            "gate": args.gate_hidden,
            "sign": args.sign_hidden,
            "amplitude": args.amplitude_hidden,
        },
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_frame)),
        "rows_val": int(len(val_frame)),
        "rows_test": int(len(test_frame)),
        "active_fraction": float(frame["active_mask"].mean()),
        "baseline_beta": args.baseline_beta,
        "active_threshold": args.active_threshold,
        "amplitude_scale": amplitude_scale,
        "gate_param_count": int(gate_model.count_params()),
        "sign_param_count": int(sign_model.count_params()),
        "amplitude_param_count": int(amplitude_model.count_params()),
        "student_param_count": int(gate_model.count_params() + sign_model.count_params() + amplitude_model.count_params()),
        "gate_best_epoch": int(np.argmin(gate_history.history["val_loss"]) + 1),
        "sign_best_epoch": int(np.argmin(sign_history.history["val_loss"]) + 1),
        "amplitude_best_epoch": int(np.argmin(amp_history.history["val_loss"]) + 1),
    }
    if teacher_model is not None:
        summary["teacher_param_count"] = int(teacher_model.count_params())
        summary["teacher_vs_raw_full_metrics"] = compute_regression_metrics(frame["beta_raw"], frame["beta_teacher"])

    split_masks = {
        "train": frame["split"] == "train",
        "val": frame["split"] == "val",
        "test": frame["split"] == "test",
        "full": np.ones(len(frame), dtype=bool),
    }
    for split_name, mask in split_masks.items():
        subset = frame.loc[mask]
        summary[f"{split_name}_target_metrics"] = compute_regression_metrics(
            subset["delta_beta_target"], subset["delta_beta_student"]
        )
        summary[f"{split_name}_raw_metrics"] = compute_regression_metrics(
            subset["beta_raw"], subset["beta_student"]
        )
        summary[f"{split_name}_gate_metrics"] = compute_binary_metrics(
            subset["active_mask"], subset["gate_student"]
        )
        active_subset = subset[subset["active_mask"] == 1]
        summary[f"{split_name}_sign_metrics"] = compute_binary_metrics(
            active_subset["sign_target"].astype(int),
            active_subset["sign_probability_student"],
        )
        summary[f"{split_name}_amplitude_metrics"] = compute_regression_metrics(
            active_subset["amplitude_target"], active_subset["amplitude_student"]
        )
        if "beta_teacher" in subset:
            summary[f"{split_name}_teacher_metrics"] = compute_regression_metrics(
                subset["beta_teacher"], subset["beta_student"]
            )

    (output_dir / "summary.json").write_text(json.dumps(convert_value(summary), indent=2))

    print("=" * 72)
    print("Staged structured student training complete")
    print("=" * 72)
    print(f"Run directory      : {output_dir}")
    print(f"Feature preset     : {args.feature_preset}")
    print(f"Branch features    : {summary['branch_features']}")
    print(f"Student params     : {summary['student_param_count']}")
    print(f"Teacher params     : {summary.get('teacher_param_count', 0)}")
    print(f"Amplitude scale    : {summary['amplitude_scale']:.6f}")
    print(
        "Validation target  : "
        f"R2={summary['val_target_metrics']['r2']:.4f}, "
        f"RMSE={summary['val_target_metrics']['rmse']:.6f}"
    )
    print(
        "Validation raw beta: "
        f"R2={summary['val_raw_metrics']['r2']:.4f}, "
        f"RMSE={summary['val_raw_metrics']['rmse']:.6f}"
    )
    print(
        "Validation gate    : "
        f"AUC={summary['val_gate_metrics']['auc']:.4f}, "
        f"ACC={summary['val_gate_metrics']['accuracy']:.4f}"
    )
    print(
        "Validation sign    : "
        f"AUC={summary['val_sign_metrics']['auc']:.4f}, "
        f"ACC={summary['val_sign_metrics']['accuracy']:.4f}"
    )
    print(
        "Validation amp     : "
        f"R2={summary['val_amplitude_metrics']['r2']:.4f}, "
        f"RMSE={summary['val_amplitude_metrics']['rmse']:.6f}"
    )


if __name__ == "__main__":
    main()
