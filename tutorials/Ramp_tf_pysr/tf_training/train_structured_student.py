#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from mpi4py import MPI
from pyofm import PYOFM
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, losses, models, optimizers, regularizers


DEFAULT_CASES = ("c1_data", "c2_data")
DEFAULT_FEATURES = ("PoD", "VoS", "PSoSS", "KoU2")
FEATURE_PRESETS = {
    "strict": {
        "gate": ("PoD",),
        "sign": ("KoU2",),
        "amplitude": ("PSoSS",),
    },
    "guided": {
        "gate": ("PoD",),
        "sign": DEFAULT_FEATURES,
        "amplitude": DEFAULT_FEATURES,
    },
    "all": {
        "gate": DEFAULT_FEATURES,
        "sign": DEFAULT_FEATURES,
        "amplitude": DEFAULT_FEATURES,
    },
}


def parse_feature_list(value, default_value):
    if value is None:
        return tuple(default_value)
    tokens = tuple(item.strip() for item in value.split(",") if item.strip())
    if not tokens:
        raise ValueError("Feature list must be non-empty")
    return tokens


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small structured student for symbolic distillation of the tf_training beta teacher."
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
    parser.add_argument("--active-delta-weight", type=float, default=8.0)
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--gate-loss-weight", type=float, default=0.25)
    parser.add_argument("--sign-loss-weight", type=float, default=0.25)
    parser.add_argument("--amplitude-loss-weight", type=float, default=0.5)
    parser.add_argument("--run-tag", default="structured_student_guided_v1")
    parser.add_argument("--output-dir", default="structured_student_runs")
    return parser.parse_args()


def read_scalar_field(ofm, field_name, case_path, n_cells):
    values = np.zeros(n_cells)
    ofm.readField(field_name, "volScalarField", case_path, values)
    return values


def load_dataset(cases, features, raw_field, n_cells):
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("Run this script with MPI size 1.")

    ofm = PYOFM(comm=comm)
    frames = []
    for case_index, case_name in enumerate(cases):
        case_dir = Path(case_name)
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")

        data = {}
        for feature_name in features:
            data[feature_name] = read_scalar_field(ofm, feature_name, str(case_dir), n_cells)
        data["beta_raw"] = read_scalar_field(ofm, raw_field, str(case_dir), n_cells)
        frame = pd.DataFrame(data)
        frame["case_name"] = case_name
        frame["case_index"] = case_index
        frame["cell_index"] = np.arange(n_cells, dtype=int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_teacher_predictions(frame, feature_names, model_path):
    teacher_model = tf.keras.models.load_model(model_path)
    teacher_beta = teacher_model.predict(frame[list(feature_names)].to_numpy(), verbose=0).reshape(-1)
    return teacher_model, teacher_beta


def make_active_labels(delta_beta, threshold):
    return (np.abs(np.asarray(delta_beta, dtype=float)) > threshold).astype(int)


def split_indices(n_rows, val_fraction, test_fraction, labels, random_state):
    total_holdout = val_fraction + test_fraction
    if total_holdout <= 0.0 or total_holdout >= 1.0:
        raise ValueError("val_fraction + test_fraction must be in (0, 1)")

    indices = np.arange(n_rows, dtype=int)
    stratify_all = labels if len(np.unique(labels)) > 1 else None
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=total_holdout,
        random_state=random_state,
        stratify=stratify_all,
    )

    holdout_ratio = test_fraction / total_holdout
    holdout_labels = labels[holdout_idx]
    stratify_holdout = holdout_labels if len(np.unique(holdout_labels)) > 1 else None
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=holdout_ratio,
        random_state=random_state,
        stratify=stratify_holdout,
    )
    return train_idx, val_idx, test_idx


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) == 0:
        return {
            "mse": None,
            "rmse": None,
            "mae": None,
            "max_error": None,
            "r2": None,
            "correlation": None,
            "rows": 0,
        }

    diff = y_true - y_pred
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    max_error = float(np.max(np.abs(diff)))
    variance = float(np.var(y_true))
    r2 = 1.0 if variance == 0.0 else float(1.0 - mse / variance)
    correlation = 1.0 if len(y_true) < 2 else float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
        "r2": r2,
        "correlation": correlation,
        "rows": int(len(y_true)),
    }


def compute_binary_metrics(y_true, score):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    prediction = (score >= 0.5).astype(int)
    accuracy = float(np.mean(prediction == y_true)) if len(y_true) else None
    out = {"accuracy": accuracy, "rows": int(len(y_true))}
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, score))
    else:
        out["auc"] = None
    return out


def convert_value(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): convert_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert_value(item) for item in value]
    return value


def resolve_branch_features(args, feature_names):
    preset = FEATURE_PRESETS[args.feature_preset]
    gate_features = parse_feature_list(args.gate_features, preset["gate"])
    sign_features = parse_feature_list(args.sign_features, preset["sign"])
    amplitude_features = parse_feature_list(args.amplitude_features, preset["amplitude"])

    valid = set(feature_names)
    for branch_name, branch_features in (
        ("gate", gate_features),
        ("sign", sign_features),
        ("amplitude", amplitude_features),
    ):
        invalid = [name for name in branch_features if name not in valid]
        if invalid:
            raise ValueError(f"Invalid {branch_name} features: {', '.join(invalid)}")

    return {
        "gate": gate_features,
        "sign": sign_features,
        "amplitude": amplitude_features,
    }


def build_branch(x, hidden_units, output_activation, name, regularizer):
    if hidden_units > 0:
        x = layers.Dense(
            hidden_units,
            activation="tanh",
            kernel_regularizer=regularizer,
            name=f"{name}_hidden",
        )(x)
    return layers.Dense(
        1,
        activation=output_activation,
        kernel_regularizer=regularizer,
        name=name,
    )(x)


def build_model(feature_names, branch_features, hidden_units, baseline_beta, l1_strength, X_train):
    regularizer = regularizers.L1(l1_strength) if l1_strength > 0.0 else None
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    inputs = layers.Input(shape=(len(feature_names),), name="features")
    normalizer = layers.Normalization(axis=-1, name="input_norm")
    normalizer.adapt(X_train)
    normalized = normalizer(inputs)

    def select_columns(x, columns):
        indices = [feature_index[name] for name in columns]
        return tf.gather(x, indices=indices, axis=-1)

    gate_input = layers.Lambda(select_columns, arguments={"columns": branch_features["gate"]}, name="gate_input")(
        normalized
    )
    sign_input = layers.Lambda(select_columns, arguments={"columns": branch_features["sign"]}, name="sign_input")(
        normalized
    )
    amplitude_input = layers.Lambda(
        select_columns,
        arguments={"columns": branch_features["amplitude"]},
        name="amplitude_input",
    )(normalized)

    gate = build_branch(gate_input, hidden_units["gate"], "sigmoid", "gate", regularizer)
    sign_probability = build_branch(
        sign_input,
        hidden_units["sign"],
        "sigmoid",
        "sign_probability",
        regularizer,
    )
    amplitude = build_branch(amplitude_input, hidden_units["amplitude"], "softplus", "amplitude", regularizer)

    sign_smooth = layers.Lambda(lambda x: 2.0 * x - 1.0, name="sign_smooth")(sign_probability)
    delta = layers.Multiply(name="delta")([gate, sign_smooth, amplitude])
    beta = layers.Lambda(lambda x: x + baseline_beta, name="beta")(delta)

    return models.Model(
        inputs=inputs,
        outputs={
            "delta": delta,
            "gate": gate,
            "sign_probability": sign_probability,
            "amplitude": amplitude,
            "beta": beta,
        },
        name="structured_student",
    )


def make_targets(frame):
    active = frame["active_mask"].to_numpy().astype(float)
    delta = frame["delta_beta_target"].to_numpy().astype(np.float32)
    sign_target = np.zeros_like(delta, dtype=np.float32)
    if np.any(active > 0.0):
        sign_target[active > 0.0] = (delta[active > 0.0] > 0.0).astype(np.float32)
    amplitude = np.abs(delta).astype(np.float32)
    return {
        "delta": delta,
        "gate": active.astype(np.float32),
        "sign_probability": sign_target,
        "amplitude": amplitude,
    }


def make_sample_weights(frame, active_delta_weight):
    active = frame["active_mask"].to_numpy().astype(np.float32)
    positive = np.sum(active > 0.0)
    negative = len(active) - positive
    positive_weight = 1.0 if positive == 0 else float(negative / positive)
    delta_weight = np.where(active > 0.0, active_delta_weight, 1.0).astype(np.float32)
    gate_weight = np.where(active > 0.0, positive_weight, 1.0).astype(np.float32)
    active_only = active.astype(np.float32)
    return {
        "delta": delta_weight,
        "gate": gate_weight,
        "sign_probability": active_only,
        "amplitude": active_only,
    }


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

    X_train = train_frame[list(feature_names)].to_numpy().astype(np.float32)
    X_val = val_frame[list(feature_names)].to_numpy().astype(np.float32)
    X_test = test_frame[list(feature_names)].to_numpy().astype(np.float32)

    model = build_model(
        feature_names,
        branch_features,
        hidden_units={
            "gate": args.gate_hidden,
            "sign": args.sign_hidden,
            "amplitude": args.amplitude_hidden,
        },
        baseline_beta=args.baseline_beta,
        l1_strength=args.l1,
        X_train=X_train,
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.learning_rate),
        loss={
            "delta": losses.MeanSquaredError(),
            "gate": losses.BinaryCrossentropy(),
            "sign_probability": losses.BinaryCrossentropy(),
            "amplitude": losses.MeanSquaredError(),
        },
        loss_weights={
            "delta": args.delta_loss_weight,
            "gate": args.gate_loss_weight,
            "sign_probability": args.sign_loss_weight,
            "amplitude": args.amplitude_loss_weight,
        },
        weighted_metrics=[],
    )

    train_targets = make_targets(train_frame)
    val_targets = make_targets(val_frame)
    train_weights = make_sample_weights(train_frame, args.active_delta_weight)
    val_weights = make_sample_weights(val_frame, args.active_delta_weight)

    history = model.fit(
        X_train,
        train_targets,
        sample_weight=train_weights,
        validation_data=(X_val, val_targets, val_weights),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_delta_loss",
                patience=args.patience,
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_delta_loss",
                factor=0.5,
                patience=max(5, args.patience // 2),
                min_lr=1.0e-5,
            ),
        ],
    )

    predictions = model.predict(frame[list(feature_names)].to_numpy().astype(np.float32), verbose=0)
    frame["gate_student"] = predictions["gate"].reshape(-1)
    frame["sign_probability_student"] = predictions["sign_probability"].reshape(-1)
    frame["sign_student"] = 2.0 * frame["sign_probability_student"] - 1.0
    frame["amplitude_student"] = predictions["amplitude"].reshape(-1)
    frame["delta_beta_student"] = predictions["delta"].reshape(-1)
    frame["beta_student"] = predictions["beta"].reshape(-1)

    summary = {
        "run_tag": args.run_tag,
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
        "student_param_count": int(model.count_params()),
        "best_epoch": int(np.argmin(history.history["val_delta_loss"]) + 1),
        "epochs_ran": int(len(history.history["loss"])),
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
            subset["active_mask"],
            subset["gate_student"],
        )

        active_subset = subset[subset["active_mask"] == 1]
        sign_true = (active_subset["sign_target"].to_numpy() > 0.5).astype(int)
        sign_score = np.clip(active_subset["sign_probability_student"].to_numpy(), 0.0, 1.0)
        summary[f"{split_name}_sign_metrics"] = compute_binary_metrics(sign_true, sign_score)
        summary[f"{split_name}_amplitude_metrics"] = compute_regression_metrics(
            active_subset["amplitude_target"], active_subset["amplitude_student"]
        )

        if "beta_teacher" in subset:
            summary[f"{split_name}_teacher_metrics"] = compute_regression_metrics(
                subset["beta_teacher"], subset["beta_student"]
            )

    model.save(output_dir / "student_model.keras")
    pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index=False)
    frame.to_csv(output_dir / "dataset_with_student_predictions.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(convert_value(summary), indent=2))

    print("=" * 72)
    print("Structured student training complete")
    print("=" * 72)
    print(f"Run directory      : {output_dir}")
    print(f"Feature preset     : {args.feature_preset}")
    print(f"Branch features    : {summary['branch_features']}")
    print(f"Student params     : {summary['student_param_count']}")
    print(f"Teacher params     : {summary.get('teacher_param_count', 0)}")
    print(f"Active fraction    : {summary['active_fraction']:.4f}")
    print(f"Best epoch         : {summary['best_epoch']}")
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
