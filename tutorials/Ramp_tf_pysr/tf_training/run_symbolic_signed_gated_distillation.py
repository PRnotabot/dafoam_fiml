#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_symbolic_distillation import (
    DEFAULT_CASES,
    DEFAULT_FEATURES,
    add_shared_pysr_args,
    build_pysr_model,
    compute_metrics,
    convert_value,
    load_dataset,
    load_teacher_predictions,
    make_active_labels,
    resolve_pysr_runtime,
    sample_training_rows,
    split_indices,
    sympy_to_numpy_string,
)
from train_structured_student import FEATURE_PRESETS, resolve_branch_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sign-aware gated symbolic distillation for the Ramp tf_training beta teacher."
    )
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--n-cells", type=int, default=20000)
    parser.add_argument("--raw-field", default="betaFIOmega")
    parser.add_argument("--model-path", default="model")
    parser.add_argument("--structured-student-run", default=None)
    parser.add_argument("--target-source", choices=("teacher", "raw", "structured_student"), default="teacher")
    parser.add_argument("--baseline-beta", type=float, default=1.0)
    parser.add_argument("--active-threshold", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--active-sample-size", type=int, default=4000)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--feature-preset", choices=tuple(FEATURE_PRESETS.keys()), default="guided")
    parser.add_argument("--gate-features", default=None)
    parser.add_argument("--sign-features", default=None)
    parser.add_argument("--amplitude-features", default=None)
    parser.add_argument("--run-tag", default="signed_gated_teacher_default")
    parser.add_argument("--output-dir", default="sr_results")
    add_shared_pysr_args(parser)
    return parser.parse_args()


def export_signed_gated_artifacts(
    gate_model,
    sign_model,
    amp_model,
    run_directory,
    feature_names,
    baseline_beta,
):
    gate_best = gate_model.get_best()
    sign_best = sign_model.get_best()
    amp_best = amp_model.get_best()

    gate_raw_expr = str(gate_best["sympy_format"])
    sign_expr = str(sign_best["sympy_format"])
    amp_expr = str(amp_best["sympy_format"])
    gate_expr = f"0.5*(1 + ({gate_raw_expr}))"
    gate_expr_clipped = f"clip({gate_expr}, 0, 1)"
    sign_expr_clipped = f"clip(({sign_expr}), -1, 1)"
    amp_expr_clipped = f"max(0, ({amp_expr}))"
    delta_expr = f"({gate_expr_clipped}) * ({sign_expr_clipped}) * ({amp_expr_clipped})"
    beta_expr = f"{baseline_beta} + ({delta_expr})"

    gate_numpy = sympy_to_numpy_string(gate_best["sympy_format"])
    sign_numpy = sympy_to_numpy_string(sign_best["sympy_format"])
    amp_numpy = sympy_to_numpy_string(amp_best["sympy_format"])

    python_lines = [
        '"""Auto-generated sign-aware gated symbolic distillation equation."""',
        "",
        "import numpy as np",
        "",
        f"def gate_raw({', '.join(feature_names)}):",
        f"    return {gate_numpy}",
        "",
        f"def gate_active({', '.join(feature_names)}):",
        f"    return np.clip(0.5 * (1.0 + gate_raw({', '.join(feature_names)})), 0.0, 1.0)",
        "",
        f"def sign_active({', '.join(feature_names)}):",
        f"    return np.clip({sign_numpy}, -1.0, 1.0)",
        "",
        f"def amplitude_active({', '.join(feature_names)}):",
        f"    return np.maximum(0.0, {amp_numpy})",
        "",
        f"def delta_beta({', '.join(feature_names)}):",
        f"    return gate_active({', '.join(feature_names)}) * sign_active({', '.join(feature_names)}) * amplitude_active({', '.join(feature_names)})",
        "",
        f"def beta_fiomega({', '.join(feature_names)}):",
        f"    return {baseline_beta} + delta_beta({', '.join(feature_names)})",
        "",
    ]
    (run_directory / "equation.py").write_text("\n".join(python_lines))
    (run_directory / "gate_equation_sympy.txt").write_text(gate_expr + "\n")
    (run_directory / "sign_equation_sympy.txt").write_text(sign_expr + "\n")
    (run_directory / "amplitude_equation_sympy.txt").write_text(amp_expr + "\n")
    (run_directory / "equation_beta.txt").write_text(beta_expr + "\n")

    pareto_payloads = {
        "pareto_gate.json": gate_model.equations_,
        "pareto_sign.json": sign_model.equations_,
        "pareto_amplitude.json": amp_model.equations_,
    }
    for file_name, dataframe in pareto_payloads.items():
        payload = []
        if dataframe is not None:
            for _, row in dataframe.iterrows():
                payload.append({column: convert_value(row[column]) for column in dataframe.columns})
        (run_directory / file_name).write_text(json.dumps(payload, indent=2))

    return {
        "gate_expression": gate_expr,
        "gate_expression_clipped": gate_expr_clipped,
        "sign_expression": sign_expr,
        "amplitude_expression": amp_expr,
        "delta_expression": delta_expr,
        "beta_expression": beta_expr,
        "gate_best_equation": str(gate_best["equation"]),
        "sign_best_equation": str(sign_best["equation"]),
        "amplitude_best_equation": str(amp_best["equation"]),
        "gate_best_loss": float(gate_best["loss"]),
        "sign_best_loss": float(sign_best["loss"]),
        "amplitude_best_loss": float(amp_best["loss"]),
        "gate_best_complexity": int(gate_best["complexity"]),
        "sign_best_complexity": int(sign_best["complexity"]),
        "amplitude_best_complexity": int(amp_best["complexity"]),
    }


def load_structured_student_reference(frame, run_dir):
    run_path = Path(run_dir)
    summary_path = run_path / "summary.json"
    dataset_path = run_path / "dataset_with_student_predictions.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Structured student summary not found: {summary_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Structured student dataset not found: {dataset_path}")

    summary = json.loads(summary_path.read_text())
    student_frame = pd.read_csv(dataset_path)
    required_columns = {"case_name", "cell_index", "beta_student"}
    missing_columns = sorted(required_columns - set(student_frame.columns))
    if missing_columns:
        raise ValueError(
            f"Structured student dataset missing required columns: {', '.join(missing_columns)}"
        )

    merge_columns = ["case_name", "cell_index"]
    merged = frame.merge(
        student_frame[merge_columns + ["beta_student"]],
        on=merge_columns,
        how="left",
        validate="one_to_one",
    )
    if merged["beta_student"].isna().any():
        raise ValueError(
            "Structured student dataset does not cover the requested cases/cells for this symbolic run."
        )
    return summary, merged["beta_student"].to_numpy()


def main():
    args = parse_args()
    feature_names = tuple(args.features)
    branch_features = resolve_branch_features(args, feature_names)
    run_directory = Path(args.output_dir) / args.run_tag
    run_directory.mkdir(parents=True, exist_ok=True)
    runtime_config = resolve_pysr_runtime(args)
    populations_note = ""
    if args.populations < runtime_config["worker_count"]:
        populations_note = (
            f"populations ({args.populations}) < workers ({runtime_config['worker_count']}); "
            "some workers may be underutilized."
        )

    frame = load_dataset(args.cases, feature_names, args.raw_field, args.n_cells)
    frame["delta_beta_raw"] = frame["beta_raw"] - args.baseline_beta

    teacher_model = None
    teacher_param_count = None
    teacher_reference = ""
    model_path = Path(args.model_path)
    if args.target_source == "teacher":
        if not model_path.exists():
            raise FileNotFoundError(f"Teacher model not found at {model_path}")
        teacher_model, teacher_beta = load_teacher_predictions(frame, feature_names, str(model_path))
        frame["beta_teacher"] = teacher_beta
        frame["delta_beta_teacher"] = frame["beta_teacher"] - args.baseline_beta
        teacher_param_count = int(teacher_model.count_params())
        teacher_reference = str(model_path)
    elif args.target_source == "structured_student":
        if args.structured_student_run is None:
            raise ValueError("`--structured-student-run` is required for `--target-source structured_student`.")
        student_summary, student_beta = load_structured_student_reference(frame, args.structured_student_run)
        frame["beta_teacher"] = student_beta
        frame["delta_beta_teacher"] = frame["beta_teacher"] - args.baseline_beta
        teacher_param_count = int(student_summary.get("student_param_count", 0))
        teacher_reference = str(Path(args.structured_student_run))

    target_column = "delta_beta_teacher" if args.target_source in ("teacher", "structured_student") else "delta_beta_raw"
    frame["delta_beta_target"] = frame[target_column]
    frame["beta_target"] = args.baseline_beta + frame["delta_beta_target"]
    frame["active_mask"] = make_active_labels(frame["delta_beta_target"].to_numpy(), args.active_threshold)
    frame["gate_target_signed"] = 2.0 * frame["active_mask"] - 1.0

    active_only = frame["active_mask"] == 1
    frame["sign_target"] = 0.0
    frame.loc[active_only, "sign_target"] = np.sign(frame.loc[active_only, "delta_beta_target"])
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
    active_train = train_frame[train_frame["active_mask"] == 1].copy()

    gate_train = sample_training_rows(train_frame, args.sample_size, args.random_state)
    sign_train = sample_training_rows(active_train, args.active_sample_size, args.random_state)
    amp_train = sample_training_rows(active_train, args.active_sample_size, args.random_state + 1)

    gate_args = argparse.Namespace(**vars(args))
    gate_args.run_tag = f"{args.run_tag}_gate"
    sign_args = argparse.Namespace(**vars(args))
    sign_args.run_tag = f"{args.run_tag}_sign"
    amp_args = argparse.Namespace(**vars(args))
    amp_args.run_tag = f"{args.run_tag}_amp"

    gate_feature_names = tuple(branch_features["gate"])
    sign_feature_names = tuple(branch_features["sign"])
    amplitude_feature_names = tuple(branch_features["amplitude"])

    gate_model = build_pysr_model(gate_args, gate_feature_names, run_directory, runtime_config=runtime_config)
    gate_model.fit(
        gate_train[list(gate_feature_names)].to_numpy(),
        gate_train["gate_target_signed"].to_numpy(),
        variable_names=list(gate_feature_names),
    )

    sign_model = build_pysr_model(sign_args, sign_feature_names, run_directory, runtime_config=runtime_config)
    sign_model.fit(
        sign_train[list(sign_feature_names)].to_numpy(),
        sign_train["sign_target"].to_numpy(),
        variable_names=list(sign_feature_names),
    )

    amp_model = build_pysr_model(amp_args, amplitude_feature_names, run_directory, runtime_config=runtime_config)
    amp_model.fit(
        amp_train[list(amplitude_feature_names)].to_numpy(),
        amp_train["amplitude_target"].to_numpy(),
        variable_names=list(amplitude_feature_names),
    )

    frame["gate_sr_signed"] = gate_model.predict(frame[list(gate_feature_names)].to_numpy())
    frame["gate_sr"] = np.clip(0.5 * (1.0 + frame["gate_sr_signed"]), 0.0, 1.0)
    frame["sign_sr"] = np.clip(sign_model.predict(frame[list(sign_feature_names)].to_numpy()), -1.0, 1.0)
    frame["amplitude_sr"] = np.maximum(
        0.0,
        amp_model.predict(frame[list(amplitude_feature_names)].to_numpy()),
    )
    frame["delta_beta_sr"] = frame["gate_sr"] * frame["sign_sr"] * frame["amplitude_sr"]
    frame["beta_sr"] = args.baseline_beta + frame["delta_beta_sr"]

    summary = {
        "run_tag": args.run_tag,
        "distill_mode": "sign_aware_gated",
        "target_source": args.target_source,
        "features": list(feature_names),
        "feature_preset": args.feature_preset,
        "branch_features": {name: list(values) for name, values in branch_features.items()},
        "cases": list(args.cases),
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_frame)),
        "rows_val": int(len(val_frame)),
        "rows_test": int(len(test_frame)),
        "rows_gate_distill": int(len(gate_train)),
        "rows_sign_distill": int(len(sign_train)),
        "rows_amplitude_distill": int(len(amp_train)),
        "active_threshold": args.active_threshold,
        "active_fraction": float(frame["active_mask"].mean()),
        "gate_distill_active_fraction": float(gate_train["active_mask"].mean()),
        "teacher_model_path": teacher_reference,
        "pysr_parallelism": runtime_config["parallelism"],
        "pysr_worker_count": int(runtime_config["worker_count"]),
        "pysr_procs": convert_value(runtime_config["procs"]),
        "pysr_julia_num_threads": int(runtime_config["julia_num_threads"]),
        "pysr_cluster_manager": runtime_config["cluster_manager"] or "",
        "pysr_batching": bool(runtime_config["batching"]),
        "pysr_batch_size": int(runtime_config["batch_size"]),
        "pysr_precision": int(runtime_config["precision"]),
        "pysr_deterministic": bool(runtime_config["deterministic"]),
        "populations_warning": populations_note,
    }

    split_masks = {
        "train": frame["split"] == "train",
        "val": frame["split"] == "val",
        "test": frame["split"] == "test",
        "full": np.ones(len(frame), dtype=bool),
    }
    for split_name, mask in split_masks.items():
        subset = frame.loc[mask]
        summary[f"{split_name}_target_metrics"] = compute_metrics(
            subset["delta_beta_target"], subset["delta_beta_sr"]
        )
        summary[f"{split_name}_raw_metrics"] = compute_metrics(
            subset["beta_raw"], subset["beta_sr"]
        )
        gate_true = subset["active_mask"].to_numpy()
        gate_score = subset["gate_sr"].to_numpy()
        gate_pred = (gate_score >= 0.5).astype(int)
        summary[f"{split_name}_gate_accuracy"] = float(np.mean(gate_pred == gate_true))
        if len(np.unique(gate_true)) > 1:
            summary[f"{split_name}_gate_auc"] = float(roc_auc_score(gate_true, gate_score))

        active_subset = subset[subset["active_mask"] == 1]
        if len(active_subset) > 0:
            sign_true = ((active_subset["sign_target"].to_numpy() + 1.0) / 2.0).astype(float)
            sign_score = np.clip(0.5 * (1.0 + active_subset["sign_sr"].to_numpy()), 0.0, 1.0)
            sign_pred = (sign_score >= 0.5).astype(int)
            summary[f"{split_name}_sign_accuracy"] = float(np.mean(sign_pred == sign_true))
            if len(np.unique(sign_true)) > 1:
                summary[f"{split_name}_sign_auc"] = float(roc_auc_score(sign_true, sign_score))
            summary[f"{split_name}_amplitude_metrics"] = compute_metrics(
                active_subset["amplitude_target"], active_subset["amplitude_sr"]
            )
        if "beta_teacher" in subset:
            summary[f"{split_name}_teacher_metrics"] = compute_metrics(
                subset["beta_teacher"], subset["beta_sr"]
            )

    if "beta_teacher" in frame:
        summary["teacher_vs_raw_full_metrics"] = compute_metrics(frame["beta_raw"], frame["beta_teacher"])
    if teacher_param_count is not None:
        summary["teacher_param_count"] = teacher_param_count

    summary.update(
        export_signed_gated_artifacts(
            gate_model,
            sign_model,
            amp_model,
            run_directory,
            feature_names,
            args.baseline_beta,
        )
    )

    dataset_columns = [
        "case_name",
        "case_index",
        "cell_index",
        *feature_names,
        "beta_raw",
        "delta_beta_raw",
    ]
    if "beta_teacher" in frame:
        dataset_columns.extend(["beta_teacher", "delta_beta_teacher"])
    dataset_columns.extend(
        [
            "beta_target",
            "delta_beta_target",
            "active_mask",
            "gate_target_signed",
            "sign_target",
            "amplitude_target",
            "gate_sr_signed",
            "gate_sr",
            "sign_sr",
            "amplitude_sr",
            "delta_beta_sr",
            "beta_sr",
            "split",
        ]
    )
    frame[dataset_columns].to_csv(run_directory / "dataset_with_predictions.csv", index=False)
    (run_directory / "summary.json").write_text(json.dumps(convert_value(summary), indent=2))

    print("=" * 72)
    print("Sign-aware gated symbolic distillation complete")
    print("=" * 72)
    print(f"Run directory        : {run_directory}")
    print(f"Target source        : {args.target_source}")
    print(f"Gate rows            : {len(gate_train)}")
    print(f"Sign rows            : {len(sign_train)}")
    print(f"Amplitude rows       : {len(amp_train)}")
    print(f"Branch features      : {summary['branch_features']}")
    print(f"Active fraction      : {summary['active_fraction']:.4f}")
    print(
        "PySR runtime         : "
        f"{summary['pysr_parallelism']} "
        f"(workers={summary['pysr_worker_count']}, "
        f"threads={summary['pysr_julia_num_threads']})"
    )
    print(f"Beta equation        : {summary['beta_expression']}")
    print(f"Gate equation        : {summary['gate_expression']}")
    print(f"Sign equation        : {summary['sign_expression']}")
    print(f"Amplitude equation   : {summary['amplitude_expression']}")
    if populations_note:
        print(f"Population note      : {populations_note}")
    print(
        "Validation target    : "
        f"R2={summary['val_target_metrics']['r2']:.4f}, "
        f"RMSE={summary['val_target_metrics']['rmse']:.6f}"
    )
    print(
        "Validation raw beta  : "
        f"R2={summary['val_raw_metrics']['r2']:.4f}, "
        f"RMSE={summary['val_raw_metrics']['rmse']:.6f}"
    )
    print(
        "Validation gate      : "
        f"AUC={summary.get('val_gate_auc', float('nan')):.4f}, "
        f"ACC={summary['val_gate_accuracy']:.4f}"
    )
    print(
        "Validation sign      : "
        f"AUC={summary.get('val_sign_auc', float('nan')):.4f}, "
        f"ACC={summary.get('val_sign_accuracy', float('nan')):.4f}"
    )
    print(
        "Validation amplitude : "
        f"R2={summary['val_amplitude_metrics']['r2']:.4f}, "
        f"RMSE={summary['val_amplitude_metrics']['rmse']:.6f}"
    )


if __name__ == "__main__":
    main()
