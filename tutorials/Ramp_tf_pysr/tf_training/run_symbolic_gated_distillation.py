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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gated symbolic distillation for the Ramp tf_training beta teacher."
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
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--value-sample-size", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--run-tag", default="gated_teacher_default")
    parser.add_argument("--output-dir", default="sr_results")
    add_shared_pysr_args(parser)
    return parser.parse_args()


def export_gated_artifacts(
    gate_model,
    value_model,
    run_directory,
    feature_names,
    baseline_beta,
):
    gate_best = gate_model.get_best()
    value_best = value_model.get_best()

    gate_raw_expr = str(gate_best["sympy_format"])
    gate_expr = f"0.5*(1 + ({gate_raw_expr}))"
    gate_expr_clipped = f"clip({gate_expr}, 0, 1)"
    value_expr = str(value_best["sympy_format"])
    delta_expr = f"({gate_expr_clipped}) * ({value_expr})"
    beta_expr = f"{baseline_beta} + ({delta_expr})"

    gate_numpy = sympy_to_numpy_string(gate_best["sympy_format"])
    value_numpy = sympy_to_numpy_string(value_best["sympy_format"])

    python_lines = [
        '"""Auto-generated gated symbolic distillation equation."""',
        "",
        "import numpy as np",
        "",
        f"def gate_raw({', '.join(feature_names)}):",
        f"    return {gate_numpy}",
        "",
        f"def gate_active({', '.join(feature_names)}):",
        f"    return np.clip(0.5 * (1.0 + gate_raw({', '.join(feature_names)})), 0.0, 1.0)",
        "",
        f"def value_active({', '.join(feature_names)}):",
        f"    return {value_numpy}",
        "",
        f"def delta_beta({', '.join(feature_names)}):",
        f"    return gate_active({', '.join(feature_names)}) * value_active({', '.join(feature_names)})",
        "",
        f"def beta_fiomega({', '.join(feature_names)}):",
        f"    return {baseline_beta} + delta_beta({', '.join(feature_names)})",
        "",
    ]
    (run_directory / "equation.py").write_text("\n".join(python_lines))
    (run_directory / "gate_equation_sympy.txt").write_text(gate_expr + "\n")
    (run_directory / "value_equation_sympy.txt").write_text(value_expr + "\n")
    (run_directory / "equation_beta.txt").write_text(beta_expr + "\n")

    gate_pareto = []
    if gate_model.equations_ is not None:
        for _, row in gate_model.equations_.iterrows():
            gate_pareto.append({column: convert_value(row[column]) for column in gate_model.equations_.columns})
    value_pareto = []
    if value_model.equations_ is not None:
        for _, row in value_model.equations_.iterrows():
            value_pareto.append({column: convert_value(row[column]) for column in value_model.equations_.columns})
    (run_directory / "pareto_gate.json").write_text(json.dumps(gate_pareto, indent=2))
    (run_directory / "pareto_value.json").write_text(json.dumps(value_pareto, indent=2))

    return {
        "gate_expression": gate_expr,
        "gate_expression_clipped": gate_expr_clipped,
        "value_expression": value_expr,
        "delta_expression": delta_expr,
        "beta_expression": beta_expr,
        "gate_best_equation": str(gate_best["equation"]),
        "value_best_equation": str(value_best["equation"]),
        "gate_best_loss": float(gate_best["loss"]),
        "value_best_loss": float(value_best["loss"]),
        "gate_best_complexity": int(gate_best["complexity"]),
        "value_best_complexity": int(value_best["complexity"]),
    }


def main():
    args = parse_args()
    feature_names = tuple(args.features)
    output_root = Path(args.output_dir)
    run_directory = output_root / args.run_tag
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
    teacher_beta = None
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
    frame["active_mask"] = make_active_labels(frame["delta_beta_target"].to_numpy(), args.active_threshold)
    frame["gate_target_signed"] = 2.0 * frame["active_mask"] - 1.0

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

    gate_train = sample_training_rows(train_frame, args.sample_size, args.random_state)
    active_train_full = train_frame[train_frame["active_mask"] == 1].copy()
    value_train = sample_training_rows(active_train_full, args.value_sample_size, args.random_state)

    gate_args = argparse.Namespace(**vars(args))
    gate_args.run_tag = f"{args.run_tag}_gate"
    value_args = argparse.Namespace(**vars(args))
    value_args.run_tag = f"{args.run_tag}_value"

    gate_model = build_pysr_model(gate_args, feature_names, run_directory, runtime_config=runtime_config)
    gate_model.fit(
        gate_train[list(feature_names)].to_numpy(),
        gate_train["gate_target_signed"].to_numpy(),
        variable_names=list(feature_names),
    )

    value_model = build_pysr_model(value_args, feature_names, run_directory, runtime_config=runtime_config)
    value_model.fit(
        value_train[list(feature_names)].to_numpy(),
        value_train["delta_beta_target"].to_numpy(),
        variable_names=list(feature_names),
    )

    feature_matrix = frame[list(feature_names)].to_numpy()
    frame["gate_sr_signed"] = gate_model.predict(feature_matrix)
    frame["gate_sr"] = np.clip(0.5 * (1.0 + frame["gate_sr_signed"]), 0.0, 1.0)
    frame["value_sr"] = value_model.predict(feature_matrix)
    frame["delta_beta_sr"] = frame["gate_sr"] * frame["value_sr"]
    frame["beta_sr"] = args.baseline_beta + frame["delta_beta_sr"]

    summary = {
        "run_tag": args.run_tag,
        "distill_mode": "gated",
        "target_source": args.target_source,
        "features": list(feature_names),
        "cases": list(args.cases),
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_frame)),
        "rows_val": int(len(val_frame)),
        "rows_test": int(len(test_frame)),
        "rows_gate_distill": int(len(gate_train)),
        "rows_value_distill": int(len(value_train)),
        "active_threshold": args.active_threshold,
        "active_fraction": float(frame["active_mask"].mean()),
        "gate_distill_active_fraction": float(gate_train["active_mask"].mean()),
        "value_train_rows_total": int(len(active_train_full)),
        "teacher_model_path": str(model_path) if teacher_model is not None else "",
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
        summary[f"{split_name}_value_active_metrics"] = compute_metrics(
            subset.loc[subset["active_mask"] == 1, "delta_beta_target"],
            subset.loc[subset["active_mask"] == 1, "value_sr"],
        )
        gate_true = subset["active_mask"].to_numpy()
        gate_score = subset["gate_sr"].to_numpy()
        gate_pred = (gate_score >= 0.5).astype(int)
        summary[f"{split_name}_gate_accuracy"] = float(np.mean(gate_pred == gate_true))
        if len(np.unique(gate_true)) > 1:
            summary[f"{split_name}_gate_auc"] = float(roc_auc_score(gate_true, gate_score))
        if "beta_teacher" in subset:
            summary[f"{split_name}_teacher_metrics"] = compute_metrics(
                subset["beta_teacher"], subset["beta_sr"]
            )

    if teacher_model is not None:
        summary["teacher_vs_raw_full_metrics"] = compute_metrics(frame["beta_raw"], frame["beta_teacher"])
        summary["teacher_param_count"] = int(teacher_model.count_params())

    summary.update(export_gated_artifacts(gate_model, value_model, run_directory, feature_names, args.baseline_beta))

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
            "gate_sr_signed",
            "gate_sr",
            "value_sr",
            "delta_beta_sr",
            "beta_sr",
            "split",
        ]
    )
    frame[dataset_columns].to_csv(run_directory / "dataset_with_predictions.csv", index=False)
    (run_directory / "summary.json").write_text(json.dumps(convert_value(summary), indent=2))

    print("=" * 72)
    print("Gated symbolic distillation complete")
    print("=" * 72)
    print(f"Run directory        : {run_directory}")
    print(f"Target source        : {args.target_source}")
    print(f"Gate rows            : {len(gate_train)}")
    print(f"Value rows           : {len(value_train)}")
    print(f"Active fraction      : {summary['active_fraction']:.4f}")
    print(f"Gate active fraction : {summary['gate_distill_active_fraction']:.4f}")
    print(
        "PySR runtime         : "
        f"{summary['pysr_parallelism']} "
        f"(workers={summary['pysr_worker_count']}, "
        f"threads={summary['pysr_julia_num_threads']})"
    )
    print(f"Beta equation        : {summary['beta_expression']}")
    print(f"Gate equation        : {summary['gate_expression']}")
    print(f"Value equation       : {summary['value_expression']}")
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
        "Validation active val: "
        f"R2={summary['val_value_active_metrics']['r2']:.4f}, "
        f"RMSE={summary['val_value_active_metrics']['rmse']:.6f}"
    )


if __name__ == "__main__":
    main()
