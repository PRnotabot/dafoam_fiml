#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from mpi4py import MPI
from pyofm import PYOFM
from sklearn.model_selection import train_test_split


DEFAULT_CASES = ("c1_data", "c2_data")
DEFAULT_FEATURES = ("PoD", "VoS", "PSoSS", "KoU2")
PARALLELISM_CHOICES = ("auto", "serial", "multithreading", "multiprocessing")
CLUSTER_MANAGER_CHOICES = ("slurm", "pbs", "lsf", "sge", "qrsh", "scyld", "htc")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill the Ramp tf_training beta teacher into a symbolic PySR expression."
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
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--run-tag", default="teacher_default")
    parser.add_argument("--output-dir", default="sr_results")
    add_shared_pysr_args(parser)
    return parser.parse_args()


def add_shared_pysr_args(parser):
    parser.add_argument("--binary-operators", default="+,-,*")
    parser.add_argument("--unary-operators", default="tanh")
    parser.add_argument("--niterations", type=int, default=18)
    parser.add_argument("--populations", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=28)
    parser.add_argument("--ncycles-per-iteration", type=int, default=60)
    parser.add_argument("--maxsize", type=int, default=18)
    parser.add_argument("--maxdepth", type=int, default=8)
    parser.add_argument("--parsimony", type=float, default=2.5e-3)
    parser.add_argument("--parallelism", choices=PARALLELISM_CHOICES, default="auto")
    parser.add_argument("--procs", type=int, default=None)
    parser.add_argument("--julia-num-threads", type=int, default=None)
    parser.add_argument("--cluster-manager", choices=CLUSTER_MANAGER_CHOICES, default=None)
    parser.add_argument("--heap-size-hint-in-bytes", type=int, default=None)
    parser.add_argument("--batching", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--precision", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--turbo", action="store_true")


def parse_operator_list(spec):
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("Operator list must be non-empty")
    return values


def read_scalar_field(ofm, field_name, case_path, n_cells):
    data = np.zeros(n_cells)
    ofm.readField(field_name, "volScalarField", case_path, data)
    return data


def load_dataset(cases, features, raw_field, n_cells):
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("Run this script with MPI size 1. PySR should not be launched with multiple MPI ranks.")

    ofm = PYOFM(comm=comm)
    frames = []
    for case_index, case_name in enumerate(cases):
        case_dir = Path(case_name)
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")

        columns = {}
        for feature_name in features:
            columns[feature_name] = read_scalar_field(ofm, feature_name, str(case_dir), n_cells)
        columns["beta_raw"] = read_scalar_field(ofm, raw_field, str(case_dir), n_cells)
        frame = pd.DataFrame(columns)
        frame["case_name"] = case_name
        frame["case_index"] = case_index
        frame["cell_index"] = np.arange(n_cells, dtype=int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_teacher_predictions(frame, feature_names, model_path):
    model = tf.keras.models.load_model(model_path)
    teacher_beta = model.predict(frame[list(feature_names)].to_numpy(), verbose=0).reshape(-1)
    return model, teacher_beta


def make_active_labels(delta_beta, threshold):
    return (np.abs(np.asarray(delta_beta, dtype=float)) > threshold).astype(int)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
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


def split_indices(n_rows, val_fraction, test_fraction, labels, random_state):
    total_holdout = val_fraction + test_fraction
    if total_holdout <= 0.0 or total_holdout >= 1.0:
        raise ValueError("val_fraction + test_fraction must be in (0, 1)")

    indices = np.arange(n_rows, dtype=int)
    unique_labels = np.unique(labels)
    stratify_all = labels if len(unique_labels) > 1 else None

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


def sample_training_rows(train_frame, sample_size, random_state):
    if sample_size <= 0 or sample_size >= len(train_frame):
        return train_frame.reset_index(drop=True)

    if "active_mask" in train_frame and len(train_frame["active_mask"].unique()) > 1:
        active = train_frame[train_frame["active_mask"] == 1]
        inactive = train_frame[train_frame["active_mask"] == 0]
        target_active = min(len(active), max(1, sample_size // 2))
        target_inactive = min(len(inactive), target_active)

        sampled_parts = []
        if target_active > 0:
            sampled_parts.append(active.sample(n=target_active, random_state=random_state))
        if target_inactive > 0:
            sampled_parts.append(inactive.sample(n=target_inactive, random_state=random_state))

        sampled = pd.concat(sampled_parts, ignore_index=False)
        return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return train_frame.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def read_env_int(name):
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value > 0 else None


def infer_local_worker_count():
    for env_name in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "PBS_NP", "NSLOTS"):
        value = read_env_int(env_name)
        if value is not None:
            return value
    return os.cpu_count() or 1


def infer_cluster_worker_count(cluster_manager):
    if cluster_manager == "slurm":
        ntasks = read_env_int("SLURM_NTASKS")
        cpus_per_task = read_env_int("SLURM_CPUS_PER_TASK") or 1
        if ntasks is not None:
            return ntasks * cpus_per_task
        return cpus_per_task
    if cluster_manager == "pbs":
        return read_env_int("PBS_NP")
    if cluster_manager in ("sge", "qrsh"):
        return read_env_int("NSLOTS")
    if cluster_manager == "lsf":
        return read_env_int("LSB_DJOB_NUMPROC")
    return None


def resolve_pysr_runtime(args):
    requested_parallelism = args.parallelism
    if args.cluster_manager and requested_parallelism == "multithreading":
        raise ValueError("`--cluster-manager` requires multiprocessing or auto parallelism.")

    if requested_parallelism == "auto":
        if args.cluster_manager or ((args.procs or 0) > 1):
            parallelism = "multiprocessing"
        else:
            parallelism = "multithreading"
    else:
        parallelism = requested_parallelism

    if parallelism == "multithreading" and (args.procs or 0) > 1:
        raise ValueError("`--procs` is for multiprocessing. Use `--julia-num-threads` for multithreading.")

    if args.deterministic and parallelism != "serial":
        raise ValueError("PySR deterministic mode requires `--parallelism serial`.")

    if args.cluster_manager and parallelism != "multiprocessing":
        raise ValueError("`--cluster-manager` can only be used with multiprocessing parallelism.")

    if parallelism == "multithreading":
        worker_count = args.julia_num_threads or infer_local_worker_count()
        procs = None
        julia_num_threads = worker_count
    elif parallelism == "multiprocessing":
        worker_count = args.procs or infer_cluster_worker_count(args.cluster_manager) or infer_local_worker_count()
        procs = worker_count
        # Keep one Julia thread per worker unless the user explicitly requests otherwise.
        julia_num_threads = args.julia_num_threads or 1
    else:
        worker_count = 1
        procs = None
        julia_num_threads = 1

    if worker_count < 1:
        raise ValueError("Resolved PySR worker count must be positive.")

    return {
        "parallelism": parallelism,
        "procs": procs,
        "cluster_manager": args.cluster_manager,
        "heap_size_hint_in_bytes": args.heap_size_hint_in_bytes,
        "batching": args.batching,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "deterministic": bool(args.deterministic and parallelism == "serial"),
        "progress": args.progress,
        "verbosity": args.verbosity,
        "worker_count": worker_count,
        "julia_num_threads": julia_num_threads,
    }


def prepare_julia_runtime(runtime_config):
    os.environ["JULIA_NUM_THREADS"] = str(runtime_config["julia_num_threads"])


def get_pysr_regressor():
    from pysr import PySRRegressor

    return PySRRegressor


def build_pysr_model(args, feature_names, run_directory, runtime_config=None):
    runtime_config = runtime_config or resolve_pysr_runtime(args)
    prepare_julia_runtime(runtime_config)
    PySRRegressor = get_pysr_regressor()
    binary_operators = parse_operator_list(args.binary_operators)
    unary_operators = parse_operator_list(args.unary_operators)

    constraints = {}
    if "/" in binary_operators:
        constraints["/"] = (6, 4)
    if "tanh" in unary_operators:
        constraints["tanh"] = 6
    if "sqrt" in unary_operators:
        constraints["sqrt"] = 4
    if "log" in unary_operators:
        constraints["log"] = 4
    if "exp" in unary_operators:
        constraints["exp"] = 4

    nested_constraints = {}
    if "tanh" in unary_operators:
        nested_constraints["tanh"] = {"tanh": 0}
    if "log" in unary_operators:
        nested_constraints["log"] = {"log": 0, "exp": 0}
    if "exp" in unary_operators:
        nested_constraints["exp"] = {"log": 0, "exp": 0}

    return PySRRegressor(
        model_selection="best",
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        constraints=constraints or None,
        nested_constraints=nested_constraints or None,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        niterations=args.niterations,
        populations=args.populations,
        population_size=args.population_size,
        tournament_selection_n=max(2, min(10, args.population_size - 1)),
        topn=max(2, min(12, args.population_size)),
        ncycles_per_iteration=args.ncycles_per_iteration,
        maxsize=args.maxsize,
        maxdepth=args.maxdepth,
        parsimony=args.parsimony,
        parallelism=runtime_config["parallelism"],
        procs=runtime_config["procs"],
        cluster_manager=runtime_config["cluster_manager"],
        heap_size_hint_in_bytes=runtime_config["heap_size_hint_in_bytes"],
        batching=runtime_config["batching"],
        batch_size=runtime_config["batch_size"],
        turbo=args.turbo,
        bumper=False,
        precision=runtime_config["precision"],
        random_state=args.random_state,
        deterministic=runtime_config["deterministic"],
        progress=runtime_config["progress"],
        verbosity=runtime_config["verbosity"],
        update=False,
        warm_start=False,
        output_directory=str(run_directory),
        run_id=args.run_tag,
        temp_equation_file=False,
        delete_tempfiles=False,
    )


def sympy_to_numpy_string(sympy_expression):
    expression = str(sympy_expression)
    replacements = {
        "Abs(": "np.abs(",
        "abs(": "np.abs(",
        "exp(": "np.exp(",
        "log(": "np.log(",
        "sqrt(": "np.sqrt(",
        "sin(": "np.sin(",
        "cos(": "np.cos(",
        "tanh(": "np.tanh(",
    }
    for old, new in replacements.items():
        expression = expression.replace(old, new)
    return expression


def convert_value(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [convert_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): convert_value(val) for key, val in value.items()}
    return str(value)


def export_artifacts(model, run_directory, feature_names, baseline_beta):
    best = model.get_best()
    sympy_expr = best["sympy_format"]
    delta_expr = str(sympy_expr)
    beta_expr = f"{baseline_beta} + ({delta_expr})"
    numpy_expr = sympy_to_numpy_string(sympy_expr)

    python_lines = [
        '"""Auto-generated symbolic distillation equation."""',
        "",
        "import numpy as np",
        "",
        f"def delta_beta({', '.join(feature_names)}):",
        f"    return {numpy_expr}",
        "",
        f"def beta_fiomega({', '.join(feature_names)}):",
        f"    return {baseline_beta} + delta_beta({', '.join(feature_names)})",
        "",
    ]
    (run_directory / "equation.py").write_text("\n".join(python_lines))
    (run_directory / "equation_sympy.txt").write_text(delta_expr + "\n")
    (run_directory / "equation_beta.txt").write_text(beta_expr + "\n")

    try:
        from sympy import latex

        latex_expr = latex(sympy_expr)
    except Exception:
        latex_expr = delta_expr
    latex_lines = [
        r"\begin{equation}",
        rf"    \Delta \beta = {latex_expr}",
        r"\end{equation}",
        "",
        r"\begin{equation}",
        rf"    \beta = {baseline_beta} + {latex_expr}",
        r"\end{equation}",
        "",
    ]
    (run_directory / "equation.tex").write_text("\n".join(latex_lines))

    equations = []
    if model.equations_ is not None:
        for _, row in model.equations_.iterrows():
            record = {column: convert_value(row[column]) for column in model.equations_.columns}
            equations.append(record)
    (run_directory / "pareto_front.json").write_text(json.dumps(equations, indent=2))

    return {
        "best_equation": str(best["equation"]),
        "best_sympy": delta_expr,
        "beta_expression": beta_expr,
        "best_loss": float(best["loss"]),
        "best_complexity": int(best["complexity"]),
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
    sampled_train = sample_training_rows(train_frame, args.sample_size, args.random_state)

    model = build_pysr_model(args, feature_names, run_directory, runtime_config=runtime_config)
    model.fit(
        sampled_train[list(feature_names)].to_numpy(),
        sampled_train["delta_beta_target"].to_numpy(),
        variable_names=list(feature_names),
    )

    frame["delta_beta_sr"] = model.predict(frame[list(feature_names)].to_numpy())
    frame["beta_sr"] = args.baseline_beta + frame["delta_beta_sr"]

    summary = {
        "run_tag": args.run_tag,
        "target_source": args.target_source,
        "features": list(feature_names),
        "cases": list(args.cases),
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_frame)),
        "rows_val": int(len(val_frame)),
        "rows_test": int(len(test_frame)),
        "rows_distill": int(len(sampled_train)),
        "active_threshold": args.active_threshold,
        "active_fraction": float(frame["active_mask"].mean()),
        "distill_active_fraction": float(sampled_train["active_mask"].mean()),
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
        if "beta_teacher" in subset:
            summary[f"{split_name}_teacher_metrics"] = compute_metrics(
                subset["beta_teacher"], subset["beta_sr"]
            )

    if teacher_model is not None:
        summary["teacher_vs_raw_full_metrics"] = compute_metrics(frame["beta_raw"], frame["beta_teacher"])
        summary["teacher_param_count"] = int(teacher_model.count_params())

    summary.update(export_artifacts(model, run_directory, feature_names, args.baseline_beta))

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
            "beta_sr",
            "delta_beta_sr",
            "active_mask",
            "split",
        ]
    )
    frame[dataset_columns].to_csv(run_directory / "dataset_with_predictions.csv", index=False)
    (run_directory / "summary.json").write_text(json.dumps(convert_value(summary), indent=2))

    print("=" * 72)
    print("Symbolic distillation complete")
    print("=" * 72)
    print(f"Run directory      : {run_directory}")
    print(f"Target source      : {args.target_source}")
    print(f"Rows used for PySR : {len(sampled_train)}")
    print(f"Active fraction    : {summary['active_fraction']:.4f}")
    print(f"Distill active frac: {summary['distill_active_fraction']:.4f}")
    print(
        "PySR runtime       : "
        f"{summary['pysr_parallelism']} "
        f"(workers={summary['pysr_worker_count']}, "
        f"threads={summary['pysr_julia_num_threads']})"
    )
    print(f"Best equation      : {summary['beta_expression']}")
    print(f"Best sympy delta   : {summary['best_sympy']}")
    print(f"Complexity         : {summary['best_complexity']}")
    print(f"Loss               : {summary['best_loss']:.6e}")
    if populations_note:
        print(f"Population note    : {populations_note}")
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
    if "val_teacher_metrics" in summary:
        print(
            "Validation teacher : "
            f"R2={summary['val_teacher_metrics']['r2']:.4f}, "
            f"RMSE={summary['val_teacher_metrics']['rmse']:.6f}"
        )


if __name__ == "__main__":
    main()
