#!/usr/bin/env python
"""
Distill branch MLP outputs with PySR. Operates on a frozen MLP teacher.

Loads branch predictions from a prior training run and distills each
branch independently with branch-specific PySR settings. Reports both
MLP and SR branch-level metrics, plus per-branch distillation gaps.

Compatible run directories:
  - fiml_mlp_runs/<tag>/          (from train_fiml_branch_mlp.py)
  - structured_student_runs/<tag>/ (from train_structured_student_staged.py)

Exports both diagnostic (clip/max) and smooth (tanh/sigmoid/softplus)
equation forms.

Usage:
    cd tf_training
    python distill_fiml_branch_mlp.py --mlp-run-dir fiml_mlp_runs/fiml_v1
    python distill_fiml_branch_mlp.py --mlp-run-dir structured_student_runs/staged_guided_heavy_v1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_symbolic_distillation import (
    build_pysr_model,
    convert_value,
    resolve_pysr_runtime,
    sample_training_rows,
    sympy_to_numpy_string,
)


# ====================================================================== #
# Branch-specific PySR defaults
# ====================================================================== #

BRANCH_PYSR_DEFAULTS = {
    "gate": {
        "binary_operators": "+,*",
        "unary_operators": "tanh",
        "maxsize": 10,
        "maxdepth": 5,
        "parsimony": 3.0e-3,
        "niterations": 15,
        "populations": 4,
        "population_size": 24,
        "ncycles_per_iteration": 50,
        "sample_size": 5000,
    },
    "sign": {
        "binary_operators": "+,-,*",
        "unary_operators": "tanh",
        "maxsize": 14,
        "maxdepth": 6,
        "parsimony": 2.0e-3,
        "niterations": 20,
        "populations": 6,
        "population_size": 28,
        "ncycles_per_iteration": 60,
        "sample_size": 3000,
    },
    "amplitude": {
        "binary_operators": "+,-,*",
        "unary_operators": "tanh",
        "maxsize": 20,
        "maxdepth": 8,
        "parsimony": 1.5e-3,
        "niterations": 25,
        "populations": 8,
        "population_size": 32,
        "ncycles_per_iteration": 80,
        "sample_size": 3000,
    },
}


# ====================================================================== #
# Data loading — auto-detects run directory format
# ====================================================================== #

# Column mapping: source format -> canonical name
_FIML_COLUMNS = {
    "gate_mlp": "gate",
    "sign_prob_mlp": "sign_prob",
    "sign_mlp": "sign",
    "amplitude_mlp": "amplitude",
    "delta_beta_mlp": "delta_beta_pred",
    "beta_mlp": "beta_pred",
}
_STUDENT_COLUMNS = {
    "gate_student": "gate",
    "sign_probability_student": "sign_prob",
    "sign_student": "sign",
    "amplitude_student": "amplitude",
    "delta_beta_student": "delta_beta_pred",
    "beta_student": "beta_pred",
}


def load_branch_predictions(run_dir):
    """Load branch predictions from a compatible run directory.

    Detects the CSV format automatically and renames columns to canonical names:
      gate, sign_prob, sign, amplitude, delta_beta_pred, beta_pred
    """
    run_dir = Path(run_dir)

    # Try both CSV names
    csv_path = run_dir / "predictions.csv"
    if not csv_path.exists():
        csv_path = run_dir / "dataset_with_student_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No predictions CSV found in {run_dir}")

    frame = pd.read_csv(csv_path)

    # Detect format and rename
    if "gate_mlp" in frame.columns:
        frame = frame.rename(columns=_FIML_COLUMNS)
    elif "gate_student" in frame.columns:
        frame = frame.rename(columns=_STUDENT_COLUMNS)
    else:
        raise ValueError(f"Unrecognized column format in {csv_path}. Expected gate_mlp or gate_student columns.")

    # Validate required columns
    required = ["gate", "sign", "amplitude", "active_mask", "split"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # Load metadata if available
    summary_path = run_dir / "summary.json"
    metadata = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    return frame, metadata


def detect_features(frame, metadata):
    """Detect feature columns and branch feature assignments."""
    if "branch_features" in metadata:
        branch_features = {k: tuple(v) for k, v in metadata["branch_features"].items()}
        all_features = []
        for feats in branch_features.values():
            all_features.extend(feats)
        feature_names = tuple(dict.fromkeys(all_features))
        return feature_names, branch_features

    # Fallback: detect from columns
    candidate_features = ("PoD", "VoS", "PSoSS", "KoU2", "chiSA", "ReWall", "pGradStream", "SCurv", "UOrth", "CoP", "TauoK")
    feature_names = tuple(f for f in candidate_features if f in frame.columns)
    # Default guided preset
    branch_features = {
        "gate": ("PoD",),
        "sign": feature_names,
        "amplitude": feature_names,
    }
    return feature_names, branch_features


# ====================================================================== #
# PySR branch distillation
# ====================================================================== #


def make_branch_pysr_args(base_args, branch_name):
    """Create a PySR args namespace with branch-specific settings."""
    defaults = BRANCH_PYSR_DEFAULTS[branch_name]
    branch_args = argparse.Namespace(**vars(base_args))

    # Apply branch defaults, then CLI overrides
    for key, default_value in defaults.items():
        if key == "sample_size":
            continue  # handled separately
        cli_key = f"{branch_name}_{key}"
        cli_value = getattr(base_args, cli_key, None)
        if cli_value is not None:
            setattr(branch_args, key, cli_value)
        else:
            setattr(branch_args, key, default_value)

    branch_args.run_tag = f"{base_args.run_tag}_{branch_name}"
    return branch_args


def distill_one_branch(branch_name, frame, train_idx, branch_features, target_column, sample_size, base_args, output_dir, runtime_config):
    """Run PySR on one branch. Returns the fitted model."""
    branch_args = make_branch_pysr_args(base_args, branch_name)
    features = list(branch_features[branch_name])

    # Sample training data
    train_df = frame.iloc[train_idx].copy()
    if branch_name in ("sign", "amplitude"):
        train_df = train_df[train_df["active_mask"] == 1].copy()

    sr_data = sample_training_rows(train_df, sample_size, base_args.random_state)

    print(f"  {branch_name:12s}: {len(sr_data)} samples, features={features}, maxsize={branch_args.maxsize}, parsimony={branch_args.parsimony}")

    model = build_pysr_model(branch_args, features, output_dir, runtime_config=runtime_config)
    model.fit(
        sr_data[features].to_numpy(),
        sr_data[target_column].to_numpy(),
        variable_names=features,
    )
    return model


# ====================================================================== #
# Metrics
# ====================================================================== #


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) == 0:
        return {"mse": None, "rmse": None, "r2": None, "rows": 0}
    diff = y_true - y_pred
    mse = float(np.mean(diff * diff))
    var = float(np.var(y_true))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(1.0 - mse / var) if var > 0 else 1.0,
        "rows": int(len(y_true)),
    }


def binary_metrics(y_true, score):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    pred = (score >= 0.5).astype(int)
    acc = float(np.mean(pred == y_true)) if len(y_true) else None
    auc = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else None
    return {"accuracy": acc, "auc": auc, "rows": int(len(y_true))}


def compute_split_metrics(frame, split_name, has_sr):
    """Compute MLP metrics, SR metrics, and distillation gaps for one split."""
    s = frame[frame["split"] == split_name]
    out = {}

    # MLP branch metrics
    out["mlp_gate"] = binary_metrics(s["active_mask"], s["gate"])
    active = s[s["active_mask"] == 1]
    if len(active) > 0 and "sign_prob" in active.columns:
        out["mlp_sign"] = binary_metrics(active["sign_target"].astype(int), active["sign_prob"])
        out["mlp_amplitude"] = regression_metrics(active["amplitude_target"], active["amplitude"])

    # MLP overall
    out["mlp_vs_target"] = regression_metrics(s["delta_beta_target"], s["delta_beta_pred"])

    if has_sr:
        # SR branch metrics
        out["sr_gate"] = binary_metrics(s["active_mask"], s["gate_sr"])
        if len(active) > 0:
            active_sign_sr = s.loc[s["active_mask"] == 1, "sign_sr"]
            out["sr_sign"] = binary_metrics(
                active["sign_target"].astype(int),
                np.clip(0.5 * (1.0 + active_sign_sr.to_numpy()), 0.0, 1.0),
            )
            out["sr_amplitude"] = regression_metrics(active["amplitude_target"], s.loc[s["active_mask"] == 1, "amplitude_sr"])

        # SR overall
        out["sr_vs_target"] = regression_metrics(s["delta_beta_target"], s["delta_beta_sr"])
        out["sr_vs_raw"] = regression_metrics(s["beta_raw"], s["beta_sr"])

        # Per-branch distillation gaps (SR vs MLP)
        out["gap_gate"] = regression_metrics(s["gate"], s["gate_sr"])
        if len(active) > 0:
            out["gap_sign"] = regression_metrics(active["sign"], active_sign_sr)
            out["gap_amplitude"] = regression_metrics(active["amplitude"], s.loc[s["active_mask"] == 1, "amplitude_sr"])
        out["gap_overall"] = regression_metrics(s["delta_beta_pred"], s["delta_beta_sr"])

    return out


# ====================================================================== #
# Equation export
# ====================================================================== #


def export_equations(sr_models, output_dir, branch_features, baseline_beta):
    """Export diagnostic (clip/max) and smooth (tanh/sigmoid/softplus) equation forms."""
    gate_best = sr_models["gate"].get_best()
    sign_best = sr_models["sign"].get_best()
    amp_best = sr_models["amplitude"].get_best()

    gate_raw = str(gate_best["sympy_format"])
    sign_raw = str(sign_best["sympy_format"])
    amp_raw = str(amp_best["sympy_format"])

    gate_np = sympy_to_numpy_string(gate_best["sympy_format"])
    sign_np = sympy_to_numpy_string(sign_best["sympy_format"])
    amp_np = sympy_to_numpy_string(amp_best["sympy_format"])

    gate_vars = ", ".join(branch_features["gate"])
    sign_vars = ", ".join(branch_features["sign"])
    amp_vars = ", ".join(branch_features["amplitude"])
    all_vars_list = list(branch_features["gate"]) + list(branch_features["sign"]) + list(branch_features["amplitude"])
    all_vars = ", ".join(dict.fromkeys(all_vars_list))

    # ---- Diagnostic form (clip/max) ---- #
    diag_lines = [
        '"""Diagnostic equation — uses clip/max for exact reconstruction."""',
        "",
        "import numpy as np",
        "",
        "",
        f"def gate({gate_vars}):",
        f"    return np.clip(0.5 * (1.0 + ({gate_np})), 0.0, 1.0)",
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
    (output_dir / "equation_diagnostic.py").write_text("\n".join(diag_lines))

    # ---- Smooth form (tanh/softplus) for gradient-based optimization ---- #
    smooth_lines = [
        '"""Smooth equation — uses tanh/softplus for gradient-safe deployment."""',
        "",
        "import numpy as np",
        "",
        "",
        f"def gate({gate_vars}):",
        f"    return 0.5 * (1.0 + np.tanh({gate_np}))",
        "",
        "",
        f"def sign_value({sign_vars}):",
        f"    return np.tanh({sign_np})",
        "",
        "",
        f"def amplitude({amp_vars}):",
        f"    return np.log(1.0 + np.exp({amp_np}))",
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
    (output_dir / "equation_smooth.py").write_text("\n".join(smooth_lines))

    # ---- Per-branch text files ---- #
    (output_dir / "gate_equation.txt").write_text(f"diagnostic: clip(0.5*(1 + ({gate_raw})), 0, 1)\nsmooth:     0.5*(1 + tanh({gate_raw}))\n")
    (output_dir / "sign_equation.txt").write_text(f"diagnostic: clip({sign_raw}, -1, 1)\nsmooth:     tanh({sign_raw})\n")
    (output_dir / "amplitude_equation.txt").write_text(f"diagnostic: max(0, {amp_raw})\nsmooth:     softplus({amp_raw})\n")

    beta_diag = f"{baseline_beta} + clip(0.5*(1+({gate_raw})),0,1) * clip({sign_raw},-1,1) * max(0,{amp_raw})"
    beta_smooth = f"{baseline_beta} + 0.5*(1+tanh({gate_raw})) * tanh({sign_raw}) * log(1+exp({amp_raw}))"
    (output_dir / "equation_beta.txt").write_text(f"diagnostic: {beta_diag}\nsmooth:     {beta_smooth}\n")

    # ---- Pareto fronts ---- #
    for name in ("gate", "sign", "amplitude"):
        model = sr_models[name]
        payload = []
        if model.equations_ is not None:
            for _, row in model.equations_.iterrows():
                payload.append({col: convert_value(row[col]) for col in model.equations_.columns})
        (output_dir / f"pareto_{name}.json").write_text(json.dumps(payload, indent=2))

    return {
        "gate_raw_expr": gate_raw,
        "sign_raw_expr": sign_raw,
        "amplitude_raw_expr": amp_raw,
        "gate_best_loss": float(gate_best["loss"]),
        "sign_best_loss": float(sign_best["loss"]),
        "amplitude_best_loss": float(amp_best["loss"]),
        "gate_best_complexity": int(gate_best["complexity"]),
        "sign_best_complexity": int(sign_best["complexity"]),
        "amplitude_best_complexity": int(amp_best["complexity"]),
    }


# ====================================================================== #
# CLI
# ====================================================================== #


def add_pysr_runtime_args(parser):
    """PySR runtime arguments (shared across branches)."""
    parser.add_argument("--parallelism", choices=("auto", "serial", "multithreading", "multiprocessing"), default="auto")
    parser.add_argument("--procs", type=int, default=None)
    parser.add_argument("--julia-num-threads", type=int, default=None)
    parser.add_argument("--cluster-manager", default=None)
    parser.add_argument("--heap-size-hint-in-bytes", type=int, default=None)
    parser.add_argument("--batching", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--precision", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--turbo", action="store_true")


def add_branch_override_args(parser):
    """Per-branch PySR overrides. Defaults come from BRANCH_PYSR_DEFAULTS."""
    for branch in ("gate", "sign", "amplitude"):
        prefix = f"--{branch}"
        parser.add_argument(f"{prefix}-maxsize", type=int, default=None)
        parser.add_argument(f"{prefix}-maxdepth", type=int, default=None)
        parser.add_argument(f"{prefix}-parsimony", type=float, default=None)
        parser.add_argument(f"{prefix}-niterations", type=int, default=None)
        parser.add_argument(f"{prefix}-populations", type=int, default=None)
        parser.add_argument(f"{prefix}-population-size", type=int, default=None)
        parser.add_argument(f"{prefix}-ncycles-per-iteration", type=int, default=None)
        parser.add_argument(f"{prefix}-sample-size", type=int, default=None)
        parser.add_argument(f"{prefix}-binary-operators", type=str, default=None)
        parser.add_argument(f"{prefix}-unary-operators", type=str, default=None)


def parse_args():
    parser = argparse.ArgumentParser(description="Distill branch MLP outputs with PySR.")
    parser.add_argument("--mlp-run-dir", required=True, help="Path to a branch MLP training run directory.")
    parser.add_argument("--baseline-beta", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--run-tag", default="distill_v1")
    parser.add_argument("--output-dir", default="fiml_distill_runs")
    add_pysr_runtime_args(parser)
    add_branch_override_args(parser)
    return parser.parse_args()


# ====================================================================== #
# Main
# ====================================================================== #


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load frozen teacher predictions ----------------------------- #
    print("=" * 72)
    print(f"Loading branch predictions from: {args.mlp_run_dir}")
    print("=" * 72)

    frame, metadata = load_branch_predictions(args.mlp_run_dir)
    feature_names, branch_features = detect_features(frame, metadata)

    # Ensure derived columns exist
    if "sign_target" not in frame.columns:
        frame["sign_target"] = 0.0
        active = frame["active_mask"] == 1
        frame.loc[active, "sign_target"] = (frame.loc[active, "delta_beta_target"] > 0.0).astype(float)
    if "amplitude_target" not in frame.columns:
        frame["amplitude_target"] = np.abs(frame["delta_beta_target"])

    # Prepare PySR targets (signed gate, raw sign, raw amplitude)
    frame["gate_signed"] = 2.0 * frame["gate"] - 1.0

    train_idx = np.where(frame["split"] == "train")[0]

    print(f"  Features       : {list(feature_names)}")
    print(f"  Branch features: { {k: list(v) for k, v in branch_features.items()} }")
    print(f"  Rows           : {len(frame)} ({len(train_idx)} train)")
    print(f"  Active fraction: {frame['active_mask'].mean():.4f}")
    print(f"  Source run     : {metadata.get('run_tag', 'unknown')}")

    # ---- Distill each branch ----------------------------------------- #
    print("\n" + "=" * 72)
    print("PySR distillation (branch-specific settings)")
    print("=" * 72)

    runtime_config = resolve_pysr_runtime(args)
    sr_models = {}

    for branch_name, target_col in [("gate", "gate_signed"), ("sign", "sign"), ("amplitude", "amplitude")]:
        sample_key = f"{branch_name}_sample_size"
        cli_sample = getattr(args, sample_key, None)
        sample_size = cli_sample if cli_sample is not None else BRANCH_PYSR_DEFAULTS[branch_name]["sample_size"]

        sr_models[branch_name] = distill_one_branch(
            branch_name, frame, train_idx, branch_features, target_col, sample_size, args, output_dir, runtime_config
        )

    # ---- SR predictions ---------------------------------------------- #
    frame["gate_sr_signed"] = sr_models["gate"].predict(frame[list(branch_features["gate"])].to_numpy())
    frame["gate_sr"] = np.clip(0.5 * (1.0 + frame["gate_sr_signed"]), 0.0, 1.0)
    frame["sign_sr"] = np.clip(sr_models["sign"].predict(frame[list(branch_features["sign"])].to_numpy()), -1.0, 1.0)
    frame["amplitude_sr"] = np.maximum(0.0, sr_models["amplitude"].predict(frame[list(branch_features["amplitude"])].to_numpy()))
    frame["delta_beta_sr"] = frame["gate_sr"] * frame["sign_sr"] * frame["amplitude_sr"]
    frame["beta_sr"] = args.baseline_beta + frame["delta_beta_sr"]

    # ---- Export equations --------------------------------------------- #
    eq_info = export_equations(sr_models, output_dir, branch_features, args.baseline_beta)

    # ---- Metrics ----------------------------------------------------- #
    summary = {
        "run_tag": args.run_tag,
        "mlp_run_dir": str(args.mlp_run_dir),
        "mlp_run_tag": metadata.get("run_tag", ""),
        "features": list(feature_names),
        "branch_features": {k: list(v) for k, v in branch_features.items()},
        "baseline_beta": args.baseline_beta,
        "rows_total": int(len(frame)),
        "active_fraction": float(frame["active_mask"].mean()),
        "mlp_param_count": metadata.get("mlp_param_count") or metadata.get("student_param_count"),
    }

    # Branch-specific PySR config used
    for branch_name in ("gate", "sign", "amplitude"):
        ba = make_branch_pysr_args(args, branch_name)
        summary[f"{branch_name}_pysr"] = {
            "maxsize": ba.maxsize,
            "maxdepth": ba.maxdepth,
            "parsimony": ba.parsimony,
            "niterations": ba.niterations,
            "binary_operators": ba.binary_operators,
            "unary_operators": ba.unary_operators,
        }

    for split_name in ("train", "val", "test"):
        summary[split_name] = compute_split_metrics(frame, split_name, has_sr=True)

    summary.update(eq_info)

    frame.to_csv(output_dir / "dataset_with_sr.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(convert_value(summary), indent=2))

    # ---- Print ------------------------------------------------------- #
    print("\n" + "=" * 72)
    print("Distillation complete")
    print("=" * 72)
    print(f"  Output         : {output_dir}")
    print(f"  MLP source     : {args.mlp_run_dir}")

    val = summary.get("val", {})

    print(f"\n  MLP branch quality (validation):")
    mg = val.get("mlp_gate", {})
    ms = val.get("mlp_sign", {})
    ma = val.get("mlp_amplitude", {})
    print(f"    Gate       AUC={mg.get('auc', 'N/A')}  ACC={mg.get('accuracy', 'N/A')}")
    print(f"    Sign       AUC={ms.get('auc', 'N/A')}  ACC={ms.get('accuracy', 'N/A')}")
    print(f"    Amplitude  R2={ma.get('r2', 'N/A')}  RMSE={ma.get('rmse', 'N/A')}")

    print(f"\n  SR branch quality (validation):")
    sg = val.get("sr_gate", {})
    ss = val.get("sr_sign", {})
    sa = val.get("sr_amplitude", {})
    print(f"    Gate       AUC={sg.get('auc', 'N/A')}  ACC={sg.get('accuracy', 'N/A')}")
    print(f"    Sign       AUC={ss.get('auc', 'N/A')}  ACC={ss.get('accuracy', 'N/A')}")
    print(f"    Amplitude  R2={sa.get('r2', 'N/A')}  RMSE={sa.get('rmse', 'N/A')}")

    print(f"\n  Per-branch distillation gap (SR vs MLP, validation):")
    dg = val.get("gap_gate", {})
    ds = val.get("gap_sign", {})
    da = val.get("gap_amplitude", {})
    print(f"    Gate       R2={dg.get('r2', 'N/A')}")
    print(f"    Sign       R2={ds.get('r2', 'N/A')}")
    print(f"    Amplitude  R2={da.get('r2', 'N/A')}")
    print(f"    Overall    R2={val.get('gap_overall', {}).get('r2', 'N/A')}")

    sr_t = val.get("sr_vs_target", {})
    print(f"\n  SR overall (validation):")
    print(f"    vs target  R2={sr_t.get('r2', 'N/A')}  RMSE={sr_t.get('rmse', 'N/A')}")

    print(f"\n  Equations (diagnostic form):")
    print(f"    Gate      : {eq_info.get('gate_raw_expr', 'N/A')}")
    print(f"    Sign      : {eq_info.get('sign_raw_expr', 'N/A')}")
    print(f"    Amplitude : {eq_info.get('amplitude_raw_expr', 'N/A')}")
    print(f"\n  Exported: equation_diagnostic.py (clip/max), equation_smooth.py (tanh/softplus)")


if __name__ == "__main__":
    main()
