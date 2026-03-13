# FIML MLP->PySR Pipeline

Two-stage pipeline for symbolic distillation of turbulence correction models:

1. **`train_fiml_branch_mlp.py`** — train branch MLPs (gate/sign/amplitude) on FI data
2. **`distill_fiml_branch_mlp.py`** — distill frozen MLP outputs with branch-specific PySR

For background on why one-shot SR fails and branchwise decomposition is needed, see `symbolic_distillation_learnings.md`.

---

## Fundamentals

The turbulence correction field `betaFIOmega` from field inversion is sparse: about 13% of cells deviate from the baseline `beta = 1` by more than the active threshold. The correct decomposition is:

```
beta(x) = 1 + g(x) * s(x) * a(x)
```

| Branch    | Role | Default features | Hidden | Activation | Output range | Params |
|-----------|------|------------------|--------|------------|--------------|--------|
| Gate      | Where is the correction active? | PoD | [4] | tanh → sigmoid | [0, 1] | 9 |
| Sign      | Which direction? | PoD, VoS, PSoSS, KoU2 | [6] | tanh → sigmoid → 2x-1 | [-1, 1] | 37 |
| Amplitude | How large? | PoD, VoS, PSoSS, KoU2 | [8] | tanh → softplus | [0, ∞) | 49 |

**Total: 95 parameters.**

**Key insight**: PySR distills the branch MLP outputs (smooth, continuous), not the raw FI targets (binary, noisy). The MLP absorbs noise; PySR gets a clean signal.

**Design principle**: The MLP teacher is frozen before PySR runs. This decouples teacher variance from SR variance — when tuning PySR settings, the teacher is constant.

---

## Comparison with existing approaches

| Approach | Script | SR target | Best val R² |
|----------|--------|-----------|-------------|
| One-shot SR | `run_symbolic_distillation.py` | raw `delta_beta` | 0.033 |
| Gated SR | `run_symbolic_gated_distillation.py` | raw binary gate + raw value | 0.120 |
| Sign-aware gated SR | `run_symbolic_signed_gated_distillation.py` | raw binary gate/sign + raw amplitude | 0.240 |
| **This pipeline** | `train_fiml_branch_mlp.py` + `distill_fiml_branch_mlp.py` | **smooth MLP branch outputs** | — |

This pipeline also differs from the prior single-script approach (`run_fiml_mlp_pysr.py`) in:

- **Decoupled stages**: freeze the MLP teacher, then iterate on PySR settings without retraining
- **SR branch metrics**: reports symbolic quality per branch (AUC, R²), not just MLP quality
- **Per-branch distillation gaps**: measures how well PySR reproduces each MLP branch
- **Branch-specific PySR settings**: gate gets simpler search, amplitude gets larger budget
- **Dual equation export**: diagnostic (clip/max) and smooth (tanh/softplus) forms
- **Compatible with existing runs**: can distill `structured_student_runs/` directly

---

## Branch-specific PySR defaults

| Setting | Gate | Sign | Amplitude |
|---------|------|------|-----------|
| Operators | `+, *` | `+, -, *` | `+, -, *` |
| Unary | `tanh` | `tanh` | `tanh` |
| maxsize | 10 | 14 | 20 |
| maxdepth | 5 | 6 | 8 |
| parsimony | 3.0e-3 | 2.0e-3 | 1.5e-3 |
| niterations | 15 | 20 | 25 |
| populations | 4 | 6 | 8 |
| population_size | 24 | 28 | 32 |
| sample_size | 5000 | 3000 (active) | 3000 (active) |

---

## Usage

### Prerequisites

```bash
source $DAFOAM_ROOT_PATH/loadDAFoam.sh
pip install pysr tensorflow scikit-learn pandas
```

Data: `c1_data/` and `c2_data/` with `PoD`, `VoS`, `PSoSS`, `KoU2`, and `betaFIOmega` fields.

### Stage 1: Train branch MLPs

```bash
cd tf_training
./train_fiml_branch_mlp.sh
```

Or directly:

```bash
python train_fiml_branch_mlp.py --target-source raw --run-tag fiml_v1
```

Check MLP quality before proceeding:

```
  Gate       AUC > 0.93
  Sign       AUC > 0.88
  Amplitude  R²  > 0.50  (on active cells)
```

### Stage 2: Distill with PySR

```bash
./distill_fiml_branch_mlp.sh fiml_mlp_runs/fiml_v1
```

Or directly:

```bash
python distill_fiml_branch_mlp.py --mlp-run-dir fiml_mlp_runs/fiml_v1 --run-tag distill_v1
```

### Distill an existing structured student run

No retraining needed — distill the validated teacher directly:

```bash
python distill_fiml_branch_mlp.py \
    --mlp-run-dir structured_student_runs/staged_guided_heavy_v1 \
    --run-tag distill_student_v1
```

### Override branch-specific PySR settings

```bash
python distill_fiml_branch_mlp.py \
    --mlp-run-dir fiml_mlp_runs/fiml_v1 \
    --amplitude-maxsize 24 \
    --amplitude-niterations 30 \
    --amplitude-parsimony 1.0e-3 \
    --run-tag distill_amp_heavy
```

---

## Outputs

### Stage 1: `fiml_mlp_runs/<tag>/`

| File | Description |
|------|-------------|
| `gate_model.keras`, `sign_model.keras`, `amplitude_model.keras` | Branch MLPs |
| `*_history.csv` | Training curves |
| `predictions.csv` | All cells with branch predictions and split labels |
| `summary.json` | MLP branch metrics |

### Stage 2: `fiml_distill_runs/<tag>/`

| File | Description |
|------|-------------|
| `equation_diagnostic.py` | Equations with clip/max (exact reconstruction) |
| `equation_smooth.py` | Equations with tanh/softplus (gradient-safe deployment) |
| `equation_beta.txt` | Full beta in both forms |
| `*_equation.txt` | Per-branch expressions (both forms) |
| `pareto_*.json` | Pareto fronts per branch |
| `dataset_with_sr.csv` | All cells with MLP and SR predictions |
| `summary.json` | MLP metrics, SR metrics, and per-branch distillation gaps |

### Key metrics in `summary.json` (distillation)

The summary nests metrics by split (`train`, `val`, `test`). Each split contains:

- `mlp_gate`, `mlp_sign`, `mlp_amplitude` — MLP branch quality
- `sr_gate`, `sr_sign`, `sr_amplitude` — **SR branch quality** (the actual symbolic model)
- `gap_gate`, `gap_sign`, `gap_amplitude` — per-branch distillation gap (SR vs MLP)
- `gap_overall` — overall distillation gap
- `sr_vs_target` — SR reconstruction vs original FI target

The **per-branch distillation gap** is the most actionable diagnostic:

- R² > 0.9 — PySR successfully compressed this branch
- R² 0.7–0.9 — increase `--<branch>-maxsize` or `--<branch>-niterations`
- R² < 0.7 — reconsider operator set or branch architecture

---

## Recommended workflow

1. **Train once**: `train_fiml_branch_mlp.py --run-tag fiml_v1`
2. **Check MLP quality**: gate AUC > 0.93, sign AUC > 0.88, amplitude R² > 0.50
3. **Distill**: `distill_fiml_branch_mlp.py --mlp-run-dir fiml_mlp_runs/fiml_v1`
4. **Check per-branch gaps**: if one branch gap is large, increase its budget and re-distill (the MLP is frozen)
5. **Deploy**: use `equation_smooth.py` for gradient-based optimization, `equation_diagnostic.py` for comparison
