# Symbolic Distillation for `tf_training`

This folder now has two PySR paths for the current `k-omega` Ramp case:

- features: `PoD`, `VoS`, `PSoSS`, `KoU2`
- raw target: `betaFIOmega`
- default symbolic target: `delta_beta_teacher = beta_teacher - 1`
- one-shot script: `run_symbolic_distillation.py`
- gated script: `run_symbolic_gated_distillation.py`
- sign-aware gated script: `run_symbolic_signed_gated_distillation.py`

## Recommended strategy

Start with the saved TensorFlow model as the teacher and distill `delta_beta` instead of raw `beta`.

Why this is the right first pass here:

- the correction is a small perturbation around `beta = 1`
- the saved TF teacher is smoother than the raw FI field
- the active correction region is sparse, so a gated model is better aligned with the data than one global expression
- the script still reports the symbolic model against the raw `betaFIOmega` field, so the distillation gap stays visible

Use the sign-aware gated script first. Keep the simpler gated and one-shot scripts as baselines.

If the teacher-target fit looks good but the raw-beta gap stays too large, rerun with `--target-source raw` to search for a more direct but potentially less stable equation.

## Recommended sign-aware run

```bash
bash run_symbolic_signed_gated_distillation.sh \
  --target-source teacher \
  --run-tag signed_gated_teacher \
  --sample-size 12000 \
  --active-sample-size 4000 \
  --niterations 18 \
  --populations 6 \
  --population-size 24
```

## Recommended gated run

```bash
bash run_symbolic_gated_distillation.sh \
  --target-source teacher \
  --run-tag gated_teacher \
  --sample-size 12000 \
  --value-sample-size 4000 \
  --niterations 18 \
  --populations 6 \
  --population-size 24
```

## Quick run

```bash
bash run_symbolic_distillation.sh \
  --target-source teacher \
  --run-tag quick_teacher \
  --sample-size 10000 \
  --niterations 12 \
  --populations 4 \
  --population-size 24
```

## Stronger run

```bash
bash run_symbolic_distillation.sh \
  --target-source teacher \
  --run-tag balanced_teacher \
  --sample-size 12000 \
  --niterations 24 \
  --populations 8 \
  --population-size 32 \
  --ncycles-per-iteration 80 \
  --maxsize 20
```

## Raw-beta run

```bash
bash run_symbolic_distillation.sh \
  --target-source raw \
  --run-tag raw_direct \
  --sample-size 12000 \
  --niterations 24 \
  --populations 8 \
  --population-size 32
```

## Parallel PySR

The symbolic scripts now expose PySR runtime controls directly:

- `--parallelism auto|serial|multithreading|multiprocessing`
- `--julia-num-threads N` for single-node multithreading
- `--procs N` for multiprocessing
- `--cluster-manager slurm|pbs|lsf|sge|qrsh|scyld|htc` for scheduler-managed workers

Recommended workstation run:

```bash
bash run_symbolic_signed_gated_distillation.sh \
  --target-source teacher \
  --run-tag signed_gated_mt16 \
  --parallelism multithreading \
  --julia-num-threads 16 \
  --populations 48 \
  --population-size 32 \
  --niterations 40 \
  --sample-size 12000 \
  --active-sample-size 4000
```

Recommended single-node multiprocessing run:

```bash
bash run_symbolic_signed_gated_distillation.sh \
  --target-source teacher \
  --run-tag signed_gated_mp16 \
  --parallelism multiprocessing \
  --procs 16 \
  --populations 48 \
  --population-size 32 \
  --niterations 40 \
  --sample-size 12000 \
  --active-sample-size 4000
```

Recommended Slurm run from within an allocation:

```bash
bash run_symbolic_signed_gated_distillation.sh \
  --target-source teacher \
  --run-tag signed_gated_slurm32 \
  --cluster-manager slurm \
  --procs 32 \
  --populations 96 \
  --population-size 32 \
  --niterations 40 \
  --sample-size 12000 \
  --active-sample-size 4000
```

Practical notes:

- `--deterministic` only works with `--parallelism serial`.
- In `multithreading` mode, use `--julia-num-threads`; `--procs` is intentionally rejected.
- In `multiprocessing` mode, the scripts force `JULIA_NUM_THREADS=1` unless you explicitly override it, to avoid nested oversubscription.
- If `populations < workers`, some workers will sit idle. PySR guidance is to use at least as many populations as workers, and often around `3 * workers`.

## Outputs

Each run writes to `sr_results/<run-tag>/`:

- `dataset_with_predictions.csv`
- `summary.json`
- `equation.py`
- `equation_sympy.txt`
- `equation_beta.txt`
- `equation.tex`
- `pareto_front.json`

## Notes

- Run with MPI size 1.
- If you refresh the TF teacher, re-run `python trainModel.py` first.
- The default operator set is intentionally narrow: `+`, `-`, `*`, `tanh`.
- Add `/`, `sqrt`, `log`, or `exp` only if the narrow search clearly underfits.
- Use `bash run_symbolic_*.sh ...` on mounted filesystems where direct execution may be blocked.
