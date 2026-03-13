# Structured Student Tutorial for `tf_training`

This folder contains the working path for turning the existing TensorFlow teacher into a better symbolic-regression teacher before running long PySR searches.

The current recommended workflow is:

1. load the existing teacher or raw `betaFIOmega` field
2. train a staged structured student on
   - `gate`
   - `sign`
   - `amplitude`
3. validate the student branch by branch
4. only then run long symbolic regression on the staged student

This README focuses on the staged student, because that is now the validated search direction for this case.

## 1. Current case

Current case:

- folder: `tutorials/Ramp_tf_pysr/tf_training`
- inputs: `PoD`, `VoS`, `PSoSS`, `KoU2`
- raw target: `betaFIOmega`
- preferred symbolic target: `delta_beta = beta - 1`
- saved TensorFlow teacher: `model/`

Validated heavy staged-student run:

- run directory:
  - [structured_student_runs/staged_guided_heavy_v1](/home/dafoamuser/mount/dafoam/tutorials/Ramp_tf_pysr/tf_training/structured_student_runs/staged_guided_heavy_v1)
- summary:
  - [structured_student_runs/staged_guided_heavy_v1/summary.json](/home/dafoamuser/mount/dafoam/tutorials/Ramp_tf_pysr/tf_training/structured_student_runs/staged_guided_heavy_v1/summary.json)

Key metrics from that run:

- student parameter count: `132`
- teacher parameter count: `2854`
- validation `R^2` vs teacher target: `0.6565`
- validation `R^2` vs raw beta: `0.4266`
- test `R^2` vs teacher target: `0.6313`
- gate AUC: `0.9528`
- sign AUC: `0.9825`
- amplitude active-set `R^2`: `0.6619`

Training-limit note:

- gate best epoch: `1200`
- sign best epoch: `1200`
- amplitude best epoch: `1199`
- this run still hit the epoch cap, so the current bottleneck is still partly optimization budget

## 2. Why this structure

The current teacher is not well-described by one global low-complexity expression.

The useful decomposition is:

```text
delta_beta(x) = gate(x) * sign(x) * amplitude(x)
beta(x) = 1 + delta_beta(x)
```

with smooth branch outputs:

- `gate(x)` in `[0, 1]`
- `sign(x)` in `[-1, 1]`
- `amplitude(x) >= 0`

The staged trainer is gradient-friendly because it uses:

- gate: sigmoid
- sign: sigmoid mapped to `[-1, 1]`
- amplitude: softplus

This is important if the final model will later be used inside gradient-based optimization.

## 3. Recommended branch features for this case

Current validated branch setup:

- `gate = [PoD]`
- `sign = [PoD, VoS, PSoSS, KoU2]`
- `amplitude = [PoD, VoS, PSoSS, KoU2]`

Why:

- `PoD` alone already gives strong activation separation, so it is the smallest good gate feature set
- restricting the sign branch too hard hurt performance
- amplitude is the hardest branch, so it keeps the full current feature set

## 4. Scripts in this folder

Recommended staged-student scripts:

- [train_structured_student_staged.py](/home/dafoamuser/mount/dafoam/tutorials/Ramp_tf_pysr/tf_training/train_structured_student_staged.py)
- [run_structured_student_staged_training.sh](/home/dafoamuser/mount/dafoam/tutorials/Ramp_tf_pysr/tf_training/run_structured_student_staged_training.sh)
- [run_structured_student_staged_heavy.sh](/home/dafoamuser/mount/dafoam/tutorials/Ramp_tf_pysr/tf_training/run_structured_student_staged_heavy.sh)

Older comparison scripts:

- [train_structured_student.py](/home/dafoamuser/mount/dafoam/tutorials/Ramp_tf_pysr/tf_training/train_structured_student.py)
- [run_structured_student_training.sh](/home/dafoamuser/mount/dafoam/tutorials/Ramp_tf_pysr/tf_training/run_structured_student_training.sh)

Symbolic-regression scripts from the earlier exploration are still available, but they should now be treated as downstream consumers of the staged student rather than the original teacher.

## 5. How to run

### 5.1 Recommended long run

This keeps the validated architecture and only increases optimization budget:

```bash
bash run_structured_student_staged_heavy.sh
```

This currently expands to:

```bash
python train_structured_student_staged.py \
  --feature-preset guided \
  --epochs 1200 \
  --patience 120 \
  --batch-size 512 \
  --run-tag staged_guided_heavy_v1
```

Use CLI overrides when needed:

```bash
bash run_structured_student_staged_heavy.sh \
  --run-tag staged_guided_heavy_v2 \
  --random-state 11
```

Use `bash` rather than `./...` on mounted filesystems where direct execution may be blocked.

### 5.2 Reproduce the validated long baseline

```bash
bash run_structured_student_staged_training.sh \
  --feature-preset guided \
  --run-tag staged_guided_long_v1 \
  --epochs 400 \
  --patience 40 \
  --batch-size 512
```

### 5.3 Change branch feature visibility explicitly

```bash
bash run_structured_student_staged_training.sh \
  --feature-preset guided \
  --gate-features PoD \
  --sign-features PoD,VoS,PSoSS,KoU2 \
  --amplitude-features PoD,VoS,PSoSS,KoU2 \
  --run-tag staged_custom_v1
```

## 6. Outputs

Each staged run writes to:

- `structured_student_runs/<run-tag>/`

Important outputs:

- `summary.json`
- `dataset_with_student_predictions.csv`
- `gate_model.keras`
- `sign_model.keras`
- `amplitude_model.keras`
- `gate_history.csv`
- `sign_history.csv`
- `amplitude_history.csv`

## 7. How to select features for other cases

This is the important part to generalize correctly.

Do not choose branch features by looking only at the final global beta fit.

Choose them by the conditional task of each branch.

### 7.1 Gate feature selection

Task:

- classify whether a cell is active or inactive
- target:
  - `active_mask = 1` if `|delta_beta| > threshold`
  - `0` otherwise

Data used:

- all cells

Selection rule:

- start from single-feature candidates
- choose the smallest feature set that gives stable held-out gate AUC and accuracy
- if one feature already saturates performance, keep only that feature

For the current case:

- `PoD` alone was enough

General guidance for other cases:

- gate features should represent activation onset or regime switching
- they should be the most compressible branch inputs
- keep the gate branch as small as possible

### 7.2 Sign feature selection

Task:

- classify the sign of the correction on active cells

Data used:

- active cells only

Target:

- `1` if `delta_beta > 0`
- `0` if `delta_beta < 0`

Selection rule:

- evaluate sign AUC and accuracy on active validation cells
- do not judge the sign branch using inactive cells
- if a restricted sign feature set hurts sign AUC materially, relax it

For the current case:

- restricting sign too hard was harmful
- allowing all 4 current features improved sign AUC materially

General guidance for other cases:

- sign features should explain whether the correction pushes up or down
- start broader than the gate branch
- prune only after sign performance is clearly stable

### 7.3 Amplitude feature selection

Task:

- regress `|delta_beta|` on active cells

Data used:

- active cells only

Selection rule:

- evaluate active-set amplitude `R^2` and RMSE
- inspect residuals after gate and sign are already accounted for
- add features only if they reduce active-set amplitude error materially

For the current case:

- amplitude is the hardest branch
- it currently justifies using the full 4-feature set

General guidance for other cases:

- amplitude features should explain correction strength, not just whether the correction exists
- choose them only from active-set regression quality
- if amplitude still underfits, then and only then consider adding new candidate features

### 7.4 Practical feature-selection protocol for a new case

Use this order:

1. define `delta_beta = beta - 1`
2. choose an active threshold
3. train gate candidates on all cells
4. fix the smallest acceptable gate feature set
5. train sign candidates on active cells only
6. fix the smallest acceptable sign feature set
7. train amplitude candidates on active cells only
8. only after branch metrics are good, train the final staged student

Important:

- do not let a good gate result trick you into thinking amplitude is solved
- do not choose amplitude features from whole-domain metrics
- do not add more features before checking whether the issue is optimization or architecture

## 8. How to tell whether the direction is right

The staged direction is right when:

- gate metrics are strong and stable
- sign metrics improve when sign visibility is relaxed
- amplitude improves when trained on active cells directly
- the combined student improves over the original symbolic baselines

That is exactly what happened in this case:

- one-shot symbolic fits were weak
- joint student training was weak
- staged branchwise training improved materially

## 9. Why the heavy setup increases compute the right way

The heavy setup intentionally increases:

- epochs
- patience
- total optimization time

It does not immediately widen the architecture further.

Reason:

- the current `132`-parameter staged student is already much smaller than the original `2854`-parameter teacher
- keeping it small preserves symbolic tractability
- extra compute is currently better spent on convergence than on more width

If the heavy run saturates and still underfits, the next change should be modest branch-width increases, not a return to a large monolithic MLP.

## 10. Next step after a strong student run

Once a staged student run is strong enough, the next symbolic step should be:

1. export dense evaluation data from the staged student
2. run symbolic regression branchwise
   - gate
   - sign
   - amplitude
3. compose the symbolic branches into one smooth deployable expression

That is a better SR target than the original TensorFlow teacher.

## 11. Related notes

Additional context in this folder:

- [symbolic_distillation_learnings.md](/home/dafoamuser/mount/dafoam/tutorials/Ramp_tf_pysr/tf_training/symbolic_distillation_learnings.md)
- [README_symbolic_distillation.md](/home/dafoamuser/mount/dafoam/tutorials/Ramp_tf_pysr/tf_training/README_symbolic_distillation.md)
