# Symbolic Distillation Learnings for `tf_training`

Date: 2026-03-12

This note captures what we learned from the symbolic-distillation work in this folder on the current `k-omega` Ramp case:

- inputs: `PoD`, `VoS`, `PSoSS`, `KoU2`
- raw target: `betaFIOmega`
- available teacher: saved TensorFlow model in `model/`

The goal of this note is to separate:

- what is directly established by the runs we completed
- what is a reasoned design recommendation for the next round

## 1. Current case summary

- Dataset size: `40000` cell samples across `c1_data` and `c2_data`
- Teacher parameter count: `2854`
- Teacher vs raw target:
  - RMSE: `0.01296`
  - `R^2`: `0.599`
  - correlation: `0.777`
- Active fraction for `delta_beta = beta - 1` using threshold `0.01`: `0.134`

Interpretation:

- the correction is sparse
- most cells are near the baseline `beta = 1`
- the teacher is smoother than the raw field, but it is not a perfect surrogate of the raw field

## 2. What is established from the runs

### 2.1 One global symbolic equation is the wrong search problem here

Direct one-shot PySR underfit badly.

Observed runs:

- `smoke_teacher`
  - validation `R^2` vs teacher target: `0.033`
  - best form collapsed to a tiny linear correction in `KoU2`
- `teacher_balanced_v1`
  - validation `R^2` vs teacher target: `-0.086`
  - balancing alone did not fix the structural mismatch

Conclusion:

- the target is not well-described by a single global low-complexity formula over all cells
- the sparsity around `beta = 1` must be treated explicitly

### 2.2 Distilling `delta_beta` is better than fitting raw `beta`

For this case, `delta_beta = beta - 1` is the better symbolic target because:

- `beta = 1` is the physical no-correction baseline
- `delta_beta = 0` cleanly represents inactive regions
- it makes the search naturally compatible with gating
- it avoids wasting symbolic capacity on rediscovering the constant baseline

This is a design conclusion supported by the behavior of the runs above.

### 2.3 The teacher field has a gated structure

Branchwise models improved over one-shot models.

Observed runs:

- `gated_teacher_v1`
  - validation `R^2` vs teacher target: `0.120`
  - validation `R^2` vs raw beta: `0.105`
  - gate AUC: `0.945`
- `signed_gated_teacher_v1`
  - validation `R^2` vs teacher target: `0.123`
  - sign branch was weak in this low-budget run
- `signed_gated_teacher_v2`
  - validation `R^2` vs teacher target: `0.240`
  - validation `R^2` vs raw beta: `0.187`
  - gate AUC: `0.945`
  - gate accuracy: `0.880`
  - sign AUC: `0.905`
  - sign accuracy: `0.872`

Conclusion:

- the current teacher is better represented as
  - gate
  - sign
  - amplitude
- than as one global expression

### 2.4 Current feature roles in the best-performing decomposition

From the best run `signed_gated_teacher_v2`:

- gate is mostly driven by `PoD`
- sign is mostly driven by `KoU2`
- amplitude is mostly driven by `PSoSS`

Best current equation:

```text
beta = 1.0
     + clip(0.5 * (1 + tanh(11.853484 * (PoD - 0.4413447))), 0, 1)
       * clip(KoU2 + 1.4019966 - tanh(2.9110258 * KoU2) / (KoU2 + 0.037450533), -1, 1)
       * max(0, 0.041860998 * (1.0799632 - PSoSS))
```

This equation is useful as a diagnostic result, but not yet the final deployable form because `clip` and `max` are not ideal for gradient-based optimization.

## 3. What we learned about teacher simplification

## 3.1 A smaller teacher is helpful, but architectural simplicity matters more than weight sparsity

The main lesson is:

- parameter count alone is not the right simplicity metric for SR
- functional simplicity matters more

Even a `20x20`, `30x30`, or `8x8x8` MLP can still represent a tangled map that is hard for SR to compress.

So the most useful inductive bias is structural:

- predict `delta_beta`, not raw `beta`
- use branchwise decomposition instead of one monolithic MLP
- enforce smooth bounded outputs for each branch
- restrict which features each branch can see

Recommended branchwise student form:

```text
g(x) = 0.5 * (1 + tanh(u_g(x)))
s(x) = tanh(u_s(x))
a(x) = softplus(u_a(x))
delta_beta(x) = g(x) * s(x) * a(x)
beta(x) = 1 + delta_beta(x)
```

This kind of structure is likely more valuable than generic unstructured pruning.

## 3.2 Generic pruning is not the main next step

Pruning may help reduce parameter count, but we do not currently have evidence that it will make the learned function more symbolically compressible by itself.

Better next step:

- large or medium teacher
- small structured student
- then PySR on the structured student

Working recommendation:

- use mild `L1` or group-sparse penalties on the student
- use smooth activations such as `tanh` and `softplus`
- use pruning only as optional cleanup after the structure is already correct

## 4. What the amplitude branch is

For the decomposition

```text
delta_beta(x) = g_active(x) * s_sign(x) * a_amp(x)
```

the amplitude branch is:

```text
a_amp(x) = |delta_beta(x)|   on active cells
```

Meaning:

- gate says whether the correction is active
- sign says whether the active correction is positive or negative
- amplitude says how large the active correction should be

## 5. What we learned about amplitude in this specific case

Important correction to an earlier hypothesis:

- based on today’s diagnostics, we do not yet have evidence that amplitude requires additional features

Reason:

- with the current 4 features only, a flexible non-symbolic model on active cells could fit amplitude well
- similarly, sign was also strongly learnable once the problem was decomposed properly

So the main bottleneck right now is:

- symbolic compression of amplitude

not clearly:

- missing information in the current feature set

This means that adding more features is not the default next move.

## 6. When would extra features become justified?

Additional features should only be added if a small structured student still underfits amplitude on held-out active cells.

The criterion is conditional, not global:

- first fit gate
- then fit sign
- then fit amplitude on active cells only
- inspect amplitude residuals
- add a feature only if it materially reduces held-out active-set amplitude error

So the right question is not:

- "does this feature improve the global fit?"

but:

- "does this feature explain amplitude residuals after activation and sign are already accounted for?"

At the current stage, that condition has not yet been demonstrated.

## 7. How to know the search space is “right”

The search space is not “right” just because one run returns a decent equation.

It is right enough when:

- short independent runs recover the same qualitative structure
- increasing budget improves the same structure instead of jumping to unrelated forms
- the residual error points to a specific missing branch or missing physics
- the final expression families are stable under resampling

That is what happened here:

- one-shot failed
- gated improved
- sign-aware gated improved again

This means the main search-space correction was structural, not simply “run PySR longer”.

## 8. Implications for gradient-based optimization

If the final symbolic model must be used in gradient-based optimization, the deployed form should be:

- smooth
- bounded
- domain-safe
- easy to differentiate

Good operator families:

- affine terms
- `+`, `-`, `*`
- `tanh`
- `sigmoid`
- `softplus`
- carefully protected division
- optionally `exp` if it is controlled

Bad or risky final operators:

- `clip`
- `max`
- `min`
- `abs`
- `sign`
- hard piecewise branching
- free-form division with unstable denominators

Therefore, the current best equation should be treated as:

- a discovery result

not yet:

- the final optimizer-facing implementation

Preferred smooth deployable form:

```text
beta(x) = 1 + g(x) * s(x) * a(x)
```

with

- `g(x)` in `[0, 1]` using `0.5*(1+tanh(.))` or sigmoid
- `s(x)` in `[-1, 1]` using `tanh(.)`
- `a(x) >= 0` using `softplus(.)`

## 9. Practical next recommendation

Highest-value next experiment:

1. Keep the current 4 features.
2. Train a tiny sign-aware branchwise student:
   - branch inputs restricted intentionally
   - smooth bounded outputs
   - mild `L1` or group sparsity
3. Distill that student with PySR using structural templates rather than a free global search.
4. Only if the amplitude branch still underfits:
   - investigate targeted additional features for amplitude only

This is more defensible than:

- adding many more features now
- relying on generic pruning alone
- running much longer one-shot PySR searches on the current global formulation

## 10. Files produced in this folder

Execution scripts:

- `run_symbolic_distillation.py`
- `run_symbolic_gated_distillation.py`
- `run_symbolic_signed_gated_distillation.py`

Best run artifacts:

- `sr_results/signed_gated_teacher_v2/summary.json`
- `sr_results/signed_gated_teacher_v2/equation.py`
- `sr_results/signed_gated_teacher_v2/equation_beta.txt`
- `sr_results/signed_gated_teacher_v2/dataset_with_predictions.csv`

## 11. Questions needing clarification before the next revision

1. Is the immediate next goal offline symbolic fidelity, or a deployable smooth equation for optimizer use?
2. Do you want this note to stay as a working lab note, or be rewritten as a more formal project memo?
3. For the next student model, do you want to preserve all 4 current features in every branch, or do you want branch-specific feature restrictions from the start?
4. Should the next experiment prioritize exact symbolic simplicity, or smooth deployability under gradient-based optimization?
