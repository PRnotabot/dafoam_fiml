#!/bin/bash
set -euo pipefail

# Stage 1: Train branch MLPs on field inversion data.
# Output is consumed by distill_fiml_branch_mlp.sh.

if [[ -f "$DAFOAM_ROOT_PATH/loadDAFoam.sh" ]]; then
    source "$DAFOAM_ROOT_PATH/loadDAFoam.sh"
fi

python train_fiml_branch_mlp.py \
    --target-source raw \
    --feature-preset guided \
    --gate-hidden 4 \
    --sign-hidden 6 \
    --amplitude-hidden 8 \
    --l1 1.0e-5 \
    --epochs 1200 \
    --patience 120 \
    --batch-size 512 \
    --run-tag fiml_branch_mlp_v1 \
    "$@"
