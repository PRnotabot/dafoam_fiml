#!/bin/bash
set -euo pipefail

# Stage 2: Distill frozen branch MLPs with PySR.
# Uses branch-specific search settings.
#
# Usage:
#   ./distill_fiml_branch_mlp.sh
#   ./distill_fiml_branch_mlp.sh --mlp-run-dir structured_student_runs/staged_guided_heavy_v1
#   ./distill_fiml_branch_mlp.sh --amplitude-maxsize 24 --amplitude-niterations 30

if [[ -f "$DAFOAM_ROOT_PATH/loadDAFoam.sh" ]]; then
    source "$DAFOAM_ROOT_PATH/loadDAFoam.sh"
fi

MLP_DIR="${1:-fiml_mlp_runs/fiml_branch_mlp_v1}"

# Shift only if the first arg looks like a directory (not a --flag)
if [[ "${1:-}" && "${1:0:2}" != "--" ]]; then
    shift
fi

python distill_fiml_branch_mlp.py \
    --mlp-run-dir "$MLP_DIR" \
    --run-tag distill_v1 \
    "$@"
