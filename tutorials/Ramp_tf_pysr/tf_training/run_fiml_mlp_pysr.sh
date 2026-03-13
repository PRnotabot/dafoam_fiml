#!/bin/bash
set -euo pipefail

# FIML MLP->PySR Pipeline
#
# Trains branch MLPs on field inversion data, then distills each
# branch's MLP output with PySR for cleaner symbolic regression.
#
# Prerequisites:
#   - c1_data/ and c2_data/ with FI features + betaFIOmega
#   - DAFoam environment (for pyofm)
#   - PySR installed (pip install pysr)
#
# Usage:
#   ./run_fiml_mlp_pysr.sh                    # full pipeline
#   ./run_fiml_mlp_pysr.sh --skip-sr          # MLP only
#   ./run_fiml_mlp_pysr.sh --target-source teacher  # distill teacher instead

if [[ -f "$DAFOAM_ROOT_PATH/loadDAFoam.sh" ]]; then
    source "$DAFOAM_ROOT_PATH/loadDAFoam.sh"
fi

python run_fiml_mlp_pysr.py \
    --target-source raw \
    --feature-preset guided \
    --gate-hidden 4 \
    --sign-hidden 8 \
    --amplitude-hidden 8 \
    --l1 1.0e-5 \
    --epochs 1200 \
    --patience 120 \
    --batch-size 512 \
    --sr-sample-size 5000 \
    --sr-active-sample-size 3000 \
    --niterations 18 \
    --populations 6 \
    --population-size 28 \
    --ncycles-per-iteration 60 \
    --maxsize 18 \
    --maxdepth 8 \
    --parsimony 2.5e-3 \
    --binary-operators "+,-,*" \
    --unary-operators "tanh" \
    --run-tag fiml_mlp_pysr_v1 \
    "$@"
