#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f /home/dafoamuser/dafoam/loadDAFoam.sh ]; then
    # The DAFoam loader references unset variables internally.
    set +u
    # shellcheck disable=SC1091
    . /home/dafoamuser/dafoam/loadDAFoam.sh
    set -u
fi

python "$SCRIPT_DIR/train_structured_student_staged.py" \
    --feature-preset guided \
    --epochs 1200 \
    --patience 120 \
    --batch-size 512 \
    --run-tag staged_guided_heavy_v1 \
    "$@"
