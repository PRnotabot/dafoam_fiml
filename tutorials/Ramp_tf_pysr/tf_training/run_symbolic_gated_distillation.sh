#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f /home/dafoamuser/dafoam/loadDAFoam.sh ]; then
    set +u
    # shellcheck disable=SC1091
    . /home/dafoamuser/dafoam/loadDAFoam.sh
    set -u
fi

python "$SCRIPT_DIR/run_symbolic_gated_distillation.py" "$@"
