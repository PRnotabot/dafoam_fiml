#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f /home/dafoamuser/dafoam/loadDAFoam.sh ]; then
    # shellcheck disable=SC1091
    . /home/dafoamuser/dafoam/loadDAFoam.sh
fi

python "$SCRIPT_DIR/train_structured_student.py" "$@"
