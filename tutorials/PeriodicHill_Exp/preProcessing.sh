#!/bin/bash

# Check if the OpenFOAM environment is loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit 1
fi

# Copy initial and boundary condition files
cp -r 0.orig 0

# Generate experimental reference data (UData and probePointCoords.json)
echo "Generating experimental reference data..."
python genExpData.py
echo "Done. Created 0/UData and probePointCoords.json"
