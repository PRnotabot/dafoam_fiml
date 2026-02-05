#!/bin/bash

# Pre-processing script for cylinder FIML tutorial
# Run this after placing mesh files in c1/constant/polyMesh/

cd c1

# Copy initial conditions
cp -r 0_orig 0

# Generate reference Cp field from experimental data
# This creates the pRef field needed for field inversion
python ../createRefField.py

echo "Pre-processing complete for c1"
