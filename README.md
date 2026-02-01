# dafoam_fiml

This repository contains documentation and notes related to Field Inversion and Machine Learning (FIML) applied to DAFoam (Discrete Adjoint with OpenFOAM).

- Docs summary:
	- `docs/FIML.md`: Overview of the FIML pipeline, feature engineering, neural-network parameterization, and how the regression model integrates with DAFoam.
	- `docs/PipelineAnatomy_DAFoamOptimization.md`: Detailed breakdown of the DAFoam optimization pipeline (configuration, mesh warping, primal/adjoint solves, and gradient computation).
	- `docs/TensorProgression.md`: Step-through tracing of vectors, Jacobians, adjoints, and the full FIML dataflow on a small example.

These notes are intended to help researchers and developers understand and reproduce the FIML workflow implemented alongside DAFoam.

See the `docs/` folder for more details.