# References: Periodic Hill FIML with Experimental Data

## Experimental Data Sources

### Primary Experimental Reference
- **Rapp, C., Manhart, M.** (2011). "Flow over periodic hills: an experimental study."
  *Experiments in Fluids*, 51, 247-269.
  - PIV and LDA measurements at Re = 5,600, 10,000, 19,000, 37,000
  - Water channel facility at TU Munich (Fachgebiet Hydromechanik)
  - Mean velocity profiles and Reynolds stresses at multiple x/H stations
  - DOI: https://doi.org/10.1007/s00348-011-1045-y
  - Springer: https://link.springer.com/article/10.1007/s00348-011-1045-y

### DNS/LES Benchmark Data (Cross-Validation)
- **Breuer, M., Peller, N., Rapp, C., Manhart, M.** (2009). "Flow over periodic hills -
  Numerical and experimental study in a wide range of Reynolds numbers."
  *Computers & Fluids*, 38(2), 433-457.
  - DNS at Re = 5,600; LES at Re = 10,595
  - Mean velocities, pressure, Reynolds stresses, anisotropy-invariant maps
  - DOI: https://doi.org/10.1016/j.compfluid.2008.05.002
  - ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S0045793008001126

## Data Download Links

### NASA Turbulence Modeling Resource
- **LES Data (Re = 10,595)**:
  https://turbmodels.larc.nasa.gov/Other_LES_Data/2dhill_periodic.html
  - Hill geometry, grid, average velocities, Reynolds stresses, budgets, Cf, Cp

- **DNS Data (Re = 5,600, Parameterized Geometries)**:
  https://turbmodels.larc.nasa.gov/Other_DNS_Data/parameterized_periodic_hills.html
  - 29 simulations at Re = 5,600 with varied geometries
  - Download: DNS_29_Periodic_Hills.zip (1.2 GB)
  - Contains mean velocities, pressure, Reynolds stresses at all grid points

- **DNS Data (Re = 2,800)**:
  https://turbmodels.larc.nasa.gov/Other_DNS_Data/2dhill_periodic_compress.html

### GitHub Repository (Parameterized DNS Dataset)
- **Xiao, H. et al.** Para-database for PIML:
  https://github.com/xiaoh/para-database-for-PIML
  - Original DNS data plus RANS-resolution interpolated fields
  - Suitable for machine learning applications

### ERCOFTAC Knowledge Base
- **2D Periodic Hill Flow (UFR 3-30)**:
  https://www.kbwiki.ercoftac.org/w/index.php/Abstr:2D_Periodic_Hill_Flow
  - Experimental and computational reference data
  - Description: https://www.kbwiki.ercoftac.org/w/index.php/UFR_3-30_Description
  - Contributors: C. Rapp, M. Breuer, M. Manhart, N. Peller (TU Munich)

### ERCOFTAC Classic Collection
- **Case 081**: http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case081
  - LES data subset

## FIML and DAFoam References

### DAFoam Field Inversion
- **Bidar, O.** (2024). "Data-driven Augmentation of Turbulence Models for Complex Fluid Flows."
  PhD Thesis, University of Sheffield.
  - Full thesis: https://etheses.whiterose.ac.uk/id/eprint/35649/
  - Supervisors: S. Anderson, N. Qin
  - Periodic hill FIML with sensor fusion, aerodynamic shape optimization

- **Bidar, O., Anderson, S., Qin, N.** (2022). "An Open-source Adjoint-based Field
  Inversion Tool for Data-driven RANS Modelling."
  - ResearchGate: https://www.researchgate.net/publication/361442133

- **Bidar, O., Anderson, S., Qin, N.** (2022). "Turbulent Mean Flow Reconstruction
  Based on Sparse Multi-sensor Data and Adjoint-based Field Inversion."
  - ResearchGate: https://www.researchgate.net/publication/361442286

- **Bidar, O., Anderson, S., Qin, N.** (2024). "Aerodynamic Shape Optimisation
  Using a Machine Learning-Augmented Turbulence Model." AIAA SciTech Forum.
  - DOI: https://doi.org/10.2514/6.2024-1231

### FIML Methodology
- **Parish, E.J., Duraisamy, K.** (2016). "A paradigm for data-driven predictive
  modeling using field inversion and machine learning." *Journal of Computational Physics*, 305, 758-774.
  - DOI: https://doi.org/10.1016/j.jcp.2015.11.012

- **Holland, J.R., Baeder, J.D., Duraisamy, K.** (2019). "Field Inversion and Machine
  Learning With Embedded Neural Networks: Physics-Consistent Neural Network Training."
  AIAA Aviation Forum.
  - DOI: https://doi.org/10.2514/6.2019-3200

### Symbolic Regression for Turbulence
- **Weatheritt, J., Sandberg, R.D.** (2016). "A novel evolutionary algorithm applied
  to algebraic modifications of the RANS stress-strain relationship."
  *Journal of Computational Physics*, 325, 22-37.

- **Zhao, Y., Akolekar, H.D., Weatheritt, J., Michelassi, V., Sandberg, R.D.** (2020).
  "RANS turbulence model development using CFD-driven machine learning."
  *Journal of Computational Physics*, 411, 109413.

- **Schmelzer, M., Dwight, R.P., Cinnella, P.** (2020). "Discovery of algebraic
  Reynolds-stress models using sparse symbolic regression."
  *Flow, Turbulence and Combustion*, 104, 579-603.

### DAFoam Framework
- **He, P., Mader, C.A., Martins, J.R.R.A., Maki, K.J.** (2018). "DAFoam: An
  Open-Source Adjoint Framework for Multidisciplinary Design Optimization with OpenFOAM."
  *AIAA Journal*, 58(3), 1304-1319.
  - DOI: https://doi.org/10.2514/1.J058853
  - GitHub: https://github.com/mdolab/dafoam
  - Documentation: https://dafoam.github.io

- **DAFoam Field Inversion Tutorial (Periodic Hill)**:
  https://dafoam.github.io/mydoc_tutorials_field_inversion_ph.html

- **DAFoam Field Inversion Tutorial (Ramp)**:
  https://dafoam.github.io/mydoc_tutorials_field_inversion_ramp.html

- **DAFoam Publications**:
  https://dafoam.github.io/mydoc_docs_publications.html

## Geometry References

### Original Periodic Hill Geometry
- **Almeida, G.P., Durao, D.F.G., Heitor, M.V.** (1993). "Wake flows behind
  two-dimensional model hills." *Experimental Thermal and Fluid Science*, 7(1), 87-101.

- **Mellen, C.P., Froehlich, J., Rodi, W.** (2000). "Large eddy simulation of the
  flow over periodic hills." 16th IMACS World Congress, Lausanne.

### Additional DNS Studies
- **Krank, B., Fehn, M., Wall, W.A., Kronbichler, M.** (2018). "Direct Numerical
  Simulation of Flow over Periodic Hills up to Re_H = 10,595."
  *Flow, Turbulence and Combustion*, 101, 521-551.
  - DOI: https://doi.org/10.1007/s10494-018-9941-3

## Software Dependencies

- **PySR** (Symbolic Regression): https://github.com/MilesCranmer/PySR
  - Install: `pip install pysr`
  - Documentation: https://astroautomata.com/PySR/

- **OpenFOAM** (v1812): https://www.openfoam.com/

- **OpenMDAO**: https://openmdao.org/

- **PETSc**: https://petsc.org/

## Contact for Experimental Data

For the original experimental data from Rapp & Manhart (2011):
- Technische Universitaet Muenchen, Fachgebiet Hydromechanik
- The data may be available through the ERCOFTAC QNET database

For the LES/DNS benchmark data:
- L. Temmerman: lionel.temmerman@numeca.be
- M. Leschziner: mike.leschziner@imperial.ac.uk
