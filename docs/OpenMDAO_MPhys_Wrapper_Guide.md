# The Definitive Guide to Creating OpenMDAO / MPhys Wrappers

**For integrating CFD, FEA, CAD, and ML solvers into multidisciplinary analysis and optimization (MDAO) frameworks.**

---

## Table of Contents

1. [Foundational Concepts](#1-foundational-concepts)
2. [OpenMDAO Component Types and When to Use Each](#2-openmdao-component-types-and-when-to-use-each)
3. [The MPhys Builder Pattern](#3-the-mphys-builder-pattern)
4. [Derivatives: The Make-or-Break of Gradient-Based MDAO](#4-derivatives-the-make-or-break-of-gradient-based-mdao)
5. [Wrapper Architecture Patterns](#5-wrapper-architecture-patterns)
6. [SU2 Wrapper](#6-su2-wrapper)
7. [FEniCS Wrapper](#7-fenics-wrapper)
8. [NVIDIA PhysicsNeMo Surrogate Wrapper](#8-nvidia-physicsnemo-surrogate-wrapper)
9. [ParaBlade Wrapper](#9-parablade-wrapper)
10. [Open-Source CAD Tool Wrappers](#10-open-source-cad-tool-wrappers)
11. [Closed-Source Tool Integration (COMSOL, ANSYS Fluent)](#11-closed-source-tool-integration-comsol-ansys-fluent)
12. [MPI and Parallel Execution](#12-mpi-and-parallel-execution)
13. [Testing and Verification](#13-testing-and-verification)
14. [Common Pitfalls and Debugging](#14-common-pitfalls-and-debugging)
15. [Reference: Existing MPhys-Compatible Wrappers](#15-reference-existing-mphys-compatible-wrappers)

---

## 1. Foundational Concepts

### What is OpenMDAO?

OpenMDAO is an open-source Python framework for multidisciplinary design, analysis,
and optimization. It provides:

- A unified component interface for wrapping any analysis code
- Automatic assembly of total derivatives across coupled systems
- Built-in optimizers and interfaces to external optimizers (SNOPT, IPOPT, etc.)
- MPI-based parallelism for large-scale problems

### What is MPhys?

MPhys (Multiphysics) is a library built on top of OpenMDAO that standardizes how
high-fidelity solvers are assembled into multiphysics problems. It provides:

- A **Builder** pattern for plugging in new solvers
- Pre-built **Scenarios** for common problem types (aerodynamic, structural, aerostructural)
- Standardized variable naming conventions (`x_aero0`, `x_struct0`, `f_aero`, etc.)
- Coupling infrastructure (load/displacement transfer, convergence)

### How They Relate

```
+---------------------------------------------------------------+
|                        OPTIMIZATION                            |
|   Driver (SNOPT, IPOPT, scipy)                                 |
|       |                                                        |
|       v                                                        |
|   +-----------------------------------------------------------+|
|   |                   OpenMDAO Problem                        ||
|   |                                                           ||
|   |   +-----------------------------------------------------+||
|   |   |              MPhys Multipoint Group                  |||
|   |   |                                                      |||
|   |   |   +-----------+  +-----------+  +-----------+        |||
|   |   |   | Scenario  |  | Scenario  |  | Scenario  |        |||
|   |   |   |  Cruise   |  |  Climb    |  | Maneuver  |        |||
|   |   |   +-----------+  +-----------+  +-----------+        |||
|   |   |        |                                             |||
|   |   |        v                                             |||
|   |   |   +---------+  +----------+  +----------+           |||
|   |   |   |  Aero   |  | Struct   |  | Transfer |           |||
|   |   |   | Builder |  | Builder  |  | Builder  |           |||
|   |   |   +---------+  +----------+  +----------+           |||
|   |   +-----------------------------------------------------+||
|   +-----------------------------------------------------------+|
+---------------------------------------------------------------+
```

OpenMDAO is the engine. MPhys is the standardized chassis that makes it easy to
swap solvers in and out without rewiring the entire problem.

---

## 2. OpenMDAO Component Types and When to Use Each

OpenMDAO offers several component base classes. Choosing the right one is the first
critical decision when building a wrapper.

### Decision Tree

```
Is your solver iterative (converges a nonlinear system)?
  |
  +-- YES --> Does it solve R(x, y) = 0 for state y given input x?
  |             |
  |             +-- YES --> ImplicitComponent
  |             |           (CFD solvers, FEA solvers, Newton-based codes)
  |             |
  |             +-- NO  --> ExplicitComponent with internal iteration
  |                         (rare; consider restructuring)
  |
  +-- NO  --> Is the code a Python-callable function?
                |
                +-- YES --> ExplicitComponent
                |           (geometry tools, surrogates, post-processing)
                |
                +-- NO  --> Is it a standalone executable?
                              |
                              +-- YES --> ExternalCodeComp
                              |           (legacy Fortran codes, compiled binaries)
                              |
                              +-- NO  --> ExternalCodeImplicitComp
                                          (iterative compiled solvers without Python API)
```

### ExplicitComponent

Outputs are **explicitly** computed from inputs: `y = f(x)`.

```python
import openmdao.api as om
import numpy as np

class MyExplicitWrapper(om.ExplicitComponent):
    """Wrap a tool that computes outputs directly from inputs."""

    def initialize(self):
        # Declare options (solver settings, file paths, etc.)
        self.options.declare('mesh_file', types=str)
        self.options.declare('solver_options', types=dict, default={})

    def setup(self):
        # Declare inputs
        self.add_input('design_vars', shape=(10,), units='m',
                       desc='Design variable vector')
        self.add_input('bc_temperature', val=300.0, units='K',
                       desc='Boundary condition temperature')

        # Declare outputs
        self.add_output('objective', val=0.0,
                        desc='Objective function value')
        self.add_output('constraint_vec', shape=(5,),
                        desc='Constraint vector')

    def setup_partials(self):
        # Option 1: Dense partials (small problems)
        self.declare_partials('*', '*')

        # Option 2: Sparse partials (large problems)
        # self.declare_partials('objective', 'design_vars',
        #                       rows=np.arange(1), cols=np.arange(10))

        # Option 3: Finite difference (no analytic derivatives)
        # self.declare_partials('*', '*', method='fd', step=1e-6)

        # Option 4: Complex step (if code supports complex arithmetic)
        # self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        dvs = inputs['design_vars']
        T_bc = inputs['bc_temperature']

        # --- Call your solver here ---
        result = my_solver.run(dvs, T_bc)

        outputs['objective'] = result['obj']
        outputs['constraint_vec'] = result['constraints']

    def compute_partials(self, inputs, partials):
        dvs = inputs['design_vars']
        T_bc = inputs['bc_temperature']

        # --- Compute Jacobian entries ---
        jac = my_solver.compute_jacobian(dvs, T_bc)

        partials['objective', 'design_vars'] = jac['dobj_ddv']
        partials['objective', 'bc_temperature'] = jac['dobj_dT']
        partials['constraint_vec', 'design_vars'] = jac['dcon_ddv']
        partials['constraint_vec', 'bc_temperature'] = jac['dcon_dT']
```

### ImplicitComponent

Defines a **residual** equation `R(x, y) = 0` and solves for state `y`.
This is the correct choice for iterative solvers (CFD, FEA).

```python
class MyImplicitWrapper(om.ImplicitComponent):
    """Wrap an iterative solver that converges a residual."""

    def setup(self):
        # Inputs: parameters that affect the system
        self.add_input('angle_of_attack', val=0.0, units='deg')
        self.add_input('x_aero', shape_by_conn=True, units='m',
                       desc='Mesh coordinates', tags=['mphys_coupling'])

        # Outputs: state variables (what the solver converges)
        self.add_output('flow_states', shape=(50000,),
                        desc='Flow field state vector',
                        tags=['mphys_coupling'])

    def setup_partials(self):
        self.declare_partials('*', '*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        """Evaluate the residual R(x, y) without solving.

        This is called by OpenMDAO's nonlinear solvers to check convergence.
        """
        alpha = inputs['angle_of_attack']
        mesh = inputs['x_aero']
        states = outputs['flow_states']

        # Compute residual: R = A(mesh, alpha) * states - b(mesh, alpha)
        residuals['flow_states'] = self.solver.compute_residual(
            mesh, alpha, states
        )

    def solve_nonlinear(self, inputs, outputs):
        """Converge the solver to find states where R = 0.

        Called when OpenMDAO needs a converged solution.
        """
        alpha = inputs['angle_of_attack']
        mesh = inputs['x_aero']

        converged_states = self.solver.solve(mesh, alpha)
        outputs['flow_states'] = converged_states

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs,
                     d_residuals, mode):
        """Provide the action of the partial Jacobians for the linear system.

        For 'fwd' mode: d_residuals += dR/dy * d_outputs + dR/dx * d_inputs
        For 'rev' mode: d_outputs  += (dR/dy)^T * d_residuals
                        d_inputs   += (dR/dx)^T * d_residuals
        """
        if mode == 'fwd':
            if 'flow_states' in d_outputs:
                d_residuals['flow_states'] += (
                    self.solver.dRdy_product(d_outputs['flow_states'])
                )
            if 'x_aero' in d_inputs:
                d_residuals['flow_states'] += (
                    self.solver.dRdx_product(d_inputs['x_aero'])
                )
        elif mode == 'rev':
            if 'flow_states' in d_outputs:
                d_outputs['flow_states'] += (
                    self.solver.dRdy_T_product(d_residuals['flow_states'])
                )
            if 'x_aero' in d_inputs:
                d_inputs['x_aero'] += (
                    self.solver.dRdx_T_product(d_residuals['flow_states'])
                )

    def solve_linear(self, d_outputs, d_residuals, mode):
        """Solve the linear system (dR/dy) * psi = RHS.

        This is the adjoint (or direct) solve.
        """
        if mode == 'rev':
            # Adjoint solve: (dR/dy)^T * psi = d_residuals
            d_outputs['flow_states'] = self.solver.solve_adjoint(
                d_residuals['flow_states']
            )
        elif mode == 'fwd':
            # Direct solve: (dR/dy) * d_outputs = d_residuals
            d_outputs['flow_states'] = self.solver.solve_direct(
                d_residuals['flow_states']
            )
```

### ExternalCodeComp

For standalone executables that must be run as subprocesses.

```python
class MyExternalWrapper(om.ExternalCodeComp):
    """Wrap a compiled binary that reads/writes files."""

    def setup(self):
        self.add_input('mach', val=0.8)
        self.add_input('altitude', val=10000.0, units='m')
        self.add_output('lift', val=0.0, units='N')
        self.add_output('drag', val=0.0, units='N')

        self.input_file = 'solver_input.dat'
        self.output_file = 'solver_output.dat'

        self.options['external_input_files'] = [self.input_file]
        self.options['external_output_files'] = [self.output_file]
        self.options['command'] = ['./my_solver', self.input_file, self.output_file]
        self.options['timeout'] = 600  # seconds

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd', step=1e-4)

    def compute(self, inputs, outputs):
        # Write input file
        with open(self.input_file, 'w') as f:
            f.write(f"{inputs['mach'][0]:.16e}\n")
            f.write(f"{inputs['altitude'][0]:.16e}\n")

        # Run the external code (handled by parent class)
        super().compute(inputs, outputs)

        # Parse output file
        with open(self.output_file, 'r') as f:
            lines = f.readlines()
            outputs['lift'] = float(lines[0])
            outputs['drag'] = float(lines[1])
```

### Summary Table

```
+------------------------+------------------+-------------------+-------------------+
| Base Class             | Use When         | Derivatives       | Data Exchange     |
+------------------------+------------------+-------------------+-------------------+
| ExplicitComponent      | Direct function  | compute_partials  | In-memory         |
|                        | evaluation       | (analytic/FD/CS)  | (Python objects)  |
+------------------------+------------------+-------------------+-------------------+
| ImplicitComponent      | Iterative solver | apply_linear +    | In-memory         |
|                        | (CFD, FEA)       | solve_linear      | (Python objects)  |
|                        |                  | (adjoint/direct)  |                   |
+------------------------+------------------+-------------------+-------------------+
| ExternalCodeComp       | Standalone       | FD only           | File I/O          |
|                        | executable       | (unless adjoint   | (read/write)      |
|                        |                  | run separately)   |                   |
+------------------------+------------------+-------------------+-------------------+
| ExternalCodeImplicit   | Iterative        | FD + external     | File I/O          |
| Comp                   | executable       | adjoint solver    |                   |
+------------------------+------------------+-------------------+-------------------+
```

---

## 3. The MPhys Builder Pattern

MPhys uses a **Builder** abstraction to decouple solver-specific logic from the
multiphysics assembly logic. Each solver you want to integrate needs one Builder class.

### MPhys Model Hierarchy

```
Problem
  |
  +-- Multipoint (or MultipointParallel)
        |
        +-- [Geometry Parameterization]         <-- shared across scenarios
        |
        +-- Scenario: "cruise"
        |     |
        |     +-- MeshComp (from builder.get_mesh_coordinate_subsystem)
        |     +-- PreCoupling (from builder.get_pre_coupling_subsystem)
        |     +-- CouplingGroup
        |     |     |
        |     |     +-- AeroSolver   (from aero_builder)
        |     |     +-- StructSolver (from struct_builder)
        |     |     +-- LoadXfer     (from xfer_builder)
        |     |     +-- DispXfer     (from xfer_builder)
        |     |     +-- NonlinearBlockGaussSeidel (convergence)
        |     |
        |     +-- PostCoupling (from builder.get_post_coupling_subsystem)
        |           (functionals: C_L, C_D, stress, etc.)
        |
        +-- Scenario: "maneuver"
              |
              +-- (same structure, different flight condition)
```

### Builder Interface

```python
from mphys import Builder

class MySolverBuilder(Builder):
    """Builder for integrating MySolver into MPhys."""

    def __init__(self, solver_options, mesh_file=None):
        """Store options. Do NOT do heavy initialization here.

        This constructor may be called before MPI is set up.
        """
        self.solver_options = solver_options
        self.mesh_file = mesh_file

    def initialize(self, comm):
        """Called once by MPhys Multipoint.setup().

        Receives the MPI communicator. Do heavy initialization here:
        - Read mesh files
        - Allocate solver objects
        - Partition mesh across processors
        """
        self.comm = comm
        self.solver = MySolverAPI(
            mesh_file=self.mesh_file,
            options=self.solver_options,
            comm=comm
        )
        self.mesh_coords = self.solver.get_surface_coordinates()
        self.nnodes = len(self.mesh_coords) // 3

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        """Return a Component that outputs initial mesh coordinates.

        Must output a variable named 'x_aero0' (for aero) or
        'x_struct0' (for structural) with shape (nnodes*3,).
        """
        return MySolverMeshComp(
            solver=self.solver,
            nnodes=self.nnodes
        )

    def get_coupling_group_subsystem(self, scenario_name=None):
        """Return the main solver Component/Group for the CouplingGroup.

        This is where the actual physics solve happens.
        For aerodynamics: receives 'x_aero' mesh coords, outputs forces.
        For structures: receives loads, outputs displacements.
        """
        return MySolverCouplingGroup(
            solver=self.solver,
            solver_options=self.solver_options
        )

    def get_pre_coupling_subsystem(self, scenario_name=None):
        """(Optional) Return subsystem that runs before coupling iterations.

        Use for mesh warping, coordinate transformations, etc.
        Return None if not needed.
        """
        return MySolverMeshWarper(solver=self.solver)

    def get_post_coupling_subsystem(self, scenario_name=None):
        """(Optional) Return subsystem that runs after coupling converges.

        Use for computing functionals (lift, drag, stress, etc.).
        """
        return MySolverFunctionals(solver=self.solver)

    def get_number_of_nodes(self):
        """Return number of surface nodes."""
        return self.nnodes

    def get_ndof(self):
        """Return degrees of freedom per node (typically 3 for 3D)."""
        return 3
```

### Mesh Coordinate Component

```python
class MySolverMeshComp(om.IndepVarComp):
    """Outputs the initial (undeformed) mesh coordinates."""

    def initialize(self):
        self.options.declare('solver', recordable=False)
        self.options.declare('nnodes', types=int)

    def setup(self):
        solver = self.options['solver']
        nnodes = self.options['nnodes']
        coords = solver.get_surface_coordinates()  # shape (nnodes*3,)

        self.add_output(
            'x_aero0',           # MPhys standard name
            val=coords,
            shape=(nnodes * 3,),
            units='m',
            desc='Initial aerodynamic surface coordinates',
            tags=['mphys_coordinates']
        )
```

### Assembling the MPhys Problem

```python
import openmdao.api as om
from mphys import Multipoint, ScenarioAerodynamic

# Create builders
aero_builder = MySolverBuilder(
    solver_options={'turbulence_model': 'SA'},
    mesh_file='wing.cgns'
)

# Assemble problem
class AeroOptimization(Multipoint):
    def setup(self):
        # Register and initialize builders
        aero_builder.initialize(self.comm)

        # Add scenarios
        self.mphys_add_scenario(
            'cruise',
            ScenarioAerodynamic(aero_builder=aero_builder)
        )

prob = om.Problem()
prob.model = AeroOptimization()

# Configure driver
prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'

# Add design variables, objective, constraints
prob.model.add_design_var('cruise.aoa', lower=-5, upper=15)
prob.model.add_objective('cruise.aero_post.C_D')
prob.model.add_constraint('cruise.aero_post.C_L', equals=0.5)

prob.setup()
prob.run_driver()

# Visualize
om.n2(prob, outfile='n2_diagram.html', show_browser=True)
```

### Data Flow Within a Scenario

```
x_aero0 (from MeshComp)
    |
    v
[Geometry Parameterization]  <-- design variables modify shape
    |
    v
x_aero (deformed coordinates)
    |
    +-----------------------------+
    |                             |
    v                             v
[Mesh Warper]              [Load/Disp Transfer]
    |                         |         ^
    v                         v         |
[Aero Solver] --forces--> [Struct Solver]
    ^    |                    |
    |    v                    v
    +--displacements----------+
         (converged via NonlinearBlockGaussSeidel)
    |
    v
[Post-Coupling: Functionals]
    |
    v
C_L, C_D, stress, weight, ...  --> to optimizer
```

---

## 4. Derivatives: The Make-or-Break of Gradient-Based MDAO

The entire value proposition of OpenMDAO is its **unified derivative framework**.
Without derivatives, you are limited to gradient-free optimizers (genetic algorithms,
Nelder-Mead), which scale poorly beyond ~20 design variables.

### Derivative Methods Comparison

```
+-------------------+------------+------------+------------------+----------------+
| Method            | Accuracy   | Cost       | Scalability      | Implementation |
+-------------------+------------+------------+------------------+----------------+
| Analytic (hand-   | Exact      | O(1) per   | Best             | Hardest        |
|  coded Jacobian)  |            | variable   |                  |                |
+-------------------+------------+------------+------------------+----------------+
| Adjoint (solver-  | Exact      | O(1) per   | Best for many    | Hard (needs    |
|  provided)        |            | output     | design vars      | adjoint solver)|
+-------------------+------------+------------+------------------+----------------+
| Algorithmic Diff. | Exact      | ~2-5x      | Good             | Moderate       |
| (AD tape)         |            | forward    |                  | (tool support) |
+-------------------+------------+------------+------------------+----------------+
| Complex Step      | ~Exact     | O(n) per   | Good for small n | Easy (if code  |
|                   | (1e-15)    | variable   |                  | is CS-safe)    |
+-------------------+------------+------------+------------------+----------------+
| Finite Difference | Approx     | O(n) per   | Poor for large n | Easiest        |
|                   | (~1e-6)    | variable   |                  |                |
+-------------------+------------+------------+------------------+----------------+
```

### Derivative Verification

Always verify your derivatives using OpenMDAO's built-in checker:

```python
prob.setup(force_alloc_complex=True)  # needed for CS check
prob.run_model()

# Check partials of all components
prob.check_partials(compact_print=True, method='cs')

# Check total derivatives (what the optimizer sees)
prob.check_totals(
    of=['cruise.aero_post.C_D', 'cruise.aero_post.C_L'],
    wrt=['cruise.aoa', 'shape_vars'],
    compact_print=True
)
```

### How OpenMDAO Assembles Total Derivatives

OpenMDAO uses the **MAUD** (Modular Analysis and Unified Derivatives) architecture.
Each component provides **partial** derivatives. OpenMDAO assembles them into
**total** derivatives via chain rule, automatically handling coupled systems.

```
Total derivative: dF/dx (what optimizer needs)
                    =
Assembled from partials of each component:

Component A: dy_A/dx,   dy_A/dy_B
Component B: dy_B/dy_A, dy_B/dx
Component C: dF/dy_A,   dF/dy_B

OpenMDAO solves the coupled linear system:

  [I        -dy_A/dy_B] [dy_A/dx]     [partial_A/dx]
  [-dy_B/dy_A   I     ] [dy_B/dx]  =  [partial_B/dx]

Then: dF/dx = dF/dy_A * dy_A/dx + dF/dy_B * dy_B/dx
```

---

## 5. Wrapper Architecture Patterns

There are three fundamental patterns for wrapping an external tool. Choose based
on the tool's API maturity and your derivative requirements.

### Pattern A: In-Memory Python API (Preferred)

```
+------------------+       Python API calls        +------------------+
|    OpenMDAO      | <---------------------------> |   Solver (e.g.   |
|    Component     |   (function calls, arrays)    |   pysu2, FEniCS) |
|                  |                               |                  |
|  compute()       |   solver.set_mesh(coords)     |                  |
|  compute_        |   solver.solve()              |                  |
|   partials()     |   forces = solver.get_forces() |                 |
+------------------+                               +------------------+
```

Best for: SU2 (pysu2), FEniCS, ParaBlade, PhysicsNeMo, CadQuery.

### Pattern B: File-Based I/O (Fallback)

```
+------------------+     write input     +-----------+     read input
|    OpenMDAO      | -----------------> | input.dat  | <----------+
|  ExternalCode    |                    +-----------+            |
|    Comp          |                                    +--------+--------+
|                  |     read output     +-----------+  |  External Code  |
|  compute()       | <----------------- | output.dat|  |  (./solver.exe) |
+------------------+                    +-----------+  +-----------------+
                                              ^               |
                                              +---------------+
                                                write output
```

Best for: Legacy Fortran codes, COMSOL (via MPh), ANSYS Fluent (batch mode).

### Pattern C: Hybrid (Python API + File I/O for Mesh)

```
+------------------+     Python API     +------------------+
|    OpenMDAO      | <---------------> |   Solver Python   |
|    Component     |  (set BCs, solve) |   Interface       |
|                  |                   |                    |
|  setup():        |     File I/O      |  reads mesh from  |
|   read mesh file | <---------------> |  mesh.cgns        |
+------------------+                   +--------------------+
```

Best for: SU2 (when mesh updates require file writes), ADflow, DAFoam.

---

## 6. SU2 Wrapper

SU2 is an open-source CFD suite with a SWIG-generated Python API (`pysu2`) that
supports in-memory data exchange and discrete adjoint computation.

### Approach: In-Memory via pysu2 (Recommended)

```python
import numpy as np
import openmdao.api as om

try:
    import pysu2
except ImportError:
    raise ImportError(
        "pysu2 not found. Build SU2 with -Denable-pywrapper=true"
    )


class SU2MeshComp(om.IndepVarComp):
    """Output initial surface mesh coordinates from SU2."""

    def initialize(self):
        self.options.declare('su2_driver', recordable=False)

    def setup(self):
        driver = self.options['su2_driver']

        # Get all marker (boundary) tags
        all_marker_ids = driver.GetAllBoundaryMarkers()

        # Collect surface coordinates
        coords = []
        for marker_tag in all_marker_ids:
            marker_id = all_marker_ids[marker_tag]
            n_vertices = driver.GetNumberVertices(marker_id)
            for i_vertex in range(n_vertices):
                x = driver.GetVertexCoordX(marker_id, i_vertex)
                y = driver.GetVertexCoordY(marker_id, i_vertex)
                z = driver.GetVertexCoordZ(marker_id, i_vertex)
                coords.extend([x, y, z])

        self.coords = np.array(coords)
        self.add_output('x_aero0', val=self.coords, units='m',
                        tags=['mphys_coordinates'])


class SU2Solver(om.ImplicitComponent):
    """Wrap SU2 as an implicit component using pysu2."""

    def initialize(self):
        self.options.declare('config_file', types=str)
        self.options.declare('mesh_file', types=str)
        self.options.declare('restart', types=bool, default=False)

    def setup(self):
        config = self.options['config_file']

        # Initialize SU2 driver
        # In production, pass comm for MPI
        self.su2_driver = pysu2.CSinglezoneDriver(config, 1, 0)

        n_nodes = self._count_surface_nodes()

        # Mesh coordinates (input from geometry)
        self.add_input('x_aero', shape=(n_nodes * 3,), units='m',
                       desc='Surface mesh coordinates',
                       tags=['mphys_coupling'])

        # Flow states (output: what the solver converges)
        n_states = self._count_states()
        self.add_output('flow_states', shape=(n_states,),
                        desc='Flow field state vector',
                        tags=['mphys_coupling'])

        # Aerodynamic forces (output for coupling)
        self.add_output('f_aero', shape=(n_nodes * 3,), units='N',
                        desc='Aerodynamic surface forces',
                        tags=['mphys_coupling'])

    def _count_surface_nodes(self):
        """Count total surface nodes across all boundaries."""
        driver = self.su2_driver
        total = 0
        markers = driver.GetAllBoundaryMarkers()
        for tag in markers:
            total += driver.GetNumberVertices(markers[tag])
        return total

    def _count_states(self):
        """Count total flow states (nCells * nVar)."""
        # SU2-specific: depends on solver type (Euler=4, RANS=5+, etc.)
        return self.su2_driver.GetNumberSolverVariables()

    def solve_nonlinear(self, inputs, outputs):
        """Run SU2 to convergence."""
        driver = self.su2_driver

        # Update surface coordinates
        self._set_surface_coords(inputs['x_aero'])

        # Run the solver
        driver.ResetConvergence()
        driver.Preprocess(0)
        driver.Run()
        driver.Postprocess()
        driver.Update()

        # Extract converged states and forces
        outputs['flow_states'] = self._get_states()
        outputs['f_aero'] = self._get_forces()

    def apply_nonlinear(self, inputs, outputs, residuals):
        """Evaluate the residual without solving."""
        self._set_surface_coords(inputs['x_aero'])
        self._set_states(outputs['flow_states'])

        # Single iteration to compute residual
        self.su2_driver.Preprocess(0)
        self.su2_driver.Run()  # single-iteration mode

        residuals['flow_states'] = self._get_residual()

    def _set_surface_coords(self, coords):
        """Push coordinates into SU2 driver."""
        driver = self.su2_driver
        markers = driver.GetAllBoundaryMarkers()
        idx = 0
        for tag in markers:
            mid = markers[tag]
            for iv in range(driver.GetNumberVertices(mid)):
                driver.SetVertexCoordX(mid, iv, coords[idx])
                driver.SetVertexCoordY(mid, iv, coords[idx + 1])
                driver.SetVertexCoordZ(mid, iv, coords[idx + 2])
                idx += 3

    def _get_forces(self):
        """Extract surface forces from SU2."""
        driver = self.su2_driver
        forces = []
        markers = driver.GetAllBoundaryMarkers()
        for tag in markers:
            mid = markers[tag]
            for iv in range(driver.GetNumberVertices(mid)):
                fx = driver.GetFlowLoad(mid, iv)[0]
                fy = driver.GetFlowLoad(mid, iv)[1]
                fz = driver.GetFlowLoad(mid, iv)[2]
                forces.extend([fx, fy, fz])
        return np.array(forces)

    # _get_states, _set_states, _get_residual follow similar patterns


class SU2AdjointDerivatives(om.ExplicitComponent):
    """Compute derivatives using SU2's discrete adjoint solver.

    In production, this would be integrated into the ImplicitComponent
    via apply_linear/solve_linear. Shown separately for clarity.
    """

    def initialize(self):
        self.options.declare('su2_driver', recordable=False)
        self.options.declare('adj_config_file', types=str)

    def setup(self):
        # ... declare same I/O as SU2Solver ...
        pass

    def compute_partials(self, inputs, partials):
        """Run SU2 discrete adjoint to compute sensitivities."""
        # Initialize adjoint driver
        adj_driver = pysu2.CDiscAdjSinglezoneDriver(
            self.options['adj_config_file'], 1, 0
        )
        adj_driver.Preprocess(0)
        adj_driver.Run()
        adj_driver.Postprocess()

        # Extract sensitivities: dObjective/dSurfaceCoords
        # These are the surface sensitivity fields
        sens = self._extract_surface_sensitivities(adj_driver)
        partials['objective', 'x_aero'] = sens
```

### SU2 MPhys Builder

```python
from mphys import Builder

class SU2Builder(Builder):
    """MPhys Builder for SU2 CFD solver."""

    def __init__(self, config_file, mesh_file, options=None):
        self.config_file = config_file
        self.mesh_file = mesh_file
        self.options = options or {}

    def initialize(self, comm):
        self.comm = comm
        # Read mesh to get node count (lightweight operation)
        self._read_mesh_info()

    def _read_mesh_info(self):
        """Parse SU2 mesh file header for node/element counts."""
        self.nnodes = 0
        with open(self.mesh_file, 'r') as f:
            for line in f:
                if line.startswith('NPOIN='):
                    self.nnodes = int(line.split('=')[1].strip().split()[0])
                    break

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return SU2MeshComp(
            su2_driver=None  # will be set during setup
        )

    def get_coupling_group_subsystem(self, scenario_name=None):
        return SU2Solver(
            config_file=self.config_file,
            mesh_file=self.mesh_file
        )

    def get_post_coupling_subsystem(self, scenario_name=None):
        return SU2Functionals(config_file=self.config_file)

    def get_number_of_nodes(self):
        return self.nnodes

    def get_ndof(self):
        return 3
```

### Alternative: File-Based SU2 Wrapper (Simpler, No pysu2 Required)

```python
class SU2ExternalWrapper(om.ExternalCodeComp):
    """Wrap SU2 via file I/O -- no pysu2 dependency."""

    def initialize(self):
        self.options.declare('config_template', types=str)
        self.options.declare('mesh_file', types=str)

    def setup(self):
        self.add_input('mach', val=0.8)
        self.add_input('aoa', val=2.0, units='deg')
        self.add_output('C_L', val=0.0)
        self.add_output('C_D', val=0.0)
        self.add_output('C_M', val=0.0)

        self.options['command'] = ['SU2_CFD', 'config_runtime.cfg']
        self.options['external_input_files'] = ['config_runtime.cfg']
        self.options['external_output_files'] = ['forces_breakdown.dat']

    def setup_partials(self):
        # No analytic derivatives available in file-based mode
        self.declare_partials('*', '*', method='fd', step=1e-4)

    def compute(self, inputs, outputs):
        # Generate runtime config from template
        self._write_config(inputs)

        # Run SU2
        super().compute(inputs, outputs)

        # Parse force breakdown file
        coeffs = self._parse_forces()
        outputs['C_L'] = coeffs['CL']
        outputs['C_D'] = coeffs['CD']
        outputs['C_M'] = coeffs['CMy']

    def _write_config(self, inputs):
        """Write SU2 config file with current design parameters."""
        with open(self.options['config_template'], 'r') as f:
            template = f.read()

        config = template.replace('__MACH__', f"{inputs['mach'][0]:.8f}")
        config = config.replace('__AOA__', f"{inputs['aoa'][0]:.8f}")
        config = config.replace('__MESH__', self.options['mesh_file'])

        with open('config_runtime.cfg', 'w') as f:
            f.write(config)

    def _parse_forces(self):
        """Parse SU2 forces_breakdown.dat."""
        coeffs = {}
        with open('forces_breakdown.dat', 'r') as f:
            for line in f:
                if 'Total CL:' in line:
                    coeffs['CL'] = float(line.split(':')[1].split('|')[0])
                elif 'Total CD:' in line:
                    coeffs['CD'] = float(line.split(':')[1].split('|')[0])
                elif 'Total CMy:' in line:
                    coeffs['CMy'] = float(line.split(':')[1].split('|')[0])
        return coeffs
```

---

## 7. FEniCS Wrapper

FEniCS is a finite element framework with automatic differentiation via
`dolfin-adjoint` / `pyadjoint`. This makes it exceptionally well-suited for
gradient-based MDAO -- the adjoint is computed automatically from the variational
form.

### Architecture

```
+---------------------+
|  OpenMDAO Component |
|                     |
|  compute():         |    +---------------------+
|    solve PDE -------+--->|  FEniCS / dolfinx    |
|                     |    |  (variational form)  |
|  compute_partials():|    +---------------------+
|    adjoint ----------+-->|  dolfin-adjoint /    |
|    sensitivities     |   |  pyadjoint (tape)    |
+---------------------+   +---------------------+
```

### Linear Elasticity FEA Wrapper

```python
import openmdao.api as om
import numpy as np

# FEniCS imports (legacy API shown; dolfinx API is similar)
from dolfin import *
from dolfin_adjoint import *


class FEniCSElasticity(om.ExplicitComponent):
    """Wrap a FEniCS linear elasticity solve with adjoint derivatives."""

    def initialize(self):
        self.options.declare('mesh_file', types=str)
        self.options.declare('degree', types=int, default=1)

    def setup(self):
        mesh_file = self.options['mesh_file']

        # Load mesh
        self.mesh = Mesh(mesh_file)
        self.V = VectorFunctionSpace(
            self.mesh, 'CG', self.options['degree']
        )

        n_nodes = self.mesh.num_vertices()
        n_dofs = self.V.dim()

        # Inputs
        self.add_input('E_field', shape=(n_nodes,), val=200e9,
                       units='Pa', desc='Young modulus at each node')
        self.add_input('applied_loads', shape=(n_dofs,), val=0.0,
                       units='N', desc='Applied load vector')

        # Outputs
        self.add_output('displacements', shape=(n_dofs,), val=0.0,
                        units='m', desc='Displacement field')
        self.add_output('compliance', val=0.0, units='J',
                        desc='Structural compliance')
        self.add_output('max_stress', val=0.0, units='Pa',
                        desc='Maximum von Mises stress')

    def setup_partials(self):
        self.declare_partials('compliance', 'E_field')
        self.declare_partials('compliance', 'applied_loads')
        self.declare_partials('max_stress', 'E_field')
        self.declare_partials('displacements', '*', method='fd')

    def compute(self, inputs, outputs):
        # Clear the dolfin-adjoint tape for a fresh recording
        set_working_tape(Tape())

        E_vals = inputs['E_field']
        load_vals = inputs['applied_loads']

        # Create spatially-varying Young's modulus
        E_space = FunctionSpace(self.mesh, 'CG', 1)
        E_func = Function(E_space, name='E')
        E_func.vector()[:] = E_vals

        nu = Constant(0.3)  # Poisson's ratio (fixed)

        # Lame parameters
        lmbda = E_func * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E_func / (2 * (1 + nu))

        # Variational form
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        def epsilon(u):
            return 0.5 * (grad(u) + grad(u).T)

        def sigma(u):
            return lmbda * tr(epsilon(u)) * Identity(3) + 2 * mu * epsilon(u)

        a = inner(sigma(u), epsilon(v)) * dx

        # Load vector
        f = Function(self.V, name='load')
        f.vector()[:] = load_vals
        L = inner(f, v) * dx

        # Boundary conditions
        bc = DirichletBC(self.V, Constant((0, 0, 0)), 'on_boundary')

        # Solve
        u_sol = Function(self.V, name='displacement')
        solve(a == L, u_sol, bc)

        # Store for adjoint
        self._u_sol = u_sol
        self._E_func = E_func

        # Extract outputs
        outputs['displacements'] = u_sol.vector().get_local()
        outputs['compliance'] = assemble(inner(f, u_sol) * dx)

        # Von Mises stress
        s = sigma(u_sol) - (1.0/3)*tr(sigma(u_sol))*Identity(3)
        von_mises = sqrt(3.0/2 * inner(s, s))
        VM = project(von_mises, FunctionSpace(self.mesh, 'CG', 1))
        outputs['max_stress'] = VM.vector().max()

    def compute_partials(self, inputs, partials):
        """Use dolfin-adjoint to compute derivatives automatically."""
        u_sol = self._u_sol
        E_func = self._E_func

        # Compliance functional
        f = Function(self.V)
        f.vector()[:] = inputs['applied_loads']
        J = assemble(inner(f, u_sol) * dx)

        # Compute dJ/dE using the adjoint method
        # dolfin-adjoint records the forward solve on a tape
        # and automatically derives the adjoint equations
        dJdE = compute_gradient(J, Control(E_func))
        partials['compliance', 'E_field'] = dJdE.vector().get_local()

        # dJ/d(loads) via adjoint
        dJdf = compute_gradient(J, Control(f))
        partials['compliance', 'applied_loads'] = dJdf.vector().get_local()
```

### FEniCS MPhys Builder (Structural)

```python
from mphys import Builder

class FEniCSStructBuilder(Builder):
    """MPhys Builder for FEniCS structural analysis."""

    def __init__(self, mesh_file, element_degree=1, material_props=None):
        self.mesh_file = mesh_file
        self.degree = element_degree
        self.material_props = material_props or {}

    def initialize(self, comm):
        self.comm = comm
        # Read mesh to determine node count
        mesh = Mesh(self.mesh_file)
        self.nnodes = mesh.num_vertices()

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return FEniCSStructMeshComp(mesh_file=self.mesh_file)

    def get_coupling_group_subsystem(self, scenario_name=None):
        return FEniCSElasticity(
            mesh_file=self.mesh_file,
            degree=self.degree
        )

    def get_post_coupling_subsystem(self, scenario_name=None):
        return FEniCSStructFunctionals(mesh_file=self.mesh_file)

    def get_number_of_nodes(self):
        return self.nnodes

    def get_ndof(self):
        return 3
```

### Notes on FEniCSx (dolfinx) vs Legacy FEniCS

```
+------------------+-------------------------------+-------------------------------+
| Feature          | Legacy FEniCS (dolfin)        | FEniCSx (dolfinx)             |
+------------------+-------------------------------+-------------------------------+
| Status           | Maintenance only              | Active development            |
| AD support       | dolfin-adjoint (mature)       | dolfinx-adjoint (in progress) |
| MPI              | Yes (via PETSc)               | Yes (native petsc4py)         |
| Recommended for  | Existing projects with AD     | New projects                  |
| OpenMDAO compat  | Well-tested (GOLDFISH, etc.)  | Requires custom AD setup      |
+------------------+-------------------------------+-------------------------------+
```

For new projects, consider FEniCSx with manual adjoint implementation or
use the GOLDFISH framework as reference for legacy FEniCS integration.

---

## 8. NVIDIA PhysicsNeMo Surrogate Wrapper

PhysicsNeMo (formerly NVIDIA Modulus) trains physics-informed neural network
surrogates that can replace expensive CFD/FEA solves. The integration strategy
is: **train offline, deploy as an OpenMDAO component**.

### Integration Architecture

```
OFFLINE (Training Phase):
+------------------+      +-----------------+      +------------------+
| Training Data    | ---> | PhysicsNeMo     | ---> | Trained Model    |
| (CFD/FEA runs,   |     | Training Loop   |      | (.onnx or .pt)   |
|  PDE residuals)  |      +-----------------+      +------------------+
+------------------+

ONLINE (MDAO Phase):
+------------------+      +------------------+      +------------------+
| OpenMDAO         | ---> | ONNX Runtime or  | ---> | Predictions      |
| Component        |      | PyTorch Inference|      | (C_L, C_D, ...)  |
| (ExplicitComp)   |      +------------------+      +------------------+
                           |
                           +-- Jacobian via torch.autograd or FD
```

### ONNX-Based Wrapper (Production Deployment)

```python
import openmdao.api as om
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("pip install onnxruntime  (or onnxruntime-gpu)")


class PhysicsNeMoSurrogate(om.ExplicitComponent):
    """Wrap a PhysicsNeMo-trained ONNX model as an OpenMDAO component.

    The model maps design parameters -> physical quantities.
    Example: (Mach, AoA, Re, shape_params) -> (C_L, C_D, C_M, Cp_distribution)
    """

    def initialize(self):
        self.options.declare('model_path', types=str,
                             desc='Path to .onnx model file')
        self.options.declare('input_names', types=list,
                             desc='ONNX model input names')
        self.options.declare('output_names', types=list,
                             desc='ONNX model output names')
        self.options.declare('input_shapes', types=dict,
                             desc='Dict of input_name: shape tuples')
        self.options.declare('output_shapes', types=dict,
                             desc='Dict of output_name: shape tuples')
        self.options.declare('use_gpu', types=bool, default=False)

    def setup(self):
        # Load ONNX model
        providers = ['CUDAExecutionProvider'] if self.options['use_gpu'] \
                    else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(
            self.options['model_path'], providers=providers
        )

        # Declare OpenMDAO inputs/outputs from model metadata
        for name in self.options['input_names']:
            shape = self.options['input_shapes'][name]
            self.add_input(name, shape=shape)

        for name in self.options['output_names']:
            shape = self.options['output_shapes'][name]
            self.add_output(name, shape=shape)

    def setup_partials(self):
        # ONNX Runtime does not natively support differentiation.
        # Options:
        #   1. Finite differences (simplest)
        #   2. Use PyTorch model directly for torch.autograd (see below)
        self.declare_partials('*', '*', method='fd', step=1e-5)

    def compute(self, inputs, outputs):
        # Build ONNX input dict
        feed = {}
        for name in self.options['input_names']:
            feed[name] = inputs[name].astype(np.float32)

        # Run inference
        results = self.session.run(
            self.options['output_names'], feed
        )

        for i, name in enumerate(self.options['output_names']):
            outputs[name] = results[i].flatten()
```

### PyTorch-Based Wrapper (With Analytic Derivatives)

```python
import openmdao.api as om
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("pip install torch")


class PhysicsNeMoTorchSurrogate(om.ExplicitComponent):
    """Wrap a PyTorch model with analytic derivatives via autograd."""

    def initialize(self):
        self.options.declare('model_path', types=str,
                             desc='Path to .pt model checkpoint')
        self.options.declare('model_class', desc='nn.Module class')
        self.options.declare('n_inputs', types=int)
        self.options.declare('n_outputs', types=int)
        self.options.declare('device', types=str, default='cpu')

    def setup(self):
        n_in = self.options['n_inputs']
        n_out = self.options['n_outputs']
        device = self.options['device']

        # Load model
        ModelClass = self.options['model_class']
        self.model = ModelClass()
        self.model.load_state_dict(
            torch.load(self.options['model_path'],
                        map_location=device)
        )
        self.model.eval()
        self.model.to(device)
        self.device = device

        self.add_input('x', shape=(n_in,))
        self.add_output('y', shape=(n_out,))

    def setup_partials(self):
        self.declare_partials('y', 'x')

    def compute(self, inputs, outputs):
        x_np = inputs['x']
        x_t = torch.tensor(x_np, dtype=torch.float32,
                           device=self.device)

        with torch.no_grad():
            y_t = self.model(x_t.unsqueeze(0))

        outputs['y'] = y_t.squeeze(0).cpu().numpy()

    def compute_partials(self, inputs, partials):
        x_np = inputs['x']
        x_t = torch.tensor(x_np, dtype=torch.float32,
                           device=self.device, requires_grad=True)

        # Compute full Jacobian using torch.autograd
        def model_fn(x):
            return self.model(x.unsqueeze(0)).squeeze(0)

        jac = torch.autograd.functional.jacobian(model_fn, x_t)
        partials['y', 'x'] = jac.detach().cpu().numpy()
```

### Usage in an MDAO Problem

```python
# Train a surrogate to replace expensive CFD
# Inputs: [Mach, AoA, Re, 10 shape params] = 13 inputs
# Outputs: [C_L, C_D, C_M] = 3 outputs

prob = om.Problem()
model = prob.model

model.add_subsystem('surrogate', PhysicsNeMoTorchSurrogate(
    model_path='airfoil_surrogate.pt',
    model_class=AirfoilNet,  # your trained nn.Module
    n_inputs=13,
    n_outputs=3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
))

model.add_design_var('surrogate.x', lower=lb, upper=ub)
model.add_objective('surrogate.y', index=1)       # minimize C_D
model.add_constraint('surrogate.y', index=0,       # C_L = 0.5
                     equals=0.5)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.setup()
prob.run_driver()
```

### PhysicsNeMo-CFD Integration Note

PhysicsNeMo-CFD specifically targets GPU-accelerated CFD using neural operators
(FNO, AFNO). The integration pattern is the same -- train the model to predict
flow fields, export to ONNX or keep as PyTorch, wrap in an ExplicitComponent.

The key advantage over traditional surrogates: PhysicsNeMo models can enforce
PDE constraints during training, giving better extrapolation and requiring
fewer training samples.

---

## 9. ParaBlade Wrapper

ParaBlade is a parametric turbomachinery blade geometry tool from TU Delft.
It generates blade surface coordinates from engineering design parameters and
provides analytic derivatives via the complex-step method.

### Architecture

```
Design Parameters (angles, thickness, stagger, ...)
    |
    v
+------------------+
|  ParaBlade       |  B-spline parameterization
|  Blade Generator |  of hub-to-tip blade sections
+------------------+
    |
    +-- Surface coordinates (x, y, z arrays)
    |
    +-- Shape derivatives (dx/d_param via complex step)
    |
    v
Connect to mesh deformation --> CFD solver (SU2, ADflow, ...)
```

### Wrapper Implementation

```python
import openmdao.api as om
import numpy as np

# Assuming ParaBlade is installed and importable
from parablade.parablade_setup import ParaBladeSetup


class ParaBladeComp(om.ExplicitComponent):
    """Wrap ParaBlade parametric blade generator for MDAO.

    Inputs: engineering design parameters
    Outputs: blade surface coordinates
    Derivatives: via complex step on ParaBlade internals
    """

    def initialize(self):
        self.options.declare('config_file', types=str,
                             desc='ParaBlade configuration file')
        self.options.declare('n_sections', types=int, default=5,
                             desc='Number of blade sections hub-to-tip')
        self.options.declare('n_points_per_section', types=int, default=100,
                             desc='Points per blade section')

    def setup(self):
        ns = self.options['n_sections']
        npts = self.options['n_points_per_section']
        n_surf = ns * npts * 3  # x, y, z for each point

        # Load ParaBlade config to determine parameter set
        self.blade_setup = ParaBladeSetup(self.options['config_file'])

        # Typical turbomachinery design parameters (per section)
        param_names = [
            'inlet_metal_angle',    # beta_in  [deg]
            'outlet_metal_angle',   # beta_out [deg]
            'stagger_angle',        # stagger  [deg]
            'max_thickness',        # t_max / chord [-]
            'leading_edge_radius',  # r_LE / chord [-]
            'trailing_edge_radius', # r_TE / chord [-]
            'chord_length',         # chord [m]
        ]

        for param in param_names:
            self.add_input(param, shape=(ns,),
                           desc=f'{param} at each spanwise section')

        # Output: blade surface coordinates
        self.add_output('blade_coords', shape=(n_surf,), units='m',
                        desc='Blade surface coordinates [x1,y1,z1,x2,...]')

        # Also output useful derived quantities
        self.add_output('blade_area', val=0.0, units='m**2',
                        desc='Total blade surface area')

    def setup_partials(self):
        # Use complex step for accurate derivatives
        # ParaBlade supports complex arithmetic internally
        self.declare_partials('*', '*', method='cs', step=1e-30)

    def compute(self, inputs, outputs):
        ns = self.options['n_sections']
        npts = self.options['n_points_per_section']

        # Update ParaBlade parameters from OpenMDAO inputs
        for i_sec in range(ns):
            self.blade_setup.set_section_params(
                section=i_sec,
                beta_in=inputs['inlet_metal_angle'][i_sec],
                beta_out=inputs['outlet_metal_angle'][i_sec],
                stagger=inputs['stagger_angle'][i_sec],
                t_max=inputs['max_thickness'][i_sec],
                r_le=inputs['leading_edge_radius'][i_sec],
                r_te=inputs['trailing_edge_radius'][i_sec],
                chord=inputs['chord_length'][i_sec],
            )

        # Generate blade geometry
        self.blade_setup.generate_blade()
        coords = self.blade_setup.get_surface_coordinates()

        outputs['blade_coords'] = coords.flatten()
        outputs['blade_area'] = self.blade_setup.get_surface_area()


class ParaBladeMeshDeformation(om.ExplicitComponent):
    """Deform a CFD mesh based on ParaBlade surface coordinate changes.

    Uses radial basis function (RBF) interpolation to propagate surface
    deformations into the volume mesh.
    """

    def initialize(self):
        self.options.declare('n_surface', types=int)
        self.options.declare('n_volume', types=int)

    def setup(self):
        ns = self.options['n_surface']
        nv = self.options['n_volume']

        self.add_input('blade_coords', shape=(ns * 3,), units='m')
        self.add_input('blade_coords_ref', shape=(ns * 3,), units='m',
                       desc='Reference (undeformed) surface coordinates')
        self.add_input('vol_mesh_ref', shape=(nv * 3,), units='m',
                       desc='Reference volume mesh')

        self.add_output('vol_mesh', shape=(nv * 3,), units='m',
                        desc='Deformed volume mesh')

    def setup_partials(self):
        self.declare_partials('vol_mesh', 'blade_coords')

    def compute(self, inputs, outputs):
        # Compute surface displacement
        disp = inputs['blade_coords'] - inputs['blade_coords_ref']

        # RBF interpolation to volume mesh
        # (simplified -- in practice use scipy.interpolate.RBFInterpolator)
        from scipy.interpolate import RBFInterpolator

        surf_pts = inputs['blade_coords_ref'].reshape(-1, 3)
        vol_pts = inputs['vol_mesh_ref'].reshape(-1, 3)
        disp_3d = disp.reshape(-1, 3)

        rbf = RBFInterpolator(surf_pts, disp_3d, kernel='thin_plate_spline')
        vol_disp = rbf(vol_pts)

        outputs['vol_mesh'] = (
            inputs['vol_mesh_ref'] + vol_disp.flatten()
        )

    def compute_partials(self, inputs, partials):
        # RBF interpolation is linear in the displacements,
        # so the Jacobian is just the interpolation matrix
        # (constant w.r.t. inputs, can be computed once and cached)
        pass  # Implementation depends on RBF library
```

### Turbomachinery Optimization Example (ParaBlade + SU2)

```python
import openmdao.api as om

prob = om.Problem()
model = prob.model

# Geometry: ParaBlade generates blade shape
model.add_subsystem('geometry', ParaBladeComp(
    config_file='naca_blade.cfg',
    n_sections=5,
    n_points_per_section=100
))

# Mesh deformation: propagate surface changes to volume mesh
model.add_subsystem('mesh_deform', ParaBladeMeshDeformation(
    n_surface=500,
    n_volume=50000
))

# CFD: SU2 computes aerodynamic performance
model.add_subsystem('cfd', SU2ExternalWrapper(
    config_template='su2_template.cfg',
    mesh_file='blade_mesh.su2'
))

# Connections
model.connect('geometry.blade_coords', 'mesh_deform.blade_coords')
model.connect('mesh_deform.vol_mesh', 'cfd.mesh_coords')

# Optimization setup
model.add_design_var('geometry.inlet_metal_angle', lower=20, upper=60)
model.add_design_var('geometry.outlet_metal_angle', lower=40, upper=80)
model.add_design_var('geometry.stagger_angle', lower=20, upper=50)
model.add_design_var('geometry.max_thickness', lower=0.04, upper=0.15)

model.add_objective('cfd.total_pressure_loss')
model.add_constraint('cfd.mass_flow_rate', equals=1.5)
model.add_constraint('geometry.max_thickness', lower=0.05)

prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'
prob.setup()
prob.run_driver()
```

---

## 10. Open-Source CAD Tool Wrappers

### Decision Tree for CAD Tool Selection

```
Need parametric geometry for MDAO?
  |
  +-- Need analytic shape derivatives?
  |     |
  |     +-- YES --> ESP/CAPS (native OpenMDAO support, derivatives via CAPS)
  |     |
  |     +-- NO  --> Any tool works
  |
  +-- Need complex CAD operations (booleans, fillets, assemblies)?
  |     |
  |     +-- YES --> CadQuery or pyOCCT (OpenCASCADE kernel)
  |     |
  |     +-- NO  --> ESP/CAPS or simple parametric scripts
  |
  +-- Need existing CAD file import (STEP, IGES)?
        |
        +-- YES --> pyOCCT, CadQuery, or FreeCAD
        |
        +-- NO  --> ESP/CAPS (OpenCSM scripting)
```

### ESP/CAPS Wrapper (Most MDAO-Native)

ESP/CAPS is the gold standard for CAD-MDAO integration. CAPS provides Analysis
Interface Modules (AIMs) that connect geometry to meshing and analysis tools.

```python
import openmdao.api as om
import pyCAPS


class ESPCAPSGeometry(om.ExplicitComponent):
    """Wrap ESP/CAPS geometry and meshing via pyCAPS.

    pyCAPS provides a direct method to create OpenMDAO components,
    but this shows the manual approach for full control.
    """

    def initialize(self):
        self.options.declare('csm_file', types=str,
                             desc='OpenCSM script file (.csm)')
        self.options.declare('aim_name', types=str, default='egadsTessAIM',
                             desc='CAPS Analysis Interface Module')

    def setup(self):
        csm_file = self.options['csm_file']

        # Initialize CAPS problem
        self.caps_problem = pyCAPS.Problem(
            problemName='mdao_geom',
            capsFile=csm_file,
            outLevel=0
        )

        # Get design parameters from CSM file
        geom = self.caps_problem.geometry
        for param_name in geom.despmtr:
            val = geom.despmtr[param_name].value
            if isinstance(val, (int, float)):
                self.add_input(param_name, val=float(val))
            elif isinstance(val, list):
                self.add_input(param_name, val=np.array(val))

        # Setup meshing AIM
        self.mesh_aim = self.caps_problem.analysis.create(
            aim=self.options['aim_name'],
            name='mesh'
        )

        # Output: tessellation / mesh coordinates
        # Size determined after first evaluation
        self.add_output('surface_mesh', shape_by_conn=True,
                        desc='Surface mesh coordinates')

    def compute(self, inputs, outputs):
        geom = self.caps_problem.geometry

        # Update design parameters
        for param_name in geom.despmtr:
            if param_name in inputs:
                geom.despmtr[param_name].value = float(inputs[param_name])

        # Run meshing
        self.mesh_aim.preAnalysis()
        # Extract mesh coordinates from AIM
        mesh_data = self.mesh_aim.output['Surface_Mesh'].value
        outputs['surface_mesh'] = mesh_data


class ESPCAPSNative(om.ExplicitComponent):
    """Use pyCAPS built-in OpenMDAO component generation.

    This is the simplest approach -- pyCAPS does most of the work.
    """

    def initialize(self):
        self.options.declare('csm_file', types=str)

    def setup(self):
        self.caps = pyCAPS.Problem(
            problemName='native_om',
            capsFile=self.options['csm_file']
        )

        # pyCAPS can auto-generate an OpenMDAO component:
        # om_comp = self.caps.analysis['myAIM'].createOpenMDAOComponent(
        #     vartype='Input',     # or 'Output'
        #     inputVar=['despmtr1', 'despmtr2'],
        #     outputVar=['area', 'volume']
        # )
        # This returns a ready-to-use OpenMDAO component.
```

### CadQuery Wrapper

```python
import openmdao.api as om
import numpy as np

try:
    import cadquery as cq
except ImportError:
    raise ImportError("pip install cadquery")


class CadQueryParametricGeometry(om.ExplicitComponent):
    """Generate parametric geometry using CadQuery and export mesh data.

    CadQuery sits on OpenCASCADE and excels at programmatic CAD.
    """

    def initialize(self):
        self.options.declare('export_format', types=str, default='stl',
                             values=['stl', 'step', 'brep'])

    def setup(self):
        # Example: parametric nozzle
        self.add_input('inlet_radius', val=0.05, units='m')
        self.add_input('outlet_radius', val=0.03, units='m')
        self.add_input('length', val=0.2, units='m')
        self.add_input('throat_radius', val=0.02, units='m')
        self.add_input('throat_position', val=0.5,
                       desc='Fraction of length where throat occurs')

        self.add_output('volume', val=0.0, units='m**3')
        self.add_output('surface_area', val=0.0, units='m**2')
        self.add_output('geometry_file', val='',
                        desc='Path to exported geometry file')

    def setup_partials(self):
        self.declare_partials('volume', '*', method='fd', step=1e-6)
        self.declare_partials('surface_area', '*', method='fd', step=1e-6)

    def compute(self, inputs, outputs):
        r_in = float(inputs['inlet_radius'])
        r_out = float(inputs['outlet_radius'])
        L = float(inputs['length'])
        r_throat = float(inputs['throat_radius'])
        x_throat = float(inputs['throat_position']) * L

        # Build nozzle as a revolved spline profile
        pts = [
            (0, r_in),
            (x_throat, r_throat),
            (L, r_out),
        ]

        # Create profile wire and revolve
        profile = cq.Workplane("XZ").spline(pts)
        profile = profile.lineTo(L, 0).lineTo(0, 0).close()
        nozzle = profile.revolve(360, (0, 0, 0), (1, 0, 0))

        # Extract properties
        props = nozzle.val().Volume(), nozzle.val().Area()
        outputs['volume'] = props[0]
        outputs['surface_area'] = props[1]

        # Export
        fmt = self.options['export_format']
        filename = f'nozzle.{fmt}'
        if fmt == 'stl':
            cq.exporters.export(nozzle, filename, exportType='STL',
                                tolerance=0.001)
        elif fmt == 'step':
            cq.exporters.export(nozzle, filename, exportType='STEP')

        outputs['geometry_file'] = filename
```

### pyOCCT Wrapper

```python
import openmdao.api as om
import numpy as np

try:
    from OCCT.gp import gp_Pnt, gp_Dir, gp_Ax2
    from OCCT.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCCT.GProp import GProp_GProps
    from OCCT.BRepGProp import brepgprop
    from OCCT.STEPControl import STEPControl_Writer, STEPControl_AsIs
except ImportError:
    raise ImportError("pip install pyOCCT")


class PyOCCTGeometry(om.ExplicitComponent):
    """Direct OpenCASCADE geometry via pyOCCT bindings."""

    def setup(self):
        self.add_input('radius', val=0.1, units='m')
        self.add_input('height', val=0.5, units='m')

        self.add_output('volume', val=0.0, units='m**3')
        self.add_output('surface_area', val=0.0, units='m**2')

    def setup_partials(self):
        # Analytic for cylinder:
        self.declare_partials('volume', 'radius')
        self.declare_partials('volume', 'height')
        self.declare_partials('surface_area', 'radius')
        self.declare_partials('surface_area', 'height')

    def compute(self, inputs, outputs):
        r = float(inputs['radius'])
        h = float(inputs['height'])

        # Build geometry using OpenCASCADE kernel
        shape = BRepPrimAPI_MakeCylinder(r, h).Shape()

        # Compute properties
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        outputs['volume'] = props.Mass()

        brepgprop.SurfaceProperties(shape, props)
        outputs['surface_area'] = props.Mass()

    def compute_partials(self, inputs, partials):
        r = float(inputs['radius'])
        h = float(inputs['height'])

        # Analytic derivatives for a cylinder
        partials['volume', 'radius'] = 2 * np.pi * r * h
        partials['volume', 'height'] = np.pi * r**2
        partials['surface_area', 'radius'] = 2 * np.pi * h + 4 * np.pi * r
        partials['surface_area', 'height'] = 2 * np.pi * r
```

---

## 11. Closed-Source Tool Integration (COMSOL, ANSYS Fluent)

Closed-source tools present unique challenges: no source code access, no adjoint
implementations, restricted APIs, and licensing constraints. The wrapping strategy
must work within these limitations.

### Integration Challenges

```
+--------------------------+--------------------+--------------------+
| Challenge                | Open-Source         | Closed-Source      |
+--------------------------+--------------------+--------------------+
| API access               | Full (modify src)  | Limited (official  |
|                          |                    | API only)          |
+--------------------------+--------------------+--------------------+
| Analytic derivatives     | Implement adjoint  | Not available      |
|                          | in source          | (FD or surrogate)  |
+--------------------------+--------------------+--------------------+
| In-memory data exchange  | Direct (shared     | Often file-based   |
|                          | memory / arrays)   | only               |
+--------------------------+--------------------+--------------------+
| MPI integration          | Link against same  | Separate process   |
|                          | MPI                | (no shared comm)   |
+--------------------------+--------------------+--------------------+
| Licensing in HPC         | No restrictions    | License server     |
|                          |                    | availability       |
+--------------------------+--------------------+--------------------+
```

### COMSOL Integration

COMSOL has no official Python API. Three viable approaches exist:

#### Approach 1: MPh Library (Python via JPype)

```python
import openmdao.api as om
import numpy as np

try:
    import mph
except ImportError:
    raise ImportError("pip install mph  (requires COMSOL installation)")


class COMSOLWrapper(om.ExplicitComponent):
    """Wrap COMSOL Multiphysics via the MPh Python library.

    MPh uses JPype to bridge Python to COMSOL's Java API.
    Requires a COMSOL installation with a valid license.
    """

    def initialize(self):
        self.options.declare('model_file', types=str,
                             desc='Path to .mph model file')
        self.options.declare('study_name', types=str, default='Study 1')
        self.options.declare('comsol_port', types=int, default=2036)

    def setup(self):
        model_file = self.options['model_file']

        # Start COMSOL server (or connect to running one)
        self.client = mph.start(cores=4)
        self.comsol_model = self.client.load(model_file)

        # Declare OpenMDAO variables based on COMSOL model parameters
        # These must match parameter names defined in the .mph file
        #
        # Example: heat transfer problem
        self.add_input('thermal_conductivity', val=50.0, units='W/(m*K)')
        self.add_input('heat_flux', val=1000.0, units='W/m**2')
        self.add_input('geometry_param_1', val=0.01, units='m')

        self.add_output('max_temperature', val=0.0, units='K')
        self.add_output('avg_temperature', val=0.0, units='K')
        self.add_output('total_heat_flow', val=0.0, units='W')

    def setup_partials(self):
        # COMSOL does not provide adjoint derivatives.
        # Finite differences are the only option.
        #
        # Use central differences for better accuracy at 2x cost.
        self.declare_partials('*', '*', method='fd',
                              step=1e-4, form='central')

    def compute(self, inputs, outputs):
        model = self.comsol_model

        # Set parameters in COMSOL model
        model.parameter('k_thermal', f"{inputs['thermal_conductivity'][0]} [W/(m*K)]")
        model.parameter('q_flux', f"{inputs['heat_flux'][0]} [W/m^2]")
        model.parameter('L_param', f"{inputs['geometry_param_1'][0]} [m]")

        # Rebuild geometry if shape parameters changed
        model.build()

        # Mesh
        model.mesh()

        # Solve
        model.solve(self.options['study_name'])

        # Extract results
        # These use COMSOL's evaluation syntax
        outputs['max_temperature'] = model.evaluate(
            'maxop1(T)', 'dataset', 'Study 1/Solution 1'
        )
        outputs['avg_temperature'] = model.evaluate(
            'aveop1(T)', 'dataset', 'Study 1/Solution 1'
        )
        outputs['total_heat_flow'] = model.evaluate(
            'intop1(ht.ntflux)', 'dataset', 'Study 1/Solution 1'
        )

    def cleanup(self):
        """Stop COMSOL server when done."""
        if hasattr(self, 'client'):
            self.client.clear()
```

#### Approach 2: COMSOL via MATLAB Engine Bridge

```python
class COMSOLMatlabBridge(om.ExternalCodeComp):
    """Call COMSOL through MATLAB LiveLink via MATLAB Engine for Python.

    Requires: COMSOL with LiveLink for MATLAB, MATLAB Engine for Python.

    Data flow:
    OpenMDAO -> write params.mat -> MATLAB script -> COMSOL solve -> results.mat -> OpenMDAO
    """

    def setup(self):
        self.add_input('design_params', shape=(5,))
        self.add_output('objectives', shape=(2,))

        self.options['command'] = [
            'matlab', '-batch',
            "run('comsol_driver.m')"
        ]
        self.options['external_input_files'] = ['params.mat']
        self.options['external_output_files'] = ['results.mat']

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        import scipy.io as sio

        # Write inputs for MATLAB
        sio.savemat('params.mat', {
            'design_params': inputs['design_params']
        })

        # Run MATLAB + COMSOL
        super().compute(inputs, outputs)

        # Read results
        results = sio.loadmat('results.mat')
        outputs['objectives'] = results['objectives'].flatten()
```

**Corresponding MATLAB script (`comsol_driver.m`):**

```matlab
% comsol_driver.m -- Called by OpenMDAO via MATLAB Engine
load('params.mat');

% Connect to COMSOL
import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.load('tag', 'my_model.mph');

% Update parameters
model.param.set('p1', num2str(design_params(1)));
model.param.set('p2', num2str(design_params(2)));
% ... etc

% Solve
model.study('std1').run;

% Extract results
T_max = mphglobal(model, 'maxop1(T)');
q_total = mphglobal(model, 'intop1(ht.ntflux)');

objectives = [T_max, q_total];
save('results.mat', 'objectives');

ModelUtil.disconnect;
```

#### COMSOL Architecture Diagram

```
+------------------+       scipy.io          +------------------+
|    OpenMDAO      | -- write params.mat --> |                  |
|    Component     |                         |  MATLAB Engine   |
|                  |                         |  (comsol_driver) |
|  (Python)        | <-- read results.mat -- |                  |
+------------------+                         +-------+----------+
                                                     |
                                             COMSOL LiveLink
                                                     |
                                                     v
                                             +------------------+
                                             |     COMSOL       |
                                             |  Multiphysics    |
                                             |  Server          |
                                             +------------------+
```

### ANSYS Fluent Integration

Fluent offers better Python integration than COMSOL through the official
PyFluent library. The adjoint solver is available in Fluent but API access
to adjoint sensitivities via PyFluent is limited.

#### Approach 1: PyFluent (Recommended)

```python
import openmdao.api as om
import numpy as np

try:
    import ansys.fluent.core as pyfluent
except ImportError:
    raise ImportError("pip install ansys-fluent-core")


class FluentPyWrapper(om.ExplicitComponent):
    """Wrap ANSYS Fluent via PyFluent for in-memory control.

    Requires: ANSYS Fluent installation with valid license.
    """

    def initialize(self):
        self.options.declare('case_file', types=str,
                             desc='Fluent .cas.h5 case file')
        self.options.declare('dimension', types=int, default=3)
        self.options.declare('precision', types=str, default='double')
        self.options.declare('n_processors', types=int, default=4)
        self.options.declare('n_iterations', types=int, default=500)

    def setup(self):
        # Launch Fluent session
        self.solver = pyfluent.launch_fluent(
            dimension=self.options['dimension'],
            precision=self.options['precision'],
            processor_count=self.options['n_processors'],
            mode='solver'
        )

        # Read case
        self.solver.file.read_case(file_name=self.options['case_file'])

        # OpenMDAO inputs: flow conditions and boundary parameters
        self.add_input('velocity_inlet', val=10.0, units='m/s')
        self.add_input('temperature_inlet', val=300.0, units='K')
        self.add_input('pressure_outlet', val=101325.0, units='Pa')
        self.add_input('turbulent_intensity', val=0.05)

        # OpenMDAO outputs: performance metrics
        self.add_output('drag_force', val=0.0, units='N')
        self.add_output('lift_force', val=0.0, units='N')
        self.add_output('pressure_drop', val=0.0, units='Pa')
        self.add_output('avg_outlet_temperature', val=0.0, units='K')

    def setup_partials(self):
        # Fluent's adjoint solver exists but extracting sensitivities
        # via PyFluent API is limited. Use FD as fallback.
        self.declare_partials('*', '*', method='fd',
                              step=1e-3, form='central')

    def compute(self, inputs, outputs):
        solver = self.solver

        # Update boundary conditions via PyFluent settings API
        inlet = solver.setup.boundary_conditions.velocity_inlet['inlet']
        inlet.momentum.velocity.value = float(inputs['velocity_inlet'])
        inlet.thermal.temperature.value = float(inputs['temperature_inlet'])
        inlet.turbulence.turbulent_intensity.value = float(
            inputs['turbulent_intensity']
        )

        outlet = solver.setup.boundary_conditions.pressure_outlet['outlet']
        outlet.momentum.gauge_pressure.value = float(inputs['pressure_outlet'])

        # Initialize and solve
        solver.solution.initialization.hybrid_initialize()
        solver.solution.run_calculation.iterate(
            iter_count=self.options['n_iterations']
        )

        # Extract results from report definitions
        # (These must be pre-defined in the Fluent case)
        outputs['drag_force'] = solver.solution.report_definitions.compute(
            report_defs=['drag-force']
        )['drag-force'][0]

        outputs['lift_force'] = solver.solution.report_definitions.compute(
            report_defs=['lift-force']
        )['lift-force'][0]

        outputs['pressure_drop'] = solver.solution.report_definitions.compute(
            report_defs=['pressure-drop']
        )['pressure-drop'][0]

        outputs['avg_outlet_temperature'] = \
            solver.solution.report_definitions.compute(
                report_defs=['avg-T-outlet']
            )['avg-T-outlet'][0]

    def cleanup(self):
        """Shutdown Fluent session."""
        if hasattr(self, 'solver') and self.solver is not None:
            self.solver.exit()
```

#### Approach 2: Journal File Wrapping (No PyFluent Dependency)

```python
class FluentJournalWrapper(om.ExternalCodeComp):
    """Wrap Fluent using TUI journal files.

    Works with any Fluent version. No Python API needed.
    """

    def initialize(self):
        self.options.declare('journal_template', types=str)
        self.options.declare('fluent_exec', types=str,
                             default='fluent')
        self.options.declare('dimension', types=str, default='3ddp')

    def setup(self):
        self.add_input('velocity', val=10.0, units='m/s')
        self.add_input('aoa', val=0.0, units='deg')
        self.add_output('C_L', val=0.0)
        self.add_output('C_D', val=0.0)

        self.options['command'] = [
            self.options['fluent_exec'],
            self.options['dimension'],
            '-g', '-i', 'runtime.jou'
        ]
        self.options['external_input_files'] = ['runtime.jou']
        self.options['external_output_files'] = ['forces.csv']
        self.options['timeout'] = 3600  # 1 hour max

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Generate journal file from template
        with open(self.options['journal_template'], 'r') as f:
            template = f.read()

        journal = template.replace('__VELOCITY__',
                                   f"{inputs['velocity'][0]:.8f}")
        journal = template.replace('__AOA__',
                                   f"{inputs['aoa'][0]:.8f}")

        with open('runtime.jou', 'w') as f:
            f.write(journal)

        # Run Fluent
        super().compute(inputs, outputs)

        # Parse output CSV
        data = np.genfromtxt('forces.csv', delimiter=',', skip_header=1)
        outputs['C_L'] = data[-1, 1]  # last iteration
        outputs['C_D'] = data[-1, 2]
```

**Corresponding Fluent journal template (`template.jou`):**

```
; Fluent TUI journal for OpenMDAO integration
/file/read-case "wing.cas.h5"

; Set boundary conditions
/define/boundary-conditions/velocity-inlet inlet
yes yes no __VELOCITY__ no 0 no no yes 5 0.1

; Initialize
/solve/initialize/hyb-initialization

; Solve
/solve/iterate 500

; Export forces
/report/forces/wall-forces yes wall () yes "forces.csv"

; Exit
/exit yes
```

#### Fluent Architecture Diagram

```
Option A: PyFluent (in-process)          Option B: Journal (subprocess)

+------------------+                     +------------------+
|    OpenMDAO      |                     |    OpenMDAO      |
|    Component     |                     | ExternalCodeComp |
+--------+---------+                     +--------+---------+
         |                                        |
    PyFluent API                          write runtime.jou
    (gRPC / Python)                               |
         |                                        v
         v                               +------------------+
+------------------+                     | fluent 3ddp -g   |
|  Fluent Solver   |                     | -i runtime.jou   |
|  (in-process or  |                     +--------+---------+
|   gRPC server)   |                              |
+------------------+                      read forces.csv
                                                  |
                                                  v
                                          +------------------+
                                          |  OpenMDAO parses |
                                          |  outputs         |
                                          +------------------+
```

### Derivative Strategies for Closed-Source Tools

Since you cannot implement adjoint solvers in closed-source tools, use these
strategies to make gradient-based optimization viable:

```
+---------------------------+------------------+---------------------------+
| Strategy                  | Design Variables | When to Use               |
+---------------------------+------------------+---------------------------+
| Finite Differences        | < 20             | Quick prototyping, low    |
|                           |                  | dimension problems        |
+---------------------------+------------------+---------------------------+
| Surrogate-Assisted        | 20 - 100+        | Train surrogate on tool  |
| Optimization              |                  | evaluations, optimize    |
|                           |                  | surrogate (with derivs)  |
+---------------------------+------------------+---------------------------+
| Gradient-Free Optimizer   | < 50             | When FD is too expensive  |
| (GA, CMA-ES, etc.)       |                  | but surrogates impractical|
+---------------------------+------------------+---------------------------+
| Simultaneous Perturbation | 50 - 200         | 2 evaluations regardless |
| Stochastic Approx. (SPSA)|                  | of dimension (noisy)     |
+---------------------------+------------------+---------------------------+
| Hybrid: FD for closed     | Mixed            | Use adjoint where you    |
| + adjoint for open        |                  | can, FD for the rest     |
+---------------------------+------------------+---------------------------+
```

---

## 12. MPI and Parallel Execution

High-fidelity solvers (SU2, FEniCS, TACS) are MPI-parallel. OpenMDAO and MPhys
support MPI natively.

### MPI Communicator Flow

```
MPI_COMM_WORLD (all processors)
    |
    +-- Multipoint.setup() splits communicator
          |
          +-- Scenario "cruise" (procs 0-15)
          |     |
          |     +-- AeroSolver (procs 0-7)
          |     +-- StructSolver (procs 8-15)
          |
          +-- Scenario "maneuver" (procs 16-31)
                |
                +-- AeroSolver (procs 16-23)
                +-- StructSolver (procs 24-31)
```

### Key Rules for MPI Wrappers

```python
class MPISolverComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('solver', recordable=False)

    def setup(self):
        solver = self.options['solver']

        # RULE 1: Use self.comm, not MPI.COMM_WORLD
        # OpenMDAO assigns the correct sub-communicator
        comm = self.comm

        # RULE 2: Only add I/O on all procs in your sub-comm
        if comm.rank == 0:
            # This is WRONG. All ranks must call add_input/add_output.
            pass

        # CORRECT: all ranks declare the same I/O
        self.add_input('x_aero', shape=(solver.get_local_nnodes() * 3,))
        self.add_output('states', shape=(solver.get_local_nstates(),))

        # RULE 3: Use distributed=True for arrays split across procs
        self.add_input('x_aero_dist', distributed=True,
                       shape=(solver.get_local_nnodes() * 3,))
        self.add_output('states_dist', distributed=True,
                        shape=(solver.get_local_nstates(),))

    def solve_nonlinear(self, inputs, outputs):
        # RULE 4: All ranks must participate in the solve
        # (even if some have zero local work)
        solver = self.options['solver']
        solver.solve(self.comm)

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs,
                     d_residuals, mode):
        # RULE 5: Distributed derivatives must be consistent
        # across processors. OpenMDAO handles the global assembly.
        pass
```

### Running with MPI

```bash
# Single scenario, 8 processors for the solver
mpirun -np 8 python my_optimization.py

# Multi-point with 2 scenarios, 4 procs each
mpirun -np 8 python my_optimization.py  # MPhys splits automatically

# Using MultipointParallel for concurrent scenarios
mpirun -np 16 python my_optimization.py
```

---

## 13. Testing and Verification

### Derivative Verification (Critical)

```python
import openmdao.api as om

prob = om.Problem()
prob.model.add_subsystem('wrapper', MyWrapper())
prob.setup(force_alloc_complex=True)  # enable complex step

# Set realistic input values (not defaults!)
prob.set_val('wrapper.x_aero', realistic_mesh_coords)
prob.set_val('wrapper.aoa', 2.0)

prob.run_model()

# Check partial derivatives
data = prob.check_partials(
    includes=['wrapper'],    # only check your component
    compact_print=True,
    method='cs',             # complex step (most accurate)
    step=1e-30,
    out_stream=None          # suppress print, return dict
)

# Automated assertion
for comp_name in data:
    for (of, wrt) in data[comp_name]:
        err = data[comp_name][(of, wrt)]
        rel_error = err['rel error']
        print(f"  d({of})/d({wrt}): rel_error = {rel_error.forward:.2e}")
        assert rel_error.forward < 1e-5, \
            f"Derivative check failed for d({of})/d({wrt})"
```

### Unit Test Template

```python
import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials


class TestMyWrapper(unittest.TestCase):

    def setUp(self):
        """Set up a problem with the wrapper for testing."""
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wrapper', MyWrapper(
            mesh_file='test_mesh.cgns',
            solver_options={'max_iter': 100}
        ))
        self.prob.setup(force_alloc_complex=True)

    def test_compute_runs(self):
        """Test that compute executes without error."""
        self.prob.set_val('wrapper.aoa', 2.0)
        self.prob.run_model()
        # Check outputs are finite and reasonable
        cl = self.prob.get_val('wrapper.C_L')
        cd = self.prob.get_val('wrapper.C_D')
        self.assertTrue(np.isfinite(cl))
        self.assertTrue(np.isfinite(cd))
        self.assertGreater(cd, 0)  # drag should be positive

    def test_partials(self):
        """Verify partial derivatives against complex step."""
        self.prob.set_val('wrapper.aoa', 2.0)
        self.prob.run_model()
        data = self.prob.check_partials(
            includes=['wrapper'],
            method='cs',
            compact_print=True
        )
        assert_check_partials(data, atol=1e-5, rtol=1e-5)

    def test_totals(self):
        """Verify total derivatives through the full model."""
        self.prob.set_val('wrapper.aoa', 2.0)
        self.prob.run_model()
        totals = self.prob.check_totals(
            of=['wrapper.C_L', 'wrapper.C_D'],
            wrt=['wrapper.aoa'],
            compact_print=True
        )
        for key in totals:
            err = totals[key]
            self.assertLess(abs(err['rel error'].forward), 1e-4)

    def test_solver_convergence(self):
        """Test that the solver converges for different conditions."""
        for aoa in [0.0, 2.0, 5.0, 8.0]:
            self.prob.set_val('wrapper.aoa', aoa)
            self.prob.run_model()
            # Check residual is small
            residual = self.prob.model.wrapper.get_residuals()
            self.assertLess(np.linalg.norm(residual), 1e-8)


if __name__ == '__main__':
    unittest.main()
```

---

## 14. Common Pitfalls and Debugging

### Pitfall 1: Forgetting to Handle Units

```python
# WRONG: units mismatch causes silent errors
self.add_input('velocity', val=10.0)        # what units??
self.add_output('force', val=0.0)           # Newtons? Pounds?

# CORRECT: always specify units
self.add_input('velocity', val=10.0, units='m/s')
self.add_output('force', val=0.0, units='N')
```

### Pitfall 2: Stale State in Iterative Solvers

```python
def solve_nonlinear(self, inputs, outputs):
    # WRONG: solver keeps state from previous call
    # This can cause convergence issues in optimization
    self.solver.solve()

    # CORRECT: reset or warm-start properly
    if self.first_call:
        self.solver.initialize()
        self.first_call = False
    else:
        # Use previous solution as initial guess (warm start)
        pass
    self.solver.solve()
```

### Pitfall 3: Not Declaring Partials as Zero

```python
def setup_partials(self):
    # WRONG: declares ALL partials, even zero ones
    self.declare_partials('*', '*')
    # OpenMDAO will try to compute/store all of them

    # CORRECT: only declare non-zero partials
    self.declare_partials('C_L', 'aoa')
    self.declare_partials('C_L', 'x_aero')
    self.declare_partials('C_D', 'aoa')
    self.declare_partials('C_D', 'x_aero')
    # C_L does not depend on structural_thickness -> not declared
```

### Pitfall 4: File I/O Race Conditions in Parallel

```python
def compute(self, inputs, outputs):
    # WRONG: all MPI ranks write to the same file
    with open('input.dat', 'w') as f:
        f.write(str(inputs['x']))

    # CORRECT: only rank 0 writes, then broadcast
    if self.comm.rank == 0:
        with open('input.dat', 'w') as f:
            f.write(str(inputs['x']))
    self.comm.Barrier()  # wait for file to be written
```

### Pitfall 5: Large Memory from Dense Jacobians

```python
# WRONG: 50000x50000 dense Jacobian = 20 GB!
self.add_output('flow_field', shape=(50000,))
self.add_input('mesh', shape=(50000,))
self.declare_partials('flow_field', 'mesh')  # dense by default

# CORRECT: use sparse format or matrix-free
# Option A: Sparse (known sparsity pattern)
self.declare_partials('flow_field', 'mesh',
                      rows=sparse_rows, cols=sparse_cols)

# Option B: Matrix-free (implement apply_linear instead)
# No declare_partials needed; provide Jacobian-vector products
```

### Debugging Checklist

```
1. [ ] Does `prob.run_model()` complete without errors?
2. [ ] Are outputs physically reasonable (not NaN, not zero)?
3. [ ] Do `check_partials()` pass with rel_error < 1e-5?
4. [ ] Do `check_totals()` pass with rel_error < 1e-4?
5. [ ] Does the solver converge for perturbed inputs?
6. [ ] Does the N2 diagram show correct connections?
7. [ ] Are units consistent across connected variables?
8. [ ] For MPI: do all ranks participate in setup/compute?
9. [ ] For file I/O: no race conditions between ranks?
10.[ ] Memory usage reasonable? (check for dense Jacobians)
```

### Useful Debugging Commands

```python
# Generate N2 diagram (visual dependency graph)
om.n2(prob, outfile='n2.html', show_browser=True)

# List all variables and their values
prob.model.list_inputs(print_arrays=True)
prob.model.list_outputs(print_arrays=True)

# Check connections
prob.model.list_connections()

# Print solver convergence history
prob.set_solver_print(level=2)

# Profile execution time
prob.setup()
with om.profiling.profile() as prof:
    prob.run_model()
prof.print_stats()
```

---

## 15. Reference: Existing MPhys-Compatible Wrappers

Use these as reference implementations when building your own wrapper.

```
+-----------------+----------+---------+------------------+---------------------------+
| Package         | Domain   | Derivs  | MPhys Builder?   | Repository                |
+-----------------+----------+---------+------------------+---------------------------+
| ADflow          | CFD      | Adjoint | Yes (AdflowBld)  | mdolab/adflow             |
| DAfoam          | CFD      | D.Adj.  | Yes              | mdolab/dafoam             |
| OpenAeroStruct  | VLM/FEA  | Analyt. | Yes              | mdolab/OpenAeroStruct     |
| TACS            | Struct.  | Adjoint | Yes (TacsBld)    | smdogroup/tacs            |
| FunToFEM        | Transfer | Analyt. | Yes              | smdogroup/funtofem        |
| pyCycle         | Thermo   | Analyt. | Yes              | OpenMDAO/pyCycle          |
| pyGeo           | Geom.    | Analyt. | Yes (DVGeoBld)   | mdolab/pygeo              |
| ESP/CAPS        | CAD/Mesh | FD/Adj. | Partial          | OpenMDAO/EngSketchPad     |
| OpenVSP         | Geom.    | FD      | Via pyGeo        | OpenVSP/OpenVSP           |
| Meld/FunToFEM   | Xfer     | Analyt. | Yes              | smdogroup/funtofem        |
+-----------------+----------+---------+------------------+---------------------------+
```

### Quick Reference: Variable Naming Conventions in MPhys

```
+----------------------------+----------------------------------------------+
| Variable Name              | Meaning                                      |
+----------------------------+----------------------------------------------+
| x_aero0                    | Initial aerodynamic surface coordinates      |
| x_struct0                  | Initial structural surface coordinates       |
| x_aero                     | Current (deformed) aero surface coordinates  |
| x_struct                   | Current (deformed) struct coordinates         |
| f_aero                     | Aerodynamic surface forces                   |
| u_struct                   | Structural displacements                     |
+----------------------------+----------------------------------------------+
| Tags:                                                                     |
| mphys_coordinates          | Mesh coordinate variables                    |
| mphys_coupling             | Variables exchanged in coupling iterations   |
| mphys_result               | Post-coupling functional outputs             |
+----------------------------+----------------------------------------------+
```

### Minimal Wrapper Checklist

```
To create a new MPhys wrapper, you need at minimum:

1. [ ] A Builder class that implements:
       [ ] __init__() -- store options (no heavy work)
       [ ] initialize(comm) -- read mesh, allocate solver
       [ ] get_mesh_coordinate_subsystem() -- returns IndepVarComp with x_*0
       [ ] get_coupling_group_subsystem() -- returns main solver component
       [ ] get_number_of_nodes()
       [ ] get_ndof()

2. [ ] A solver Component (Explicit or Implicit) that implements:
       [ ] setup() -- declare inputs/outputs with correct shapes and units
       [ ] compute() or solve_nonlinear() -- run the solver
       [ ] Derivative method (one of):
           [ ] compute_partials() -- for ExplicitComponent
           [ ] apply_linear() + solve_linear() -- for ImplicitComponent
           [ ] declare_partials(method='fd') -- finite difference fallback

3. [ ] Tests:
       [ ] test_compute() -- solver runs and gives reasonable outputs
       [ ] test_partials() -- derivatives verified via check_partials()
       [ ] test_totals() -- total derivatives verified via check_totals()
```

---

## Appendix A: Installation Notes

### SU2 with Python Wrapper

```bash
# Build SU2 with Python wrapper enabled
git clone https://github.com/su2code/SU2.git
cd SU2
python meson.py build -Denable-pywrapper=true -Dwith-mpi=enabled
cd build && ninja install
# Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/SU2/build/SU2_PY
```

### FEniCS with dolfin-adjoint

```bash
# Via conda (recommended)
conda create -n fenics-mdao -c conda-forge fenics dolfin-adjoint
conda activate fenics-mdao
pip install openmdao

# Or via Docker
# docker pull quay.io/fenicsproject/stable:latest
```

### PhysicsNeMo

```bash
pip install nvidia-physicsnemo
# For ONNX export:
pip install onnxruntime-gpu  # or onnxruntime for CPU
```

### ParaBlade

```bash
git clone https://github.com/NAnand-TUD/parablade.git
cd parablade
pip install -e .
```

### CadQuery

```bash
conda install -c conda-forge cadquery
# or: pip install cadquery
```

### ESP/CAPS

```bash
# Build from source (requires OpenCASCADE)
git clone https://github.com/OpenMDAO/EngSketchPad.git
cd EngSketchPad
# Follow build instructions in README
# pyCAPS is included in the build
```

### MPh (COMSOL Python bridge)

```bash
pip install mph
# Requires COMSOL Multiphysics installed on the system
```

### PyFluent

```bash
pip install ansys-fluent-core
# Requires ANSYS Fluent installed on the system
```

---

## Appendix B: Recommended Reading

- **OpenMDAO Documentation**: https://openmdao.org/newdocs/versions/latest/main.html
- **MPhys Documentation**: https://openmdao.github.io/mphys/
- **MDO Lab GitHub**: https://github.com/mdolab (reference implementations)
- **Martins & Ning, "Engineering Design Optimization"** (Cambridge University Press, 2021) -- the textbook on MDAO
- **Kenway et al., "Effective Adjoint Approaches for Computational Fluid Dynamics"** (Progress in Aerospace Sciences, 2019)
- **SU2 Python Wrapper**: https://su2code.github.io/docs/Python-Wrapper-Build/
- **dolfin-adjoint**: https://www.dolfin-adjoint.org/
- **NVIDIA PhysicsNeMo**: https://github.com/NVIDIA/physicsnemo
- **ParaBlade**: https://github.com/NAnand-TUD/parablade
- **ESP/CAPS**: https://acdl.mit.edu/ESP/
