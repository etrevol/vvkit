# vvkit Demonstration Suite

This directory contains the official pitch and verification demonstration suite for `vvkit`. The suite is designed to showcase the framework's capabilities in automating the Method of Manufactured Solutions (MMS), grid convergence studies, and strict mathematical verification across a variety of partial differential equations (PDEs), coordinate systems, and solver topologies.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites & Execution](#prerequisites--execution)
3. [The Mock Solver](#the-mock-solver)
4. [Verification Cases](#verification-cases)
5. [Understanding the Output](#understanding-the-output)
6. [Generating MMS Code](#generating-mms-code)

---

## Overview

The `vvkit` demonstration suite proves that a numerical discretization achieves its theoretical order of accuracy. The suite relies entirely on declarative YAML configurations (`vvcase.yaml`) rather than custom Python orchestration code. 

By running this suite, you will witness:
- **Automated Symbolic MMS**: Generation of exact analytical forcing terms $S = L(u_{exact})$ from SymPy string representations.
- **Multi-Dimensional Support**: Verification of 1D, 2D, and 3D solvers.
- **Coordinate System Agnosticism**: Handling of Cartesian, Spherical, and Cylindrical domains.
- **Rigorous Quadrature**: Utilization of $N$-dimensional Gauss-Legendre cell-averaging to compute true $L_2$ norms.
- **ASME V&V 20 Metrics**: Estimation of the Grid Convergence Index (GCI), asymptotic range indicator, and expected order validation.
- **Advanced Diagnostics**: Automatic detection of mathematical round-off floors (noise exclusion from GCI metrics), identification of physically invalid (non-positive) solutions, and catching "vanishing terms" in analytical MMS source evaluations.

---

## Prerequisites & Execution

To run the pitch demonstration suite in fully automated mode:

```bash
uv run python examples/demo/run_pitch.py --auto
```

This master script performs the following operations:
1. Iterates through all `.yaml` definitions in `examples/demo/cases/`.
2. Invokes `vvkit mms` to generate a mock C/C++ source term file.
3. Invokes `vvkit run` to execute the spatial or temporal verification sweep.
4. Orchestrates the external `mock_solver.py` via subprocesses.
5. Generates the final convergence HTML reports in `workdir_{case_name}/reports/`.

---

## The Mock Solver

The script `mock_solver.py` acts as a stand-in for a compiled C++ or Fortran numerical solver. 

**How it works:**
- `vvkit` uses Jinja2 (`templates/solver.in.j2`) to generate an input configuration file (e.g., `solver.in`) containing parameters like the mesh size (`n_cells`) or timestep (`n_steps`).
- `vvkit` calls the solver using the command specified in the YAML configuration: `python mock_solver.py solver.in`.
- The mock solver reads the input, determines the equation type, and generates a numerical solution $u_{num}$ that artificially includes an exact theoretical truncation error (e.g., $O(\Delta x^2)$).
- The solution is saved in NumPy `.npz` format containing the coordinate meshgrids (`x`, `y`, `z`, or `r`) and the solution scalar field (`u`).
- `vvkit` reads the `.npz` file, evaluates the exact solution $u_{exact}$, calculates the discrete $L_2$ error norm, and updates the regression metrics.

---

## Verification Cases

The following test suites are executed:

### 1. `advection_1d.yaml`
- **Domain**: 1D Cartesian ($x$).
- **Physics**: Linear advection-diffusion.
- **Verification Target**: 2nd-order spatial accuracy ($O(\Delta x^2)$).
- **Norm**: Discrete point-value $L_2$.

### 2. `burgers_1d_temporal.yaml`
- **Domain**: 1D Cartesian time-dependent ($x, t$).
- **Physics**: Non-linear Burgers equation. Validates `vvkit`'s ability to symbolically differentiate non-linear terms such as $u \partial_x u$.
- **Verification Target**: 1st-order temporal accuracy (Forward Euler, $O(\Delta t)$).
- **Conservation Check**: Validates the discrete budget over the integration domain.

### 3. `cylindrical_2d.yaml`
- **Domain**: 2D Cylindrical ($r, z$).
- **Physics**: Poisson equation.
- **Verification Target**: 2nd-order spatial accuracy.
- **Key Feature**: Utilizes a specific `cell_measures: R * dr * dz` weighting to correctly compute the $L_2$ norm in cylindrical coordinates.

### 4. `diffusion_1d_temporal.yaml`
- **Domain**: 1D Cartesian time-dependent ($x, t$).
- **Physics**: Linear diffusion.
- **Verification Target**: 1st-order temporal accuracy.

### 5. `poisson_2d.yaml`
- **Domain**: 2D Cartesian ($x, y$).
- **Physics**: Poisson equation.
- **Verification Target**: 2nd-order spatial accuracy over a tensor-product meshgrid.
- **Key Feature**: High-order Gauss-Legendre `cell_average` quadrature integration. Demonstrates that `vvkit` correctly integrates the analytical solution over $N$-dimensional finite volumes, ensuring $L_2$ convergence doesn't artificially cap at 2nd order.

### 6. `poisson_3d.yaml`
- **Domain**: 3D Cartesian ($x, y, z$).
- **Physics**: Poisson equation.
- **Verification Target**: 2nd-order spatial accuracy.
- **Key Feature**: Demonstrates large-scale 3D mesh support and multi-dimensional array slicing.

### 7. `spherical_1d.yaml`
- **Domain**: 1D Spherical ($r$).
- **Physics**: Laplacian in spherical geometry.
- **Verification Target**: 2nd-order spatial accuracy.
- **Key Feature**: Spherical cell measures $r^2 dr$ for correct integration bounds.

---

## Understanding the Output

Upon completion, `vvkit` generates a `reports/` directory containing:

- **HTML Report (`{name}.html`)**: A human-readable verification document containing the log-log grid convergence plot, Least-Squares regression statistics ($R^2$, slope standard error), and ASME V&V 20 Grid Convergence Index (GCI) metrics. It also surfaces critical diagnostic warnings for Divergent/Oscillatory convergence states and Asymptotic Ratio violations ($|R - 1| > 0.1$).
- **JSON Report (`{name}.json`)**: A machine-readable contract structured according to the `schema_version: 1` specification, ideal for CI/CD tracking and regression alerting.
- **PNG Plot (`{name}_plot.png`)**: A standalone log-log error vs. element size plot. If the discretization hits the machine precision limits on the finest grids, `vvkit` dynamically detects this "round-off floor" and plots these excluded points in red (`x--`).

A passing test suite signifies that the observed convergence order rigorously matches the expected theoretical order within the defined tolerance (e.g., $2.0 \pm 0.2$), and that the GCI asymptotic limits hold true.

---

## Generating MMS Code

If you are developing a real C++ or C solver, you can use `vvkit mms` to emit the source code for the generated source term:

```bash
vvkit mms --config-path cases/poisson_2d.yaml --language cpp --output mms_source.cpp
```

This translates the SymPy abstract syntax tree directly into standard `<cmath>` C++ functions, ready to be compiled into your solver binary.
