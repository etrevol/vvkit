<p align="center">
  <img src="design/assets/banner.svg" alt="vvkit banner" width="100%">
</p>

# vvkit — Verification Harness for Numerical Solvers

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checking: mypy strict](https://img.shields.io/badge/mypy-strict-brightgreen.svg)](https://mypy.readthedocs.io/)
[![Wiki](https://img.shields.io/badge/docs-wiki-brightgreen.svg)](https://github.com/etrevol/vvkit/wiki)

> **`pytest` proves your code runs. `vvkit` proves your discretization converges at the order you claim.**

`vvkit` is the professional verification framework for computational scientists and numerical solver authors (CFD, thermal-hydraulics, FEM, custom PDE codes). It automates formal code verification through the **Method of Manufactured Solutions (MMS)**, systematic grid refinement sweeps, **Grid Convergence Index (GCI)** estimation per ASME V&V 20, discrete conservation budget checks, and regression baseline drift detection.

## 📖 Comprehensive Documentation (Wiki)

For detailed installation instructions, architectural overview, mathematical theory, and plugins, please visit the **[vvkit Wiki](https://github.com/etrevol/vvkit/wiki)**.

- **[Setup Guide](https://github.com/etrevol/vvkit/wiki/Home)**: How to install and configure `vvkit`.
- **[Mathematical Theory](https://github.com/etrevol/vvkit/wiki/Theory)**: Deep dive into the verification mathematics ($L_1$, $L_2$, $L_\infty$ norms, Grid Convergence Index).
- **[Verification Suite Demos](https://github.com/etrevol/vvkit/wiki/Demos)**: Explanation of the Python verification suites.
- **[Athena++ Integration](https://github.com/etrevol/vvkit/wiki/Athena++-Integration)**: Full documentation of the automated Athena++ MHD verification plugin.
- **[Plugins and Adapters](https://github.com/etrevol/vvkit/wiki/Plugins-and-Adapters)**: How to write your own adapter for any C/C++/Fortran numerical solver.

---

## Key Features

- **Symbolic MMS Engine**: Automatic derivation of non-trivial source terms $S = L(u_m)$, boundary data, and initial conditions from SymPy PDE operator definitions.
- **Multi-Language Code Emitters**: Automatic code emission of manufactured source terms to **Python**, **C**, and **C++** with mathematical standard library compatibility.
- **Rigorous Order of Accuracy**:
  - Pairwise order and log-log **Least-Squares fit** with standard error and $R^2$.
  - **Roache transcendental root-finding** for non-constant grid refinement ratios $r = h_i / h_j$.
  - **Gauss-Legendre cell-averaging** ($\geq$ 4th-order quadrature) to prevent $O(h^2)$ capping on Finite Volume solvers.
- **ASME V&V 20 Compliance**:
  - Grid Convergence Index ($\text{GCI}_{\text{fine}}$) with automatic safety factor selection ($F_s = 1.25$ vs $F_s = 3.0$).
  - Asymptotic range indicator ($R$) and convergence state classification ($R_c$: Monotonic, Oscillatory, Divergent).
  - Round-off error floor detection to exclude noise from asymptotic convergence slopes.
- **Universal Solver Adapters**:
  - `CallableAdapter`: Direct in-process Python solver functions.
  - `CommandAdapter`: External executable solvers driven via Jinja2 templated input files in isolated temporary workdirs.
  - Built-in readers for **NumPy (`.npz`)**, **HDF5 (`.h5`)**, **CSV**, **Text**, and custom callables via entry points.
- **Athena++ Dedicated Plugin**:
  - Seamless, zero-configuration bridging between Windows and WSL for native compilation and execution.
  - Automatic `pathlib`-based output discovery and intelligent column mapping for Athena++ `.tab` dumps.
- **Conservation & Invariant Checks**: Time-series discrete budget closure monitoring to pinpoint the exact time step where conservation departs from round-off bounds.
- **CI-Ready Reporting**: Offline-capable HTML reports, machine-readable JSON contracts (`schema_version: 1`), and JUnit XML for continuous integration pipelines.
- **Pytest Integration**: Native `@pytest.mark.convergence(case="...")` test marker support.

---

## Quickstart

Detailed setup instructions are available in the [Wiki Setup Guide](https://github.com/etrevol/vvkit/wiki/Home). 

If you already have `uv` installed, simply:

```bash
# Clone the repository with its wiki submodule
git clone --recurse-submodules https://github.com/etrevol/vvkit.git
cd vvkit

# Install dependencies and sync virtual environment
uv sync

# Run the full internal Python demonstration suite
uv run python examples/demo/run_pitch.py --auto
```

The reports will be written to `examples/demo/workdir_{case_name}/reports/`. Open the `.html` files in any web browser to view the interactive Plotly convergence plots and diagnostic metrics!

---

## License

`vvkit` is licensed under the **Apache-2.0 License**. See [LICENSE](LICENSE) for details.
