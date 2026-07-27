# vvkit — Verification Harness for Numerical Solvers

**Status:** greenfield. Nothing exists yet.
**Owner:** Artem Holovashchenko (human). You are the implementing agent.
**Primary language:** Python 3.12+.
**License:** Apache-2.0.

---

## 1. Problem statement

Authors of numerical solvers (CFD, thermal-hydraulics, MHD, FEM, custom PDE codes)
routinely ship code whose *order of accuracy has never been measured*. The standard
practice is ad-hoc: a few plots, an eyeballed comparison against an analytical case,
and a spreadsheet. Formal verification — the Method of Manufactured Solutions (MMS),
systematic grid-convergence studies, Grid Convergence Index per ASME V&V 20 — is
well-documented in the literature but has almost no tooling. Every group re-implements
it badly.

`vvkit` is the missing tool. It turns solver verification into a declarative,
reproducible, CI-runnable artifact.

### The one-sentence pitch

> `pytest` proves your code runs. `vvkit` proves your discretization converges at the
> order you claim.

---

## 2. Scope

### In scope (v1.0)

1. **MMS generation** — symbolic derivation of source terms and boundary/initial
   conditions from a user-declared PDE operator and a chosen manufactured solution,
   with code emission to Python, C, C++, and C#.
2. **Convergence studies** — spatial and temporal refinement sweeps, error norms,
   observed order of accuracy, GCI, asymptotic-range diagnostics.
3. **Solver adapters** — run the user's solver, whatever it is: an in-process Python
   callable, or an external executable driven by templated input files.
4. **Conservation & invariant checks** — discrete budget closure for conserved
   quantities; user-declarable invariants.
5. **Regression baselines** — golden-value storage with tolerances, drift detection.
6. **Reporting** — human-readable HTML report, machine-readable JSON, JUnit XML for CI.
7. **CLI + pytest plugin.**

### Explicitly out of scope (v1.0)

- Validation against experimental data (that is a different, later product).
- Uncertainty quantification / sensitivity analysis / Monte Carlo sampling.
- Mesh generation. `vvkit` never builds a mesh; it asks the solver for a sequence of
  meshes or passes a refinement parameter through.
- Any GUI. CLI and generated HTML only.
- Parallel/cluster job scheduling. `vvkit` may run cases concurrently on one machine
  (process pool), nothing more.
- Solving PDEs itself. `vvkit` is never a solver.

Do not implement out-of-scope items. If you believe one is necessary,
write a concise ADR (Architecture Decision Record) detailing: 1. Context, 2. Proposed Decision, 3. Consequences.
Stop implementation and wait for human review.

---

## 3. Domain background you must get right

This section is the technical core. Implementation must match it exactly. If any
formula below appears to conflict with a reference you find, **stop and ask the human**
rather than silently choosing.

### 3.1 Method of Manufactured Solutions

Given a differential operator `L` and a governing equation `L(u) = 0`, MMS proceeds:

1. Pick an analytic function `u_m(x, t)` — the *manufactured solution*. It need not be
   physical. It must be smooth, non-trivial (no term of the operator may vanish
   identically), and must exercise every derivative present in `L`.
2. Compute the residual `S = L(u_m)` symbolically. `S` is nonzero.
3. Add `S` as a source term to the solver's governing equation. The modified problem
   `L(u) = S` now has `u_m` as its exact solution.
4. Derive Dirichlet/Neumann boundary data and the initial condition by evaluating
   `u_m` (or its derivatives) on the domain boundary and at `t = 0`.
5. Run the solver on a sequence of refined meshes; measure `‖u_h − u_m‖` and its
   convergence rate.

**Design consequences.** The user declares the operator symbolically in SymPy terms.
`vvkit` derives `S`, the BCs, and the IC — the user never hand-differentiates. Common
solution families (trigonometric, exponential, polynomial, and a manufactured
boundary-layer form) ship as presets.

**Failure modes to detect and warn about:**
- A manufactured solution that makes a term identically zero → that term is untested.
  Detect symbolically and warn, listing which operator terms vanished.
- Negative density/pressure/temperature over the domain when the target solver is a
  physical code that will crash on non-positive states. Sample the solution over the
  domain and warn.
- Source terms so large they dominate the physical terms (poorly scaled MMS).
  Report the ratio of `‖S‖` to `‖L_i(u_m)‖` per term.

### 3.2 Error norms

For a discrete solution on a grid with cell measures `V_i` (length, area, volume —
weights supplied by the adapter, defaulting to uniform):

```
L1   = Σ |e_i| V_i / Σ V_i
L2   = sqrt( Σ e_i² V_i / Σ V_i )
Linf = max |e_i|
```

Weighting by cell measure is mandatory for non-uniform grids; unweighted norms give
wrong orders. When the adapter cannot supply weights, log a warning that results are
valid only for uniform grids.

For cell-averaged (finite-volume) solvers, the exact reference must be the *cell
average* of `u_m`, not its point value at the cell centre. Point sampling caps the
observed order at 2. Provide both modes; default to cell-average for FV adapters, with
Gauss quadrature of order ≥ 4 for the averaging. This detail is the single most common
reason a correct 3rd-order scheme appears to be 2nd-order — it must be handled.

### 3.3 Observed order of accuracy

**Three-grid formula, constant refinement ratio `r = h_coarse / h_fine`:**

```
p = ln( (f_3 − f_2) / (f_2 − f_1) ) / ln(r)
```

where `f_1` is the finest. When the exact solution is known (MMS), use errors `e_k`
directly instead of solution differences:

```
p = ln( e_3 / e_2 ) / ln(r)      (pairwise)
```

**Non-constant refinement ratio.** Solve Roache's transcendental equation for `p` by
fixed-point iteration:

```
p = (1/ln(r_21)) * | ln|ε_32/ε_21| + q(p) |
q(p) = ln( (r_21^p − s) / (r_32^p − s) )
s    = sign(ε_32 / ε_21)
```

with `ε_21 = f_2 − f_1`, `ε_32 = f_3 − f_2`, `r_ij = h_i/h_j`. Initialize the root-finding
with p_0 = expected_order (or p_0 = 1.0 if not defined).
Iterate to a relative tolerance of 1e-8, cap at 100 iterations,
and report non-convergence explicitly rather than returning a silently wrong number.

**Least-squares estimate** for ≥ 3 grids: fit `ln(e) = p·ln(h) + c`. Report `p`, its
standard error, and R². This is the headline number when enough grids exist; the
pairwise values go in the detail table so the user can see whether `p` is drifting
(a sign the coarse grids are outside the asymptotic range).

### 3.4 Grid Convergence Index (ASME V&V 20 / Roache)

```
GCI_fine = Fs · |ε_21| / (r_21^p − 1)        
with ε_21 normalized globally: (f_2 − f_1) / Q_scale.
Q_scale must be explicitly defined (e.g., max(f_1) - min(f_1) over the domain, or RMS of f_1) to avoid division by zero when the solution crosses zero.
```

Safety factor `Fs = 1.25` when `p` is computed from three or more grids;
`Fs = 3.0` when only two grids are available and `p` is assumed rather than measured.
Report which case applies — never silently use 1.25 with an assumed order.

**Asymptotic-range indicator:**

```
R = GCI_32 / ( r_21^p · GCI_21 )
```

`R ≈ 1` (report the deviation; flag if `|R − 1| > 0.1`) indicates the solutions lie in
the asymptotic range. Outside it, the reported `p` and GCI are not meaningful, and the
report must say so prominently rather than printing a confident number.

Also compute and report the *convergence condition* `R_c = ε_21 / ε_32`:
- `0 < R_c < 1` — monotonic convergence (the good case)
- `R_c < 0` — oscillatory convergence; report oscillation amplitude, and note that GCI
  is unreliable
- `|R_c| > 1` — divergence; the study has failed, say so loudly

### 3.5 Temporal order

Measured separately from spatial order. Fix a spatial grid fine enough that spatial
error is negligible relative to temporal error (require the user to state the fine
grid, and verify by checking that the finest-`Δt` error has not plateaued — a plateau
means spatial error dominates and the temporal study is invalid). Then refine `Δt`.
For a coupled space-time refinement (`Δt ∝ h`), the measured order is the minimum of
the two; document this and warn if the user's config appears to be doing it
accidentally.

### 3.6 Expected-order caveats that must appear in the report

- Limiters and shock-capturing schemes degrade formal order near extrema even for
  smooth solutions. A 2nd-order-TVD scheme commonly measures ~1.7 on a smooth
  manufactured solution with an extremum. The report must state the manufactured
  solution's smoothness and whether extrema lie inside the domain.
- Boundary-condition implementation order frequently limits global order. Provide an
  option to compute norms over the interior only, excluding `n` boundary cells, so the
  user can isolate interior scheme order from BC order. Report both.
- Round-off floor: at very fine grids the error stops falling and rises. Detect the
  minimum of the error curve and exclude post-minimum points from the fit, with a
  clear note in the report.

### 3.7 Conservation checks

For a conserved quantity `Q` with fluxes `F` and sources `S`:

```
imbalance = ( Q(t_end) − Q(t_0) − ∫ (F_in − F_out) dt − ∫ S dV dt ) / Q_scale
```

`Q_scale` is user-declared (default `max(|Q(t_0)|, |Q(t_end)|)`). Compare against a
tolerance that the user sets relative to machine epsilon and the number of time steps
(a reasonable default: `100 · N_steps · eps`, since round-off accumulates). Report
imbalance as a time series, not only a final scalar — a monotonically growing
imbalance is diagnostically different from a jump at one step.

---

## 4. Architecture

```
vvkit/
  mms/          symbolic MMS: operator DSL, source derivation, codegen
  norms/        error norms, cell-average quadrature, weighting
  convergence/  order estimation, GCI, asymptotic diagnostics
  runner/       solver adapters, case matrix expansion, process pool
  checks/       conservation, invariants, positivity
  regression/   baseline store, tolerance comparison, drift detection
  report/       Jinja2 templates, matplotlib figures, JSON/JUnit emitters
  config/       pydantic models, YAML loading, validation
  cli/          typer app
```

### 4.1 Dependency rules (enforce these; they are non-negotiable)

- `mms`, `norms`, `convergence`, `checks` are **pure**: no I/O, no subprocess, no
  filesystem. They take arrays and dataclasses, return dataclasses. This is what makes
  them unit-testable against closed-form answers.
- `runner` is the only module that touches subprocesses and the filesystem.
- `report` depends on everything but nothing depends on `report`.
- `config` depends on nothing.
- No module imports `cli`.

### 4.2 Solver adapters

An adapter implements:

```python
class SolverAdapter(Protocol):
    def run(self, case: CaseSpec, workdir: Path) -> SolverResult: ...
```

`CaseSpec` carries the refinement parameter, the MMS source (as generated code or a
callable), boundary data, and arbitrary user parameters. `SolverResult` carries the
solution field(s), coordinates, cell measures, wall time, exit status, and raw stdout
path.

Ship three adapters:

1. **`CallableAdapter`** — the solver is a Python function. Zero-friction path, used by
   all of the library's own tests.
2. **`CommandAdapter`** — the solver is an executable. The adapter renders an input
   file from a Jinja2 template, runs the command in an isolated working directory,
   and reads the output through a configurable reader. This is the adapter that must
   work well; it is how real users will adopt the tool.
3. **`PlynsimAdapter`** *(stretch, only if the human provides the interface)* — a
   worked example of a domain-specific adapter, used as documentation.

Output readers to support: HDF5 (h5py), CSV, NumPy `.npz`, plain-text columns, and a
user-supplied Python callable. Keep readers pluggable via entry points so third parties
can add VTK without `vvkit` depending on VTK.

### 4.3 Configuration format

A study is declared in `vvcase.yaml`. Full schema must be pydantic-validated with
helpful errors (point at the offending key, suggest the closest valid key on typos).

```yaml
version: 1
name: burgers_1d_muscl

solver:
  type: command
  command: ["./burgers", "{input_file}"]
  template: templates/burgers.in.j2
  reader:
    type: hdf5
    fields: {u: /solution/u, x: /grid/x, dx: /grid/dx}
  timeout_s: 600

mms:
  operator: |
    Eq(Derivative(u, t) + u*Derivative(u, x), nu*Derivative(u, x, 2))
  solution: "sin(2*pi*x) * exp(-t)"
  symbols: {nu: 0.01}
  domain: {x: [0.0, 1.0], t: [0.0, 0.5]}
  emit:
    language: c
    path: generated/mms_source.c

study:
  type: spatial
  refinement: {parameter: n_cells, values: [32, 64, 128, 256, 512]}
  reference: cell_average          # or point_value
  quadrature_order: 5
  norms: [L1, L2, Linf]
  exclude_boundary_cells: 2
  expected_order: 2.0
  order_tolerance: 0.2

checks:
  conservation:
    - quantity: mass
      field: u
      tolerance_mode: roundoff
      factor: 100

report:
  formats: [html, json, junit]
  output_dir: reports/
```

### 4.4 Result data model

Everything computed is serialized into one `StudyResult` JSON document. This document
is the contract: the HTML report is a rendering of it, the JUnit XML is a projection of
it, and regression baselines are diffs against it. Version this schema from day one
(`schema_version` field) and never break it without a migration.

---

## 5. Milestones

Each milestone ends with: green tests, updated docs, a tagged commit, and an updated
`docs/STATE.md`. Do not start M(n+1) before M(n) is tagged.

### M0 — Repository skeleton
- `uv`-managed project, `pyproject.toml`, `src/` layout, Apache-2.0 license.
- `ruff` (lint + format), `mypy --strict` on `src/`, `pytest` with coverage.
- GitHub Actions: lint, type-check, test on Python 3.12 and 3.13, Ubuntu + Windows.
- Pre-commit hooks.
- **Acceptance:** `uv run pytest`, `uv run ruff check`, `uv run mypy src` all pass on a
  trivial placeholder module; CI green on the first push.

### M1 — Norms and convergence core
- Norm implementations with cell-measure weighting.
- Cell-average reference via Gauss–Legendre quadrature (1D, 2D, 3D tensor product).
- Pairwise order, least-squares order with standard error and R², Roache iteration for
  non-constant `r`, GCI with correct `Fs` selection, `R` and `R_c` diagnostics,
  round-off floor detection.
- **Acceptance:** unit tests on *synthetic* error sequences with analytically known
  answers. Construct `e_k = C·h_k^p` for `p ∈ {1, 1.5, 2, 3, 4}` and assert recovery to
  1e-10. Test oscillatory and divergent sequences produce the correct classification,
  not an exception. Test the round-off floor detector on a sequence that turns upward.
  ≥ 95% branch coverage on `convergence/`.

### M2 — MMS engine
- Operator DSL over SymPy; source-term derivation; BC/IC extraction.
- Vanishing-term detection; scaling diagnostics; domain sampling for positivity.
- Preset solution families.
- Code emitters: Python (as a callable and as source), C, C++. 
Use sympy.codegen / ccode. (C# is excluded from direct code emission in v1.0 to avoid writing a custom SymPy AST visitor;
C# solvers will interface with emitted C code via P/Invoke or external DLLs). Emitted code must compile — test this, do not assume.
- **Acceptance:** for a small library of operators (1D advection, 1D advection–
  diffusion, Burgers, 2D Poisson, 1D Euler), the derived source term is verified
  symbolically: substitute `u_m` back into `L(u) − S` and assert SymPy simplifies it to
  zero. Emitted C and C++ are compiled with `cc`/`c++` in a test and evaluated against
  the SymPy-lambdified reference at 1000 random points to 1e-12. Skip the C# compile
  test gracefully when `dotnet` is absent, but run it in CI on the Ubuntu job.

### M3 — Runner and adapters
- `CallableAdapter`, `CommandAdapter`, case-matrix expansion, isolated per-case
  workdirs, process pool with configurable concurrency, timeouts, artifact capture.
- Readers: HDF5, CSV, npz, text columns, callable. Entry-point plugin mechanism.
- Deterministic workdir naming and a `--keep-artifacts` flag.
- **Acceptance:** an end-to-end test using a deliberately *2nd-order* toy FV solver for
  1D advection–diffusion written in the test suite, driven through `CallableAdapter`;
  the measured order must land in [1.9, 2.1]. A second toy solver, deliberately made
  1st-order by using upwind reconstruction, must measure in [0.9, 1.1]. A third, with
  an intentionally wrong flux, must *fail* the expected-order check — the harness must
  catch a broken solver, and there must be a test proving it does. Then repeat the
  2nd-order case through `CommandAdapter` with a real subprocess (a small C program
  compiled by the test) to prove the external path works.

### M4 — Checks and regression
- Conservation budget with time-series imbalance; user invariants; positivity checks.
- Baseline store (JSON under `baselines/`), tolerance comparison, `vv baseline update`
  with explicit confirmation, drift report.
- **Acceptance:** conservation test on a solver that conserves to round-off (passes)
  and one with a seeded 1e-6 leak (fails, and the report identifies the time step at
  which the imbalance departs).

### M5 — Reporting and CLI
- `vv run`, `vv mms generate`, `vv report`, `vv baseline update`, `vv init` (scaffold a
  `vvcase.yaml` interactively).
- HTML report: summary verdict, order table, log–log convergence plot with fitted
  slope and reference slopes, GCI table, asymptotic diagnostics, MMS diagnostics,
  conservation time series, environment provenance (solver git hash if available,
  `vvkit` version, platform, timestamp, full config echo).
- JSON and JUnit emitters. Non-zero exit code on failure so CI fails.
- **Acceptance:** golden-file test of the JSON output (with volatile fields masked);
  HTML validates as well-formed and renders offline with no CDN dependency.

### M6 — pytest plugin, docs, release
- `pytest-vvkit`: `@pytest.mark.convergence(case="...")` collecting a study as a test.
- Documentation: quickstart (< 10 minutes to a first converging study), the theory
  section from §3 rewritten for users with references, adapter-authoring guide, full
  config reference, and three worked examples.
- Publish to PyPI as a release candidate.
- **Acceptance:** a fresh clone, `uv sync`, and following the quickstart verbatim
  produces a correct report. Have the human do this run before tagging 1.0.0-rc1.

---

## 6. Quality bar

- `mypy --strict` clean on `src/`. No `Any` in public signatures, no `# type: ignore`
  without an inline justification comment.
- Public functions have docstrings citing the formula or reference they implement.
- Test coverage ≥ 90% overall, ≥ 95% on `convergence/` and `mms/`.
- **Numerical code is tested against closed-form answers, never against its own
  output.** A test that merely records what the code currently does is a regression
  test, not a correctness test; both are welcome but the former never substitutes for
  the latter.
- No network access at runtime. Ever. This tool runs inside air-gapped engineering
  environments.
- Determinism: identical config plus identical solver produces byte-identical JSON
  except for explicitly marked volatile fields (timestamps, wall times, paths).
- Errors are actionable. "Study failed" is unacceptable; "observed order 0.98 against
  expected 2.00 ± 0.20; convergence is monotonic and within asymptotic range, so the
  discretization — not the study setup — is the likely cause" is the standard.

---

## 7. Non-negotiable constraints

1. Never claim a result the mathematics does not support. If the asymptotic-range test
   fails, the order estimate is reported as unreliable, not as a number with a green
   checkmark. This tool's entire value is trustworthiness; a plausible-looking wrong
   answer is worse than a refusal.
2. No dependency additions beyond the approved list without an ADR:
   `numpy`, `scipy`, `sympy`, `pydantic`, `pyyaml`, `jinja2`, `matplotlib`, `typer`,
   `rich`, `h5py`. Dev: `pytest`, `pytest-cov`, `ruff`, `mypy`, `hypothesis`.
3. Code comments in English, concise, only where the code is not self-explanatory. No
   separator-style comment banners (no `# ---`, no `# ===`).
4. Do not vendor or copy code from GPL sources.

---

## 8. Reference reading

Cite these in docstrings where relevant. Do not fetch them at runtime.

- Roache, *Verification and Validation in Computational Science and Engineering*, 1998.
- Roache, "Code Verification by the Method of Manufactured Solutions", J. Fluids Eng.,
  2002.
- Salari & Knupp, "Code Verification by the Method of Manufactured Solutions",
  SAND2000-1444.
- Oberkampf & Roy, *Verification and Validation in Scientific Computing*, 2010.
- ASME V&V 20-2009, *Standard for Verification and Validation in Computational Fluid
  Dynamics and Heat Transfer*.
- Eça & Hoekstra, "A procedure for the estimation of the numerical uncertainty of CFD
  calculations based on grid refinement studies", JCP, 2014.
