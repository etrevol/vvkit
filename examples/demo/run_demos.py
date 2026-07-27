"""Demonstration suite for vvkit verification harness.

Demonstrates:
1. 2nd-order Finite Volume solver verification (Order ~ 2.0, Pass).
2. 1st-order Upwind solver verification (Order ~ 1.0, Catches order degradation).
3. Conservation budget check (Detects exact departure time step).
"""

import sys
from pathlib import Path

import numpy as np

from vvkit.checks import check_conservation
from vvkit.convergence import (
    compute_gci,
    compute_least_squares_order,
)
from vvkit.norms import compute_l2_norm
from vvkit.report import (
    StudyResultSummary,
    emit_html_report,
    emit_json_report,
    emit_junit_xml,
    generate_convergence_plot,
)
from vvkit.runner import CallableAdapter, CaseSpec, SolverResult

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


def solver_2nd_order_fv(case: CaseSpec, workdir: Path) -> SolverResult:
    """Correct 2nd-order Finite Volume discretization for 1D advection."""
    n_cells = int(case.refinement_value)
    dx = 1.0 / n_cells
    x_centers = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_cells)

    u_exact = np.sin(2 * np.pi * x_centers)
    u_num = u_exact + 0.5 * (dx**2) * np.sin(2 * np.pi * x_centers)

    res = SolverResult(
        case_id=case.case_id,
        solution_fields={"u": u_num},
        coordinates={"x": x_centers},
        cell_measures=np.full(n_cells, dx),
    )

    # Save artifact in case directory
    np.savez(
        workdir / "solution.npz",
        x=x_centers,
        u=u_num,
        cell_measures=res.cell_measures,
    )
    return res


def solver_1st_order_upwind(case: CaseSpec, workdir: Path) -> SolverResult:
    """1st-order Upwind discretization for 1D advection."""
    n_cells = int(case.refinement_value)
    dx = 1.0 / n_cells
    x_centers = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_cells)

    u_exact = np.sin(2 * np.pi * x_centers)
    u_num = u_exact + 0.8 * dx * np.cos(2 * np.pi * x_centers)

    res = SolverResult(
        case_id=case.case_id,
        solution_fields={"u": u_num},
        coordinates={"x": x_centers},
        cell_measures=np.full(n_cells, dx),
    )

    np.savez(
        workdir / "solution.npz",
        x=x_centers,
        u=u_num,
        cell_measures=res.cell_measures,
    )
    return res


def run_demo_suite(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("      vvkit — Verification Harness Demo Suite Execution")
    print("=" * 70)

    # --- Case 1: 2nd-Order FV Solver Sweep ---
    print("\n[Case 1] Running 2nd-Order Finite Volume Verification Study...")
    adapter_2nd = CallableAdapter(solver_2nd_order_fv)
    grids = [32, 64, 128, 256]
    h_vals = []
    errors_2nd = []
    solutions_2nd = []

    for n in grids:
        case = CaseSpec(
            case_id=f"2nd_fv_{n}",
            refinement_parameter="n_cells",
            refinement_value=n,
        )
        res = adapter_2nd.run(case, output_dir / case.case_id)
        dx = 1.0 / n
        u_exact = np.sin(2 * np.pi * res.coordinates["x"])
        err = compute_l2_norm(res.solution_fields["u"] - u_exact, res.cell_measures)
        h_vals.append(dx)
        errors_2nd.append(err)
        solutions_2nd.append(res.solution_fields["u"][0])

    fit_2nd = compute_least_squares_order(np.array(h_vals), np.array(errors_2nd))
    gci_2nd = compute_gci(
        solutions_2nd[-3],
        solutions_2nd[-2],
        solutions_2nd[-1],
        r21=2.0,
        r32=2.0,
        p=fit_2nd.order,
    )

    plot_path_2nd = output_dir / "reports" / "2nd_order_plot.png"
    generate_convergence_plot(
        np.array(h_vals),
        np.array(errors_2nd),
        fit_2nd.order,
        expected_slope=2.0,
        output_path=plot_path_2nd,
    )

    passed_2nd = bool(abs(fit_2nd.order - 2.0) <= 0.2)
    print(f"  -> Observed Order: {fit_2nd.order:.3f} (Expected: 2.000)")
    print(f"  -> R^2 Fit Quality: {fit_2nd.r_squared:.4f}")
    print(f"  -> GCI Fine Grid:  {gci_2nd.gci_fine:.3e}")
    asymp_str = f"{gci_2nd.asymptotic_ratio:.3f}" if gci_2nd.asymptotic_ratio else "N/A"
    print(f"  -> Asymptotic R:   {asymp_str} (Is Asymptotic: {bool(gci_2nd.is_asymptotic)})")
    print(f"  -> Plot saved:     {plot_path_2nd}")
    print(f"  -> Verdict:        {'PASSED' if passed_2nd else 'FAILED'}")

    asymp_val = float(gci_2nd.asymptotic_ratio) if gci_2nd.asymptotic_ratio is not None else None
    summary_2nd = StudyResultSummary(
        name="1d_advection_2nd_order_fv",
        observed_order=float(fit_2nd.order),
        expected_order=2.0,
        order_passed=passed_2nd,
        gci_fine=float(gci_2nd.gci_fine),
        asymptotic_ratio=asymp_val,
        is_asymptotic=bool(gci_2nd.is_asymptotic),
        convergence_state=str(gci_2nd.convergence_state.value),
        plot_image_path=plot_path_2nd,
    )
    emit_html_report(summary_2nd, output_dir / "reports" / "2nd_order_report.html")
    emit_json_report(summary_2nd, output_dir / "reports" / "2nd_order_report.json")
    emit_junit_xml(summary_2nd, output_dir / "reports" / "2nd_order_report.xml")

    # --- Case 2: 1st-Order Upwind Solver Sweep ---
    print("\n[Case 2] Running 1st-Order Upwind Verification Study...")
    adapter_1st = CallableAdapter(solver_1st_order_upwind)
    errors_1st = []
    for n in grids:
        case = CaseSpec(
            case_id=f"1st_upwind_{n}",
            refinement_parameter="n_cells",
            refinement_value=n,
        )
        res = adapter_1st.run(case, output_dir / case.case_id)
        dx = 1.0 / n
        u_exact = np.sin(2 * np.pi * res.coordinates["x"])
        err = compute_l2_norm(res.solution_fields["u"] - u_exact, res.cell_measures)
        errors_1st.append(err)

    fit_1st = compute_least_squares_order(np.array(h_vals), np.array(errors_1st))
    passed_1st_expected_2nd = bool(abs(fit_1st.order - 2.0) <= 0.2)
    print(f"  -> Observed Order: {fit_1st.order:.3f} (Claimed: 2.000, Actual: 1.000)")
    v_msg = "PASSED" if passed_1st_expected_2nd else "FAILED (Correctly caught order degradation!)"
    print(f"  -> Verdict:        {v_msg}")

    # --- Case 3: Discrete Conservation Check ---
    print("\n[Case 3] Running Conservation Budget Check (Leaking Mass Solver)...")
    q_series = np.full(100, 10.0, dtype=np.float64)
    q_series[42:] += np.linspace(0.0, 1e-3, 58)
    cons_res = check_conservation(q_series, factor=100.0)

    print(f"  -> Conserved:        {cons_res.is_conserved}")
    print(f"  -> Departure Step:   {cons_res.departure_step}")
    print(f"  -> Final Imbalance:  {cons_res.final_imbalance:.3e}")

    print("\n" + "=" * 70)
    print(f"Demo complete! Generated reports saved under: {output_dir / 'reports'}")
    print("=" * 70)


if __name__ == "__main__":
    demo_dir = Path(__file__).parent / "workdir"
    run_demo_suite(demo_dir)
