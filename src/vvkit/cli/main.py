# Copyright 2026 Artem Holovashchenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typer CLI application entry point for vvkit."""

import shutil
from pathlib import Path
from typing import Annotated

import numpy as np
import sympy as sp
import typer
from rich.console import Console

from vvkit.config import load_config
from vvkit.convergence import compute_gci, compute_least_squares_order
from vvkit.mms.emitters import emit_c_source, emit_cpp_source, emit_python_source
from vvkit.norms.norms import compute_l1_norm, compute_l2_norm, compute_linf_norm
from vvkit.report.emitters import (
    StudyResultSummary,
    NormResultSummary,
    emit_html_report,
    emit_json_report,
    emit_junit_xml,
)
from vvkit.report.plots import generate_convergence_plot
from vvkit.runner.adapters import CaseSpec
from vvkit.runner.matrix import create_adapter

app = typer.Typer(
    name="vv",
    help="vvkit — Verification Harness for Numerical Solvers",
    add_completion=False,
)
console = Console()


@app.command()
def init(
    path: Annotated[Path, typer.Option(help="Path to output yaml")] = Path("vvcase.yaml"),
) -> None:
    """Scaffold an initial vvcase.yaml configuration file."""
    template = """version: 1
name: burgers_1d_example

solver:
  type: command
  command: ["./burgers", "{input_file}"]
  template: templates/burgers.in.j2
  reader:
    type: npz
    file: solution.npz
    coords: {x: x}
    fields: {u: u}

mms:
  operator: "Derivative(u(x, t), t) + u(x, t)*Derivative(u(x, t), x) - nu*Derivative(u(x, t), x, 2)"
  solution: "sin(2*pi*x) * exp(-t)"
  symbols: {nu: 0.01}
  domain: {x: [0.0, 1.0], t: [0.0, 0.5]}

study:
  type: spatial
  refinement:
    parameter: n_cells
    values: [32, 64, 128, 256]
  reference: cell_average
  expected_order: 2.0
  order_tolerance: 0.2

report:
  formats: [html, json, junit]
  output_dir: reports/
"""
    path.write_text(template, encoding="utf-8")
    console.print(f"[green]Created initial configuration template at {path}[/green]")


@app.command()
def run(
    config_path: Annotated[Path, typer.Option(help="Path to vvcase.yaml")] = Path("vvcase.yaml"),
    workdir_base: Annotated[Path, typer.Option(help="Base working directory")] = Path("workdir"),
) -> None:
    """Run a verification study from a configuration file."""
    if not config_path.exists():
        console.print(f"[bold red]Error:[/] Configuration file {config_path} not found.")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Loading configuration from {config_path}...[/cyan]")
    config = load_config(config_path)

    # 1. Parse MMS exact solution for error computation
    u_sym = sp.sympify(config.mms.solution)
    u_sym = u_sym.subs(config.mms.symbols)

    # 2. Setup adapter
    adapter = create_adapter(config.solver)

    grids = config.study.refinement.values
    h_vals = []
    errors_by_norm = {n: [] for n in config.study.norms}
    solutions = []

    console.print(f"[cyan]Executing {len(grids)} cases...[/cyan]")

    for n in grids:
        case = CaseSpec(
            case_id=f"case_{n}",
            refinement_parameter=config.study.refinement.parameter,
            refinement_value=n,
            user_params=config.study.user_params,
        )
        case_workdir = workdir_base / case.case_id
        console.print(f"  -> Running {case.case_id} in {case_workdir}")

        try:
            res = adapter.run(case, case_workdir)
            if res.exit_status != 0:
                console.print(f"[bold red]Solver failed for {case.case_id} (exit code {res.exit_status})[/]")
                raise typer.Exit(code=1)

            if config.solver.reader and config.solver.reader.fields:
                field_names = list(config.solver.reader.fields.keys())
            else:
                field_names = ["u"]
                
            if config.solver.reader and config.solver.reader.coords:
                coord_names = list(config.solver.reader.coords.keys())
            else:
                coord_names = ["x"]

            # Calculate h_val (mesh size or timestep)
            if config.study.type == "temporal":
                domain_val = config.mms.domain.get("t", [0.0, 1.0])
            else:
                domain_val = config.mms.domain.get(coord_names[0], [0.0, 1.0])
            h_val = (domain_val[1] - domain_val[0]) / n

            u_num = res.solution_fields[field_names[0]]
            
            # Evaluate exact solution
            u_sym_eval = u_sym
            for fs in u_sym.free_symbols:
                fs_name = str(fs)
                if fs_name not in coord_names and fs_name in config.mms.domain:
                    u_sym_eval = u_sym_eval.subs(fs, config.mms.domain[fs_name][1])
            
            eval_symbols = [sp.Symbol(c) for c in coord_names]
            u_exact_func = sp.lambdify(eval_symbols, u_sym_eval, modules="numpy")
            
            if config.study.reference == "cell_average":
                from vvkit.norms.quadrature import cell_average_nd
                bounds_list = []
                for c in coord_names:
                    c_vals = res.coordinates[c]
                    c_vals_flat = c_vals.ravel()
                    domain_val = config.mms.domain.get(c, [0.0, 1.0])
                    h_c = (domain_val[1] - domain_val[0]) / n
                    c_bounds = np.zeros((len(c_vals_flat), 2), dtype=np.float64)
                    c_bounds[:, 0] = c_vals_flat - h_c / 2.0
                    c_bounds[:, 1] = c_vals_flat + h_c / 2.0
                    bounds_list.append(c_bounds)
                
                u_exact = cell_average_nd(u_exact_func, bounds_list, order=config.study.quadrature_order)
                u_exact = u_exact.reshape(u_num.shape)
                if np.isscalar(u_exact):
                    u_exact = np.full_like(u_num, u_exact)
            else:
                kwargs = {c: res.coordinates[c] for c in coord_names}
                u_exact = u_exact_func(**kwargs)
                if np.isscalar(u_exact):
                    u_exact = np.full_like(u_num, u_exact)

            if res.cell_measures is not None:
                measures = res.cell_measures
            else:
                # If spatial 2D, dx*dy, if 1D, dx
                if config.study.type == "spatial" and len(coord_names) > 1:
                    measures = np.full_like(u_num, h_val**len(coord_names))
                else:
                    measures = np.full_like(u_num, h_val)
            if config.study.exclude_boundary_cells > 0:
                n_ex = config.study.exclude_boundary_cells
                if len(coord_names) == 1:
                    u_num = u_num[n_ex:-n_ex]
                    u_exact = u_exact[n_ex:-n_ex]
                    measures = measures[n_ex:-n_ex]
                else:
                    slices = tuple(slice(n_ex, -n_ex) for _ in range(len(coord_names)))
                    u_num = u_num[slices]
                    u_exact = u_exact[slices]
                    measures = measures[slices]

            err_diff = u_num - u_exact
            for norm_name in config.study.norms:
                if norm_name == "L1":
                    err = compute_l1_norm(err_diff, measures)
                elif norm_name == "L2":
                    err = compute_l2_norm(err_diff, measures)
                elif norm_name == "Linf":
                    err = compute_linf_norm(err_diff)
                else:
                    console.print(f"[bold yellow]Warning: unknown norm {norm_name}[/]")
                    continue
                errors_by_norm[norm_name].append(err)

            if config.checks.conservation:
                from vvkit.checks.conservation import check_conservation
                for cons_cfg in config.checks.conservation:
                    if cons_cfg.field in res.solution_fields:
                        field_data = res.solution_fields[cons_cfg.field]
                        q_final = float(np.sum(field_data * measures))
                        
                        # Use the exact solution as the reference initial state mass
                        q_initial = float(np.sum(u_exact * measures))
                        
                        cons_res = check_conservation(np.array([q_initial, q_final]), factor=cons_cfg.factor)
                        if not cons_res.is_conserved:
                            console.print(f"[bold yellow]Conservation check failed for {cons_cfg.quantity} on {case.case_id} (imbalance: {cons_res.final_imbalance:.2e})[/]")

            h_vals.append(h_val)
            solutions.append(u_num.flatten()[len(u_num.flatten())//2]) # Sample center point for GCI


        except Exception as e:
            console.print(f"[bold red]Failed during processing {case.case_id}: {e}[/]")
            raise typer.Exit(code=1) from e

    console.print("[cyan]Computing convergence metrics...[/cyan]")

    from vvkit.convergence.diagnostics import detect_roundoff_floor
    norm_summaries = []
    fitted_slopes = {}
    excluded_idxs = {}

    for norm_name in config.study.norms:
        errors_arr = np.array(errors_by_norm[norm_name])
        if len(errors_arr) == 0:
            continue
            
        min_idx = detect_roundoff_floor(errors_arr)
        
        if min_idx < len(errors_arr) - 1:
            excluded_count = len(errors_arr) - min_idx - 1
            if norm_name == config.study.norms[0]:
                console.print(f"[bold yellow]Warning ({norm_name}): Round-off floor detected at grid {grids[min_idx]}. Excluding {excluded_count} finer grids from metrics.[/bold yellow]")
            h_vals_fit = h_vals[:min_idx + 1]
            errors_fit = errors_by_norm[norm_name][:min_idx + 1]
            sols_fit = solutions[:min_idx + 1]
            grids_fit = grids[:min_idx + 1]
            excluded_idxs[norm_name] = min_idx
        else:
            h_vals_fit = h_vals
            errors_fit = errors_by_norm[norm_name]
            sols_fit = solutions
            grids_fit = grids
            excluded_idxs[norm_name] = None

        fit = compute_least_squares_order(np.array(h_vals_fit), np.array(errors_fit))
        fitted_slopes[norm_name] = fit.order
        
        if len(sols_fit) >= 3:
            gci = compute_gci(
                f1=sols_fit[-1],
                f2=sols_fit[-2],
                f3=sols_fit[-3],
                r21=grids_fit[-1]/grids_fit[-2],
                r32=grids_fit[-2]/grids_fit[-3],
                p=fit.order,
            )
        elif len(sols_fit) == 2:
            gci = compute_gci(
                f1=sols_fit[-1],
                f2=sols_fit[-2],
                f3=None,
                r21=grids_fit[-1]/grids_fit[-2],
                p=fit.order,
            )
        else:
            console.print("[bold red]Not enough points for GCI.[/bold red]")
            raise typer.Exit(code=1)

        passed = bool(abs(fit.order - config.study.expected_order) <= config.study.order_tolerance)
        asymp_val = float(gci.asymptotic_ratio) if gci.asymptotic_ratio is not None else None
        
        norm_summaries.append(NormResultSummary(
            norm_name=norm_name,
            observed_order=float(fit.order),
            expected_order=config.study.expected_order,
            std_err=float(fit.std_err) if fit.std_err is not None else None,
            r_squared=float(fit.r_squared) if fit.r_squared is not None else None,
            order_passed=passed,
            gci_fine=float(gci.gci_fine),
            asymptotic_ratio=asymp_val,
            is_asymptotic=bool(gci.is_asymptotic),
            convergence_state=str(gci.convergence_state.value),
        ))
        
        expected = config.study.expected_order
        console.print(f"  [{norm_name}] -> Observed Order: {fit.order:.3f} (Expected: {expected:.3f}), R^2: {fit.r_squared:.4f}")

    all_passed = all(ns.order_passed for ns in norm_summaries)
    verdict_str = "[bold green]PASSED[/]" if all_passed else "[bold red]FAILED[/]"
    console.print(f"  -> Overall Verdict: {verdict_str}")

    report_dir = Path(config.report.output_dir)
    plot_path = report_dir / f"{config.name}_plot.png"

    generate_convergence_plot(
        np.array(h_vals),
        errors_by_norm=errors_by_norm,
        fitted_slopes=fitted_slopes,
        expected_slope=config.study.expected_order,
        output_path=plot_path,
        excluded_idxs=excluded_idxs
    )

    summary = StudyResultSummary(
        name=config.name,
        norms=norm_summaries,
        plot_image_path=plot_path,
    )

    if "html" in config.report.formats:
        emit_html_report(summary, report_dir / f"{config.name}.html")
    if "json" in config.report.formats:
        emit_json_report(summary, report_dir / f"{config.name}.json")
    if "junit" in config.report.formats:
        emit_junit_xml(summary, report_dir / f"{config.name}.xml")

    baseline_path = Path("baselines") / f"{config.name}_baseline.json"
    if baseline_path.exists():
        console.print("\n[cyan]Checking regression baseline...[/cyan]")
        try:
            with baseline_path.open("r", encoding="utf-8") as bf:
                baseline_data = json.load(bf)
            b_summary = baseline_data.get("summary", {})
            b_norms = {n["norm_name"]: n for n in b_summary.get("norms", [])}
            
            for ns in norm_summaries:
                b_ns = b_norms.get(ns.norm_name)
                if b_ns:
                    drift_p = ns.observed_order - b_ns.get("observed_order", 0.0)
                    drift_gci = ns.gci_fine - b_ns.get("gci_fine", 0.0)
                    console.print(f"  [{ns.norm_name}] Baseline Drift: Δp = {drift_p:+.3f}, ΔGCI = {drift_gci:+.2e}")
                else:
                    console.print(f"  [{ns.norm_name}] No baseline data available.")
        except Exception as e:
            console.print(f"[bold yellow]Failed to read baseline for drift report: {e}[/]")

    if not all_passed:
        raise typer.Exit(code=1)


@app.command(name="mms")
def generate_mms(
    config_path: Annotated[Path, typer.Option(help="Path to vvcase.yaml")] = Path("vvcase.yaml"),
    language: Annotated[str, typer.Option(help="Target language (c, cpp, python)")] = "c",
    output: Annotated[Path, typer.Option(help="Output source file")] = Path("mms_source.c"),
) -> None:
    """Generate MMS source terms from a configuration file."""
    if not config_path.exists():
        console.print(f"[bold red]Error:[/] Configuration file {config_path} not found.")
        raise typer.Exit(code=1)

    config = load_config(config_path)

    # Simple derivation of source term S = L(u_m) using sympy
    # For full MMS, substitute u(x,t) with solution.
    # In vvkit mms, we sympify operator with u substituted.
    u_sym = sp.sympify(config.mms.solution)
    u_sym = u_sym.subs(config.mms.symbols)

    # Convert 'Derivative(u(x, t), t)' -> differentiate u_sym
    # For a full implementation, this parses the DSL. Here we assume the user wrote the operator
    # directly as a sympy expression with u(x,t) which we substitute.
    op_str = config.mms.operator
    # Very basic evaluation for demo purposes:
    # A complete MMS engine would parse the DSL.
    try:
        from vvkit.mms.dsl import parse_mms_problem
        prob = parse_mms_problem(
            config.mms.operator,
            config.mms.solution,
            symbols_dict=config.mms.symbols,
            domain_dict=config.mms.domain,
        )

        if prob.vanished_terms:
            console.print(f"[bold yellow]Warning: The following operator terms vanish with the chosen manufactured solution: {prob.vanished_terms}[/bold yellow]")

        from vvkit.mms.preset import check_domain_positivity
        
        all_syms = prob.variables.copy()
        if prob.time_var and prob.time_var not in all_syms:
            all_syms.append(prob.time_var)
            
        domain_bounds = {}
        for s in all_syms:
            if str(s) in config.mms.domain:
                domain_bounds[str(s)] = tuple(config.mms.domain[str(s)])
                
        is_positive = check_domain_positivity(
            prob.manufactured_sol,
            symbols=all_syms,
            domain=domain_bounds,
        )
        if not is_positive:
            console.print("[bold yellow]Warning: The manufactured solution is not strictly positive over the domain. Physical solvers may crash![/bold yellow]")

        op_expr = prob.source_term

        if language == "c":
            code = emit_c_source(op_expr)
        elif language == "cpp":
            code = emit_cpp_source(op_expr)
        else:
            code = emit_python_source(op_expr)

        output.write_text(code, encoding="utf-8")
        console.print(f"[green]MMS source term successfully generated in {output}[/green]")

    except Exception as e:
        console.print(f"[bold red]Failed to generate MMS source:[/] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def report(
    json_path: Annotated[Path, typer.Argument(help="Path to saved study JSON result")],
    output_dir: Annotated[Path, typer.Option(help="Directory to output reports")] = Path("reports"),
) -> None:
    """Regenerate reports from a saved JSON contract."""
    import json
    if not json_path.exists():
        console.print(f"[bold red]Error:[/] JSON file {json_path} not found.")
        raise typer.Exit(code=1)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    summary_data = data["summary"]
    # We must deserialize the nested norms objects explicitly or let the class do it.
    if "norms" in summary_data:
        summary_data["norms"] = [NormResultSummary(**ns) for ns in summary_data["norms"]]
        
    # Handle plot path string
    if summary_data.get("plot_image_path"):
        summary_data["plot_image_path"] = Path(summary_data["plot_image_path"])

    summary = StudyResultSummary(**summary_data)

    emit_html_report(summary, output_dir / f"{summary.name}.html")
    emit_junit_xml(summary, output_dir / f"{summary.name}.xml")
    console.print(f"[green]Reports successfully regenerated in {output_dir}/[/green]")


@app.command()
def baseline_update(
    json_path: Annotated[Path, typer.Argument(help="Path to current study JSON result")],
) -> None:
    """Update regression baselines with the current study result."""
    import json
    if not json_path.exists():
        console.print(f"[bold red]Error:[/] JSON file {json_path} not found.")
        raise typer.Exit(code=1)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    name = data["summary"]["name"]

    baselines_dir = Path("baselines")
    baselines_dir.mkdir(exist_ok=True)
    baseline_path = baselines_dir / f"{name}_baseline.json"

    if baseline_path.exists():
        typer.confirm(f"Baseline {baseline_path} already exists. Overwrite?", abort=True)

    shutil.copy(json_path, baseline_path)
    console.print(f"[green]Baseline updated at {baseline_path}[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
