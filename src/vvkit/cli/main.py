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
from vvkit.norms import compute_l2_norm
from vvkit.report.emitters import (
    StudyResultSummary,
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
    errors = []
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
                    
            err = compute_l2_norm(u_num - u_exact, measures)

            h_vals.append(h_val)
            errors.append(err)
            solutions.append(u_num.flatten()[len(u_num.flatten())//2]) # Sample center point for GCI

        except Exception as e:
            console.print(f"[bold red]Failed during processing {case.case_id}: {e}[/]")
            raise typer.Exit(code=1) from e

    console.print("[cyan]Computing convergence metrics...[/cyan]")

    fit = compute_least_squares_order(np.array(h_vals), np.array(errors))
    gci = compute_gci(
        solutions[-3],
        solutions[-2],
        solutions[-1],
        r21=grids[-1]/grids[-2],
        r32=grids[-2]/grids[-3],
        p=fit.order,
    )

    passed = bool(abs(fit.order - config.study.expected_order) <= config.study.order_tolerance)

    report_dir = Path(config.report.output_dir)
    plot_path = report_dir / f"{config.name}_plot.png"

    generate_convergence_plot(
        np.array(h_vals),
        np.array(errors),
        fit.order,
        expected_slope=config.study.expected_order,
        output_path=plot_path,
    )

    asymp_val = float(gci.asymptotic_ratio) if gci.asymptotic_ratio is not None else None
    summary = StudyResultSummary(
        name=config.name,
        observed_order=float(fit.order),
        expected_order=config.study.expected_order,
        order_passed=passed,
        gci_fine=float(gci.gci_fine),
        asymptotic_ratio=asymp_val,
        is_asymptotic=bool(gci.is_asymptotic),
        convergence_state=str(gci.convergence_state.value),
        plot_image_path=plot_path,
    )

    if "html" in config.report.formats:
        emit_html_report(summary, report_dir / f"{config.name}.html")
    if "json" in config.report.formats:
        emit_json_report(summary, report_dir / f"{config.name}.json")
    if "junit" in config.report.formats:
        emit_junit_xml(summary, report_dir / f"{config.name}.xml")

    expected = config.study.expected_order
    console.print(f"  -> Observed Order: {fit.order:.3f} (Expected: {expected:.3f})")
    console.print(f"  -> R^2 Fit Quality: {fit.r_squared:.4f}")
    verdict_str = "[bold green]PASSED[/]" if passed else "[bold red]FAILED[/]"
    console.print(f"  -> Verdict: {verdict_str}")

    if not passed:
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
        from sympy.parsing.sympy_parser import parse_expr
        import re
        op_str_replaced = re.sub(r'u\([^)]+\)', f"({config.mms.solution})", op_str)
        op_expr = parse_expr(op_str_replaced)
        op_expr = op_expr.subs(config.mms.symbols).doit()

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
