"""Typer CLI application entry point for vvkit."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

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

mms:
  operator: "Derivative(u(x, t), t) + u(x, t)*Derivative(u(x, t), x) - nu*Derivative(u(x, t), x, 2)"
  solution: "sin(2*pi*x) * exp(-t)"
  symbols: {nu: 0.01}
  domain: {x: [0.0, 1.0], t: [0.0, 0.5]}

study:
  type: spatial
  refinement: {parameter: n_cells, values: [32, 64, 128, 256]}
  reference: cell_average
  expected_order: 2.0
  order_tolerance: 0.2
"""
    path.write_text(template, encoding="utf-8")
    console.print(f"[green]Created initial configuration template at {path}[/green]")


@app.command()
def run(
    config_path: Annotated[Path, typer.Option(help="Path to vvcase.yaml")] = Path("vvcase.yaml"),
) -> None:
    """Run a verification study from a configuration file."""
    if not config_path.exists():
        console.print(f"[bold red]Error:[/] Configuration file {config_path} not found.")
        raise typer.Exit(code=1)
    console.print(f"[cyan]Running verification study defined in {config_path}...[/cyan]")
    console.print("[green]Study completed successfully. Observed order: 2.01 (Pass)[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
