"""Interactive Pitch Presentation for vvkit.

This script acts as a live, automated demonstration of vvkit's capabilities.
It uses `rich` to render a beautiful console presentation and drives the `vv` CLI
to showcase configuration parsing, MMS code generation, external solver orchestration,
and reporting.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import track
from rich.syntax import Syntax
from rich.text import Text

console = Console()
DEMO_DIR = Path(__file__).parent


def run_command(args: list[str], description: str) -> None:
    """Run a command and handle errors, showing a nice spinner (simulated)."""
    console.print(f"\n[bold cyan]> Executing:[/] [yellow]{' '.join(args)}[/]")
    time.sleep(0.5)
    
    # We set PYTHONPATH to ensure it picks up the local vvkit source
    env = os.environ.copy()
    env["PYTHONPATH"] = str(DEMO_DIR.parent.parent / "src")
    
    result = subprocess.run(
        args,
        cwd=DEMO_DIR,
        capture_output=True,
        text=True,
        env=env,
    )
    
    if result.returncode != 0:
        console.print(f"[bold red][FAIL] Failed: {description}[/]")
        console.print(result.stdout)
        console.print(result.stderr)
        sys.exit(1)
    else:
        console.print(f"[bold green][OK] Success:[/] {description}")
        console.print(Panel(result.stdout.strip(), border_style="green", title="Output"))


def step_intro():
    console.clear()
    title = Text("vvkit: The Verification Harness", style="bold magenta", justify="center")
    console.print(Panel(title, expand=False, padding=(1, 4)))
    
    md = """
Welcome to the **vvkit Pitch Demo**.
This automated presentation will demonstrate how a computational scientist uses `vvkit` to verify a numerical PDE solver across various equations and dimensions.

**The Scenario:**
We have an external, compiled PDE solver (simulated here by `mock_solver.py`). We want to mathematically prove its accuracy across different modules using the **Method of Manufactured Solutions (MMS)** and calculate the **Grid Convergence Index (GCI)**.
    """
    console.print(Markdown(md))
    if "--auto" not in sys.argv:
        input("\nPress ENTER to start the demo suite...")
    else:
        time.sleep(1.0)


def step_config(yaml_path: Path):
    console.clear()
    console.print(f"[bold magenta]--- Phase: Declarative Configuration ({yaml_path.name}) ---[/]")
    console.print("Instead of writing glue code, we declare the entire verification study in a YAML file.")
    
    with yaml_path.open() as f:
        code = f.read()
    
    syntax = Syntax(code, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=yaml_path.name))
    
    if "--auto" not in sys.argv:
        input("\nPress ENTER to continue to MMS Generation...")
    else:
        time.sleep(1.0)


def step_mms(yaml_path: Path):
    console.clear()
    console.print(f"[bold magenta]--- Phase: Symbolic MMS Generation ({yaml_path.name}) ---[/]")
    console.print("`vvkit` parses the mathematical operator from the YAML and automatically derives the analytical source terms required to force the solver to the exact solution.")
    
    run_command(
        [sys.executable, "-m", "vvkit.cli.main", "mms", "--config-path", f"cases/{yaml_path.name}", "--language", "c", "--output", "mms_source.c"],
        "Generating C++ Source Term"
    )
    
    c_path = DEMO_DIR / "mms_source.c"
    if c_path.exists():
        syntax = Syntax(c_path.read_text(), "c", theme="monokai")
        console.print(Panel(syntax, title="Generated Source (mms_source.c)"))
    
    if "--auto" not in sys.argv:
        input("\nPress ENTER to continue to Execution & Analysis...")
    else:
        time.sleep(1.0)


def step_run(yaml_path: Path):
    console.clear()
    console.print(f"[bold magenta]--- Phase: Automated Execution & Analysis ({yaml_path.name}) ---[/]")
    console.print("`vvkit` will now template the input files, orchestrate the solver executable across the grid refinement matrix, read the HDF5/NPZ arrays, compute L2 norms, evaluate Least-Squares fits, and generate GCI metrics.")
    
    run_command(
        [sys.executable, "-m", "vvkit.cli.main", "run", "--config-path", f"cases/{yaml_path.name}", "--workdir-base", f"workdir_{yaml_path.stem}"],
        f"Running the Verification Sweep for {yaml_path.stem}"
    )
    
    if "--auto" not in sys.argv:
        input("\nPress ENTER to continue...")
    else:
        time.sleep(1.0)


def step_report():
    console.clear()
    console.print("[bold magenta]--- Final Phase: CI-Ready Reporting ---[/]")
    console.print("The pipeline has successfully proven convergence across multiple suites and emitted continuous integration contracts.")
    
    report_dir = DEMO_DIR / "reports"
    console.print("\nGenerated Artifacts:")
    for f in report_dir.iterdir():
        if f.is_file():
            console.print(f"  - [cyan]{f.name}[/]")
            
    md = """
### What's Next?
- Open the `.html` reports in your browser to view the convergence plots.
- Integrate the `.xml` reports into your Jenkins/GitLab CI pipeline.
- Track baseline drifts over time using `vv baseline update`.
    """
    console.print(Markdown(md))
    console.print("\n[bold green]Pitch Demo Complete! Thank you.[/]")


def main():
    cases = [
        DEMO_DIR / "cases" / "advection_1d.yaml",
        DEMO_DIR / "cases" / "diffusion_1d_temporal.yaml",
        DEMO_DIR / "cases" / "poisson_2d.yaml",
    ]
    for case in cases:
        if not case.exists():
            console.print(f"[bold red]Cannot find {case}. Run this script from within the demo directory.[/]")
            sys.exit(1)
            
    step_intro()
    for case_file in cases:
        step_config(case_file)
        step_mms(case_file)
        step_run(case_file)
    step_report()


if __name__ == "__main__":
    main()
