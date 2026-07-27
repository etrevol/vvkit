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
    time.sleep(1.0)
    
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
This automated presentation will demonstrate how a computational scientist uses `vvkit` to verify a numerical PDE solver.

**The Scenario:**
We have an external, compiled PDE solver (simulated here by `mock_solver.py`). We claim it is 2nd-order accurate in space. We want to mathematically prove this using the **Method of Manufactured Solutions (MMS)** and calculate the **Grid Convergence Index (GCI)** per ASME V&V 20.
    """
    console.print(Markdown(md))
    if "--auto" not in sys.argv:
        input("\nPress ENTER to continue to Step 1 (Configuration)...")
    else:
        time.sleep(2.0)


def step_config():
    console.clear()
    console.print("[bold magenta]--- Step 1: Declarative Configuration ---[/]")
    console.print("Instead of writing glue code, we declare the entire verification study in a `vvcase.yaml` file.")
    
    yaml_path = DEMO_DIR / "burgers_case.yaml"
    with yaml_path.open() as f:
        code = f.read()
    
    syntax = Syntax(code, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="burgers_case.yaml"))
    
    if "--auto" not in sys.argv:
        input("\nPress ENTER to continue to Step 2 (MMS Generation)...")
    else:
        time.sleep(2.0)


def step_mms():
    console.clear()
    console.print("[bold magenta]--- Step 2: Symbolic MMS Generation ---[/]")
    console.print("`vvkit` parses the mathematical operator from the YAML and automatically derives the analytical source terms required to force the solver to the exact solution.")
    
    run_command(
        [sys.executable, "-m", "vvkit.cli.main", "mms", "--config-path", "burgers_case.yaml", "--language", "c", "--output", "mms_source.c"],
        "Generating C++ Source Term"
    )
    
    c_path = DEMO_DIR / "mms_source.c"
    if c_path.exists():
        syntax = Syntax(c_path.read_text(), "c", theme="monokai")
        console.print(Panel(syntax, title="Generated Source (mms_source.c)"))
    
    if "--auto" not in sys.argv:
        input("\nPress ENTER to continue to Step 3 (Execution & Analysis)...")
    else:
        time.sleep(2.0)


def step_run():
    console.clear()
    console.print("[bold magenta]--- Step 3: Automated Execution & Analysis ---[/]")
    console.print("`vvkit` will now template the input files, orchestrate the solver executable across the grid refinement matrix, read the HDF5/NPZ arrays, compute L2 norms, evaluate Least-Squares fits, and generate GCI metrics.")
    
    run_command(
        [sys.executable, "-m", "vvkit.cli.main", "run", "--config-path", "burgers_case.yaml", "--workdir-base", "workdir"],
        "Running the Verification Sweep"
    )
    
    if "--auto" not in sys.argv:
        input("\nPress ENTER to continue to Step 4 (Reporting)...")
    else:
        time.sleep(2.0)


def step_report():
    console.clear()
    console.print("[bold magenta]--- Step 4: CI-Ready Reporting ---[/]")
    console.print("The pipeline has successfully proven 2nd-order convergence and emitted continuous integration contracts.")
    
    report_dir = DEMO_DIR / "reports"
    console.print("\nGenerated Artifacts:")
    for f in report_dir.iterdir():
        if f.is_file():
            console.print(f"  - [cyan]{f.name}[/]")
            
    md = """
### What's Next?
- Open `reports/advection_1d_demo.html` in your browser to view the convergence plots.
- Integrate `advection_1d_demo.xml` into your Jenkins/GitLab CI pipeline.
- Track baseline drifts over time using `vv baseline update`.
    """
    console.print(Markdown(md))
    console.print("\n[bold green]Pitch Demo Complete! Thank you.[/]")


def main():
    if not (DEMO_DIR / "burgers_case.yaml").exists():
        console.print("[bold red]Run this script from within its directory, or ensure the demo files exist.[/]")
        sys.exit(1)
        
    step_intro()
    step_config()
    step_mms()
    step_run()
    step_report()


if __name__ == "__main__":
    main()
