"""Implementations of CallableAdapter and CommandAdapter."""

import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jinja2

from vvkit.runner.adapters import CaseSpec, SolverResult


class CallableAdapter:
    """Adapter for solvers supplied as Python callables."""

    def __init__(
        self,
        solver_func: Callable[[CaseSpec, Path], SolverResult],
    ) -> None:
        self.solver_func = solver_func

    def run(self, case: CaseSpec, workdir: Path) -> SolverResult:
        workdir.mkdir(parents=True, exist_ok=True)
        start_time = time.perf_counter()
        result = self.solver_func(case, workdir)
        result.wall_time = time.perf_counter() - start_time
        return result


class CommandAdapter:
    """Adapter for external executable solvers driven by input templates."""

    def __init__(
        self,
        command_template: list[str],
        input_template_path: Path | None = None,
        reader_func: Callable[[Path], dict[str, Any]] | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.command_template = command_template
        self.input_template_path = input_template_path
        self.reader_func = reader_func
        self.timeout_s = timeout_s

    def run(self, case: CaseSpec, workdir: Path) -> SolverResult:
        workdir.mkdir(parents=True, exist_ok=True)

        input_file_path = workdir / "solver.in"
        if self.input_template_path and self.input_template_path.exists():
            template_str = self.input_template_path.read_text(encoding="utf-8")
            rendered = jinja2.Template(template_str).render(
                case=case,
                refinement_value=case.refinement_value,
                params=case.user_params,
            )
            input_file_path.write_text(rendered, encoding="utf-8")

        cmd = [
            arg.format(input_file=str(input_file_path), workdir=str(workdir))
            for arg in self.command_template
        ]

        stdout_file = workdir / "stdout.log"
        stderr_file = workdir / "stderr.log"

        start_time = time.perf_counter()
        with (
            stdout_file.open("w", encoding="utf-8") as out,
            stderr_file.open("w", encoding="utf-8") as err,
        ):
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                stdout=out,
                stderr=err,
                timeout=self.timeout_s,
                check=False,
            )
        wall_time = time.perf_counter() - start_time

        fields: dict[str, Any] = {}
        coords: dict[str, Any] = {}
        measures = None

        if self.reader_func:
            read_data = self.reader_func(workdir)
            fields = read_data.get("fields", {})
            coords = read_data.get("coords", {})
            measures = read_data.get("cell_measures")

        return SolverResult(
            case_id=case.case_id,
            solution_fields=fields,
            coordinates=coords,
            cell_measures=measures,
            wall_time=wall_time,
            exit_status=proc.returncode,
            stdout_path=stdout_file,
            stderr_path=stderr_file,
        )


def create_adapter(config_solver: Any) -> CommandAdapter | CallableAdapter:
    """Factory to create an adapter from SolverConfig."""
    from vvkit.runner.readers import get_reader

    if config_solver.type == "callable":
        # Callable adapter is currently only for internal python scripts and tests
        raise ValueError("CallableAdapter cannot be initialized directly from config file yet.")

    reader_func = None
    if config_solver.reader:
        r_type = config_solver.reader.type
        r_file = config_solver.reader.file
        r_coords = config_solver.reader.coords
        r_fields = config_solver.reader.fields

        base_reader = get_reader(r_type)

        def wrapped_reader(workdir: Path) -> dict[str, Any]:
            file_path = workdir / r_file
            if r_type == "npz":
                return base_reader(file_path)
            elif r_type in ["csv", "txt"]:
                return base_reader(file_path, list(r_coords.values()), list(r_fields.values()))
            elif r_type == "hdf5":
                return base_reader(file_path, r_coords, r_fields)
            else:
                # Custom readers get the file path
                return base_reader(file_path)
        reader_func = wrapped_reader

    template_path = Path(config_solver.template) if config_solver.template else None

    return CommandAdapter(
        command_template=config_solver.command,
        input_template_path=template_path,
        reader_func=reader_func,
        timeout_s=config_solver.timeout_s,
    )
