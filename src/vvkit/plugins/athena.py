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

"""Athena++ specialized solver adapter plugin."""

import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import jinja2
import numpy as np

from vvkit.config import SolverConfig
from vvkit.runner.adapters import CaseSpec, SolverResult


class AthenaPlusPlusAdapter:
    """Specialized adapter for Athena++."""

    def __init__(self, config: SolverConfig) -> None:
        self.config = config
        self.executable = config.plugin_args.get("executable", "athena")
        self.use_wsl = config.plugin_args.get("use_wsl", False)
        self.wsl_distro = config.plugin_args.get("wsl_distro", "Ubuntu")
        self.timeout_s = config.timeout_s
        self.template_path = Path(config.template) if config.template else None

    def run(self, case: CaseSpec, workdir: Path) -> SolverResult:
        workdir = workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        input_file_path = workdir / "solver.in"
        if self.template_path and self.template_path.exists():
            template_str = self.template_path.read_text(encoding="utf-8")
            rendered = jinja2.Template(template_str).render(
                case=case,
                refinement_value=case.refinement_value,
                params=case.user_params,
            )
            input_file_path.write_text(rendered, encoding="utf-8")

        # Prepare the command
        athena_cmd = f"{self.executable} -i solver.in"
        
        if self.use_wsl:
            cmd = ["wsl", "-d", self.wsl_distro, "--", "bash", "-c", athena_cmd]
        else:
            cmd = shlex.split(athena_cmd)

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

        if proc.returncode == 0:
            # Find the latest generated .tab file
            tab_files = list(workdir.glob("*.tab"))
            if not tab_files:
                raise FileNotFoundError(f"No .tab files found in {workdir} after Athena++ execution.")
            
            # Sort by name (e.g. Sod.block0.out1.00000.tab, Sod.block0.out1.00001.tab)
            # The last one is the final state
            tab_files.sort(key=lambda x: x.name)
            final_tab = tab_files[-1]

            coords, fields = self._parse_athena_tab(final_tab)

        return SolverResult(
            case_id=case.case_id,
            solution_fields=fields,
            coordinates=coords,
            wall_time=wall_time,
            exit_status=proc.returncode,
            stdout_path=stdout_file,
            stderr_path=stderr_file,
        )

    def _parse_athena_tab(self, filepath: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        """Parse Athena++ .tab file and extract coordinates and fields."""
        lines = filepath.read_text(encoding="utf-8").strip().splitlines()
        
        header = ""
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("# i"):
                header = line
            elif not line.startswith("#"):
                data_start_idx = i
                break
                
        if header.startswith("#"):
            header = header[1:].strip()
            
        columns = header.split()
        
        # Determine coordinates vs fields
        coord_names = [col for col in columns if col in ["x1v", "x2v", "x3v"]]
        # Standard variables
        field_names = [col for col in columns if col not in coord_names and col not in ["i", "j", "k"]]
        
        # Read the numerical data
        data = np.loadtxt(lines[data_start_idx:])
        
        # If the file had 1 row (very unlikely), reshape it
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        coords: dict[str, Any] = {}
        fields: dict[str, Any] = {}
        
        coord_map = self.config.plugin_args.get("coords", {"x1v": "x", "x2v": "y", "x3v": "z"})
        field_map = self.config.plugin_args.get("fields", None)
        
        for idx, col in enumerate(columns):
            if col in coord_names:
                mapped_name = coord_map.get(col, col)
                coords[mapped_name] = data[:, idx]
            elif col in field_names:
                mapped_name = field_map.get(col, col) if field_map else col
                fields[mapped_name] = data[:, idx]
                
        return coords, fields
