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
    
    adapter_name = "Athena++"
    adapter_description = "Fully automated adapter for the Athena++ GRMHD AMR code (https://www.athena-astro.app/index.html)."
    
    adapter_help = """\
# Athena++ Plugin

This plugin provides a fully automated compilation and execution environment for the Athena++ astrophysical GRMHD AMR code.

## Plugin Arguments (`plugin_args`)
When configuring your `vvcase.yaml`, you can pass the following arguments under `solver.plugin_args`:

- `athena_source` *(string, optional)*: Path to the Athena++ source code. If provided, the plugin will compile Athena++ on-the-fly and cache the binary.
- `configure_args` *(list of strings, optional)*: List of arguments to pass to `configure.py` when compiling (e.g., `["--prob=shock_tube", "--coord=cylindrical", "--nscalars=1"]`).
- `executable` *(string, optional)*: Path to a pre-compiled Athena++ executable. Ignored if `athena_source` is provided. Default is `"athena"`.
- `output_stream` *(string, optional)*: The output stream block identifier to parse for fields (e.g., `"out1"`, `"out2"`). Default is `"out1"`.
- `use_wsl` *(boolean, optional)*: If `true`, compilation and execution are routed through Windows Subsystem for Linux natively.
- `wsl_distro` *(string, optional)*: The WSL distribution to use. Default is `"Ubuntu"`.

## Features
- Intelligently parses the output `.tab` files directly, automatically extracting coordinate bounds and field names. No need for a `reader` block in your `vvcase.yaml`!
- Caches compiled binaries using an MD5 hash of `configure_args` to avoid unnecessary rebuilds across multiple studies.
"""
    
    @classmethod
    def get_init_template(cls) -> str:
        return """version: 1
name: athena_shock_tube_example

solver:
  type: athena++
  template: athinput.sod.template
  plugin_args:
    athena_source: "/home/user/athena-collab"
    configure_args: ["--prob=shock_tube"]
    use_wsl: true
    wsl_distro: Ubuntu

mms:
  operator: "0"  # Shock tube doesn't strictly use MMS source terms
  solution: "0"
  symbols: {}
  domain: {x1: [-0.5, 0.5]}

study:
  type: spatial
  refinement:
    parameter: nx1
    values: [32, 64, 128, 256]
  reference: cell_average
  expected_order: 1.0
  order_tolerance: 0.2

report:
  formats: [html, json, junit]
  output_dir: reports/
"""

    def __init__(self, config: SolverConfig) -> None:
        self.config = config
        self.use_wsl = config.plugin_args.get("use_wsl", False)
        self.wsl_distro = config.plugin_args.get("wsl_distro", "Ubuntu")
        self.timeout_s = config.timeout_s
        self.template_path = Path(config.template) if config.template else None

        self.athena_source = config.plugin_args.get("athena_source")
        self.configure_args = config.plugin_args.get("configure_args", [])
        
        if self.athena_source:
            self.executable = self._build_athena()
        else:
            self.executable = config.plugin_args.get("executable", "athena")

    def _build_athena(self) -> str:
        """Build Athena++ if source is provided and return the path to the executable."""
        import hashlib
        
        src_path = self.athena_source
        
        # Create a unique binary name based on the configure args
        args_str = " ".join(self.configure_args)
        args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
        binary_name = f"athena_{args_hash}"
        
        # If binary already exists in the source bin/, use it
        if self.use_wsl:
            check_cmd = ["wsl", "-d", self.wsl_distro, "--", "bash", "-c", f"test -f {src_path}/bin/{binary_name}"]
            exists = subprocess.run(check_cmd, check=False).returncode == 0
        else:
            bin_path = Path(src_path).expanduser() / "bin" / binary_name
            exists = bin_path.exists()
            
        if not exists:
            print(f"[vvkit] Compiling Athena++ from {src_path} with args: {args_str}")
            config_cmd = f"python3 configure.py {args_str}"
            make_cmd = f"make clean && make -j4 && cp bin/athena bin/{binary_name}"
            full_cmd = f"cd {src_path} && {config_cmd} && {make_cmd}"
            
            if self.use_wsl:
                build_cmd = ["wsl", "-d", self.wsl_distro, "--", "bash", "-c", full_cmd]
            else:
                build_cmd = ["bash", "-c", full_cmd]
                
            res = subprocess.run(build_cmd, check=False, capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stderr)
                raise RuntimeError(f"Failed to compile Athena++.\n{res.stderr}")
                
        return f"{src_path}/bin/{binary_name}"

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
            stream_pattern = self.config.plugin_args.get("output_stream", "out1")
            tab_files = list(workdir.glob(f"*.{stream_pattern}.*.tab"))
            if not tab_files:
                # Fallback to general search if the output_stream pattern doesn't match
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
        
        # Read the numerical data by flattening and reshaping
        all_floats = []
        for line in lines[data_start_idx:]:
            all_floats.extend([float(x) for x in line.split()])
        data = np.array(all_floats).reshape(-1, len(columns))
            
        coords: dict[str, Any] = {}
        fields: dict[str, Any] = {}
        
        coord_map = self.config.plugin_args.get("coords", {"x1v": "x", "x2v": "y", "x3v": "z"})
        field_map = self.config.plugin_args.get("fields", None)
        
        for idx, col in enumerate(columns):
            if col in coord_names:
                mapped_name = coord_map.get(col, col)
                coords[mapped_name] = data[:, idx]
            elif col in field_names:
                if field_map is not None:
                    if col in field_map:
                        fields[field_map[col]] = data[:, idx]
                else:
                    fields[col] = data[:, idx]
                
        return coords, fields
