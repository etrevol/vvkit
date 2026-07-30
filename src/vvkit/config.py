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

"""Pydantic configuration models for vvcase.yaml."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ReaderConfig(BaseModel):
    type: str
    file: str = "solution.npz"
    fields: dict[str, str] = Field(default_factory=dict)
    coords: dict[str, str] = Field(default_factory=dict)


class SolverConfig(BaseModel):
    type: str = "command"
    command: list[str] = Field(default_factory=list)
    template: str | None = None
    reader: ReaderConfig | None = None
    timeout_s: float = 600.0
    plugin_args: dict[str, Any] = Field(default_factory=dict)


class MMSConfig(BaseModel):
    operator: str | dict[str, str]
    solution: str | dict[str, str]
    symbols: dict[str, float] = Field(default_factory=dict)
    domain: dict[str, list[float]] = Field(default_factory=dict)


class RefinementConfig(BaseModel):
    parameter: str
    values: list[float]


class StudyConfig(BaseModel):
    type: Literal["spatial", "temporal"] = "spatial"
    refinement: RefinementConfig
    user_params: dict[str, Any] = Field(default_factory=dict)
    reference: Literal["cell_average", "point_value"] = "cell_average"
    quadrature_order: int = 5
    coordinate_system: Literal["cartesian", "cylindrical", "spherical_polar"] = "cartesian"
    norms: list[str] = Field(default_factory=lambda: ["L1", "L2", "Linf"])
    exclude_boundary_cells: int = 0
    expected_order: float = 2.0
    order_tolerance: float = 0.2


class ConservationCheckConfig(BaseModel):
    quantity: str
    field: str
    tolerance_mode: Literal["roundoff", "absolute"] = "roundoff"
    factor: float = 100.0


class ChecksConfig(BaseModel):
    conservation: list[ConservationCheckConfig] = Field(default_factory=list)


class ReportConfig(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["html", "json", "junit"])
    output_dir: str = "reports"


class VVCaseConfig(BaseModel):
    version: int = 1
    name: str
    solver: SolverConfig
    mms: MMSConfig
    study: StudyConfig
    checks: ChecksConfig = Field(default_factory=ChecksConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


def load_config(path: Path) -> VVCaseConfig:
    """Load and validate a vvcase configuration file."""
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return VVCaseConfig.model_validate(data)
