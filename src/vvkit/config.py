"""Pydantic configuration models for vvcase.yaml."""

from typing import Literal

from pydantic import BaseModel, Field


class ReaderConfig(BaseModel):
    type: str
    fields: dict[str, str] = Field(default_factory=dict)
    coords: dict[str, str] = Field(default_factory=dict)


class SolverConfig(BaseModel):
    type: Literal["callable", "command"] = "command"
    command: list[str] = Field(default_factory=list)
    template: str | None = None
    reader: ReaderConfig | None = None
    timeout_s: float = 600.0


class MMSConfig(BaseModel):
    operator: str
    solution: str
    symbols: dict[str, float] = Field(default_factory=dict)
    domain: dict[str, list[float]] = Field(default_factory=dict)


class RefinementConfig(BaseModel):
    parameter: str
    values: list[float]


class StudyConfig(BaseModel):
    type: Literal["spatial", "temporal"] = "spatial"
    refinement: RefinementConfig
    reference: Literal["cell_average", "point_value"] = "cell_average"
    quadrature_order: int = 5
    norms: list[str] = Field(default_factory=lambda: ["L1", "L2", "Linf"])
    expected_order: float = 2.0
    order_tolerance: float = 0.2


class VVCaseConfig(BaseModel):
    version: int = 1
    name: str
    solver: SolverConfig
    mms: MMSConfig
    study: StudyConfig
