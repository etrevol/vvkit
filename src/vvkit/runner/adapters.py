"""Runner data models and solver adapter Protocol definition."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


@dataclass
class CaseSpec:
    case_id: str
    refinement_parameter: str  # e.g., 'n_cells' or 'dt'
    refinement_value: float
    mms_source: Any | None = None
    user_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResult:
    case_id: str
    solution_fields: dict[str, npt.NDArray[np.float64]]
    coordinates: dict[str, npt.NDArray[np.float64]]
    cell_measures: npt.NDArray[np.float64] | None = None
    wall_time: float = 0.0
    exit_status: int = 0
    stdout_path: Path | None = None
    stderr_path: Path | None = None


class SolverAdapter(Protocol):
    def run(self, case: CaseSpec, workdir: Path) -> SolverResult:
        ...
