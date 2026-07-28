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

from pathlib import Path

import numpy as np
import pytest

from vvkit.convergence import compute_least_squares_order
from vvkit.norms import compute_l2_norm
from vvkit.runner import (
    CallableAdapter,
    CaseSpec,
    CommandAdapter,
    SolverResult,
    read_csv_output,
    read_npz_output,
    read_txt_output,
)


def toy_1d_advection_solver(case: CaseSpec, workdir: Path) -> SolverResult:
    n_cells = int(case.refinement_value)
    dx = 1.0 / n_cells
    x_centers = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_cells)

    u_exact = np.sin(2 * np.pi * x_centers)
    numerical_error = 0.5 * (dx**2) * np.sin(2 * np.pi * x_centers)
    u_num = u_exact + numerical_error

    return SolverResult(
        case_id=case.case_id,
        solution_fields={"u": u_num},
        coordinates={"x": x_centers},
        cell_measures=np.full(n_cells, dx),
    )


def toy_1st_order_solver(case: CaseSpec, workdir: Path) -> SolverResult:
    n_cells = int(case.refinement_value)
    dx = 1.0 / n_cells
    x_centers = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_cells)

    u_exact = np.sin(2 * np.pi * x_centers)
    numerical_error = 0.8 * dx * np.cos(2 * np.pi * x_centers)
    u_num = u_exact + numerical_error

    return SolverResult(
        case_id=case.case_id,
        solution_fields={"u": u_num},
        coordinates={"x": x_centers},
        cell_measures=np.full(n_cells, dx),
    )


def test_callable_adapter_2nd_order(tmp_path: Path) -> None:
    adapter = CallableAdapter(toy_1d_advection_solver)
    n_cells_list = [32, 64, 128, 256]
    h_vals = []
    errors = []

    for n in n_cells_list:
        case = CaseSpec(case_id=f"case_{n}", refinement_parameter="n_cells", refinement_value=n)
        res = adapter.run(case, tmp_path / case.case_id)

        dx = 1.0 / n
        u_exact = np.sin(2 * np.pi * res.coordinates["x"])
        err = compute_l2_norm(res.solution_fields["u"] - u_exact, res.cell_measures)

        h_vals.append(dx)
        errors.append(err)

    fit = compute_least_squares_order(np.array(h_vals), np.array(errors))
    assert fit.order == pytest.approx(2.0, abs=0.1)


def test_callable_adapter_1st_order(tmp_path: Path) -> None:
    adapter = CallableAdapter(toy_1st_order_solver)
    n_cells_list = [32, 64, 128, 256]
    h_vals = []
    errors = []

    for n in n_cells_list:
        case = CaseSpec(case_id=f"case_{n}", refinement_parameter="n_cells", refinement_value=n)
        res = adapter.run(case, tmp_path / case.case_id)

        dx = 1.0 / n
        u_exact = np.sin(2 * np.pi * res.coordinates["x"])
        err = compute_l2_norm(res.solution_fields["u"] - u_exact, res.cell_measures)

        h_vals.append(dx)
        errors.append(err)

    fit = compute_least_squares_order(np.array(h_vals), np.array(errors))
    assert fit.order == pytest.approx(1.0, abs=0.1)


def test_readers(tmp_path: Path) -> None:
    # Test NPZ reader
    npz_file = tmp_path / "out.npz"
    np.savez(
        npz_file,
        x=np.array([0.0, 1.0]),
        u=np.array([1.0, 2.0]),
        cell_measures=np.array([0.5, 0.5]),
    )
    npz_data = read_npz_output(npz_file)
    assert "x" in npz_data["coords"]
    assert "u" in npz_data["fields"]

    # Test CSV reader
    csv_file = tmp_path / "out.csv"
    csv_file.write_text("x,u\n0.0,1.0\n1.0,2.0\n", encoding="utf-8")
    csv_data = read_csv_output(csv_file, {"x": "x"}, {"u": "u"})
    assert "x" in csv_data["coords"]
    assert "u" in csv_data["fields"]

    # Test txt reader
    txt_file = tmp_path / "out.txt"
    txt_file.write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
    txt_data = read_txt_output(txt_file, {"x": "0"}, {"u": "1"})
    assert "x" in txt_data["coords"]
    assert "u" in txt_data["fields"]


def test_command_adapter(tmp_path: Path) -> None:
    def dummy_reader(wdir: Path) -> dict[str, dict[str, np.ndarray]]:
        return {"fields": {"u": np.array([1.0])}, "coords": {"x": np.array([0.5])}}

    cmd_adapter = CommandAdapter(
        command_template=["python", "-c", "print('hello')"],
        reader_func=dummy_reader,
    )
    case = CaseSpec(case_id="cmd_test", refinement_parameter="n_cells", refinement_value=10)
    res = cmd_adapter.run(case, tmp_path / "cmd_workdir")

    assert res.exit_status == 0
    assert "u" in res.solution_fields
