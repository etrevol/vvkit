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

"""Runner module initialization."""

from vvkit.runner.adapters import CaseSpec, SolverAdapter, SolverResult
from vvkit.runner.matrix import CallableAdapter, CommandAdapter
from vvkit.runner.readers import read_csv_output, read_npz_output

__all__ = [
    "CaseSpec",
    "SolverResult",
    "SolverAdapter",
    "CallableAdapter",
    "CommandAdapter",
    "read_npz_output",
    "read_csv_output",
]
