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

"""Convergence module initialization."""

from vvkit.convergence.diagnostics import detect_roundoff_floor
from vvkit.convergence.gci import ConvergenceState, GCIResult, compute_gci
from vvkit.convergence.order import (
    LeastSquaresOrderResult,
    compute_least_squares_order,
    compute_pairwise_order,
    compute_roache_order,
)

__all__ = [
    "compute_pairwise_order",
    "compute_roache_order",
    "compute_least_squares_order",
    "LeastSquaresOrderResult",
    "compute_gci",
    "GCIResult",
    "ConvergenceState",
    "detect_roundoff_floor",
]
