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

"""Error norms, cell-measure weighting, and cell-average reference quadrature."""

import numpy as np
import numpy.typing as npt


def compute_l1_norm(
    errors: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None = None,
) -> float:
    """Compute L1 norm of discrete error.

    L1 = sum(|e_i| * V_i) / sum(V_i)

    Cites: PROJECT_SPEC.md Section 3.2.
    """
    abs_errors = np.abs(errors)
    if weights is None:
        return float(np.mean(abs_errors))
    total_weight = float(np.sum(weights))
    if total_weight == 0.0:
        raise ValueError("Total cell measure/weight cannot be zero.")
    return float(np.sum(abs_errors * weights) / total_weight)


def compute_l2_norm(
    errors: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None = None,
) -> float:
    """Compute L2 norm of discrete error.

    L2 = sqrt(sum(e_i^2 * V_i) / sum(V_i))

    Cites: PROJECT_SPEC.md Section 3.2.
    """
    sq_errors = errors**2
    if weights is None:
        return float(np.sqrt(np.mean(sq_errors)))
    total_weight = float(np.sum(weights))
    if total_weight == 0.0:
        raise ValueError("Total cell measure/weight cannot be zero.")
    return float(np.sqrt(np.sum(sq_errors * weights) / total_weight))


def compute_linf_norm(errors: npt.NDArray[np.float64]) -> float:
    """Compute Linf (max absolute) norm of discrete error.

    Linf = max |e_i|

    Cites: PROJECT_SPEC.md Section 3.2.
    """
    if errors.size == 0:
        return 0.0
    return float(np.max(np.abs(errors)))
