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

"""Conservation and invariant budget checks."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class ConservationResult:
    is_conserved: bool
    imbalance_series: npt.NDArray[np.float64]
    final_imbalance: float
    departure_step: int | None  # Step where imbalance exceeds tolerance


def check_conservation(
    q_time_series: npt.NDArray[np.float64],
    flux_in_series: npt.NDArray[np.float64] | None = None,
    flux_out_series: npt.NDArray[np.float64] | None = None,
    source_series: npt.NDArray[np.float64] | None = None,
    q_scale: float | None = None,
    factor: float = 100.0,
    eps: float = 2.220446049250313e-16,
) -> ConservationResult:
    """Check conservation budget closure over time.

    imbalance = ( Q(t) - Q(t0) - ∫(F_in - F_out)dt - ∫S dV dt ) / Q_scale
    Cites: PROJECT_SPEC.md Section 3.7.
    """
    n_steps = len(q_time_series)
    if n_steps == 0:
        raise ValueError("q_time_series cannot be empty.")

    if q_scale is None or q_scale == 0.0:
        q_scale = max(float(np.max(np.abs(q_time_series))), 1.0)

    q0 = q_time_series[0]
    imbalance_series = np.zeros(n_steps, dtype=np.float64)

    cum_flux = 0.0
    cum_source = 0.0

    for i in range(n_steps):
        if flux_in_series is not None and flux_out_series is not None:
            cum_flux += flux_in_series[i] - flux_out_series[i]
        if source_series is not None:
            cum_source += source_series[i]

        imbalance_series[i] = (q_time_series[i] - q0 - cum_flux - cum_source) / q_scale

    tol = factor * n_steps * eps
    final_imbalance = float(imbalance_series[-1])

    departure_step = None
    exceeded_indices = np.where(np.abs(imbalance_series) > tol)[0]
    if len(exceeded_indices) > 0:
        departure_step = int(exceeded_indices[0])

    is_conserved = departure_step is None

    return ConservationResult(
        is_conserved=is_conserved,
        imbalance_series=imbalance_series,
        final_imbalance=final_imbalance,
        departure_step=departure_step,
    )
