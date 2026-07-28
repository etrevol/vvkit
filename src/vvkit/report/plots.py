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

"""Log-log convergence plotting module using matplotlib."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def generate_convergence_plot(
    h_values: npt.NDArray[np.float64],
    errors_by_norm: dict[str, npt.NDArray[np.float64]],
    fitted_slopes: dict[str, float],
    expected_slope: float,
    output_path: Path,
    excluded_idxs: dict[str, int | None] | None = None,
) -> None:
    """Generate log-log convergence plot comparing measured error vs grid size h.

    Cites: PROJECT_SPEC.md Section 4 & Milestone M5.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    
    if excluded_idxs is None:
        excluded_idxs = {}

    colors = {"L1": "#16a34a", "L2": "#0284c7", "Linf": "#9333ea"}
    markers = {"L1": "^-", "L2": "o-", "Linf": "s-"}

    # Track maximum error to place the theoretical slope line
    max_err_for_ref = 0.0

    for norm_name, errors in errors_by_norm.items():
        slope = fitted_slopes.get(norm_name, 0.0)
        c = colors.get(norm_name, "#475569")
        m = markers.get(norm_name, "d-")
        
        label_meas = f"{norm_name} (slope={slope:.2f})"
        ex_idx = excluded_idxs.get(norm_name)
        
        if ex_idx is not None and ex_idx < len(errors) - 1:
            ax.loglog(h_values[:ex_idx+1], errors[:ex_idx+1], m, label=label_meas, color=c, lw=2)
            ax.loglog(h_values[ex_idx+1:], errors[ex_idx+1:], "x--", color="#ef4444", lw=1.5, markersize=8)
        else:
            ax.loglog(h_values, errors, m, label=label_meas, color=c, lw=2)
            
        if len(errors) > 0 and errors[-1] > max_err_for_ref:
            max_err_for_ref = errors[-1]

    # Draw reference slope starting from the max error value at the finest grid
    if max_err_for_ref > 0:
        ref_errors = max_err_for_ref * ((h_values / h_values[-1]) ** expected_slope)
        label_ref = f"Reference O(h^{expected_slope:.1f})"
        ax.loglog(h_values, ref_errors, "--", label=label_ref, color="#64748b", lw=1.5)

    ax.set_xlabel("Grid spacing h")
    ax.set_ylabel("Error norm")
    ax.set_title("Grid Convergence Study")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    fig.savefig(output_path)
    plt.close(fig)
