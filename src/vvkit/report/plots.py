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

"""Log-log convergence and conservation time-series plotting using matplotlib."""

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
    study_name: str = "Grid Convergence Study",
) -> None:
    """Generate log-log convergence plot with fitted slopes and reference slope lines.

    Shows measured error vs grid spacing h for each norm, with:
    - Fitted slope lines (dashed, matching data color)
    - Reference slope lines O(h^1), O(h^2), O(h^3) for visual comparison
    - Round-off floor exclusion markers

    Cites: PROJECT_SPEC.md Section 3.3, Milestone M5.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    if excluded_idxs is None:
        excluded_idxs = {}

    colors = {"L1": "#22c55e", "L2": "#3b82f6", "Linf": "#a855f7"}
    markers = {"L1": "^", "L2": "o", "Linf": "s"}

    for norm_name, errors in errors_by_norm.items():
        slope = fitted_slopes.get(norm_name, 0.0)
        c = colors.get(norm_name, "#94a3b8")
        m = markers.get(norm_name, "d")

        label_meas = f"{norm_name} (p={slope:.3f})"
        ex_idx = excluded_idxs.get(norm_name)

        if ex_idx is not None and ex_idx < len(errors) - 1:
            ax.loglog(
                h_values[: ex_idx + 1], errors[: ex_idx + 1],
                marker=m, linestyle="-", label=label_meas, color=c, lw=2, markersize=7,
            )
            ax.loglog(
                h_values[ex_idx + 1 :], errors[ex_idx + 1 :],
                marker="x", linestyle="--", color="#ef4444", lw=1.5, markersize=8,
                label=f"{norm_name} (round-off)",
            )
            fit_h = h_values[: ex_idx + 1]
            fit_e = errors[: ex_idx + 1]
        else:
            ax.loglog(
                h_values, errors,
                marker=m, linestyle="-", label=label_meas, color=c, lw=2, markersize=7,
            )
            fit_h = h_values
            fit_e = errors

        if len(fit_h) >= 2 and slope != 0.0:
            anchor_h = fit_h[-1]
            anchor_e = fit_e[-1]
            fitted_line = anchor_e * (h_values / anchor_h) ** slope
            ax.loglog(
                h_values, fitted_line, linestyle=":", color=c, lw=1.2, alpha=0.6,
            )

    ref_slopes = sorted({1, 2, 3, int(expected_slope), int(expected_slope) + 1})
    ref_slopes = [s for s in ref_slopes if 0 < s <= 5]
    ref_colors = {1: "#475569", 2: "#64748b", 3: "#475569", 4: "#475569", 5: "#475569"}
    any_errors = next(iter(errors_by_norm.values()), np.array([1.0]))
    ref_anchor = float(any_errors[-1]) if len(any_errors) > 0 else 1.0
    h_min_val = float(h_values[-1]) if len(h_values) > 0 else 1.0

    for s in ref_slopes:
        ref_line = ref_anchor * (h_values / h_min_val) ** s
        ax.loglog(
            h_values, ref_line, linestyle="--", color=ref_colors.get(s, "#475569"),
            lw=1.0, alpha=0.4, label=f"O(h$^{s}$)",
        )

    ax.set_xlabel("Grid spacing h", fontsize=11)
    ax.set_ylabel("Error norm", fontsize=11)
    ax.set_title(study_name, fontsize=13, fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)
    fig.tight_layout()

    fig.savefig(output_path)
    plt.close(fig)


def generate_conservation_plot(
    imbalance_series: npt.NDArray[np.float64],
    tolerance: float,
    departure_step: int | None,
    output_path: Path,
    quantity_name: str = "Q",
) -> None:
    """Generate conservation imbalance time-series plot.

    Shows imbalance vs time step with tolerance band and departure marker.
    Cites: PROJECT_SPEC.md Section 3.7, Milestone M5.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    steps = np.arange(len(imbalance_series))
    ax.plot(steps, imbalance_series, color="#3b82f6", lw=2, label="Imbalance")

    ax.axhline(y=tolerance, color="#22d3ee", ls="--", lw=1.2, alpha=0.7, label=f"Tolerance (\u00b1{tolerance:.2e})")
    ax.axhline(y=-tolerance, color="#22d3ee", ls="--", lw=1.2, alpha=0.7)

    if departure_step is not None:
        ax.axvline(x=departure_step, color="#ef4444", ls="-", lw=2, alpha=0.8, label=f"Departure (step {departure_step})")

    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Imbalance (normalized)", fontsize=11)
    ax.set_title(f"Conservation Check: {quantity_name}", fontsize=13, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend(fontsize=9, loc="best", framealpha=0.8)
    fig.tight_layout()

    fig.savefig(output_path)
    plt.close(fig)
