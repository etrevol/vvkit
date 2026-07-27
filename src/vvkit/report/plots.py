"""Log-log convergence plotting module using matplotlib."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def generate_convergence_plot(
    h_values: npt.NDArray[np.float64],
    errors: npt.NDArray[np.float64],
    fitted_slope: float,
    expected_slope: float,
    output_path: Path,
) -> None:
    """Generate log-log convergence plot comparing measured error vs grid size h.

    Cites: PROJECT_SPEC.md Section 4 & Milestone M5.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    label_meas = f"Measured (slope={fitted_slope:.2f})"
    ax.loglog(h_values, errors, "o-", label=label_meas, color="#0284c7", lw=2)

    ref_errors = errors[-1] * ((h_values / h_values[-1]) ** expected_slope)
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
