"""Convergence order estimation algorithms."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import linregress


@dataclass
class LeastSquaresOrderResult:
    order: float
    std_err: float
    r_squared: float


def compute_pairwise_order(
    err1: float,
    err2: float,
    r: float,
) -> float:
    """Compute pairwise observed order of accuracy for constant refinement ratio r.

    Formula: p = ln(err_coarse / err_fine) / ln(r)
    Cites: PROJECT_SPEC.md Section 3.3.
    """
    if err1 <= 0 or err2 <= 0:
        raise ValueError("Errors must be positive for log-based order computation.")
    if r <= 1.0:
        raise ValueError("Refinement ratio r = h_coarse / h_fine must be > 1.0.")
    return float(np.log(err1 / err2) / np.log(r))


def compute_roache_order(
    f1: float,
    f2: float,
    f3: float,
    h1: float,
    h2: float,
    h3: float,
    expected_order: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Solve Roache transcendental equation for order p with non-constant r.

    r21 = h2 / h1, r32 = h3 / h2
    e21 = f2 - f1, e32 = f3 - f2
    p = (1 / ln(r21)) * | ln|e32 / e21| + q(p) |
    q(p) = ln((r21^p - s) / (r32^p - s))
    s = sign(e32 / e21)

    Cites: Roache (1998), PROJECT_SPEC.md Section 3.3.
    """
    e21 = f2 - f1
    e32 = f3 - f2

    if abs(e21) < 1e-15 or abs(e32) < 1e-15:
        raise ValueError("Solution differences are zero or near round-off floor.")

    r21 = h2 / h1
    r32 = h3 / h2

    ratio = e32 / e21
    s = 1.0 if ratio > 0 else -1.0
    abs_ratio = abs(ratio)

    p = expected_order if expected_order > 0 else 1.0

    for _ in range(max_iter):
        num = (r21**p) - s
        den = (r32**p) - s
        if den == 0.0 or num / den <= 0.0:
            break
        q_p = np.log(num / den)
        p_next = (1.0 / np.log(r21)) * abs(np.log(abs_ratio) + q_p)

        if abs(p_next - p) / max(abs(p), 1e-10) < tol:
            return float(p_next)
        p = p_next

    raise RuntimeError(
        f"Roache fixed-point iteration did not converge within {max_iter} iterations (p={p})."
    )


def compute_least_squares_order(
    h_values: npt.NDArray[np.float64],
    errors: npt.NDArray[np.float64],
) -> LeastSquaresOrderResult:
    """Fit ln(e) = p * ln(h) + c using linear regression.

    Cites: PROJECT_SPEC.md Section 3.3.
    """
    if len(h_values) < 3 or len(errors) < 3:
        raise ValueError("At least 3 grid data points are required for least-squares fit.")

    log_h = np.log(h_values)
    log_e = np.log(errors)

    res = linregress(log_h, log_e)
    return LeastSquaresOrderResult(
        order=float(res.slope),
        std_err=float(res.stderr) if res.stderr is not None else 0.0,
        r_squared=float(res.rvalue**2),
    )
