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
