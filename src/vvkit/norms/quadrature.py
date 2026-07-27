"""Gauss-Legendre quadrature for computing cell-averaged reference values in 1D, 2D, and 3D."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.special import roots_legendre


def cell_average_1d(
    func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x_bounds: npt.NDArray[np.float64],  # shape (N, 2): [[x_min, x_max], ...]
    order: int = 5,
) -> npt.NDArray[np.float64]:
    """Compute cell average of 1D function over N intervals using Gauss-Legendre quadrature.

    Order >= 4 Gauss quadrature ensures exactness for polynomials up to degree 2*order - 1.
    Cites: PROJECT_SPEC.md Section 3.2.
    """
    nodes, weights = roots_legendre(order)
    x_min = x_bounds[:, 0]
    x_max = x_bounds[:, 1]
    dx = x_max - x_min
    mid = 0.5 * (x_min + x_max)
    half_dx = 0.5 * dx

    averages = np.zeros(len(x_bounds), dtype=np.float64)
    for xi, w in zip(nodes, weights, strict=True):
        x_pts = mid + half_dx * xi
        f_vals = func(x_pts)
        averages += w * f_vals

    averages *= 0.5
    return averages
