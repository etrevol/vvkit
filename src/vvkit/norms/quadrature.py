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


def cell_average_nd(
    func: Callable[..., npt.NDArray[np.float64]],
    bounds_list: list[npt.NDArray[np.float64]],
    order: int = 5,
) -> npt.NDArray[np.float64]:
    """Compute cell average of nD function over N intervals using tensor-product Gauss-Legendre quadrature.

    Args:
        func: Vectorized function taking ndim positional arguments.
        bounds_list: List of dimension bounds, each array of shape (N, 2).
        order: Gauss-Legendre quadrature order per dimension.
    """
    import itertools

    ndim = len(bounds_list)
    if ndim == 0:
        raise ValueError("Must provide at least one dimension for quadrature.")

    n_cells = bounds_list[0].shape[0]
    nodes, weights = roots_legendre(order)

    mids = []
    half_dxs = []
    for bounds in bounds_list:
        mids.append(0.5 * (bounds[:, 0] + bounds[:, 1]))
        half_dxs.append(0.5 * (bounds[:, 1] - bounds[:, 0]))

    averages = np.zeros(n_cells, dtype=np.float64)

    for idx_tuple in itertools.product(range(order), repeat=ndim):
        w_prod = 1.0
        pts = []
        for d in range(ndim):
            i = idx_tuple[d]
            w_prod *= weights[i]
            pts.append(mids[d] + half_dxs[d] * nodes[i])

        f_vals = func(*pts)
        averages += w_prod * f_vals

    averages *= (0.5**ndim)
    return averages

