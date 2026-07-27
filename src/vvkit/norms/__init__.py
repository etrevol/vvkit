"""Norms module initialization."""

from vvkit.norms.norms import compute_l1_norm, compute_l2_norm, compute_linf_norm
from vvkit.norms.quadrature import cell_average_1d

__all__ = ["compute_l1_norm", "compute_l2_norm", "compute_linf_norm", "cell_average_1d"]
