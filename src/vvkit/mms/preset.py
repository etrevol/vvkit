"""MMS preset solution families and scaling diagnostics."""

import numpy as np
import sympy as sp


def preset_trigonometric_1d(
    x_sym: sp.Symbol,
    t_sym: sp.Symbol | None = None,
    freq: float = 2.0,
) -> sp.Expr:
    """Trigonometric manufactured solution preset: sin(freq * pi * x) * exp(-t)."""
    if t_sym is not None:
        return sp.sin(freq * sp.pi * x_sym) * sp.exp(-t_sym)
    return sp.sin(freq * sp.pi * x_sym)


def check_domain_positivity(
    expr: sp.Expr,
    sym_x: sp.Symbol,
    x_range: tuple[float, float],
    sym_t: sp.Symbol | None = None,
    t_range: tuple[float, float] | None = None,
    num_samples: int = 100,
) -> bool:
    """Sample solution over the domain and return True if all values are strictly positive (> 0).

    Cites: PROJECT_SPEC.md Section 3.1.
    """
    func = sp.lambdify([sym_x] + ([sym_t] if sym_t is not None else []), expr, "numpy")
    xs = np.linspace(x_range[0], x_range[1], num_samples)

    if sym_t is not None and t_range is not None:
        ts = np.linspace(t_range[0], t_range[1], num_samples)
        grid_x, grid_t = np.meshgrid(xs, ts)
        vals = func(grid_x, grid_t)
    else:
        vals = func(xs)

    return bool(np.all(vals > 0))
