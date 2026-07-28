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


def preset_exponential_1d(
    x_sym: sp.Symbol,
    t_sym: sp.Symbol | None = None,
    alpha: float = 1.0,
) -> sp.Expr:
    """Exponential manufactured solution preset: exp(alpha * x) * exp(-t)."""
    if t_sym is not None:
        return sp.exp(alpha * x_sym) * sp.exp(-t_sym)
    return sp.exp(alpha * x_sym)


def preset_polynomial_1d(
    x_sym: sp.Symbol,
    t_sym: sp.Symbol | None = None,
    degree: int = 3,
) -> sp.Expr:
    """Polynomial manufactured solution preset: x^degree * (1 + t)."""
    if t_sym is not None:
        return (x_sym**degree) * (1 + t_sym)
    return x_sym**degree


def preset_boundary_layer_1d(
    x_sym: sp.Symbol,
    L: float = 1.0,
    thickness: float = 0.05,
) -> sp.Expr:
    """Boundary-layer manufactured solution preset: 1 - exp(-x / thickness)."""
    return 1 - sp.exp(-x_sym / thickness)


def check_domain_positivity(
    expr: sp.Expr,
    symbols: list[sp.Symbol],
    domain: dict[str, tuple[float, float]],
    num_samples: int = 20,
) -> bool:
    """Sample solution over the domain and return True if all values are strictly positive (> 0).

    Cites: PROJECT_SPEC.md Section 3.1.
    """
    if not symbols:
        return True
    
    # We only sample symbols that have domain bounds
    valid_syms = [s for s in symbols if str(s) in domain]
    if not valid_syms:
        return True
        
    func = sp.lambdify(valid_syms, expr, modules="numpy")
    
    grids_1d = []
    for s in valid_syms:
        bounds = domain[str(s)]
        grids_1d.append(np.linspace(bounds[0], bounds[1], num_samples))
        
    mesh = np.meshgrid(*grids_1d, indexing="ij")
    vals = func(*mesh)
    
    return bool(np.all(vals > 0))
