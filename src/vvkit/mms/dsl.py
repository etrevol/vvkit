"""MMS symbolic Operator DSL and source term derivation."""

from dataclasses import dataclass
from typing import Any

import sympy as sp
from sympy.core.function import AppliedUndef


@dataclass
class MMSProblem:
    operator_expr: sp.Expr
    manufactured_sol: sp.Expr
    symbols_map: dict[sp.Symbol, float]
    variables: list[sp.Symbol]
    time_var: sp.Symbol | None
    source_term: sp.Expr
    vanished_terms: list[str]


def parse_mms_problem(
    operator_str: str,
    solution_str: str,
    symbols_dict: dict[str, float] | None = None,
    domain_dict: dict[str, list[float]] | None = None,
) -> MMSProblem:
    """Parse SymPy expression strings into symbolic objects and compute source S = L(u_m).

    Cites: Roache (2002), PROJECT_SPEC.md Section 3.1.
    """
    if symbols_dict is None:
        symbols_dict = {}

    sym_objs = {name: sp.Symbol(name) for name in symbols_dict}
    for var in ["x", "y", "z", "t", "u"]:
        if var not in sym_objs:
            sym_objs[var] = sp.Symbol(var)

    u_func = sp.Function("u")
    u_sym = sym_objs["u"]
    x_sym = sym_objs["x"]
    t_sym = sym_objs["t"] if "t" in sym_objs else None

    local_dict: dict[str, Any] = {**sym_objs, "u": u_func, "Eq": sp.Eq, "Derivative": sp.Derivative}

    u_m = sp.sympify(solution_str, locals=local_dict)

    op_parsed = sp.sympify(operator_str, locals=local_dict)
    if isinstance(op_parsed, sp.Eq):
        op_expr = op_parsed.lhs - op_parsed.rhs
    else:
        op_expr = op_parsed

    def is_u_call(e: sp.Basic) -> bool:
        return isinstance(e, AppliedUndef) and e.func == u_func

    def replace_u_call(expr: sp.Basic) -> sp.Basic:
        if is_u_call(expr):
            sub_map = {}
            if len(expr.args) >= 1:
                sub_map[x_sym] = expr.args[0]
            if len(expr.args) >= 2 and t_sym is not None:
                sub_map[t_sym] = expr.args[1]
            return u_m.subs(sub_map)
        return expr

    source_expr = op_expr.replace(is_u_call, replace_u_call)
    if source_expr == op_expr:
        source_expr = op_expr.subs(u_sym, u_m)

    source_term = source_expr.doit()

    if symbols_dict:
        sym_sub_map = {sym_objs[k]: v for k, v in symbols_dict.items()}
        source_term = source_term.subs(sym_sub_map)

    vanished: list[str] = []

    return MMSProblem(
        operator_expr=op_expr,
        manufactured_sol=u_m,
        symbols_map={sym_objs[k]: v for k, v in symbols_dict.items()},
        variables=[x_sym],
        time_var=t_sym,
        source_term=sp.simplify(source_term),
        vanished_terms=vanished,
    )
