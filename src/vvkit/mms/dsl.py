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

    def get_initial_condition(self) -> sp.Expr | None:
        """Evaluate manufactured solution at t=0."""
        if self.time_var is None:
            return None
        return sp.simplify(self.manufactured_sol.subs(self.time_var, 0.0))

    def get_boundary_condition(self, var: sp.Symbol, value: float) -> sp.Expr:
        """Evaluate manufactured solution at a specific domain boundary."""
        return sp.simplify(self.manufactured_sol.subs(var, value))


def parse_mms_problem(
    operator_str: str | dict[str, str],
    solution_str: str | dict[str, str],
    symbols_dict: dict[str, float] | None = None,
    domain_dict: dict[str, list[float]] | None = None,
) -> MMSProblem | dict[str, MMSProblem]:
    """Parse SymPy expression strings into symbolic objects and compute source S = L(u_m).

    Cites: Roache (2002), PROJECT_SPEC.md Section 3.1.
    """
    if symbols_dict is None:
        symbols_dict = {}
    if domain_dict is None:
        domain_dict = {}

    is_scalar = isinstance(operator_str, str)
    operators = {"u": operator_str} if is_scalar else operator_str
    solutions = {"u": solution_str} if isinstance(solution_str, str) else solution_str

    sym_objs = {name: sp.Symbol(name) for name in symbols_dict}
    for var in domain_dict:
        if var not in sym_objs:
            sym_objs[var] = sp.Symbol(var)
            
    # Fallback if domain_dict was empty
    for var in ["x", "y", "z", "t"]:
        if var not in sym_objs:
            sym_objs[var] = sp.Symbol(var)

    funcs = {}
    for k in solutions:
        funcs[k] = sp.Function(k)
        if k not in sym_objs:
            sym_objs[k] = sp.Symbol(k)

    local_dict: dict[str, Any] = {**sym_objs, **funcs, "Eq": sp.Eq, "Derivative": sp.Derivative}

    u_m = {}
    for k, sol_str in solutions.items():
        u_m[k] = sp.sympify(sol_str, locals=local_dict)

    op_parsed = {}
    for k, op_str in operators.items():
        op_str_parsed = sp.sympify(op_str, locals=local_dict)
        if isinstance(op_str_parsed, sp.Eq):
            op_parsed[k] = op_str_parsed.lhs - op_str_parsed.rhs
        else:
            op_parsed[k] = op_str_parsed

    # Infer variables from function calls
    from sympy import preorder_traversal
    func_args_vars = {}
    for k, op_expr in op_parsed.items():
        for node in preorder_traversal(op_expr):
            if isinstance(node, AppliedUndef) and node.func.__name__ in funcs:
                if node.func.__name__ not in func_args_vars:
                    func_args_vars[node.func.__name__] = list(node.args)

    def replace_call(expr: sp.Basic) -> sp.Basic:
        if isinstance(expr, AppliedUndef) and expr.func.__name__ in funcs:
            func_name = expr.func.__name__
            sub_map = {}
            expected_args = func_args_vars.get(func_name, [])
            for i, arg_val in enumerate(expr.args):
                if i < len(expected_args):
                    sub_map[expected_args[i]] = arg_val
            return u_m[func_name].subs(sub_map)
        return expr

    # Variables for the problem
    prob_vars = []
    t_sym = None
    for var_str in domain_dict:
        if var_str == "t":
            t_sym = sym_objs["t"]
        else:
            prob_vars.append(sym_objs[var_str])

    # If domain_dict wasn't provided or didn't contain anything, guess from func_args
    if not prob_vars and not t_sym:
        for args in func_args_vars.values():
            for v in args:
                if str(v) == "t":
                    t_sym = v
                elif v not in prob_vars:
                    prob_vars.append(v)
    
    # Ultimate fallback
    if not prob_vars:
        prob_vars = [sym_objs.get("x", sp.Symbol("x"))]

    problems = {}
    
    sym_sub_map = {sym_objs[sym]: val for sym, val in symbols_dict.items()} if symbols_dict else {}

    for k, op_expr in op_parsed.items():
        source_expr = op_expr.replace(
            lambda e: isinstance(e, AppliedUndef) and e.func.__name__ in funcs, 
            replace_call
        )
        if source_expr == op_expr:
            # Fallback if no explicit function calls, maybe direct symbols
            sub_dict = {sym_objs[fn]: u_m[fn] for fn in funcs if fn in sym_objs}
            source_expr = op_expr.subs(sub_dict)

        source_term = source_expr.doit()
        if sym_sub_map:
            source_term = source_term.subs(sym_sub_map)
        
        vanished: list[str] = []
        if isinstance(op_expr, sp.Add):
            terms_to_check = op_expr.args
        else:
            terms_to_check = (op_expr,)
            
        for term in terms_to_check:
            term_subbed = term.replace(
                lambda e: isinstance(e, AppliedUndef) and e.func.__name__ in funcs, 
                replace_call
            )
            if term_subbed == term:
                sub_dict = {sym_objs[fn]: u_m[fn] for fn in funcs if fn in sym_objs}
                term_subbed = term.subs(sub_dict)
            term_eval = term_subbed.doit()
            if sym_sub_map:
                term_eval = term_eval.subs(sym_sub_map)
            term_eval = sp.simplify(term_eval)
            if term_eval == 0:
                vanished.append(str(term))

        problems[k] = MMSProblem(
            operator_expr=op_expr,
            manufactured_sol=u_m.get(k, sp.Integer(0)),
            symbols_map={sym_objs[sym]: val for sym, val in symbols_dict.items()} if symbols_dict else {},
            variables=prob_vars,
            time_var=t_sym,
            source_term=sp.simplify(source_term),
            vanished_terms=vanished,
        )

    if is_scalar:
        return problems["u"]
    return problems
