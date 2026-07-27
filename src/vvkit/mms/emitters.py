"""Code emission to C, C++, and Python."""

import sympy as sp
from sympy.printing.c import ccode


def emit_c_source(
    expr: sp.Expr,
    func_name: str = "mms_source",
    arg_names: list[str] | None = None,
) -> str:
    """Emit C code for a symbolic SymPy expression.

    Cites: PROJECT_SPEC.md Section 3.1 & Milestone M2.
    """
    if arg_names is None:
        arg_names = ["x", "t"]
    args_str = ", ".join([f"double {arg}" for arg in arg_names])
    c_body = ccode(expr)
    return f"""#include <math.h>

double {func_name}({args_str}) {{
    return {c_body};
}}
"""


def emit_cpp_source(
    expr: sp.Expr,
    func_name: str = "mms_source",
    arg_names: list[str] | None = None,
) -> str:
    """Emit C++ code for a symbolic SymPy expression."""
    if arg_names is None:
        arg_names = ["x", "t"]
    args_str = ", ".join([f"double {arg}" for arg in arg_names])
    c_body = ccode(expr)
    return f"""#include <cmath>

extern "C" double {func_name}({args_str}) {{
    return {c_body};
}}
"""


def emit_python_source(
    expr: sp.Expr,
    func_name: str = "mms_source",
    arg_names: list[str] | None = None,
) -> str:
    """Emit standalone Python code for a symbolic SymPy expression."""
    if arg_names is None:
        arg_names = ["x", "t"]
    args_str = ", ".join(arg_names)
    py_body = sp.pycode(expr)
    return f"""import numpy as np

def {func_name}({args_str}):
    return {py_body}
"""
