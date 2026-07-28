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
