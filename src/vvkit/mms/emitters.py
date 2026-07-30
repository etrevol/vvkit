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
    expr: sp.Expr | dict[str, sp.Expr],
    func_name: str = "mms_source",
    arg_names: list[str] | None = None,
) -> str:
    """Emit C code for a symbolic SymPy expression or multiple expressions."""
    if arg_names is None:
        arg_names = ["x", "t"]
    args_str = ", ".join([f"double {arg}" for arg in arg_names])

    if isinstance(expr, dict):
        funcs = []
        for k, v in expr.items():
            c_body = ccode(v)
            funcs.append(f"double {func_name}_{k}({args_str}) {{\n    return {c_body};\n}}")
        body = "\n\n".join(funcs)
    else:
        c_body = ccode(expr)
        body = f"double {func_name}({args_str}) {{\n    return {c_body};\n}}"

    return f"""#include <math.h>

{body}
"""


def emit_cpp_source(
    expr: sp.Expr | dict[str, sp.Expr],
    func_name: str = "mms_source",
    arg_names: list[str] | None = None,
) -> str:
    """Emit C++ code for a symbolic SymPy expression or multiple expressions."""
    if arg_names is None:
        arg_names = ["x", "t"]
    args_str = ", ".join([f"double {arg}" for arg in arg_names])

    if isinstance(expr, dict):
        funcs = []
        for k, v in expr.items():
            c_body = ccode(v)
            funcs.append(f"extern \"C\" double {func_name}_{k}({args_str}) {{\n    return {c_body};\n}}")
        body = "\n\n".join(funcs)
    else:
        c_body = ccode(expr)
        body = f"extern \"C\" double {func_name}({args_str}) {{\n    return {c_body};\n}}"

    return f"""#include <cmath>

{body}
"""


def emit_python_source(
    expr: sp.Expr | dict[str, sp.Expr],
    func_name: str = "mms_source",
    arg_names: list[str] | None = None,
) -> str:
    """Emit standalone Python code for a symbolic SymPy expression."""
    if arg_names is None:
        arg_names = ["x", "t"]
    args_str = ", ".join(arg_names)

    if isinstance(expr, dict):
        funcs = []
        for k, v in expr.items():
            py_body = sp.pycode(v)
            funcs.append(f"def {func_name}_{k}({args_str}):\n    return {py_body}")
        body = "\n\n".join(funcs)
    else:
        py_body = sp.pycode(expr)
        body = f"def {func_name}({args_str}):\n    return {py_body}"

    return f"""import numpy as np

{body}
"""

