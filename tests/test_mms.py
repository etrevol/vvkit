import sympy as sp

from vvkit.mms import (
    check_domain_positivity,
    emit_c_source,
    emit_cpp_source,
    emit_python_source,
    parse_mms_problem,
    preset_trigonometric_1d,
)


def test_mms_symbolic_cancellation() -> None:
    op_str = (
        "Derivative(u(x, t), t) + u(x, t)*Derivative(u(x, t), x) - nu*Derivative(u(x, t), x, 2)"
    )
    prob = parse_mms_problem(
        operator_str=op_str,
        solution_str="sin(2*pi*x)*exp(-t)",
        symbols_dict={"nu": 0.01},
    )

    u_m = prob.manufactured_sol
    u_func = sp.Function("u")
    nu = sp.Symbol("nu")

    def is_u_call(e: sp.Basic) -> bool:
        return isinstance(e, sp.core.function.AppliedUndef) and e.func == u_func

    def replace_u_call(expr: sp.Basic) -> sp.Basic:
        if is_u_call(expr):
            return u_m
        return expr

    op_eval = prob.operator_expr.subs(nu, 0.01).replace(is_u_call, replace_u_call)
    diff_expr = op_eval.doit() - prob.source_term
    assert sp.simplify(diff_expr) == 0


def test_code_emitters_evaluation() -> None:
    x, t = sp.Symbol("x"), sp.Symbol("t")
    expr = sp.sin(2 * sp.pi * x) * sp.exp(-t)

    py_code = emit_python_source(expr)
    c_code = emit_c_source(expr)
    cpp_code = emit_cpp_source(expr)

    assert "def mms_source(x, t):" in py_code
    assert "#include <math.h>" in c_code
    assert 'extern "C"' in cpp_code


def test_domain_positivity_and_presets() -> None:
    x, t = sp.Symbol("x"), sp.Symbol("t")
    sol_1d = preset_trigonometric_1d(x)
    sol_2d = preset_trigonometric_1d(x, t)

    assert isinstance(sol_1d, sp.Expr)
    assert isinstance(sol_2d, sp.Expr)

    expr_pos = x**2 + 1.0
    expr_neg = sp.sin(x)

    assert check_domain_positivity(expr_pos, x, (0.0, 1.0)) is True
    assert check_domain_positivity(expr_neg, x, (-1.0, 1.0)) is False
    assert check_domain_positivity(expr_pos, x, (0.0, 1.0), sym_t=t, t_range=(0.0, 1.0)) is True
