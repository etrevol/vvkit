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

import numpy as np
import pytest

from vvkit.convergence import (
    ConvergenceState,
    compute_gci,
    compute_least_squares_order,
    compute_pairwise_order,
    compute_roache_order,
)
from vvkit.norms import cell_average_1d, compute_l1_norm, compute_l2_norm, compute_linf_norm


def test_norms_uniform() -> None:
    errors = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float64)
    assert compute_l1_norm(errors) == 2.5
    assert compute_l2_norm(errors) == pytest.approx(np.sqrt(7.5))
    assert compute_linf_norm(errors) == 4.0
    assert compute_linf_norm(np.array([], dtype=np.float64)) == 0.0


def test_norms_weighted() -> None:
    errors = np.array([1.0, 3.0], dtype=np.float64)
    weights = np.array([1.0, 3.0], dtype=np.float64)
    assert compute_l1_norm(errors, weights) == 2.5
    assert compute_l2_norm(errors, weights) == pytest.approx(np.sqrt((1 + 27) / 4.0))

    with pytest.raises(ValueError, match="Total cell measure/weight cannot be zero."):
        compute_l1_norm(errors, np.array([0.0, 0.0], dtype=np.float64))

    with pytest.raises(ValueError, match="Total cell measure/weight cannot be zero."):
        compute_l2_norm(errors, np.array([0.0, 0.0], dtype=np.float64))


def test_cell_average_1d() -> None:
    bounds = np.array([[0.0, 2.0]], dtype=np.float64)
    avg = cell_average_1d(lambda x: x**2, bounds, order=5)
    assert avg[0] == pytest.approx(4.0 / 3.0, abs=1e-12)


def test_synthetic_order_recovery() -> None:
    for p_exact in [1.0, 1.5, 2.0, 3.0, 4.0]:
        h_vals = np.array([0.1, 0.05, 0.025, 0.0125], dtype=np.float64)
        c = 2.5
        errors = c * (h_vals**p_exact)

        p_pair = compute_pairwise_order(errors[0], errors[1], r=2.0)
        assert p_pair == pytest.approx(p_exact, abs=1e-10)

        ls_res = compute_least_squares_order(h_vals, errors)
        assert ls_res.order == pytest.approx(p_exact, abs=1e-10)
        assert ls_res.r_squared == pytest.approx(1.0, abs=1e-10)

    with pytest.raises(ValueError, match="Errors must be positive"):
        compute_pairwise_order(0.0, 0.1, 2.0)

    with pytest.raises(ValueError, match="Refinement ratio r"):
        compute_pairwise_order(0.2, 0.1, 1.0)

    with pytest.raises(ValueError, match="At least 3 grid data points"):
        compute_least_squares_order(np.array([0.1, 0.05]), np.array([0.1, 0.05]))


def test_roache_order() -> None:
    p_exact = 2.0
    h1, h2, h3 = 0.01, 0.02, 0.05
    f1 = 1.0 + 0.5 * (h1**p_exact)
    f2 = 1.0 + 0.5 * (h2**p_exact)
    f3 = 1.0 + 0.5 * (h3**p_exact)

    p_roache = compute_roache_order(f1, f2, f3, h1, h2, h3, expected_order=2.0)
    assert p_roache == pytest.approx(p_exact, abs=1e-6)

    with pytest.raises(ValueError, match="Solution differences are zero"):
        compute_roache_order(1.0, 1.0, 1.0, h1, h2, h3)

    with pytest.raises(RuntimeError, match="Roache fixed-point iteration did not converge"):
        # Force non-convergence by passing bad parameters with low max_iter
        compute_roache_order(1.0, 2.0, 2.1, 0.01, 0.02, 0.05, expected_order=100.0, max_iter=1)


def test_gci_diagnostics() -> None:
    # Monotonic
    h1, h2, h3 = 0.01, 0.02, 0.04
    f1 = 1.0 + 0.1 * (h1**2)
    f2 = 1.0 + 0.1 * (h2**2)
    f3 = 1.0 + 0.1 * (h3**2)

    res = compute_gci(f1, f2, f3, r21=2.0, r32=2.0, p=2.0)
    assert res.convergence_state == ConvergenceState.MONOTONIC
    assert res.safety_factor == 1.25
    assert res.is_asymptotic is True

    # Oscillatory: e21/e32 < 0 and |e21/e32| < 1
    res_osc = compute_gci(1.0, 0.95, 1.1, r21=2.0, r32=2.0, p=2.0)
    assert res_osc.convergence_state == ConvergenceState.OSCILLATORY

    # Divergent: |e21/e32| >= 1
    res_div = compute_gci(1.0, 3.0, 1.1, r21=2.0, r32=2.0, p=2.0)
    assert res_div.convergence_state == ConvergenceState.DIVERGENT

    # 2 grids case
    res_2g = compute_gci(1.0, 1.05)
    assert res_2g.safety_factor == 3.0
    assert res_2g.asymptotic_ratio is None


@pytest.mark.parametrize("expected_p", [1.0, 1.5, 3.0, 4.0])
def test_order_recovery_multiple_orders(expected_p: float) -> None:
    """Verify order recovery for various p values (spec §M1 acceptance)."""
    C = 0.5
    h_values = np.array([1/8, 1/16, 1/32, 1/64, 1/128], dtype=np.float64)
    errors = C * h_values ** expected_p
    fit = compute_least_squares_order(h_values, errors)
    assert abs(fit.order - expected_p) < 1e-10, f"Expected p={expected_p}, got {fit.order}"


def test_oscillatory_convergence_test() -> None:
    # e21 = -0.1, e32 = 0.2 -> e21/e32 = -0.5
    res = compute_gci(1.0, 0.9, 1.1, r21=2.0, r32=2.0, p=2.0)
    assert res.convergence_state == ConvergenceState.OSCILLATORY


def test_divergent_sequence_test() -> None:
    # e21 = 0.2, e32 = 0.1 -> e21/e32 = 2.0
    res = compute_gci(1.0, 1.2, 1.1, r21=2.0, r32=2.0, p=2.0)
    assert res.convergence_state == ConvergenceState.DIVERGENT


def test_roundoff_floor_detection() -> None:
    """U-shaped error: decreasing then increasing (spec §M1 acceptance)."""
    from vvkit.convergence.diagnostics import detect_roundoff_floor
    errors = np.array([1e-2, 1e-3, 1e-4, 1e-5, 2e-5, 5e-5], dtype=np.float64)
    min_idx = detect_roundoff_floor(errors)
    assert min_idx == 3, f"Expected minimum at index 3, got {min_idx}"


def test_gci_computation_hand_calculated() -> None:
    # f1=1.0, f2=1.2, f3=2.0
    # e21 = 0.2, e32 = 0.8
    # r=2, p=2 -> r^p - 1 = 3
    # GCI_fine = Fs * |e21| / (r^p - 1) = 1.25 * 0.2 / 3 = 0.08333333333333333
    res = compute_gci(1.0, 1.2, 2.0, r21=2.0, r32=2.0, p=2.0)
    assert res.convergence_state == ConvergenceState.MONOTONIC
    assert res.is_asymptotic is True
    assert abs(res.gci_fine - (1.25 * 0.2 / 3.0)) < 1e-10
