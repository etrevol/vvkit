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

from pathlib import Path

import numpy as np

from vvkit.checks import check_conservation
from vvkit.regression import BaselineStore


def test_conservation_pass_and_leak() -> None:
    # Perfect conservation to roundoff
    q_clean = np.full(100, 1.0, dtype=np.float64)
    res_clean = check_conservation(q_clean)
    assert res_clean.is_conserved is True
    assert res_clean.departure_step is None

    # Seeded leak at step 40
    q_leak = np.full(100, 1.0, dtype=np.float64)
    q_leak[40:] += 1e-4
    res_leak = check_conservation(q_leak)
    assert res_leak.is_conserved is False
    assert res_leak.departure_step == 40


def test_baseline_store_and_drift(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path / "baselines")
    metrics_base = {"L2_order": 2.01, "L2_error": 1.2e-4}
    store.save_baseline("burgers_1d", metrics_base)

    # Clean match within rtol
    res_good = store.compare("burgers_1d", {"L2_order": 2.0101, "L2_error": 1.2001e-4}, rtol=1e-2)
    assert res_good.is_drifted is False

    # Drifted match
    res_drift = store.compare("burgers_1d", {"L2_order": 1.5, "L2_error": 1.2e-4}, rtol=1e-2)
    assert res_drift.is_drifted is True
