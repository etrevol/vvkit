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

"""Diagnostics for round-off floor detection and asymptotic validity."""

import numpy as np
import numpy.typing as npt


def detect_roundoff_floor(
    errors: npt.NDArray[np.float64],
) -> int:
    """Detect index where grid error reaches minimum before rising due to round-off.

    Returns index of minimum error. Points after this index should be excluded.
    Cites: PROJECT_SPEC.md Section 3.6.
    """
    if len(errors) == 0:
        return 0
    return int(np.argmin(errors))
