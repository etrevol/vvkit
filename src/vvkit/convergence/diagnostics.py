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
