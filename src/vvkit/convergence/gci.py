"""Grid Convergence Index (GCI) and asymptotic-range diagnostics per ASME V&V 20 / Roache."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ConvergenceState(Enum):
    MONOTONIC = "monotonic"
    OSCILLATORY = "oscillatory"
    DIVERGENT = "divergent"


@dataclass
class GCIResult:
    gci_fine: float
    safety_factor: float
    q_scale: float
    convergence_state: ConvergenceState
    asymptotic_ratio: float | None  # R = GCI_32 / (r21^p * GCI_21)
    is_asymptotic: bool  # |R - 1| <= 0.1


def compute_gci(
    f1: float,
    f2: float,
    f3: float | None = None,
    r21: float = 2.0,
    r32: float = 2.0,
    p: float | None = None,
    q_scale: float | None = None,
) -> GCIResult:
    """Compute Grid Convergence Index per ASME V&V 20 / Roache.

    Convergence condition Rc = e21 / e32:
    - 0 < Rc < 1: monotonic convergence
    - Rc < 0: oscillatory convergence
    - Rc >= 1 or Rc <= -1 (with |Rc| >= 1): divergence

    Cites: ASME V&V 20-2009, Roache (1998), PROJECT_SPEC.md Section 3.4.
    """
    e21 = f2 - f1
    if q_scale is None or q_scale == 0.0:
        q_scale = max(abs(f1), 1.0)

    norm_e21 = abs(e21) / q_scale

    if f3 is not None:
        e32 = f3 - f2
        r_c = e21 / e32 if e32 != 0.0 else 0.0

        if abs(r_c) >= 1.0:
            state = ConvergenceState.DIVERGENT
        elif r_c > 0:
            state = ConvergenceState.MONOTONIC
        else:
            state = ConvergenceState.OSCILLATORY

        fs = 1.25
        if p is None:
            p = float(np.log(abs(e32 / e21)) / np.log(r21))
    else:
        state = ConvergenceState.MONOTONIC
        fs = 3.0
        if p is None:
            p = 2.0

    gci_21 = fs * norm_e21 / (r21**p - 1.0)

    asymptotic_ratio = None
    is_asymptotic = False

    if f3 is not None:
        norm_e32 = abs(f3 - f2) / q_scale
        gci_32 = fs * norm_e32 / (r32**p - 1.0)
        asymptotic_ratio = gci_32 / ((r21**p) * gci_21) if gci_21 > 0 else None
        if asymptotic_ratio is not None:
            is_asymptotic = abs(asymptotic_ratio - 1.0) <= 0.1

    return GCIResult(
        gci_fine=float(gci_21),
        safety_factor=fs,
        q_scale=q_scale,
        convergence_state=state,
        asymptotic_ratio=asymptotic_ratio,
        is_asymptotic=is_asymptotic,
    )
