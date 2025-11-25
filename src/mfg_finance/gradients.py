"""
Common gradient operators shared between HJB and FP solvers.
"""

from __future__ import annotations

import numpy as np

from .ops import backward_difference, forward_difference

__all__ = ["lax_friedrichs_gradient"]


def lax_friedrichs_gradient(
    values: np.ndarray,
    dx: float,
    *,
    max_dissipation: float | None = None,
) -> np.ndarray:
    """
    Compute a monotone gradient approximation using a Lax-Friedrichs stencil.
    """

    forward = forward_difference(values, dx)
    backward = backward_difference(values, dx)

    a = float(np.max(np.abs(np.concatenate((forward, backward)))))
    if not np.isfinite(a) or a < 1e-8:
        a = 1e-3
    if max_dissipation is not None:
        a = min(a, float(max_dissipation))

    grad = 0.5 * (forward + backward) - 0.5 * a * (forward - backward)
    grad[0] = forward[0]
    grad[-1] = backward[-1]
    return grad
