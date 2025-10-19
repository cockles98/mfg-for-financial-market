"""
Price clearing utilities for endogenous price determination.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


AlphaField = Callable[[int, float], np.ndarray]

__all__ = ["solve_price_clearing"]


def solve_price_clearing(
    alpha_field: AlphaField,
    densities: np.ndarray,
    supply: Sequence[float],
    dx: float,
    *,
    bracket: tuple[float, float] = (-10.0, 10.0),
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Solve for the price path that clears instantaneous flow imbalances.

    Parameters
    ----------
    alpha_field :
        Callable returning the control profile for a given time index and price.
    densities :
        Density trajectory with shape ``(nt + 1, nx)``.
    supply :
        Target supply schedule matching the temporal grid.
    dx :
        Spatial step used for numerical integration.
    bracket :
        Initial search interval for the price.
    max_iter :
        Maximum number of iterations in the bisection search.
    tol :
        Desired absolute tolerance on the clearing condition.

    Returns
    -------
    numpy.ndarray
        Price path ``P_n`` of shape ``(nt + 1,)``.
    """

    densities = np.asarray(densities, dtype=np.float64)
    supply = np.asarray(supply, dtype=np.float64)
    if densities.shape[0] != supply.shape[0]:
        msg = "Supply and density trajectories must share the same temporal dimension."
        raise ValueError(msg)

    a, b = bracket
    prices = np.zeros(densities.shape[0], dtype=np.float64)

    for n in range(densities.shape[0]):
        target = supply[n]

        def imbalance(price: float) -> float:
            control = alpha_field(n, price)
            return float(np.sum(control * densities[n]) * dx - target)

        lower, upper = a, b
        f_lower, f_upper = imbalance(lower), imbalance(upper)

        # Expand bracket if necessary.
        expansion = 0
        while f_lower * f_upper > 0.0 and expansion < 5:
            width = upper - lower
            lower -= width
            upper += width
            f_lower, f_upper = imbalance(lower), imbalance(upper)
            expansion += 1

        if f_lower * f_upper > 0.0:
            # Fallback: choose the side with smaller absolute imbalance.
            prices[n] = lower if abs(f_lower) < abs(f_upper) else upper
            continue

        for _ in range(max_iter):
            mid = 0.5 * (lower + upper)
            f_mid = imbalance(mid)

            if abs(f_mid) < tol or 0.5 * (upper - lower) < tol:
                prices[n] = mid
                break

            if f_lower * f_mid <= 0.0:
                upper = mid
                f_upper = f_mid
            else:
                lower = mid
                f_lower = f_mid
        else:
            prices[n] = mid  # type: ignore[misc]

    return prices
