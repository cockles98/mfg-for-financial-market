"""
Tests for the backward HJB solver.
"""

from __future__ import annotations

import numpy as np

from mfg_finance.grid import Grid1D
from mfg_finance.hjb import solve_hjb_backward, terminal_condition_U
from mfg_finance.models.hft import HFTParams


def test_terminal_condition_matches_quadratic_payoff() -> None:
    params = HFTParams(gamma_T=2.0)
    grid = Grid1D(-1.0, 1.0, nx=11, T=1.0, nt=4, bc="neumann")

    terminal = terminal_condition_U(grid, params)

    expected = params.gamma_T * grid.x**2
    assert np.allclose(terminal, expected)


def test_hjb_solver_produces_finite_solution() -> None:
    grid = Grid1D(-1.0, 1.0, nx=31, T=0.2, nt=6, bc="neumann")
    params = HFTParams()

    density = np.ones((grid.nt + 1, grid.nx), dtype=np.float64)
    density /= density.sum(axis=1, keepdims=True) * grid.dx

    solution = solve_hjb_backward(
        density,
        grid,
        params,
        progress=False,
        max_inner=2,
    )

    assert solution.shape == density.shape
    assert np.all(np.isfinite(solution))
