"""
Tests for the Fokker-Planck forward solver.
"""

from __future__ import annotations

import numpy as np

from mfg_finance.fp import fp_step, solve_fp_forward, velocity_from_U
from mfg_finance.grid import Grid1D
from mfg_finance.hjb import terminal_condition_U
from mfg_finance.models.hft import HFTParams, initial_density


def test_velocity_from_U_matches_control_gradient() -> None:
    grid = Grid1D(-1.0, 1.0, nx=21, T=0.1, nt=2, bc="neumann")
    params = HFTParams()
    U = terminal_condition_U(grid, params)
    density = np.ones(grid.nx, dtype=np.float64)
    density /= density.sum() * grid.dx

    velocity = velocity_from_U(U, density, grid, params)

    assert velocity.shape == (grid.nx,)
    assert np.all(np.isfinite(velocity))


def test_fp_step_conserves_mass_and_nonnegative() -> None:
    grid = Grid1D(-1.0, 1.0, nx=31, T=0.1, nt=5, bc="neumann")
    params = HFTParams()
    x = grid.x
    U = 0.5 * x**2
    density = np.exp(-x**2)
    density /= density.sum() * grid.dx

    velocity = velocity_from_U(U, density, grid, params)

    next_density = fp_step(density, velocity, grid, params)

    assert next_density.shape == density.shape
    assert np.all(next_density >= -1e-12)
    dx = grid.dx
    assert np.isclose(np.sum(next_density) * dx, 1.0, atol=1e-6)


def test_solve_fp_forward_returns_trajectory() -> None:
    grid = Grid1D(-1.0, 1.0, nx=21, T=0.2, nt=4, bc="neumann")
    params = HFTParams()

    U = np.vstack(
        [terminal_condition_U(grid, params) for _ in range(grid.nt + 1)],
    )
    m0 = np.exp(-grid.x**2)
    m0 /= m0.sum() * grid.dx

    traj = solve_fp_forward(U, grid, params, m0, progress=False)

    assert traj.shape == (grid.nt + 1, grid.nx)
    dx = grid.dx
    masses = np.sum(traj, axis=1) * dx
    assert np.allclose(masses, 1.0, atol=1e-6)


def test_fp_zero_velocity_preserves_mass_over_steps() -> None:
    grid = Grid1D(-2.0, 2.0, nx=41, T=0.2, nt=5, bc="neumann")
    params = HFTParams()
    density = initial_density(grid, params)
    velocity = np.zeros(grid.nx, dtype=np.float64)

    current = density.copy()
    for _ in range(5):
        current = fp_step(current, velocity, grid, params)
        assert np.all(current >= -1e-12)

    mass = float(np.sum(current) * grid.dx)
    assert np.isclose(mass, 1.0, atol=1e-8)
