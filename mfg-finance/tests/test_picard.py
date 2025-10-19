"""
Tests for the Picard solver orchestrating the MFG system.
"""

from __future__ import annotations

import numpy as np

from mfg_finance.grid import Grid1D
from mfg_finance.models.hft import HFTParams, initial_density
from mfg_finance.solver import compute_alpha_traj, solve_mfg_picard


def test_compute_alpha_traj_shapes() -> None:
    grid = Grid1D(-1.0, 1.0, nx=11, T=0.1, nt=4, bc="neumann")
    params = HFTParams()

    U = np.zeros((grid.nt + 1, grid.nx))
    M = np.ones_like(U) / (grid.nx * grid.dx)

    alpha = compute_alpha_traj(U, M, grid, params)

    assert alpha.shape == U.shape
    assert np.all(np.isfinite(alpha))


def test_solve_mfg_picard_runs_small_iteration() -> None:
    grid = Grid1D(-1.0, 1.0, nx=101, T=0.5, nt=50, bc="neumann")
    params = HFTParams(
        nu=0.1,
        phi=0.0,
        gamma_T=0.0,
        eta0=1.0,
        eta1=0.0,
        m0_mean=0.0,
        m0_std=1.0,
    )
    initial = initial_density(grid, params)

    U, M, alpha, errors, metrics = solve_mfg_picard(
        grid,
        params,
        max_iter=30,
        tol=1e-5,
        mix=0.8,
        m0=initial,
        hjb_kwargs={"max_inner": 1, "tol": 1e-8},
        fp_kwargs={},
        eta_callback=None,
    )

    assert U.shape == (grid.nt + 1, grid.nx)
    assert M.shape == (grid.nt + 1, grid.nx)
    assert alpha.shape == (grid.nt + 1, grid.nx)
    assert len(errors) >= 1
    assert np.all(np.isfinite(U))
    assert np.all(np.isfinite(M))
    assert np.all(np.isfinite(alpha))
    masses = np.sum(M, axis=1) * grid.dx
    assert np.allclose(masses, 1.0, atol=1e-6)
    assert errors, "Picard solver should record convergence errors."
    assert errors[-1] < 1e-5
    assert metrics["mean_abs_alpha"] >= 0.0
    assert metrics["liquidity_proxy"] <= 1.0

    if len(errors) >= 3:
        for prev, cur in zip(errors[1:], errors[2:]):
            assert cur <= prev * (1.0 + 1e-3)
