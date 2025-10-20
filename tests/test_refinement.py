"""
Refinement tests for grid convergence.
"""

from __future__ import annotations

import numpy as np

from mfg_finance.grid import Grid1D
from mfg_finance.models.hft import HFTParams, initial_density
from mfg_finance.solver import solve_mfg_picard


def _final_density(nx: int, nt: int) -> np.ndarray:
    grid = Grid1D(-1.0, 1.0, nx=nx, T=0.3, nt=nt, bc="neumann")
    params = HFTParams(
        nu=0.05,
        phi=0.05,
        gamma_T=0.2,
        eta0=1.0,
        eta1=0.0,
        m0_mean=0.0,
        m0_std=0.8,
    )
    m0 = initial_density(grid, params)
    _, M_all, _, _, _ = solve_mfg_picard(
        grid,
        params,
        max_iter=6,
        tol=1e-4,
        mix=0.7,
        m0=m0,
        hjb_kwargs={"max_inner": 1, "tol": 1e-8},
        eta_callback=None,
    )
    return M_all[-1]


def test_refinement_l2_difference_decreases() -> None:
    coarse = _final_density(21, 20)
    medium = _final_density(41, 40)
    fine = _final_density(81, 80)

    diff_coarse_medium = np.linalg.norm(medium[::2] - coarse)
    diff_medium_fine = np.linalg.norm(fine[::2] - medium)

    assert diff_medium_fine < diff_coarse_medium
