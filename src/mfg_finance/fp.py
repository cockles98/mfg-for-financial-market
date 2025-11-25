"""
Fokker-Planck solver routines.

The forward equation is discretised with an explicit upwind advection term and
an implicit diffusion step to guarantee stability and conservation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm.auto import tqdm

from .grid import Grid1D
from .hamiltonian import alpha_star
from .models.hft import HFTParams
from .ops import divergence_upwind, laplacian_matrix, project_positive_and_renormalize
from .gradients import lax_friedrichs_gradient

__all__ = [
    "velocity_from_U",
    "fp_step",
    "solve_fp_forward",
    "FPSolver",
]


def velocity_from_U(
    U_n: np.ndarray,
    m_n: np.ndarray,
    grid: Grid1D,
    params: HFTParams,
    *,
    max_dissipation: float | None = None,
    alpha_cap: float | None = None,
    drift_strength: float = 0.0,
) -> np.ndarray:
    """
    Compute the transport velocity derived from the HJB control.

    Parameters
    ----------
    U_n :
        Value function at time level ``n``.
    m_n :
        Density at time level ``n``.
    grid :
        Space-time grid definition.
    params :
        Model parameters.

    Returns
    -------
    numpy.ndarray
        Velocity field aligned with the spatial grid.
    """

    if U_n.shape != m_n.shape or U_n.shape != (grid.nx,):
        msg = "U_n and m_n must both match the spatial grid shape."
        raise ValueError(msg)

    gradient = lax_friedrichs_gradient(U_n, grid.dx, max_dissipation=max_dissipation)

    alpha = alpha_star(gradient, m_n, params, mean_alpha=None)
    if drift_strength != 0.0:
        alpha = alpha - float(drift_strength) * grid.x
    if alpha_cap is not None:
        alpha = np.clip(alpha, -float(alpha_cap), float(alpha_cap))
    return alpha


def fp_step(
    m_n: np.ndarray,
    v_n: np.ndarray,
    grid: Grid1D,
    params: HFTParams,
) -> np.ndarray:
    """
    Advance the density one step forward using semi-implicit diffusion.

    Parameters
    ----------
    m_n :
        Density at time level ``n``.
    v_n :
        Velocity computed from the HJB solution.
    grid :
        Space-time grid definition.
    params :
        Model parameters providing the diffusion coefficient.

    Returns
    -------
    numpy.ndarray
        Updated density at time level ``n+1``.
    """

    if m_n.shape != v_n.shape or m_n.shape != (grid.nx,):
        msg = "Density and velocity must match the spatial grid shape."
        raise ValueError(msg)

    lap = laplacian_matrix(grid)
    identity = sp.eye(grid.nx, format="csr")

    dt = grid.dt
    diffusion_matrix = identity - dt * params.nu * lap

    div = divergence_upwind(m_n, v_n, grid.dx)
    rhs = m_n + dt * div

    m_next = spla.spsolve(diffusion_matrix, rhs)
    m_next = project_positive_and_renormalize(m_next, dx=grid.dx, atol=1e-12)

    return m_next


def solve_fp_forward(
    U_all: np.ndarray,
    grid: Grid1D,
    params: HFTParams,
    m0: np.ndarray,
    *,
    progress: bool = True,
    max_dissipation: float | None = None,
    alpha_cap: float | None = None,
    drift_strength: float = 0.0,
) -> np.ndarray:
    """
    Solve the Fokker-Planck equation forward in time.

    Parameters
    ----------
    U_all :
        Value function trajectory with shape ``(nt + 1, nx)``.
    grid :
        Space-time grid definition.
    params :
        Model parameters.
    m0 :
        Initial density sampled on the spatial grid.
    progress :
        Whether to display a progress bar.

    Returns
    -------
    numpy.ndarray
        Density trajectory with shape ``(nt + 1, nx)``.
    """

    U_all = np.asarray(U_all, dtype=np.float64)
    if U_all.shape[0] != grid.nt + 1 or U_all.shape[1] != grid.nx:
        msg = "U_all must have shape (nt + 1, nx)."
        raise ValueError(msg)

    density = np.zeros_like(U_all)
    density[0] = project_positive_and_renormalize(
        np.asarray(m0, dtype=np.float64),
        dx=grid.dx,
    )

    iterator: Iterable[int] = range(grid.nt)
    iterator = tqdm(iterator, desc="FP forward", leave=False, total=grid.nt) if progress else iterator

    for n in iterator:
        v_n = velocity_from_U(
            U_all[n],
            density[n],
            grid,
            params,
            max_dissipation=max_dissipation,
            alpha_cap=alpha_cap,
            drift_strength=drift_strength,
        )
        density[n + 1] = fp_step(density[n], v_n, grid, params)

    return density


@dataclass(slots=True)
class FPSolver:
    """
    High-level wrapper around :func:`solve_fp_forward`.
    """

    grid: Grid1D
    params: HFTParams
    show_progress: bool = True

    def solve(self, value_function: np.ndarray, initial_density: np.ndarray) -> np.ndarray:
        """
        Propagate the density forward using the provided value function trajectory.
        """

        return solve_fp_forward(
            value_function,
            self.grid,
            self.params,
            initial_density,
            progress=self.show_progress,
            drift_strength=0.0,
        )
