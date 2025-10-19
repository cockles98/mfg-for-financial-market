"""
Hamilton-Jacobi-Bellman solver routines.

This module contains a high-level API that will later host monotone numerical
schemes (e.g. Lax-Friedrichs). The current scaffold focuses on type-safe
interfaces that align with the rest of the code base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm.auto import tqdm

from .grid import Grid1D
from .hamiltonian import (
    alpha_star,
    effective_eta,
    hamiltonian_value,
    mean_alpha_from,
    EtaCallback,
)
from .models.hft import HFTParams
from .ops import backward_difference, forward_difference, laplacian_matrix

__all__ = [
    "HJBSolver",
    "terminal_condition_U",
    "hjb_step",
    "solve_hjb_backward",
]


def terminal_condition_U(grid: Grid1D, params: HFTParams) -> np.ndarray:
    """
    Compute the terminal payoff :math:`U_T(x) = \\gamma_T x^2`.

    Parameters
    ----------
    grid :
        Discretised space-time grid.
    params :
        Model parameters containing :math:`\\gamma_T`.

    Returns
    -------
    numpy.ndarray
        Terminal value function sampled on the spatial grid.
    """

    return params.gamma_T * (grid.x**2)


def _lax_friedrichs_gradient(
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


def hjb_step(
    U_next: np.ndarray,
    m_at_n: np.ndarray,
    grid: Grid1D,
    params: HFTParams,
    *,
    max_inner: int = 4,
    tol: float = 1e-8,
    eta_callback: EtaCallback | None = None,
    max_dissipation: float | None = None,
    alpha_cap: float | None = None,
    value_cap: float | None = None,
    value_relaxation: float | None = None,
) -> np.ndarray:
    """
    Perform a single backward HJB step using a monotone discretisation.

    Parameters
    ----------
    U_next :
        Value function at the next time level (n+1).
    m_at_n :
        Agent density at the current time level.
    grid :
        Space-time grid definition.
    params :
        Model parameters.
    max_inner :
        Number of inner fixed-point iterations used to resolve the non-linearity.
    tol :
        Convergence tolerance on the inner iterations (infinity norm).

    Returns
    -------
    numpy.ndarray
        Updated value function at time level n.
    """

    if U_next.shape != m_at_n.shape:
        msg = "U_next and m_at_n must share the same shape."
        raise ValueError(msg)

    lap = laplacian_matrix(grid)
    identity = sp.eye(grid.nx, format="csr")
    dt = grid.dt

    system_matrix = identity - dt * params.nu * lap
    rhs_static = U_next.astype(np.float64, copy=True)

    U_iter = rhs_static.copy()
    mean_alpha = 0.0

    for _ in range(max_inner):
        gradient = _lax_friedrichs_gradient(U_iter, grid.dx, max_dissipation=max_dissipation)
        alpha = alpha_star(
            gradient,
            m_at_n,
            params,
            mean_alpha=mean_alpha,
            eta_callback=eta_callback,
        )
        if alpha_cap is not None:
            alpha = np.clip(alpha, -float(alpha_cap), float(alpha_cap))
        mean_alpha = mean_alpha_from(m_at_n, alpha, grid.dx)
        alpha = alpha_star(
            gradient,
            m_at_n,
            params,
            mean_alpha=mean_alpha,
            eta_callback=eta_callback,
        )
        if alpha_cap is not None:
            alpha = np.clip(alpha, -float(alpha_cap), float(alpha_cap))

        eta_value = effective_eta(mean_alpha, params)
        if eta_callback is not None:
            eta_value = eta_callback(m_at_n, alpha, params)

        h_val = hamiltonian_value(
            gradient,
            grid.x,
            m_at_n,
            params,
            mean_alpha=mean_alpha,
            eta_callback=eta_callback,
            alpha=alpha,
            eta_value=eta_value,
        )
        rhs = rhs_static + dt * h_val

        U_new = spla.spsolve(system_matrix, rhs)

        if value_cap is not None:
            U_new = np.clip(U_new, -float(value_cap), float(value_cap))

        if value_relaxation is not None:
            relax = float(np.clip(value_relaxation, 0.0, 1.0))
            U_relaxed = relax * U_new + (1.0 - relax) * U_iter
        else:
            U_relaxed = U_new

        if not np.all(np.isfinite(U_relaxed)):
            msg = "Non-finite value encountered during HJB iteration."
            raise FloatingPointError(msg)

        if np.max(np.abs(U_relaxed - U_iter)) < tol:
            U_iter = U_relaxed
            break

        U_iter = U_relaxed

    return np.asarray(U_iter, dtype=np.float64)


def solve_hjb_backward(
    all_m: np.ndarray,
    grid: Grid1D,
    params: HFTParams,
    *,
    terminal: np.ndarray | None = None,
    max_inner: int = 4,
    tol: float = 1e-8,
    progress: bool = True,
    eta_callback: EtaCallback | None = None,
    max_dissipation: float | None = None,
    alpha_cap: float | None = None,
    value_cap: float | None = None,
    value_relaxation: float | None = None,
) -> np.ndarray:
    """
    Solve the HJB equation backward in time given a density trajectory.

    Parameters
    ----------
    all_m :
        Density trajectory with shape ``(nt + 1, nx)`` ordered from t=0 to T.
    grid :
        Space-time grid definition.
    params :
        Model parameters.
    terminal :
        Optional terminal payoff. Defaults to :func:`terminal_condition_U`.
    max_inner :
        Number of inner fixed-point iterations passed to :func:`hjb_step`.
    tol :
        Convergence tolerance forwarded to :func:`hjb_step`.
    progress :
        Whether to show a progress bar for the backward sweep.

    Returns
    -------
    numpy.ndarray
        Value function trajectory with shape ``(nt + 1, nx)``.
    """

    all_m = np.asarray(all_m, dtype=np.float64)
    if all_m.shape[0] != grid.nt + 1 or all_m.shape[1] != grid.nx:
        msg = (
            "Density trajectory must have shape (nt + 1, nx); "
            f"received {all_m.shape} expected ({grid.nt + 1}, {grid.nx})."
        )
        raise ValueError(msg)

    U = np.zeros_like(all_m)
    U[-1] = terminal_condition_U(grid, params) if terminal is None else np.asarray(terminal)

    indices: Iterable[int] = range(grid.nt - 1, -1, -1)
    iterator = tqdm(indices, desc="HJB backward", leave=False, total=grid.nt) if progress else indices

    for n in iterator:
        U[n] = hjb_step(
            U[n + 1],
            all_m[n],
            grid,
            params,
            max_inner=max_inner,
            tol=tol,
            eta_callback=eta_callback,
            max_dissipation=max_dissipation,
            alpha_cap=alpha_cap,
            value_cap=value_cap,
            value_relaxation=value_relaxation,
        )
    return U


@dataclass(slots=True)
class HJBSolver:
    """
    Wrapper around :func:`solve_hjb_backward` for convenience.
    """

    grid: Grid1D
    params: HFTParams
    max_inner: int = 4
    tol: float = 1e-8
    show_progress: bool = True
    max_dissipation: float | None = None
    alpha_cap: float | None = None
    value_cap: float | None = None
    value_relaxation: float | None = None

    def solve(
        self,
        density_path: np.ndarray,
        terminal_payoff: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Run the backward HJB solve.

        Parameters
        ----------
        density_path :
            Density trajectory with shape ``(nt + 1, nx)``.
        terminal_payoff :
            Optional terminal payoff overriding :func:`terminal_condition_U`.

        Returns
        -------
        numpy.ndarray
            Value function trajectory ordered from t=0 to T.
        """

        return solve_hjb_backward(
            density_path,
            self.grid,
            self.params,
            terminal=terminal_payoff,
            max_inner=self.max_inner,
            tol=self.tol,
            progress=self.show_progress,
            max_dissipation=self.max_dissipation,
            alpha_cap=self.alpha_cap,
            value_cap=self.value_cap,
            value_relaxation=self.value_relaxation,
        )
