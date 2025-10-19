"""
Finite-difference operators used across the HJB and FP solvers.

The routines implemented here are deliberately lightweight so they can be
reused by both equations without coupling to model specifics.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

try:
    from .grid import Grid1D
except ImportError:  # pragma: no cover - fallback when executed as script
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from mfg_finance.grid import Grid1D

__all__ = [
    "forward_difference",
    "backward_difference",
    "central_difference",
    "second_central_difference",
    "laplacian_matrix",
    "central_gradient",
    "divergence_upwind",
    "project_positive_and_renormalize",
]


def forward_difference(values: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute the first-order forward difference.

    Parameters
    ----------
    values :
        Scalar field sampled on a uniform grid.
    dx :
        Spatial step size.

    Returns
    -------
    numpy.ndarray
        Array of forward differences with the same shape as `values`. The last
        entry duplicates the penultimate derivative to maintain shape.
    """

    diff = np.empty_like(values, dtype=np.float64)
    diff[:-1] = (values[1:] - values[:-1]) / dx
    diff[-1] = diff[-2]
    return diff


def backward_difference(values: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute the first-order backward difference.

    Parameters
    ----------
    values :
        Scalar field sampled on a uniform grid.
    dx :
        Spatial step size.

    Returns
    -------
    numpy.ndarray
        Array of backward differences. The first entry duplicates the second
        derivative estimate to maintain shape.
    """

    diff = np.empty_like(values, dtype=np.float64)
    diff[1:] = (values[1:] - values[:-1]) / dx
    diff[0] = diff[1]
    return diff


def central_difference(values: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute the symmetric first derivative using central differences.

    Parameters
    ----------
    values :
        Scalar field sampled on a uniform grid.
    dx :
        Spatial step size.

    Returns
    -------
    numpy.ndarray
        Array of central difference approximations. Boundary values fall back
        to first-order one-sided estimates.
    """

    diff = np.empty_like(values, dtype=np.float64)
    diff[1:-1] = (values[2:] - values[:-2]) / (2.0 * dx)
    diff[0] = forward_difference(values, dx)[0]
    diff[-1] = backward_difference(values, dx)[-1]
    return diff


def second_central_difference(values: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute the second derivative using the classic central stencil.

    Parameters
    ----------
    values :
        Scalar field sampled on a uniform grid.
    dx :
        Spatial step size.

    Returns
    -------
    numpy.ndarray
        Array of second-derivative approximations. Boundary values reuse the
        nearest interior result to preserve shape.
    """

    diff = np.empty_like(values, dtype=np.float64)
    diff[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / (dx**2)
    diff[0] = diff[1]
    diff[-1] = diff[-2]
    return diff


def laplacian_matrix(grid: Grid1D) -> sp.csr_matrix:
    """
    Assemble the one-dimensional Laplacian with configurable boundary conditions.

    Parameters
    ----------
    grid :
        Grid definition providing spatial spacing and boundary condition.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix representing the Laplacian operator.
    """

    n = grid.nx
    dx2 = grid.dx**2

    main = np.full(n, -2.0, dtype=np.float64)
    off = np.ones(n - 1, dtype=np.float64)

    diags = [main, off, off]
    offsets = [0, -1, 1]
    lap = sp.diags(diags, offsets, shape=(n, n), format="lil")

    if grid.bc == "neumann":
        lap[0, 0] = -2.0
        lap[0, 1] = 2.0
        lap[-1, -1] = -2.0
        lap[-1, -2] = 2.0
    else:
        lap[0, :] = 0.0
        lap[-1, :] = 0.0
        lap[:, 0] = 0.0
        lap[:, -1] = 0.0
        lap[0, 0] = 1.0
        lap[-1, -1] = 1.0

    return lap.tocsr() / dx2


def central_gradient(u: np.ndarray, dx: float, bc: str = "neumann") -> np.ndarray:
    """
    Compute the gradient via central differences with boundary handling.

    Parameters
    ----------
    u :
        Scalar field.
    dx :
        Spatial step.
    bc :
        Boundary condition type. Supports ``"neumann"`` (default) and
        ``"dirichlet"``.

    Returns
    -------
    numpy.ndarray
        Gradient approximation.
    """

    grad = np.empty_like(u, dtype=np.float64)
    grad[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
    if bc == "dirichlet":
        grad[0] = (u[1] - u[0]) / dx
        grad[-1] = (u[-1] - u[-2]) / dx
    elif bc == "neumann":
        grad[0] = (u[1] - u[0]) / dx
        grad[-1] = (u[-1] - u[-2]) / dx
    else:
        msg = "Unsupported boundary condition for central_gradient."
        raise ValueError(msg)
    return grad


def divergence_upwind(m: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute the conservative divergence of the advective flux using Godunov upwinding.

    Parameters
    ----------
    m :
        Density values located at cell centres.
    v :
        Velocity field at the same locations as `m`.
    dx :
        Spatial step size.

    Returns
    -------
    numpy.ndarray
        Discrete divergence satisfying mass conservation (sum approximately zero).
    """

    if m.shape != v.shape:
        msg = "Density and velocity must share the same shape."
        raise ValueError(msg)

    mv = m * v
    mv_pos = np.maximum(mv, 0.0)
    mv_neg = np.minimum(mv, 0.0)

    n = m.size
    flux_faces = np.zeros(n + 1, dtype=np.float64)
    flux_faces[1:-1] = mv_pos[:-1] + mv_neg[1:]

    divergence = (flux_faces[1:] - flux_faces[:-1]) / dx
    divergence -= float(divergence.sum()) / n

    return divergence


def project_positive_and_renormalize(
    m: np.ndarray,
    dx: float = 1.0,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Project density onto the non-negative simplex and renormalise.

    Parameters
    ----------
    m :
        Candidate density vector.
    dx :
        Spatial step size to enforce integral normalisation.
    atol :
        Absolute tolerance; values below this after clipping trigger an error.

    Returns
    -------
    numpy.ndarray
        Non-negative vector whose entries sum to one.
    """

    clipped = np.maximum(m, 0.0)
    mass = float(clipped.sum() * dx)
    if mass <= atol:
        msg = "Mass is too small after projection."
        raise ValueError(msg)
    return clipped / mass


if __name__ == "__main__":
    grid = Grid1D(-1.0, 1.0, 101, 1.0, 100, "neumann")
    lap = laplacian_matrix(grid)
    assert lap.shape == (grid.nx, grid.nx)
    constants = np.ones(grid.nx)
    assert np.allclose(lap @ constants, 0.0, atol=1e-12)

    x = grid.x
    m = np.exp(-x**2)
    m = project_positive_and_renormalize(m, dx=grid.dx)
    v = np.sin(np.pi * x)

    div = divergence_upwind(m, v, grid.dx)
    assert abs(div.sum()) < 1e-12, "Mass is not conserved."

    grad = central_gradient(x, grid.dx)
    assert np.allclose(grad[1:-1], 1.0, atol=1e-6)
