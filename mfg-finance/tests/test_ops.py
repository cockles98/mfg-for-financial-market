"""
Tests for finite-difference operators.
"""

from __future__ import annotations

import numpy as np

from mfg_finance.grid import Grid1D
from mfg_finance.ops import (
    backward_difference,
    central_gradient,
    divergence_upwind,
    forward_difference,
    laplacian_matrix,
    project_positive_and_renormalize,
    second_central_difference,
)


def test_forward_difference_on_linear_profile() -> None:
    grid = np.linspace(0.0, 1.0, 5)
    values = 3.0 * grid + 1.0
    dx = grid[1] - grid[0]

    diff = forward_difference(values, dx)

    assert np.allclose(diff[:-1], 3.0)


def test_backward_difference_on_linear_profile() -> None:
    grid = np.linspace(-1.0, 1.0, 11)
    values = -2.5 * grid + 0.2
    dx = grid[1] - grid[0]

    diff = backward_difference(values, dx)

    assert np.allclose(diff[1:], -2.5)


def test_second_central_difference_on_quadratic_profile() -> None:
    grid = np.linspace(-1.0, 1.0, 17)
    values = grid**2
    dx = grid[1] - grid[0]

    diff = second_central_difference(values, dx)

    assert np.allclose(diff[1:-1], 2.0, atol=1e-6)


def test_laplacian_matrix_neumann_has_constant_in_nullspace() -> None:
    grid = Grid1D(-1.0, 1.0, nx=21, T=1.0, nt=10, bc="neumann")
    lap = laplacian_matrix(grid)
    constant = np.ones(grid.nx)

    residual = lap @ constant

    assert np.allclose(residual, 0.0, atol=1e-10)


def test_divergence_upwind_conserves_mass() -> None:
    grid = Grid1D(-1.0, 1.0, nx=51, T=1.0, nt=10, bc="neumann")
    x = grid.x
    density = np.exp(-x**2)
    density /= density.sum() * grid.dx
    velocity = np.sin(np.pi * x)

    div = divergence_upwind(density, velocity, grid.dx)

    assert np.isclose(div.sum() * grid.dx, 0.0, atol=1e-12)


def test_project_positive_and_renormalize_simplex() -> None:
    data = np.array([0.2, -1e-9, 0.3, 0.5])

    projected = project_positive_and_renormalize(data, dx=0.25)

    assert np.all(projected >= 0.0)
    assert np.isclose(projected.sum() * 0.25, 1.0, atol=1e-12)


def test_central_gradient_linear_field() -> None:
    dx = 0.1
    x = np.linspace(0.0, 1.0, 11)
    field = 4.0 * x - 1.0

    grad = central_gradient(field, dx)

    assert np.allclose(grad[1:-1], 4.0, atol=1e-12)
