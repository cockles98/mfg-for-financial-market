"""
Tests for quadratic Hamiltonian utilities.
"""

from __future__ import annotations

import numpy as np

from mfg_finance.hamiltonian import (
    alpha_star,
    effective_eta,
    hamiltonian_value,
    mean_alpha_from,
    running_cost,
)
from mfg_finance.models.hft import HFTParams


def test_mean_alpha_from_recovers_weighted_average() -> None:
    x = np.linspace(-1.0, 1.0, 11)
    dx = x[1] - x[0]
    density = np.maximum(0.0, 1.0 - x**2)
    density /= density.sum() * dx
    control = 2.0 * x

    mean_ctrl = mean_alpha_from(density, control, dx)

    expected = float(np.sum(density * control) * dx)
    assert np.isclose(mean_ctrl, expected)


def test_alpha_star_and_hamiltonian_match_closed_form() -> None:
    params = HFTParams()
    x = np.linspace(-0.5, 0.5, 21)
    dx = x[1] - x[0]
    density = np.ones_like(x)
    density /= density.sum() * dx
    momentum = 0.3 * x

    mean_ctrl = 0.1
    eta = effective_eta(mean_ctrl, params)

    alpha = alpha_star(momentum, density, params, mean_alpha=mean_ctrl)
    ham = hamiltonian_value(momentum, x, density, params, mean_alpha=mean_ctrl)
    running = running_cost(x, alpha, params, mean_alpha=mean_ctrl)

    assert np.allclose(alpha, -momentum / eta)
    assert np.allclose(ham, 0.5 * (momentum**2) / eta + params.phi * x**2)
    assert np.all(running >= 0.0)
