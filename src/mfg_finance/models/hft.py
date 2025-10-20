"""
Reference model for high-frequency trading Mean Field Games.

The model parameters follow stylised dynamics where agents trade to manage
inventory while reacting to aggregate order flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mfg_finance.grid import Grid1D

__all__ = ["HFTParams", "initial_density", "load_default_model", "eta_from_m_alpha"]


@dataclass(slots=True)
class HFTParams:
    """
    Parameter collection for the high-frequency trading model.

    Parameters
    ----------
    nu :
        Idiosyncratic diffusion intensity in the inventory dynamics.
    phi :
        Running inventory penalty applied during the horizon.
    gamma_T :
        Terminal inventory penalty at maturity.
    eta0 :
        Baseline execution cost.
    eta1 :
        Sensitivity of execution cost to the mean flow.
    m0_mean :
        Mean of the initial density in inventory space.
    m0_std :
        Standard deviation of the initial density (must be positive).
    """

    nu: float = 0.2
    phi: float = 0.1
    gamma_T: float = 1.0
    eta0: float = 0.05
    eta1: float = 0.5
    m0_mean: float = 0.0
    m0_std: float = 1.0

    def __post_init__(self) -> None:
        if self.m0_std <= 0.0:
            msg = "m0_std must be strictly positive."
            raise ValueError(msg)
        if self.nu < 0.0:
            msg = "nu must be non-negative."
            raise ValueError(msg)


def initial_density(grid: Grid1D, params: HFTParams) -> np.ndarray:
    """
    Generate a normalised Gaussian initial density over the spatial grid.

    Parameters
    ----------
    grid :
        Discrete grid definition.
    params :
        HFT parameter set providing the Gaussian moments.

    Returns
    -------
    numpy.ndarray
        Non-negative density vector summing to one (within 1e-12).
    """

    deviations = (grid.x - params.m0_mean) / params.m0_std
    gaussian = np.exp(-0.5 * deviations**2)
    gaussian = np.maximum(gaussian, 0.0)

    total_mass = float(gaussian.sum() * grid.dx)
    if total_mass <= 0.0:
        msg = "Gaussian evaluation produced zero total mass."
        raise ValueError(msg)

    density = gaussian / total_mass
    if np.any(density < -1e-12):
        msg = "Initial density contains negative values beyond tolerance."
        raise ValueError(msg)

    density = np.clip(density, 0.0, None)
    density_sum = float(density.sum() * grid.dx)
    if not np.isclose(density_sum, 1.0, atol=1e-12):
        density /= density_sum

    return density


def load_default_model() -> HFTParams:
    """
    Return the default configuration used across examples.

    Returns
    -------
    HFTParams
        Default parameter set for the high-frequency trading scenario.
    """

    return HFTParams()


def eta_from_m_alpha(m: np.ndarray, alpha: np.ndarray, params: HFTParams) -> float:
    """
    Compute the execution cost coefficient eta(m) = eta0 + eta1 * |mean_alpha|.
    """

    if m.shape != alpha.shape:
        msg = "Density and control arrays must have matching shapes."
        raise ValueError(msg)

    # Local import to avoid circular dependency.
    from mfg_finance.hamiltonian import mean_alpha_from

    mean_ctrl = mean_alpha_from(m, alpha, 1.0)
    eta = params.eta0 + params.eta1 * abs(mean_ctrl)
    if eta <= 0.0:
        msg = "Computed eta is non-positive."
        raise ValueError(msg)
    return float(eta)
