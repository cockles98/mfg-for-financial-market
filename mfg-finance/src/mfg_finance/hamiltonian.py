"""
Hamiltonian definitions for the Hamilton-Jacobi-Bellman equation.

The module stores reusable Hamiltonian parameterisations that can be plugged
into the solver without modifying numerical routines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Callable

import numpy as np

from .models.hft import HFTParams

__all__ = [
    "Hamiltonian",
    "ControlLaw",
    "QuadraticHamiltonian",
    "EtaCallback",
    "mean_alpha_from",
    "effective_eta",
    "running_cost",
    "alpha_star",
    "hamiltonian_value",
]

EtaCallback = Callable[[np.ndarray, np.ndarray, HFTParams], float]


def mean_alpha_from(m: np.ndarray, alpha: np.ndarray, dx: float, atol: float = 1e-12) -> float:
    """
    Return the population-averaged control weighted by the density.

    Parameters
    ----------
    m :
        Density evaluated on the spatial grid.
    alpha :
        Control evaluated on the same spatial grid.
    dx :
        Spatial step size.
    atol :
        Tolerance safeguarding division by near-zero mass.

    Returns
    -------
    float
        Weighted average of the control.
    """

    if m.shape != alpha.shape:
        msg = "Density and control arrays must share the same shape."
        raise ValueError(msg)
    if dx <= 0.0:
        msg = "dx must be strictly positive."
        raise ValueError(msg)

    mass = float(np.sum(m) * dx)
    if mass <= atol:
        msg = "Total mass is too small to compute a meaningful mean control."
        raise ValueError(msg)

    return float(np.sum(alpha * m) * dx / mass)


def effective_eta(mean_alpha: float, params: HFTParams) -> float:
    """
    Compute the execution cost coefficient ``eta(m)``.

    Parameters
    ----------
    mean_alpha :
        Aggregate control mean, typically produced by :func:`mean_alpha_from`.
    params :
        Model parameters defining ``eta0`` and ``eta1``.

    Returns
    -------
    float
        Positive control cost coefficient.
    """

    eta = params.eta0 + params.eta1 * abs(mean_alpha)
    if eta <= 0.0:
        msg = "Effective eta must remain positive."
        raise ValueError(msg)
    return float(eta)


def running_cost(
    x: np.ndarray,
    alpha: np.ndarray,
    params: HFTParams,
    *,
    mean_alpha: float | None = None,
    density: np.ndarray | None = None,
    eta_value: float | None = None,
    eta_callback: EtaCallback | None = None,
) -> np.ndarray:
    """
    Evaluate the instantaneous running cost ``L(x, alpha, m, params)``.

    Parameters
    ----------
    x :
        Spatial grid points.
    alpha :
        Control evaluated at the same locations.
    params :
        Model parameters.
    mean_alpha :
        Optional aggregate control mean. Defaults to zero when omitted.

    Returns
    -------
    numpy.ndarray
        Running cost sampled on the grid.
    """

    if x.shape != alpha.shape:
        msg = "State and control arrays must share the same shape."
        raise ValueError(msg)

    if eta_value is not None:
        eta = eta_value
    elif eta_callback is not None and density is not None:
        eta = eta_callback(density, alpha, params)
    else:
        eta = effective_eta(mean_alpha if mean_alpha is not None else 0.0, params)
    if eta <= 0.0:
        msg = "Running cost requires a positive eta."
        raise ValueError(msg)
    return 0.5 * eta * (alpha**2) + params.phi * (x**2)


def alpha_star(
    momentum: np.ndarray,
    density: np.ndarray,
    params: HFTParams,
    *,
    mean_alpha: float | None = None,
    eta_callback: EtaCallback | None = None,
    iterations: int = 3,
) -> np.ndarray:
    """
    Compute the closed-form optimal control ``alpha_star``.

    Parameters
    ----------
    momentum :
        Spatial gradient of the value function.
    density :
        Density sampled on the spatial grid (included for interface symmetry).
    params :
        Model parameters.
    mean_alpha :
        Optional aggregate control mean. Defaults to zero when omitted.

    Returns
    -------
    numpy.ndarray
        Optimal control evaluated at each grid node.
    """

    if momentum.shape != density.shape:
        msg = "Momentum and density arrays must share the same shape."
        raise ValueError(msg)

    eta = effective_eta(mean_alpha if mean_alpha is not None else 0.0, params)
    alpha = -momentum / eta

    if eta_callback is None:
        return alpha

    for _ in range(max(iterations, 1)):
        eta = eta_callback(density, alpha, params)
        if eta <= 0.0:
            eta = params.eta0
        alpha = -momentum / eta

    return alpha


def hamiltonian_value(
    momentum: np.ndarray,
    x: np.ndarray,
    density: np.ndarray,
    params: HFTParams,
    *,
    mean_alpha: float | None = None,
    eta_value: float | None = None,
    eta_callback: EtaCallback | None = None,
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    """
    Evaluate the Hamiltonian ``H(p, m, params)`` in closed form.

    Parameters
    ----------
    momentum :
        Spatial gradient of the value function.
    x :
        Spatial grid points.
    density :
        Density sampled on the spatial grid (included for interface symmetry).
    params :
        Model parameters.
    mean_alpha :
        Optional aggregate control mean. Defaults to zero when omitted.

    Returns
    -------
    numpy.ndarray
        Hamiltonian values at each grid point.
    """

    if momentum.shape != x.shape or momentum.shape != density.shape:
        msg = "Momentum, state, and density arrays must share the same shape."
        raise ValueError(msg)

    if eta_value is not None:
        eta = eta_value
    elif eta_callback is not None and alpha is not None:
        eta = eta_callback(density, alpha, params)
    else:
        eta = effective_eta(mean_alpha if mean_alpha is not None else 0.0, params)
    if eta <= 0.0:
        msg = "Hamiltonian requires a positive eta."
        raise ValueError(msg)
    return 0.5 * (momentum**2) / eta + params.phi * (x**2)


class ControlLaw(Protocol):
    """
    Protocol describing control laws derived from Hamiltonians.
    """

    def __call__(self, momentum: np.ndarray, density: np.ndarray | None = None) -> np.ndarray:
        """
        Evaluate the optimal control.

        Parameters
        ----------
        momentum :
            Spatial gradient of the value function.
        density :
            Current agent density, optional for mean-field coupling.

        Returns
        -------
        numpy.ndarray
            Optimal control evaluated point-wise.
        """


class Hamiltonian(ABC):
    """
    Base class for Hamiltonians used in the HJB equation.
    """

    @abstractmethod
    def value(self, momentum: np.ndarray, density: np.ndarray | None = None) -> np.ndarray:
        """
        Evaluate the Hamiltonian.

        Parameters
        ----------
        momentum :
            Spatial gradient of the value function.
        density :
            Current agent density used for mean-field interactions.

        Returns
        -------
        numpy.ndarray
            Hamiltonian value evaluated point-wise on the grid.
        """

    @abstractmethod
    def optimal_control(self) -> ControlLaw:
        """
        Return the optimal control associated with the Hamiltonian.

        Returns
        -------
        ControlLaw
            Callable that maps momentum and optional density to the optimal
            control.
        """

    def flux_bound(self, magnitude: np.ndarray) -> float:
        """
        Estimate a Lax-Friedrichs dissipation bound.

        Parameters
        ----------
        magnitude :
            Array containing absolute values of the solution gradient.

        Returns
        -------
        float
            Upper bound for the numerical dissipation coefficient.
        """

        return float(np.max(magnitude))


@dataclass(slots=True)
class QuadraticHamiltonian(Hamiltonian):
    """
    Quadratic Hamiltonian tailored to the LQ high-frequency trading setup.

    Parameters
    ----------
    x :
        Spatial grid points used to evaluate the running cost component.
    params :
        High-frequency trading parameter set.
    mean_alpha :
        Current estimate of the population-average control.
    """

    x: np.ndarray = field(repr=False)
    params: HFTParams
    eta_callback: EtaCallback | None = None
    mean_alpha: float = 0.0
    last_alpha: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)

    def update_mean_alpha(self, density: np.ndarray, control: np.ndarray, dx: float) -> float:
        """
        Update and store the population-average control.

        Parameters
        ----------
        density :
            Density evaluated on the spatial grid.
        control :
            Control evaluated on the same grid.
        dx :
            Spatial step size.

        Returns
        -------
        float
            Updated mean control.
        """

        self.mean_alpha = mean_alpha_from(density, control, dx)
        return self.mean_alpha

    def value(self, momentum: np.ndarray, density: np.ndarray | None = None) -> np.ndarray:
        if momentum.shape != self.x.shape:
            msg = "Momentum array must match the spatial grid shape."
            raise ValueError(msg)
        if density is not None and density.shape != momentum.shape:
            msg = "Density array must match the momentum shape."
            raise ValueError(msg)
        eta_value = None
        if self.last_alpha is not None and self.eta_callback is not None and density is not None:
            eta_value = self.eta_callback(density, self.last_alpha, self.params)
        return hamiltonian_value(
            momentum,
            self.x,
            density if density is not None else np.zeros_like(momentum),
            self.params,
            mean_alpha=self.mean_alpha,
            eta_callback=self.eta_callback,
            alpha=self.last_alpha,
            eta_value=eta_value,
        )

    def optimal_control(self) -> ControlLaw:
        def _law(momentum: np.ndarray, density: np.ndarray | None = None) -> np.ndarray:
            if momentum.shape != self.x.shape:
                msg = "Momentum array must match the spatial grid shape."
                raise ValueError(msg)
            if density is not None and density.shape != momentum.shape:
                msg = "Density array must match the momentum shape."
                raise ValueError(msg)
            effective_density = density if density is not None else np.zeros_like(momentum)
            alpha = alpha_star(
                momentum,
                effective_density,
                self.params,
                mean_alpha=self.mean_alpha,
                eta_callback=self.eta_callback,
            )
            self.last_alpha = alpha
            return alpha

        return _law
