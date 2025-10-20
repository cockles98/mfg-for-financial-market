"""
Grid utilities for Mean Field Game discretisations.

The module provides simple one-dimensional spatial and temporal grids that are
compatible with conservative finite-difference schemes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

__all__ = ["Grid1D", "SpatialGrid", "TimeGrid", "linspace_including"]


@dataclass(slots=True)
class Grid1D:
    """
    Joint space-time grid used across the HJB and FP solvers.

    Parameters
    ----------
    x_min :
        Lower spatial bound.
    x_max :
        Upper spatial bound, strictly greater than `x_min`.
    nx :
        Number of spatial nodes (including boundaries), must be >= 2.
    T :
        Final time horizon, must be positive.
    nt :
        Number of time steps. The time grid has `nt + 1` nodes, so `nt >= 1`.
    bc :
        Boundary condition type, either `"neumann"` or `"dirichlet"`.
    """

    x_min: float
    x_max: float
    nx: int
    T: float
    nt: int
    bc: Literal["neumann", "dirichlet"]
    dx: float = field(init=False)
    dt: float = field(init=False)
    x: np.ndarray = field(init=False)
    t: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.nx < 2:
            msg = "Grid1D requires at least two spatial nodes."
            raise ValueError(msg)
        if self.nt < 1:
            msg = "Grid1D requires at least one time step."
            raise ValueError(msg)
        if self.x_max <= self.x_min:
            msg = "x_max must be strictly greater than x_min."
            raise ValueError(msg)
        if self.T <= 0.0:
            msg = "T must be positive."
            raise ValueError(msg)
        if self.bc not in {"neumann", "dirichlet"}:
            msg = "Boundary condition must be 'neumann' or 'dirichlet'."
            raise ValueError(msg)

        self.x = linspace_including(self.x_min, self.x_max, self.nx)
        self.dx = float(self.x[1] - self.x[0])
        self.t = linspace_including(0.0, self.T, self.nt + 1)
        self.dt = float(self.t[1] - self.t[0])

    def check_stability(self, dt_max: float | None = None) -> bool:
        """
        Check a CFL-like stability restriction for the time step.

        Parameters
        ----------
        dt_max :
            Optional maximum allowable time step.

        Returns
        -------
        bool
            True when the current time step satisfies the requested bound.
        """

        if dt_max is None:
            return True
        if dt_max <= 0.0:
            msg = "dt_max must be positive when provided."
            raise ValueError(msg)
        if self.dt <= dt_max:
            return True

        warnings.warn(
            (
                "Time step dt=%1.3e exceeds the requested maximum %1.3e. "
                "This may violate the CFL stability condition."
            )
            % (self.dt, dt_max),
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    def apply_dirichlet(self, values: np.ndarray, left: float, right: float) -> np.ndarray:
        """
        Apply Dirichlet boundary values to a copy of the input vector.

        Parameters
        ----------
        values :
            Input vector of length `nx`.
        left, right :
            Boundary values to impose at the left and right endpoints.

        Returns
        -------
        numpy.ndarray
            Copy of `values` with Dirichlet conditions enforced.
        """

        vector = self._validate_vector(values)
        result = vector.copy()
        result[0] = left
        result[-1] = right
        return result

    def apply_neumann(
        self,
        values: np.ndarray,
        left_gradient: float = 0.0,
        right_gradient: float = 0.0,
    ) -> np.ndarray:
        """
        Apply Neumann boundary conditions (first derivative) to a copy of the vector.

        Parameters
        ----------
        values :
            Input vector of length `nx`.
        left_gradient, right_gradient :
            Desired gradients at each boundary. Defaults impose zero flux.

        Returns
        -------
        numpy.ndarray
            Copy of `values` with Neumann conditions enforced via first-order stencils.
        """

        vector = self._validate_vector(values)
        result = vector.copy()
        result[0] = result[1] - left_gradient * self.dx
        result[-1] = result[-2] + right_gradient * self.dx
        return result

    def apply_boundary(
        self,
        values: np.ndarray,
        left: float = 0.0,
        right: float = 0.0,
    ) -> np.ndarray:
        """
        Apply the configured boundary condition to the provided vector.

        Parameters
        ----------
        values :
            Input vector of length `nx`.
        left, right :
            Boundary specification. For Dirichlet, they are values; for Neumann,
            they represent gradients.

        Returns
        -------
        numpy.ndarray
            Vector with the configured boundary condition enforced.
        """

        if self.bc == "dirichlet":
            return self.apply_dirichlet(values, left, right)
        return self.apply_neumann(values, left, right)

    def _validate_vector(self, values: np.ndarray) -> np.ndarray:
        """
        Validate that the provided vector matches the spatial discretisation.

        Parameters
        ----------
        values :
            Candidate vector.

        Returns
        -------
        numpy.ndarray
            The original array when validation succeeds.
        """

        if values.shape != (self.nx,):
            msg = f"Expected vector of shape ({self.nx},), received {values.shape}."
            raise ValueError(msg)
        return values

    def suggest_cfl_limits(self, nu: float, max_velocity: float | None = None) -> dict[str, float]:
        """
        Provide heuristic CFL limits for diffusion and advection effects.

        Parameters
        ----------
        nu :
            Diffusion coefficient.
        max_velocity :
            Optional estimate for the maximum transport velocity.

        Returns
        -------
        dict
            Dictionary containing `dt` and optional `diffusion_dt` and
            `advection_dt` entries.
        """

        limits: dict[str, float] = {"dt": self.dt}
        if nu > 0.0:
            limits["diffusion_dt"] = self.dx**2 / (2.0 * nu)
        if max_velocity is not None and max_velocity > 0.0:
            limits["advection_dt"] = self.dx / max_velocity
        return limits


@dataclass(slots=True)
class SpatialGrid:
    """
    Uniform one-dimensional spatial grid.

    Parameters
    ----------
    lower :
        Lower spatial bound.
    upper :
        Upper spatial bound.
    num_points :
        Number of grid nodes including endpoints. Must be >= 2.
    """

    lower: float
    upper: float
    num_points: int
    points: np.ndarray = field(init=False)
    dx: float = field(init=False)

    def __post_init__(self) -> None:
        if self.num_points < 2:
            msg = "SpatialGrid requires at least two points."
            raise ValueError(msg)
        if self.upper <= self.lower:
            msg = "Upper bound must be greater than lower bound."
            raise ValueError(msg)

        self.points = linspace_including(self.lower, self.upper, self.num_points)
        self.dx = float(self.points[1] - self.points[0])

    @property
    def interior_slice(self) -> slice:
        """
        Return a slice pointing to interior nodes only.

        Returns
        -------
        slice
            Slice selecting indices `[1:-1]`.
        """

        return slice(1, -1)


@dataclass(slots=True)
class TimeGrid:
    """
    Uniform time grid suitable for explicit or implicit schemes.

    Parameters
    ----------
    start :
        Initial time (t=0).
    end :
        Final time (t=T), must satisfy `end > start`.
    num_steps :
        Number of time steps (intervals). Must be >= 1.
    """

    start: float
    end: float
    num_steps: int
    nodes: np.ndarray = field(init=False)
    dt: float = field(init=False)

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            msg = "TimeGrid requires at least one time step."
            raise ValueError(msg)
        if self.end <= self.start:
            msg = "End time must exceed start time."
            raise ValueError(msg)

        self.nodes = linspace_including(self.start, self.end, self.num_steps + 1)
        self.dt = float(self.nodes[1] - self.nodes[0])


def linspace_including(lower: float, upper: float, num_points: int) -> np.ndarray:
    """
    Return a `numpy.linspace` array guaranteeing floating-point consistency.

    Parameters
    ----------
    lower :
        Lower bound of the interval.
    upper :
        Upper bound of the interval.
    num_points :
        Number of points to generate.

    Returns
    -------
    numpy.ndarray
        Equally spaced vector including both endpoints.
    """

    return np.linspace(lower, upper, num_points, dtype=np.float64)
