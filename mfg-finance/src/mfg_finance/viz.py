"""
Visualisation helpers for Mean Field Game simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from .grid import Grid1D

__all__ = [
    "PlotConfig",
    "plot_value_function",
    "plot_density",
    "plot_density_time",
    "plot_value_time",
    "plot_alpha_cuts",
    "plot_convergence",
    "plot_price",
]


@dataclass(slots=True)
class PlotConfig:
    """
    Configuration parameters for plotting.

    Parameters
    ----------
    figsize :
        Tuple storing figure width and height.
    cmap :
        Matplotlib colormap name.
    """

    figsize: tuple[float, float] = (10.0, 4.0)
    cmap: str = "viridis"


def plot_value_function(time: Iterable[float], grid: Iterable[float], values: np.ndarray, cfg: PlotConfig | None = None) -> plt.Figure:
    """
    Plot the value function as an image.

    Parameters
    ----------
    time :
        Temporal grid.
    grid :
        Spatial grid.
    values :
        Value function trajectory shaped `(len(time), len(grid))`.
    cfg :
        Plotting configuration.

    Returns
    -------
    matplotlib.figure.Figure
        Figure handle for further customisation.
    """

    cfg = cfg or PlotConfig()
    fig, ax = plt.subplots(figsize=cfg.figsize)
    im = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        extent=(min(grid), max(grid), min(time), max(time)),
        cmap=cfg.cmap,
    )
    ax.set_xlabel("state")
    ax.set_ylabel("time")
    ax.set_title("Value function")
    fig.colorbar(im, ax=ax)
    return fig


def plot_density(time: Iterable[float], grid: Iterable[float], density: np.ndarray, cfg: PlotConfig | None = None) -> plt.Figure:
    """
    Plot the density evolution as an image.

    Parameters
    ----------
    time :
        Temporal grid.
    grid :
        Spatial grid.
    density :
        Density trajectory shaped `(len(time), len(grid))`.
    cfg :
        Plotting configuration.

    Returns
    -------
    matplotlib.figure.Figure
        Figure handle for further customisation.
    """

    cfg = cfg or PlotConfig(cmap="magma")
    fig, ax = plt.subplots(figsize=cfg.figsize)
    im = ax.imshow(
        density,
        aspect="auto",
        origin="lower",
        extent=(min(grid), max(grid), min(time), max(time)),
        cmap=cfg.cmap,
    )
    ax.set_xlabel("state")
    ax.set_ylabel("time")
    ax.set_title("Density")
    fig.colorbar(im, ax=ax)
    return fig


def _prepare_path(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _heatmap(
    data: np.ndarray,
    extent: tuple[float, float, float, float],
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    figsize: tuple[float, float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    return fig


def plot_density_time(M_all: np.ndarray, grid: Grid1D, path: pathlib.Path | str, cfg: PlotConfig | None = None) -> None:
    """
    Save a density heatmap over time and space.
    """

    cfg = cfg or PlotConfig(cmap="magma")
    extent = (grid.t[0], grid.t[-1], grid.x[0], grid.x[-1])
    fig = _heatmap(
        M_all.T,
        extent,
        "Density evolution",
        "time",
        "state",
        cfg.cmap,
        cfg.figsize,
    )
    fig.savefig(_prepare_path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_value_time(U_all: np.ndarray, grid: Grid1D, path: pathlib.Path | str, cfg: PlotConfig | None = None) -> None:
    """
    Save a value function heatmap over time and space.
    """

    cfg = cfg or PlotConfig()
    extent = (grid.t[0], grid.t[-1], grid.x[0], grid.x[-1])
    fig = _heatmap(
        U_all.T,
        extent,
        "Value function evolution",
        "time",
        "state",
        cfg.cmap,
        cfg.figsize,
    )
    fig.savefig(_prepare_path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_cuts(
    alpha_all: np.ndarray,
    grid: Grid1D,
    times: Sequence[float],
    path: pathlib.Path | str,
    cfg: PlotConfig | None = None,
) -> None:
    """
    Save selected temporal cuts of the control trajectory.
    """

    cfg = cfg or PlotConfig(figsize=(10.0, 5.0))
    fig, ax = plt.subplots(figsize=cfg.figsize)

    times = list(times)
    for target in times:
        idx = int(np.clip(np.searchsorted(grid.t, target), 0, len(grid.t) - 1))
        ax.plot(
            grid.x,
            alpha_all[idx],
            label=f"t={grid.t[idx]:.3f}",
        )

    ax.set_xlabel("state")
    ax.set_ylabel("alpha")
    ax.set_title("Control cuts")
    if times:
        ax.legend()
    fig.savefig(_prepare_path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(errors: Sequence[float], path: pathlib.Path | str, figsize: tuple[float, float] = (8.0, 4.0)) -> None:
    """
    Save the convergence curve of the Picard iteration.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(len(errors)), errors, marker="o", linestyle="-")
    ax.set_xlabel("iteration")
    ax.set_ylabel("||delta M||_2")
    ax.set_title("Picard convergence")
    ax.set_yscale("log")
    fig.savefig(_prepare_path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_price(time: Sequence[float], price: Sequence[float], path: pathlib.Path | str, figsize: tuple[float, float] = (8.0, 4.0)) -> None:
    """
    Save the price trajectory.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, price, marker="o")
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.set_title("Endogenous price")
    fig.savefig(_prepare_path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
