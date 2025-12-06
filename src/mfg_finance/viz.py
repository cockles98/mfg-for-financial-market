"""
Visualisation helpers for Mean Field Game simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    "plot_speed_heatmap",
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
    inventory_scale :
        Optional multiplicative factor to convert the state grid into real units
        (e.g. number of shares). When provided, axes will be shown in those units.
    inventory_label :
        Optional label for the state axis. Defaults to "inventory" when
        `inventory_scale` is set, otherwise "state".
    time_scale :
        Optional multiplicative factor to convert the time grid into real units
        (e.g. hours). Leave as None to keep the normalised time.
    time_offset :
        Optional offset added after scaling time (e.g. market open hour).
    time_label :
        Optional label for the time axis. Defaults to "time" or "time (scaled)"
        when `time_scale` is set.
    time_tick_suffix :
        Optional suffix appended to tick labels (e.g. "h" for hours).
    """

    figsize: tuple[float, float] = (10.0, 4.0)
    cmap: str = "viridis"
    inventory_scale: float | None = None
    inventory_label: str | None = None
    time_scale: float | None = None
    time_offset: float | None = None
    time_label: str | None = None
    time_tick_suffix: str | None = None


def _scaled_state(grid: Grid1D, cfg: PlotConfig) -> tuple[np.ndarray, str]:
    scale = cfg.inventory_scale
    if scale is None or scale <= 0.0:
        return grid.x, cfg.inventory_label or "state"
    return grid.x * scale, cfg.inventory_label or "inventory"


def _scaled_time(time: np.ndarray, cfg: PlotConfig) -> tuple[np.ndarray, str]:
    scale = cfg.time_scale
    offset = cfg.time_offset or 0.0
    if scale is None or scale <= 0.0:
        return time, cfg.time_label or "time"
    scaled = time * scale + offset
    return scaled, cfg.time_label or "time (scaled)"


def _format_time_axis(ax: plt.Axes, cfg: PlotConfig) -> None:
    if cfg.time_tick_suffix:
        suffix = cfg.time_tick_suffix

        def _fmt(val: float, _pos: int) -> str:
            return f"{val:g}{suffix}"

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))


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
    ax.set_xlabel("invetário")
    ax.set_ylabel("tempo")
    ax.set_title("Densidade")
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
    scaled_t, x_label = _scaled_time(grid.t, cfg)
    scaled_x, y_label = _scaled_state(grid, cfg)
    extent = (scaled_t[0], scaled_t[-1], scaled_x[0], scaled_x[-1])
    fig = _heatmap(
        M_all.T,
        extent,
        "Densidade das posições ao longo do pregão",
        x_label,
        y_label,
        cfg.cmap,
        cfg.figsize,
    )
    _format_time_axis(fig.axes[0], cfg)
    fig.savefig(_prepare_path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_value_time(U_all: np.ndarray, grid: Grid1D, path: pathlib.Path | str, cfg: PlotConfig | None = None) -> None:
    """
    Save a value function heatmap over time and space.
    """

    cfg = cfg or PlotConfig()
    scaled_t, x_label = _scaled_time(grid.t, cfg)
    scaled_x, y_label = _scaled_state(grid, cfg)
    extent = (scaled_t[0], scaled_t[-1], scaled_x[0], scaled_x[-1])
    fig = _heatmap(
        U_all.T,
        extent,
        "Custo futuro por posição ao longo do pregão",
        x_label,
        y_label,
        cfg.cmap,
        cfg.figsize,
    )
    _format_time_axis(fig.axes[0], cfg)
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

    scaled_x, x_label = _scaled_state(grid, cfg)
    scaled_t, _ = _scaled_time(grid.t, cfg)

    times = list(times)
    for target in times:
        idx = int(np.clip(np.searchsorted(grid.t, target), 0, len(grid.t) - 1))
        ax.plot(
            scaled_x,
            alpha_all[idx],
            label=f"t={scaled_t[idx]:.0f}{cfg.time_tick_suffix or ''}",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("velocidade de negociação (alfa)")
    ax.set_title("Velocidade de negociação ao longo do pregão")
    if times:
        ax.legend()
    fig.savefig(_prepare_path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_speed_heatmap(alpha_all: np.ndarray, grid: Grid1D, path: pathlib.Path | str, cfg: PlotConfig | None = None) -> None:
    """
    Save a heatmap of the trading speed (control) over time and inventory.
    """

    cfg = cfg or PlotConfig(cmap="RdBu_r")
    scaled_t, x_label = _scaled_time(grid.t, cfg)
    scaled_x, y_label = _scaled_state(grid, cfg)
    extent = (scaled_t[0], scaled_t[-1], scaled_x[0], scaled_x[-1])
    vmax = float(np.max(np.abs(alpha_all))) if alpha_all.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=cfg.figsize)
    im = ax.imshow(
        alpha_all.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cfg.cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title("Mapa de calor da velocidade de negociação")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _format_time_axis(ax, cfg)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("velocidade (alfa)")
    fig.savefig(_prepare_path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(errors: Sequence[float], path: pathlib.Path | str, figsize: tuple[float, float] = (8.0, 4.0)) -> None:
    """
    Save the convergence curve of the Picard iteration.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(len(errors)), errors, marker="o", linestyle="-")
    ax.set_xlabel("iterações")
    ax.set_ylabel("erro (norma)")
    ax.set_title("Erro ao longo das iterações (convergência de Picard)")
    ax.set_yscale("log")
    fig.savefig(_prepare_path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_price(
    time: Sequence[float],
    price: Sequence[float],
    path: pathlib.Path | str,
    figsize: tuple[float, float] = (8.0, 4.0),
    cfg: PlotConfig | None = None,
) -> None:
    """
    Save the price trajectory.
    """

    cfg = cfg or PlotConfig(figsize=figsize)
    scaled_t, x_label = _scaled_time(np.asarray(time), cfg)
    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.plot(scaled_t, price, marker="o")
    ax.set_xlabel(x_label)
    _format_time_axis(ax, cfg)
    ax.set_ylabel("preço (BRL)")
    ax.set_title("Preço de equilíbrio simulado")
    fig.savefig(_prepare_path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
