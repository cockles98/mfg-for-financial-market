"""
Utilities for validating and calibrating the MFG solver against market data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .grid import Grid1D
from .models.hft import HFTParams

__all__ = [
    "MarketTargets",
    "SimulationMetrics",
    "estimate_market_targets",
    "compute_simulation_metrics",
    "metric_gaps",
    "relative_metric_gaps",
    "adjust_parameters",
]


@dataclass(slots=True)
class MarketTargets:
    """
    Container for the empirical metrics we want to match.
    """

    intraday_vol: float
    flow_return_corr: float
    inventory_std: float
    inventory_scale: float = 1.0
    inventory_scale_shares: float = 1.0
    inventory_std_raw: float | None = None
    flow_corr_scale: float = 0.05
    price_scale: float = 1.0
    intraday_vol_raw: float = 0.0


@dataclass(slots=True)
class SimulationMetrics:
    """
    Container for the simulated metrics extracted from the solver outputs.
    """

    price_vol: float
    flow_return_corr: float
    inventory_std: float
    inventory_std_path: float = 0.0
    inventory_mean: float = 0.0
    price_vol_raw: float = 0.0
    price_scale: float = 1.0


def _safe_std(values: pd.Series | np.ndarray) -> float:
    """
    Return the standard deviation ignoring NaN values.
    """

    if isinstance(values, np.ndarray):
        values = pd.Series(values.ravel())
    cleaned = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return 0.0
    return float(cleaned.std())


def estimate_market_targets(
    data_dir: Path | str, min_obs: int = 250
) -> MarketTargets:
    """
    Estimate the empirical targets from the processed COTAHIST tables.
    """

    base = Path(data_dir)
    returns_path = base / "cotahist_equities_returns.parquet"
    extended_path = base / "cotahist_equities_extended.parquet"

    if not returns_path.exists():
        msg = f"Missing processed returns table at {returns_path}"
        raise FileNotFoundError(msg)
    if not extended_path.exists():
        msg = f"Missing extended equities table at {extended_path}"
        raise FileNotFoundError(msg)

    returns = pd.read_parquet(
        returns_path,
        columns=["date", "asset", "simple_return", "roll_vol_20", "close"],
    )
    volumes = pd.read_parquet(
        extended_path,
        columns=["date", "asset", "volume_shares"],
    )

    merged = returns.merge(volumes, on=["date", "asset"], how="inner")
    merged = merged.dropna(subset=["simple_return", "volume_shares"])
    if merged.empty:
        msg = "Joined table is empty; check the processed datasets."
        raise ValueError(msg)

    merged = merged.sort_values(["asset", "date"])
    merged["flow_proxy"] = merged["volume_shares"].astype(float) * np.sign(
        merged["simple_return"].astype(float)
    )
    merged["inventory_proxy"] = (
        merged.groupby("asset", sort=False)["flow_proxy"].cumsum()
    )

    # Filter out very short histories to avoid noisy statistics.
    valid_assets = (
        merged.groupby("asset")["simple_return"].count().loc[lambda s: s >= min_obs]
    )
    merged = merged[merged["asset"].isin(valid_assets.index)]

    vol_series = returns["roll_vol_20"].dropna()
    intraday_vol_raw = float(vol_series.median()) if not vol_series.empty else _safe_std(
        merged["simple_return"]
    )
    price_scale = float(returns["close"].median())
    if not np.isfinite(price_scale) or price_scale <= 0.0:
        price_scale = 1.0
    intraday_vol = intraday_vol_raw / max(price_scale, 1e-6)

    inventory_std_series = (
        merged.groupby("asset")["inventory_proxy"].std().dropna()
    )
    inventory_std = (
        float(inventory_std_series.median())
        if not inventory_std_series.empty
        else _safe_std(merged["inventory_proxy"])
    )

    inventory_std_raw = inventory_std
    inventory_scale_shares = float(
        merged.groupby("asset")["volume_shares"]
        .median()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .median()
    )
    if not np.isfinite(inventory_scale_shares) or inventory_scale_shares <= 0.0:
        inventory_scale_shares = 1.0

    inventory_scale = max(inventory_std_raw, inventory_scale_shares, 1.0)
    inventory_std = inventory_std_raw / inventory_scale

    flow = merged["flow_proxy"].to_numpy()
    rets = merged["simple_return"].to_numpy()
    flow_return_corr = 0.0
    if flow.size and np.std(flow) > 0.0 and np.std(rets) > 0.0:
        flow_return_corr = float(np.corrcoef(flow, rets)[0, 1])

    flow_corr_scale = max(abs(flow_return_corr), 0.01)

    return MarketTargets(
        intraday_vol=intraday_vol,
        flow_return_corr=flow_return_corr,
        inventory_std=inventory_std,
        inventory_scale=inventory_scale,
        inventory_scale_shares=inventory_scale_shares,
        inventory_std_raw=inventory_std_raw,
        flow_corr_scale=flow_corr_scale,
        price_scale=price_scale,
        intraday_vol_raw=intraday_vol_raw,
    )


def compute_simulation_metrics(
    densities: np.ndarray,
    controls: np.ndarray,
    grid: Grid1D,
    prices: np.ndarray | None = None,
) -> SimulationMetrics:
    """
    Compute the simulated counterparts of the empirical targets.
    """

    densities = np.asarray(densities, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)

    if densities.shape != controls.shape:
        msg = "Density and control arrays must have matching shapes."
        raise ValueError(msg)

    weights = grid.x.astype(np.float64)
    mean_inventory = np.sum(densities * weights, axis=1) * grid.dx
    mean_flow = np.sum(controls * densities, axis=1) * grid.dx

    second_moment = np.sum(densities * (weights**2), axis=1) * grid.dx
    cross_sectional_var = np.clip(second_moment - mean_inventory**2, 0.0, None)
    inventory_std = float(np.mean(np.sqrt(cross_sectional_var)))
    inventory_std_path = float(np.std(mean_inventory))
    inventory_mean = float(np.mean(mean_inventory))

    price_vol_raw = 0.0
    price_vol = 0.0
    flow_return_corr = 0.0
    price_scale = 1.0

    if prices is not None:
        prices = np.asarray(prices, dtype=np.float64)
        if prices.shape[0] != densities.shape[0]:
            msg = "Price path must match the temporal dimension of the densities."
            raise ValueError(msg)
        returns = np.diff(prices, prepend=prices[0])
        price_scale = float(np.max(np.abs(prices)))
        price_scale = max(price_scale, 1e-6)
        norm_returns = returns / price_scale
        price_vol_raw = float(np.std(returns))
        price_vol = float(np.std(norm_returns))
        if np.std(norm_returns) > 0.0 and np.std(mean_flow) > 0.0:
            flow_return_corr = float(np.corrcoef(mean_flow, norm_returns)[0, 1])
    else:
        price_vol = float(np.std(mean_flow))
        price_vol_raw = price_vol

    return SimulationMetrics(
        price_vol=price_vol,
        flow_return_corr=flow_return_corr,
        inventory_std=inventory_std,
        inventory_std_path=inventory_std_path,
        inventory_mean=inventory_mean,
        price_vol_raw=price_vol_raw,
        price_scale=price_scale,
    )


def metric_gaps(targets: MarketTargets, simulation: SimulationMetrics) -> dict[str, float]:
    """
    Return absolute gaps between simulated metrics and the empirical targets.
    """

    return {
        "intraday_vol": simulation.price_vol - targets.intraday_vol,
        "flow_return_corr": simulation.flow_return_corr - targets.flow_return_corr,
        "inventory_std": simulation.inventory_std - targets.inventory_std,
    }


def relative_metric_gaps(
    targets: MarketTargets,
    simulation: SimulationMetrics,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Return relative gaps to help with convergence diagnostics.
    """

    rel_errors = {}
    rel_errors["intraday_vol"] = (
        simulation.price_vol - targets.intraday_vol
    ) / max(abs(targets.intraday_vol), eps)
    rel_errors["flow_return_corr"] = (
        simulation.flow_return_corr - targets.flow_return_corr
    ) / max(targets.flow_corr_scale, 0.01)
    rel_errors["inventory_std"] = (
        simulation.inventory_std - targets.inventory_std
    ) / max(abs(targets.inventory_std), 1.0)
    return rel_errors


def adjust_parameters(
    params: HFTParams,
    targets: MarketTargets,
    simulation: SimulationMetrics,
    *,
    learning_rate: float = 0.2,
) -> HFTParams:
    """
    Produce a new parameter set nudged towards the empirical targets.
    """

    updated = replace(params)

    if targets.intraday_vol > 0.0:
        vol_ratio = (simulation.price_vol - targets.intraday_vol) / max(
            targets.intraday_vol, 1e-3
        )
        updated.nu = float(
            np.clip(updated.nu * (1.0 + learning_rate * vol_ratio), 5e-4, 1.0)
        )

    if targets.inventory_std > 0.0:
        inv_error = simulation.inventory_std - targets.inventory_std
        inv_ratio = inv_error / max(targets.inventory_std, 1.0)
        gamma_adjust = 1.0 + 1.5 * learning_rate * inv_ratio
        phi_adjust = 1.0 + learning_rate * inv_ratio
        updated.gamma_T = float(
            np.clip(updated.gamma_T * gamma_adjust, 1e-4, 12.0)
        )
        updated.phi = float(np.clip(updated.phi * phi_adjust, 1e-4, 5.0))
        updated.nu = float(
            np.clip(updated.nu * (1.0 - 0.3 * learning_rate * inv_ratio), 5e-4, 1.0)
        )
        penalty = np.clip(inv_ratio, -5.0, 5.0)
        updated.gamma_T += 0.5 * penalty

    flow_ref = max(targets.flow_corr_scale, 0.01)
    corr_error = simulation.flow_return_corr - targets.flow_return_corr
    corr_ratio = corr_error / flow_ref
    updated.eta1 = float(
        np.clip(updated.eta1 * (1.0 + 3.0 * learning_rate * corr_ratio), 1e-6, 20.0)
    )

    eta1_cap = 0.05
    updated.eta1 = min(updated.eta1, eta1_cap)

    if hasattr(simulation, "inventory_mean"):
        mean_error = float(simulation.inventory_mean)
        if abs(mean_error) > 1e-3:
            updated.m0_mean = float(
                np.clip(updated.m0_mean - learning_rate * mean_error, -5.0, 5.0)
            )

    return updated
