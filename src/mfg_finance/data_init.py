from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .grid import Grid1D

__all__ = ["empirical_initial_density"]

_MODULE_ROOT = Path(__file__).resolve().parents[2]


def _resolve_data_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        _MODULE_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def empirical_initial_density(
    grid: Grid1D,
    *,
    returns_path: str | Path = "data/processed/cotahist_equities_returns.parquet",
    volumes_path: str | Path = "data/processed/cotahist_equities_extended.parquet",
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    liquidity_bucket: str = "all",
    bucket_quantiles: tuple[float, float] | list[float] = (0.33, 0.66),
    spread_scale: float = 1.0,
) -> np.ndarray:
    returns_path = _resolve_data_path(Path(returns_path))
    volumes_path = _resolve_data_path(Path(volumes_path))
    if not returns_path.exists() or not volumes_path.exists():
        raise FileNotFoundError("Processed data missing for empirical density.")

    returns = pd.read_parquet(
        returns_path,
        columns=["date", "asset", "simple_return"],
    )
    volumes = pd.read_parquet(
        volumes_path,
        columns=["date", "asset", "volume_shares"],
    )
    merged = returns.merge(volumes, on=["date", "asset"], how="inner")
    merged = merged.sort_values(["asset", "date"])
    merged["signed_volume"] = merged["volume_shares"] * np.sign(
        merged["simple_return"].fillna(0.0)
    )
    merged["inventory_proxy"] = merged.groupby("asset")["signed_volume"].cumsum()
    liquidity_bucket = str(liquidity_bucket or "all").lower()
    vol_stats = (
        volumes.groupby("asset")["volume_shares"].mean().rename("avg_volume")
    )
    merged = merged.merge(vol_stats, on="asset", how="left")
    bucket_mask: pd.Series | None = None
    if (
        liquidity_bucket in {"low", "mid", "high"}
        and not vol_stats.empty
        and vol_stats.notna().any()
    ):
        if isinstance(bucket_quantiles, (list, tuple)) and len(bucket_quantiles) == 2:
            q_low, q_high = sorted(bucket_quantiles)
        else:
            q_low, q_high = (0.33, 0.66)
        q_low = float(np.clip(q_low, 0.0, 1.0))
        q_high = float(np.clip(q_high, 0.0, 1.0))
        if q_high < q_low:
            q_low, q_high = q_high, q_low
        vol_q_low = float(vol_stats.quantile(q_low))
        vol_q_high = float(vol_stats.quantile(q_high))
        if liquidity_bucket == "low":
            bucket_mask = merged["avg_volume"] <= vol_q_low
        elif liquidity_bucket == "mid":
            bucket_mask = (merged["avg_volume"] > vol_q_low) & (
                merged["avg_volume"] < vol_q_high
            )
        elif liquidity_bucket == "high":
            bucket_mask = merged["avg_volume"] >= vol_q_high
        if bucket_mask is not None and bucket_mask.any():
            merged = merged[bucket_mask]
        else:
            bucket_mask = None

    spread_scale = max(float(spread_scale), 1e-3)
    inventory = merged["inventory_proxy"].to_numpy(dtype=np.float64)
    inv = inventory[~np.isnan(inventory)]
    if inv.size == 0:
        raise ValueError("inventory_proxy is empty after filtering.")
    q_low, q_high = np.quantile(inv, [lower_quantile, upper_quantile])
    scale = max(abs(float(q_low)), abs(float(q_high)), 1e-6)
    scale = scale / spread_scale
    normalized = np.clip(inv / scale, -1.0, 1.0)
    mapped = normalized * max(abs(grid.x_min), abs(grid.x_max))

    bins = np.concatenate((grid.x, [grid.x[-1] + grid.dx]))
    hist, _ = np.histogram(mapped, bins=bins, density=False)
    density = hist.astype(np.float64)
    if density.sum() <= 0.0:
        density = np.ones_like(density)
    density = np.clip(density, 0.0, None)
    density = density / (density.sum() * grid.dx)
    return density
