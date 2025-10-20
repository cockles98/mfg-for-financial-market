#!/usr/bin/env python
"""
Derive an empirical supply curve from the processed COTAHIST dataset.

The script reads ``data/processed/cotahist_equities_extended.parquet`` (or a
custom table provided by ``--input``) and computes quantiles of traded volume,
notional volume and relative spread. The result replicates the structure used
by the pipeline (`data/processed/supply_curve.csv`).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class SupplyCurveConfig:
    """Parameters that govern the curve construction."""

    quantiles: Iterable[float] = (0.10, 0.25, 0.50, 0.75, 0.90)
    minimum_observations: int = 10_000


def _validate_quantiles(values: Iterable[float]) -> np.ndarray:
    quantiles = np.asarray(list(values), dtype=float)
    if quantiles.size == 0:
        raise ValueError("At least one quantile must be provided.")
    if np.any((quantiles <= 0.0) | (quantiles >= 1.0)):
        raise ValueError("Quantiles must lie in (0, 1).")
    if not np.all(np.diff(quantiles) > 0):
        raise ValueError("Quantiles must be strictly increasing.")
    return quantiles


def _load_extended_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {path}")
    df = pd.read_parquet(path)
    expected = {"volume_shares", "volume_money", "best_bid", "best_ask"}
    missing = expected - set(df.columns)
    if missing:
        raise KeyError(
            f"Input table is missing required columns: {', '.join(sorted(missing))}"
        )
    return df


def build_supply_curve(
    df: pd.DataFrame,
    *,
    quantiles: Iterable[float],
    min_obs: int,
) -> pd.DataFrame:
    """
    Construct the empirical supply curve.
    """

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["volume_shares", "volume_money", "best_bid", "best_ask"])
    if df.shape[0] < min_obs:
        raise ValueError(
            f"Insufficient observations after cleaning ({df.shape[0]}). "
            f"Need at least {min_obs} rows; check the input dataset."
        )

    quantiles_array = _validate_quantiles(quantiles)

    volume = df["volume_shares"].astype(float).to_numpy()
    dollar_volume = df["volume_money"].astype(float).to_numpy()

    mid_price = np.where(
        np.abs(df["best_bid"].to_numpy() + df["best_ask"].to_numpy()) > 0.0,
        0.5 * (df["best_bid"].to_numpy() + df["best_ask"].to_numpy()),
        np.nan,
    )
    spread = (df["best_ask"].to_numpy() - df["best_bid"].to_numpy()) / mid_price
    spread = np.where(np.isfinite(spread), spread, np.nan)
    spread = spread[~np.isnan(spread)]
    if spread.size == 0:
        raise ValueError("Could not compute relative spreads; check bid/ask columns.")

    result = pd.DataFrame(
        {
            "quantile": quantiles_array,
            "volume_shares": np.quantile(volume, quantiles_array),
            "dollar_volume": np.quantile(dollar_volume, quantiles_array),
            "spread_rel": np.quantile(spread, quantiles_array),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute empirical supply curve statistics from B3 data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/cotahist_equities_extended.parquet"),
        help="Path to the processed parquet file with bid/ask and volume data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/supply_curve.csv"),
        help="Destination CSV file for the supply curve summary.",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50, 0.75, 0.90],
        help="Quantiles (0 < q < 1) used for the curve.",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=SupplyCurveConfig.minimum_observations,
        help="Minimum number of rows required after cleaning.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = SupplyCurveConfig(
        quantiles=args.quantiles,
        minimum_observations=args.min_observations,
    )

    try:
        data = _load_extended_table(args.input)
        supply_curve = build_supply_curve(
            data,
            quantiles=cfg.quantiles,
            min_obs=cfg.minimum_observations,
        )
    except Exception as exc:  # pragma: no cover - user feedback
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    supply_curve.to_csv(args.output, index=False)
    print(f"Supply curve saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
