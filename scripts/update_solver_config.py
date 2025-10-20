#!/usr/bin/env python
"""
Update ``configs/baseline.yaml`` with a supply schedule derived from
``data/processed/supply_curve.csv``.

The mapping mirrors the logic used during development: volumes are normalised in
log-space, recentered, interpolated to the solver grid and rescaled by
``--scale`` before being written back to the configuration file alongside the
requested price-sensitivity hyper-parameter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def build_schedule(
    df: pd.DataFrame,
    *,
    scale: float,
    samples: int,
) -> list[float]:
    """
    Convert the coarse summary (quantile → volume) into a dense schedule that
    matches the solver grid.
    """

    if samples < 3:
        raise ValueError("At least three samples are required to build the schedule.")

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["quantile", "volume_shares"])
    if df.empty:
        raise ValueError("Supply curve table is empty after cleaning.")

    quantiles = df["quantile"].to_numpy(dtype=float)
    if (quantiles <= 0).any() or (quantiles >= 1).any():
        raise ValueError("Quantile values must live strictly inside (0, 1).")
    if not np.all(np.diff(quantiles) > 0):
        raise ValueError("Quantiles must be strictly increasing.")

    volumes = np.log1p(df["volume_shares"].to_numpy(dtype=float))
    volumes -= np.mean(volumes)
    max_abs = np.max(np.abs(volumes))
    if max_abs == 0.0:
        raise ValueError("Volume distribution is degenerate (all values identical).")
    volumes /= max_abs

    grid = np.linspace(0.0, 1.0, samples)
    schedule = np.interp(grid, quantiles, volumes)
    schedule[0] = 0.0
    schedule[-1] = 0.0
    schedule -= schedule.mean()
    schedule *= scale
    return schedule.tolist()


def update_config(
    config_path: Path,
    *,
    supply_schedule: list[float],
    price_sensitivity: float,
    price_bracket: tuple[float, float],
    scale: float,
) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    solver_cfg = cfg.setdefault("solver", {})
    solver_cfg["supply"] = supply_schedule
    solver_cfg["price_sensitivity"] = float(price_sensitivity)
    solver_cfg["price_bracket"] = list(price_bracket)
    solver_cfg["compute_price"] = True
    solver_cfg["supply_scale"] = float(scale)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the solver configuration using the empirical supply curve."
    )
    parser.add_argument(
        "--supply",
        type=Path,
        default=Path("data/processed/supply_curve.csv"),
        help="Path to the CSV created by scripts/build_supply_curve.py.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Configuration file to update.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0e-05,
        help="Amplitude applied to the normalised supply curve.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=151,
        help="Number of temporal samples (should match grid.nt + 1).",
    )
    parser.add_argument(
        "--price-sensitivity",
        type=float,
        default=30.0,
        help="Clearing sensitivity hyper-parameter.",
    )
    parser.add_argument(
        "--price-bracket",
        type=float,
        nargs=2,
        default=(-1.0, 1.0),
        metavar=("LOWER", "UPPER"),
        help="Initial bracket used by the root finder in the price solver.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.supply.exists():
        raise SystemExit(f"Supply curve file not found: {args.supply}")

    df = pd.read_csv(args.supply)
    schedule = build_schedule(df, scale=args.scale, samples=args.samples)
    update_config(
        args.config,
        supply_schedule=schedule,
        price_sensitivity=args.price_sensitivity,
        price_bracket=tuple(args.price_bracket),
        scale=args.scale,
    )
    print(
        f"Updated {args.config} with {len(schedule)} supply points, "
        f"scale={args.scale:g} and price_sensitivity={args.price_sensitivity:g}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
