#!/usr/bin/env python
"""
Estimate execution cost and inventory risk parameters from historical series.

The script ingests the processed COTAHIST tables, runs simple regressions
over relative spreads, signed flows and cumulative inventory proxies, and
derives heuristic values for (eta0, eta1, phi, gamma_T). The resulting
parameters can optionally be written back to a YAML config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from mfg_finance.validation import estimate_market_targets


def _safe_regression(x: pd.Series, y: pd.Series) -> float:
    """
    Ordinary least squares slope between x and y with basic filtering.
    """

    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    mask = ~(x.isna() | y.isna())
    mask &= np.isfinite(x) & np.isfinite(y)
    mask &= x != 0.0
    x = x.loc[mask]
    y = y.loc[mask]
    if x.empty:
        return 0.0
    X = x.to_numpy().reshape(-1, 1)
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    return float(beta[0])


def _compute_signed_flow(merged: pd.DataFrame) -> pd.Series:
    returns = merged["simple_return"].fillna(0.0).astype(float)
    volumes = merged["volume_shares"].fillna(0.0).astype(float)
    return volumes * np.sign(returns)


def _estimate_parameters(data_dir: Path) -> Dict[str, float]:
    returns = pd.read_parquet(
        data_dir / "cotahist_equities_returns.parquet",
        columns=["date", "asset", "simple_return", "close", "roll_vol_20"],
    )
    extended = pd.read_parquet(
        data_dir / "cotahist_equities_extended.parquet",
        columns=["date", "asset", "volume_shares"],
    )
    merged = returns.merge(extended, on=["date", "asset"], how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["simple_return", "close", "volume_shares"]
    )

    targets = estimate_market_targets(data_dir)
    price_scale = max(targets.price_scale, 1.0)

    supply_path = data_dir / "supply_curve.csv"
    eta0 = 1e-4
    if supply_path.exists():
        supply = pd.read_csv(supply_path)
        rel_spread = supply.get("spread_rel")
        if rel_spread is not None and rel_spread.notna().any():
            eta0 = float(np.clip(rel_spread.median() * 0.5, 1e-4, 0.25))

    signed_flow = _compute_signed_flow(merged)
    signed_flow = signed_flow.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    flow_scale = float(np.nanmedian(np.abs(signed_flow)))
    if not np.isfinite(flow_scale) or flow_scale <= 0.0:
        flow_scale = float(np.nanstd(signed_flow)) or 1.0
    norm_flow = signed_flow / flow_scale
    norm_returns = merged["simple_return"].fillna(0.0) / max(price_scale, 1e-6)
    impact_beta = _safe_regression(norm_flow, norm_returns)
    impact_scaled = abs(impact_beta) * (flow_scale / max(targets.inventory_scale_shares, 1.0))
    corr_based = abs(targets.flow_return_corr) * 10.0
    eta1 = float(np.clip(max(impact_scaled, corr_based), 1e-4, 1.0))

    merged = merged.assign(signed_flow=signed_flow)
    merged["inventory_proxy"] = merged.sort_values(["asset", "date"]).groupby("asset")[
        "signed_flow"
    ].cumsum()
    inv_scale = float(np.median(np.abs(merged["inventory_proxy"]))) or 1.0
    inv_norm = merged["inventory_proxy"] / inv_scale
    returns_abs = merged["simple_return"].abs()
    phi_beta = _safe_regression(inv_norm.abs(), returns_abs / max(price_scale, 1e-6))
    phi_empirical = float(np.clip(phi_beta, 0.0, 1.0))
    phi_target = float(
        np.clip(targets.intraday_vol / max(targets.inventory_std, 1e-3), 1e-4, 0.5)
    )
    phi = phi_empirical if phi_empirical > 1e-4 else phi_target

    gamma_T = float(np.clip(phi * 5.0, 0.1, 10.0))

    return {
        "eta0": eta0,
        "eta1": eta1,
        "phi": phi,
        "gamma_T": gamma_T,
        "price_scale": price_scale,
        "flow_scale": flow_scale,
        "inventory_scale_shares": targets.inventory_scale_shares,
    }


def _update_config(config_path: Path, params: Dict[str, float]) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg.setdefault("params", {})
    for key in ("eta0", "eta1", "phi", "gamma_T"):
        cfg["params"][key] = float(params[key])
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate execution cost and risk parameters from historical series."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing the processed parquet tables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cost_risk_params.json"),
        help="Where the estimated parameters will be stored as JSON.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Optional YAML config to update with the new parameters.",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Persist eta0/eta1/phi/gamma_T back to the provided config.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    params = _estimate_parameters(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Estimated parameters -> {json.dumps(params, indent=2)}")
    if args.update_config:
        _update_config(args.config, params)
        print(f"Updated {args.config} with eta0/eta1/phi/gamma_T.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
