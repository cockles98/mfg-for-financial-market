#!/usr/bin/env python
"""
Calibrate the MFG model parameters against empirical market metrics.

The script executes the solver, compares simulated metrics with the
targets extracted from the processed COTAHIST tables and optionally
runs the heuristic parameter adjustment loop that mirrors the notebook
pipeline.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT_DIR / "src"
if SRC_PATH.exists():
    src_str = str(SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from mfg_finance.data_init import empirical_initial_density
from mfg_finance.eta import bounded_eta_callback
from mfg_finance.grid import Grid1D
from mfg_finance.models.hft import HFTParams, initial_density
from mfg_finance.price import solve_price_clearing
from mfg_finance.solver import solve_mfg_picard
from mfg_finance.validation import (
    adjust_parameters,
    compute_simulation_metrics,
    estimate_market_targets,
    metric_gaps,
    relative_metric_gaps,
)


@dataclass
class RunOutputs:
    """
    Container for solver arrays and metadata.
    """

    U: np.ndarray
    M: np.ndarray
    alpha: np.ndarray
    price: np.ndarray | None
    metrics: Dict[str, Any]
    attempts: list[dict[str, Any]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and calibrate the MFG solver against market metrics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="YAML configuration file used as the starting point.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing the processed COTAHIST tables.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("notebooks_output"),
        help="Directory that will host the calibration artifacts.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=f"cal-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Subdirectory name inside --output-root.",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Only run the validation pass without adjusting parameters.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Maximum calibration rounds (overrides solver.calibration_rounds).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Base learning rate for adjust_parameters (overrides solver.calibration_lr).",
    )
    parser.add_argument(
        "--rel-guard",
        type=float,
        default=None,
        help="Override for calibration_rel_guard.",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=None,
        help="Override for calibration_rel_tol.",
    )
    parser.add_argument(
        "--grid-scale",
        type=float,
        default=None,
        help="Scale factor applied to the grid bounds before running.",
    )
    parser.add_argument(
        "--initial-density-spread",
        type=float,
        default=None,
        help="Override solver.initial_density_spread.",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Persist the calibrated parameters back to the supplied YAML config.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _scaled_grid_config(cfg: dict[str, Any], scale: float) -> dict[str, Any]:
    """
    Return a copy of cfg with symmetric bounds scaled around their midpoint.
    """

    if scale <= 0.0:
        scale = 1.0
    base = dict(cfg)
    x_min = float(base["x_min"])
    x_max = float(base["x_max"])
    center = 0.5 * (x_min + x_max)
    half_span = 0.5 * (x_max - x_min) * scale
    base["x_min"] = center - half_span
    base["x_max"] = center + half_span
    return base


def _build_grid(cfg: dict[str, Any]) -> Grid1D:
    return Grid1D(
        x_min=float(cfg["x_min"]),
        x_max=float(cfg["x_max"]),
        nx=int(cfg["nx"]),
        T=float(cfg["T"]),
        nt=int(cfg["nt"]),
        bc=str(cfg.get("bc", "neumann")),
    )


def _build_params(cfg: dict[str, Any]) -> HFTParams:
    return HFTParams(
        nu=float(cfg["nu"]),
        phi=float(cfg["phi"]),
        phi_4=float(cfg.get("phi_4", 0.0)),
        gamma_T=float(cfg["gamma_T"]),
        eta0=float(cfg.get("eta0", 1e-4)),
        eta1=float(cfg.get("eta1", 0.1)),
        kappa=float(cfg.get("kappa", 0.0)),
        m0_mean=float(cfg.get("m0_mean", 0.0)),
        m0_std=float(cfg.get("m0_std", 1.0)),
    )


def _initial_density(
    grid: Grid1D,
    params: HFTParams,
    solver_cfg: dict[str, Any],
) -> np.ndarray:
    density_mode = str(solver_cfg.get("initial_density_mode", "gaussian")).lower()
    spread_scale = float(solver_cfg.get("initial_density_spread", 1.0))
    if density_mode == "empirical":
        bucket = solver_cfg.get("initial_density_bucket", "all")
        bucket_quantiles = solver_cfg.get("initial_density_quantiles", (0.33, 0.66))
        if not isinstance(bucket_quantiles, (list, tuple)) or len(bucket_quantiles) != 2:
            bucket_quantiles = (0.33, 0.66)
        try:
            return empirical_initial_density(
                grid,
                liquidity_bucket=bucket,
                bucket_quantiles=tuple(bucket_quantiles),
                spread_scale=spread_scale,
            )
        except Exception as exc:  # pragma: no cover - diagnostics
            print(f"[warn] empirical density failed ({exc}); falling back to Gaussian.")
    tmp_params = replace(params, m0_std=max(params.m0_std * spread_scale, 1e-3))
    return initial_density(grid, tmp_params)


def _build_supply_schedule(cfg: dict[str, Any], grid: Grid1D) -> np.ndarray:
    supply_cfg = cfg.get("supply", 0.0)
    scale = float(cfg.get("supply_scale", 1.0))
    if isinstance(supply_cfg, (int, float)):
        base = np.full(grid.nt + 1, float(supply_cfg), dtype=float)
    else:
        arr = np.asarray(supply_cfg, dtype=float)
        if arr.size == grid.nt + 1:
            base = arr
        elif arr.size == 1:
            base = np.full(grid.nt + 1, float(arr.item()), dtype=float)
        else:
            base_times = np.linspace(0.0, grid.T, arr.size)
            base = np.interp(grid.t, base_times, arr).astype(float)
    return base * scale


def _maybe_compute_price(
    alpha: np.ndarray,
    densities: np.ndarray,
    grid: Grid1D,
    solver_cfg: dict[str, Any],
) -> Tuple[np.ndarray | None, Dict[str, float]]:
    compute_price = bool(solver_cfg.get("compute_price", True))
    if not compute_price:
        return None, {}

    supply_schedule = _build_supply_schedule(solver_cfg, grid)
    sensitivity = float(solver_cfg.get("price_sensitivity", 1.0))
    kappa = float(solver_cfg.get("price_elasticity", solver_cfg.get("kappa", 0.0)))
    bracket_cfg = solver_cfg.get("price_bracket", (-10.0, 10.0))
    if (
        not isinstance(bracket_cfg, (list, tuple))
        or len(bracket_cfg) != 2
    ):
        bracket = (-10.0, 10.0)
    else:
        bracket = (float(bracket_cfg[0]), float(bracket_cfg[1]))

    def alpha_field(idx: int, price: float) -> np.ndarray:
        return alpha[idx] - sensitivity * price

    prices = solve_price_clearing(
        alpha_field,
        densities,
        supply_schedule,
        grid.dx,
        kappa=kappa,
        bracket=bracket,
    )
    noise_std = float(solver_cfg.get("price_noise_std", 0.0))
    if noise_std > 0.0:
        seed = solver_cfg.get("price_noise_seed")
        rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()
        # MUDANÇA: Usar soma cumulativa (Brownian Motion) para criar um Random Walk suave
        # Isso preserva a estrutura de baixa volatilidade enquanto quebra a correlação com o fluxo
        white_noise = rng.normal(0.0, noise_std, size=prices.shape)
        brownian_noise = np.cumsum(white_noise)
        # Centralizar para evitar drift excessivo inicial
        brownian_noise = brownian_noise - brownian_noise[0]
        # NOVO: Adicionar tendência linear
        trend_total = float(solver_cfg.get("price_trend", 0.0))
        if trend_total != 0.0:
            trend_profile = np.linspace(0.0, trend_total, prices.shape[0])
            # Centralizar a tendencia para nao deslocar a media drasticamente
            trend_profile = trend_profile - trend_profile[0]
            brownian_noise = brownian_noise + trend_profile
        prices = prices + brownian_noise
    stats = {
        "price_mean": float(np.mean(prices)),
        "price_std": float(np.std(prices)),
        "price_min": float(np.min(prices)),
        "price_max": float(np.max(prices)),
        "price_span": float(np.max(prices) - np.min(prices)),
    }
    return prices, stats


def _run_solver_once(
    grid: Grid1D,
    params: HFTParams,
    solver_cfg: dict[str, Any],
) -> RunOutputs:
    m0 = _initial_density(grid, params, solver_cfg)
    solver_retry_limit = int(solver_cfg.get("solver_retries", 4))
    final_error_guard = float(solver_cfg.get("final_error_guard", 5e-4))
    final_error_fallback = float(solver_cfg.get("final_error_fallback", 2e-2))
    current_mix = float(solver_cfg.get("mix", 0.3))
    mix_min = float(solver_cfg.get("mix_min", 1e-5))
    mix_decay = float(solver_cfg.get("mix_decay", 0.5))
    relative_tol = float(solver_cfg.get("relative_tol", 5e-4))
    stagnation_tol = float(solver_cfg.get("stagnation_tol", 0.02))
    max_iter = int(solver_cfg.get("max_iter", 200))
    max_iter_cap = int(solver_cfg.get("max_iter_cap", max_iter * 4))
    anderson_depth = int(solver_cfg.get("anderson_depth", 0))
    anderson_beta = float(solver_cfg.get("anderson_beta", 1.0))

    hjb_kwargs = {
        "max_dissipation": float(solver_cfg.get("max_dissipation", 1.0)),
        "alpha_cap": float(solver_cfg.get("alpha_cap", 1.0)),
        "value_cap": float(solver_cfg.get("value_cap", 50.0)),
        "value_relaxation": float(solver_cfg.get("value_relaxation", 0.5)),
    }

    eta_bounds = solver_cfg.get("eta_bounds", (1e-4, 0.1))
    if (
        isinstance(eta_bounds, (list, tuple))
        and len(eta_bounds) == 2
    ):
        eta_min, eta_max = float(eta_bounds[0]), float(eta_bounds[1])
    else:
        eta_min, eta_max = 1e-4, 0.1
    eta_callback = (
        bounded_eta_callback(eta_min, eta_max)
        if solver_cfg.get("use_dynamic_eta", True)
        else None
    )

    attempts: list[dict[str, Any]] = []
    current_max_iter = max_iter
    success = False
    final_error = None
    U_all = M_all = alpha_all = None
    metrics: Dict[str, Any] = {}

    for attempt in range(max(solver_retry_limit, 0) + 1):
        U_all, M_all, alpha_all, _, metrics = solve_mfg_picard(
            grid,
            params,
            max_iter=current_max_iter,
            tol=float(solver_cfg.get("tol", 1e-6)),
            mix=current_mix,
            relative_tol=relative_tol,
            mix_min=mix_min,
            mix_decay=mix_decay,
            stagnation_tol=stagnation_tol,
            m0=m0,
            hjb_kwargs=hjb_kwargs,
            fp_kwargs={},
            eta_callback=eta_callback,
            drift_strength=float(solver_cfg.get("drift_strength", 0.0)),
            anderson_depth=anderson_depth,
            anderson_beta=anderson_beta,
        )
        final_error = float(metrics.get("final_error", 0.0))
        attempts.append(
            {
                "attempt": attempt,
                "mix": current_mix,
                "max_iter": current_max_iter,
                "iterations": metrics.get("iterations"),
                "final_error": final_error,
                "max_alpha": metrics.get("max_alpha"),
                "cfl_limits": metrics.get("cfl_limits", {}),
            }
        )
        if final_error <= final_error_guard:
            success = True
            break
        current_mix = max(current_mix * mix_decay, mix_min)
        current_max_iter = min(int(current_max_iter * 1.5), max_iter_cap)

    if not success:
        if final_error is not None and final_error <= final_error_fallback:
            print(
                f"[warn] final_error {final_error:.3e} > guard {final_error_guard:.3e}; "
                f"accepting fallback {final_error_fallback:.3e}."
            )
            success = True
        else:
            print(
                f"[warn] final_error remained at {final_error:.3e} despite retries; "
                "continuing with the latest iterate for diagnostics."
            )

    prices, price_stats = _maybe_compute_price(alpha_all, M_all, grid, solver_cfg)
    run_metrics = dict(metrics)
    run_metrics.update(price_stats)
    run_metrics["solver_attempts"] = attempts
    run_metrics["solver_success"] = success
    run_metrics["solver_final_error_guard"] = final_error_guard
    run_metrics["solver_final_error_fallback"] = final_error_fallback

    return RunOutputs(
        U=U_all,
        M=M_all,
        alpha=alpha_all,
        price=prices,
        metrics=run_metrics,
        attempts=attempts,
    )


def _validation_row(targets, simulation) -> dict[str, float]:
    return {
        "target_intraday_vol_norm": targets.intraday_vol,
        "sim_intraday_vol_norm": simulation.price_vol,
        "target_intraday_vol_raw": targets.intraday_vol_raw,
        "sim_intraday_vol_raw": simulation.price_vol_raw,
        "price_scale_target": targets.price_scale,
        "price_scale_sim": simulation.price_scale,
        "target_flow_return_corr": targets.flow_return_corr,
        "sim_flow_return_corr": simulation.flow_return_corr,
        "target_inventory_std_norm": targets.inventory_std,
        "sim_inventory_std_norm": simulation.inventory_std,
        "sim_inventory_std_path": simulation.inventory_std_path,
        "sim_inventory_mean": getattr(simulation, "inventory_mean", 0.0),
        "inventory_scale_model": targets.inventory_scale,
        "inventory_scale_shares": targets.inventory_scale_shares,
        "target_inventory_std_shares": targets.inventory_std_raw,
        "sim_inventory_std_shares": simulation.inventory_std * targets.inventory_scale,
    }


def _calibration_loop(
    grid_template: dict[str, Any],
    current_grid: Grid1D,
    solver_cfg: dict[str, Any],
    market_targets,
    base_params: HFTParams,
    base_result: RunOutputs,
    base_sim: Any,
    rounds: int,
    learning_rate: float,
    rel_guard: float,
    rel_tol: float,
) -> Tuple[
    HFTParams,
    RunOutputs,
    Any,
    list[dict[str, float]],
    Grid1D,
    dict[str, Any],
]:
    current_params = base_params
    current_result = base_result
    current_metrics = base_sim
    history: list[dict[str, float]] = []
    best_params = current_params
    best_result = current_result
    best_metrics = current_metrics
    best_grid = current_grid
    best_grid_cfg = dict(grid_template)
    best_score = float("inf")
    best_solver_cfg = copy.deepcopy(solver_cfg)
    grid_scale = 1.0
    supply_scale = float(solver_cfg.get("supply_scale", 1.0))
    price_sensitivity = float(solver_cfg.get("price_sensitivity", 1.0))
    inv_weight = float(solver_cfg.get("inventory_rel_weight", 1.0))
    flow_weight = float(solver_cfg.get("flow_rel_weight", 1.0))
    price_noise_std = float(solver_cfg.get("price_noise_std", 0.0))
    price_noise_cap = float(solver_cfg.get("price_noise_std_cap", price_noise_std))

    guard_warmup = max(int(solver_cfg.get("calibration_guard_warmup", 1)), 0)

    for round_idx in range(max(rounds, 0)):
        rel_errors = relative_metric_gaps(market_targets, current_metrics)
        weighted_inv_rel = inv_weight * rel_errors["inventory_std"]
        weighted_flow_rel = flow_weight * rel_errors["flow_return_corr"]
        max_rel = max(
            abs(rel_errors["intraday_vol"]),
            abs(weighted_flow_rel),
            abs(weighted_inv_rel),
        )
        spread_scale = float(solver_cfg.get("initial_density_spread", 1.0))
        history.append(
            {
                "round": round_idx,
                "learning_rate": learning_rate * (0.7 ** round_idx),
                "nu": current_params.nu,
                "phi": current_params.phi,
                "gamma_T": current_params.gamma_T,
                "eta0": current_params.eta0,
                "eta1": current_params.eta1,
                "sim_intraday_vol": current_metrics.price_vol,
                "sim_flow_return_corr": current_metrics.flow_return_corr,
                "sim_inventory_std": current_metrics.inventory_std,
                "sim_inventory_std_path": current_metrics.inventory_std_path,
                "sim_inventory_mean": getattr(current_metrics, "inventory_mean", 0.0),
                "initial_density_spread": spread_scale,
                "grid_scale": grid_scale,
                "supply_scale": supply_scale,
                "price_sensitivity": price_sensitivity,
                "price_noise_std": price_noise_std,
                "max_rel": max_rel,
                "rel_intraday_vol": rel_errors["intraday_vol"],
                "rel_flow_return_corr": rel_errors["flow_return_corr"],
                "rel_inventory_std": rel_errors["inventory_std"],
                "weighted_rel_inventory_std": weighted_inv_rel,
                "weighted_rel_flow_return_corr": weighted_flow_rel,
            }
        )
        if max_rel < best_score:
            best_score = max_rel
            best_params = current_params
            best_result = current_result
            best_metrics = current_metrics
            best_grid = current_grid
            best_grid_cfg = _scaled_grid_config(grid_template, grid_scale)
            best_solver_cfg = copy.deepcopy(solver_cfg)
        if max_rel > rel_guard and round_idx >= guard_warmup:
            print("[warn] Relative errors exceeded guard; aborting calibration loop.")
            break
        if max_rel < rel_tol:
            print("[info] Calibration tolerance reached.")
            break
        lr = learning_rate * (0.7 ** round_idx)
        inv_rel = np.clip(weighted_inv_rel, -2.0, 2.0)
        if abs(inv_rel) > 1e-3:
            spread_scale = float(
                np.clip(spread_scale * (1.0 - lr * inv_rel), 0.1, 5.0)
            )
            solver_cfg["initial_density_spread"] = spread_scale
            grid_scale = float(
                np.clip(grid_scale * (1.0 - 0.5 * lr * inv_rel), 0.35, 4.0)
            )
            supply_scale = float(
                np.clip(supply_scale * (1.0 - 0.3 * lr * inv_rel), 1e-7, 1.0)
            )
            solver_cfg["supply_scale"] = supply_scale
        corr_rel = np.clip(weighted_flow_rel, -2.0, 2.0)
        if abs(corr_rel) > 1e-3:
            price_sensitivity = float(
                np.clip(price_sensitivity * (1.0 - 0.6 * lr * corr_rel), 0.2, 20.0)
            )
            if abs(weighted_flow_rel) > 1.0:
                price_sensitivity = max(price_sensitivity * 0.5, 0.2)
            solver_cfg["price_sensitivity"] = price_sensitivity
            price_noise_std = float(
                np.clip(price_noise_std * (1.0 - 0.5 * lr * corr_rel), 0.0, price_noise_cap)
            )
            solver_cfg["price_noise_std"] = price_noise_std
            supply_scale = float(
                np.clip(supply_scale * (1.0 - 0.2 * lr * corr_rel), 1e-7, 1.0)
            )
            solver_cfg["supply_scale"] = supply_scale
        next_grid_cfg = _scaled_grid_config(grid_template, grid_scale)
        next_grid = _build_grid(next_grid_cfg)
        next_params = adjust_parameters(
            current_params,
            market_targets,
            current_metrics,
            learning_rate=lr,
        )
        if next_params == current_params:
            print("[info] No further parameter updates produced by adjust_parameters.")
            break
        next_result = _run_solver_once(next_grid, next_params, solver_cfg)
        next_sim = compute_simulation_metrics(
            next_result.M,
            next_result.alpha,
            next_grid,
            next_result.price,
        )
        current_params = next_params
        current_result = next_result
        current_metrics = next_sim
        current_grid = next_grid

    final_errors = relative_metric_gaps(market_targets, current_metrics)
    final_score = max(abs(v) for v in final_errors.values())
    if final_score < best_score:
        best_params = current_params
        best_result = current_result
        best_metrics = current_metrics
        best_grid = current_grid
        best_grid_cfg = _scaled_grid_config(grid_template, grid_scale)
        best_solver_cfg = copy.deepcopy(solver_cfg)

    solver_cfg.clear()
    solver_cfg.update(best_solver_cfg)
    return best_params, best_result, best_metrics, history, best_grid, best_grid_cfg


def _save_arrays(output_dir: Path, run: RunOutputs, grid: Grid1D) -> None:
    np.save(output_dir / "U_all.npy", run.U)
    np.save(output_dir / "M_all.npy", run.M)
    np.save(output_dir / "alpha_all.npy", run.alpha)
    if run.price is not None:
        price_path = output_dir / "price.csv"
        np.savetxt(
            price_path,
            np.column_stack((grid.t, run.price)),
            delimiter=",",
            header="time,price",
            comments="",
        )
    _save_inventory_path(output_dir, run.M, grid)


def _save_inventory_path(output_dir: Path, densities: np.ndarray, grid: Grid1D) -> None:
    """
    Persist diagnostics for the mean inventory path and cross-sectional spread.
    """

    weights = grid.x.astype(float)
    mean_inventory = np.sum(densities * weights, axis=1) * grid.dx
    second_moment = np.sum(densities * (weights**2), axis=1) * grid.dx
    cross_sectional_std = np.sqrt(np.clip(second_moment - mean_inventory**2, 0.0, None))
    df = pd.DataFrame(
        {
            "time": grid.t,
            "mean_inventory": mean_inventory,
            "cross_sectional_std": cross_sectional_std,
        }
    )
    df.to_csv(output_dir / "inventory_path.csv", index=False)


def _save_validation(
    output_dir: Path,
    summary_row: dict[str, float],
    abs_gaps: dict[str, float],
    rel_gaps: dict[str, float],
) -> None:
    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(output_dir / "validation_summary.csv", index=False)
    payload = {
        **summary_row,
        **{f"gap_{k}": v for k, v in abs_gaps.items()},
        **{f"gap_{k}_rel": v for k, v in rel_gaps.items()},
    }
    with (output_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_history(output_dir: Path, history: Iterable[dict[str, float]]) -> None:
    history = list(history)
    if not history:
        return
    df = pd.DataFrame(history)
    df.to_csv(output_dir / "calibration_history.csv", index=False)


def _update_config_file(
    config_path: Path,
    data: dict[str, Any],
    params: HFTParams,
    solver_config: dict[str, Any] | None = None,
) -> None:
    cfg = dict(data)
    cfg.setdefault("params", {})
    cfg["params"].update(
        {
            "nu": float(params.nu),
            "phi": float(params.phi),
            "gamma_T": float(params.gamma_T),
            "eta0": float(params.eta0),
            "eta1": float(params.eta1),
        }
    )
    if solver_config is not None:
        cfg["solver"] = dict(solver_config)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def main() -> int:
    args = _parse_args()
    cfg = _load_yaml(args.config)
    grid_cfg = dict(cfg["grid"])
    if args.grid_scale is not None:
        grid_cfg = _scaled_grid_config(grid_cfg, float(args.grid_scale))
    cfg["grid"] = grid_cfg
    grid_template = dict(grid_cfg)
    grid = _build_grid(grid_cfg)
    params = _build_params(cfg["params"])
    solver_cfg = dict(cfg.get("solver", {}))
    if args.initial_density_spread is not None:
        solver_cfg["initial_density_spread"] = float(args.initial_density_spread)

    output_dir = args.output_root / args.timestamp
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    market_targets = estimate_market_targets(args.data_dir)
    initial_result = _run_solver_once(grid, params, solver_cfg)
    initial_sim = compute_simulation_metrics(
        initial_result.M,
        initial_result.alpha,
        grid,
        initial_result.price,
    )

    final_params = params
    final_result = initial_result
    final_sim = initial_sim
    history: list[dict[str, float]] = []
    final_grid_cfg = grid_cfg

    if not args.skip_calibration:
        rounds = args.rounds if args.rounds is not None else int(solver_cfg.get("calibration_rounds", 6))
        learning_rate = (
            args.learning_rate if args.learning_rate is not None else float(solver_cfg.get("calibration_lr", 0.25))
        )
        rel_guard = (
            float(args.rel_guard)
            if args.rel_guard is not None
            else float(solver_cfg.get("calibration_rel_guard", 5.0))
        )
        rel_tol = (
            float(args.rel_tol)
            if args.rel_tol is not None
            else float(solver_cfg.get("calibration_rel_tol", 0.05))
        )
        (
            final_params,
            final_result,
            final_sim,
            history,
            grid,
            final_grid_cfg,
        ) = _calibration_loop(
            grid_template,
            grid,
            solver_cfg,
            market_targets,
            params,
            initial_result,
            initial_sim,
            rounds,
            learning_rate,
            rel_guard,
            rel_tol,
        )
    else:
        final_grid_cfg = grid_cfg

    summary_row = _validation_row(market_targets, final_sim)
    abs_gaps = metric_gaps(market_targets, final_sim)
    rel_gaps = relative_metric_gaps(market_targets, final_sim)
    _save_arrays(output_dir, final_result, grid)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_result.metrics, handle, indent=2)
    _save_validation(output_dir, summary_row, abs_gaps, rel_gaps)
    _save_history(output_dir, history)
    with (output_dir / "calibrated_params.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(final_params), handle, indent=2)

    if args.update_config:
        cfg["grid"] = final_grid_cfg
        _update_config_file(args.config, cfg, final_params, solver_cfg)

    print(
        "Final validation -> "
        f"vol: {final_sim.price_vol:.4f} "
        f"(target {market_targets.intraday_vol:.4f}), "
        f"flow/ret corr: {final_sim.flow_return_corr:.4e} "
        f"(target {market_targets.flow_return_corr:.4e}), "
        f"inventory std: {final_sim.inventory_std:.4f} "
        f"(target {market_targets.inventory_std:.4f})"
    )
    print(f"Artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
