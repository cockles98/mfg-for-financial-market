"""
Command-line interface for running Mean Field Game simulations.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import pathlib
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml

from .grid import Grid1D
from .models.hft import HFTParams, eta_from_m_alpha, initial_density
from .price import solve_price_clearing
from .solver import solve_mfg_picard
from .viz import (
    PlotConfig,
    plot_alpha_cuts,
    plot_convergence,
    plot_density_time,
    plot_price,
    plot_value_time,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="mfg-finance",
        description="Mean Field Game solver for financial markets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute a simulation from a YAML configuration file.")
    run_parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    run_parser.add_argument(
        "--endogenous-price",
        action="store_true",
        help="Enable price clearing after solving for controls.",
    )

    sweep_parser = subparsers.add_parser("sweep", help="Run a parameter sweep overriding config values.")
    sweep_parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Base configuration file used for the sweep.",
    )
    sweep_parser.add_argument(
        "--phi",
        type=str,
        default="",
        help="Comma-separated list of phi values (e.g. 0.05,0.1).",
    )
    sweep_parser.add_argument(
        "--gamma_T",
        type=str,
        default="",
        help="Comma-separated list of gamma_T values (e.g. 1.0,2.0).",
    )

    return parser


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_config(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data: Dict[str, Any] = yaml.safe_load(handle)
    return data


def _build_grid(grid_cfg: Dict[str, Any]) -> Grid1D:
    return Grid1D(
        x_min=float(grid_cfg["x_min"]),
        x_max=float(grid_cfg["x_max"]),
        nx=int(grid_cfg["nx"]),
        T=float(grid_cfg["T"]),
        nt=int(grid_cfg["nt"]),
        bc=str(grid_cfg.get("bc", "neumann")),
    )


def _build_params(params_cfg: Dict[str, Any]) -> HFTParams:
    return HFTParams(
        nu=float(params_cfg["nu"]),
        phi=float(params_cfg["phi"]),
        gamma_T=float(params_cfg["gamma_T"]),
        eta0=float(params_cfg["eta0"]),
        eta1=float(params_cfg["eta1"]),
        m0_mean=float(params_cfg.get("m0_mean", 0.0)),
        m0_std=float(params_cfg.get("m0_std", 1.0)),
    )


def _save_numpy_array(path: pathlib.Path, array: np.ndarray) -> None:
    np.save(path, array)


def _write_errors_csv(
    path: pathlib.Path,
    errors: Iterable[float],
    *,
    relative: Iterable[float] | None = None,
    mix: Iterable[float] | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["iteration", "error_l2"]
        if relative is not None:
            header.append("error_relative")
        if mix is not None:
            header.append("mix")
        writer.writerow(header)

        abs_list = list(errors)
        rel_list = list(relative) if relative is not None else []
        mix_list = list(mix) if mix is not None else []
        for idx, value in enumerate(abs_list):
            row = [idx, value]
            if relative is not None:
                rel_value = rel_list[idx] if idx < len(rel_list) else ""
                row.append(rel_value)
            if mix is not None:
                mix_value = mix_list[idx] if idx < len(mix_list) else ""
                row.append(mix_value)
            writer.writerow(row)


def _build_supply_schedule(solver_cfg: Dict[str, Any], grid: Grid1D) -> np.ndarray:
    supply_cfg = solver_cfg.get("supply", 0.0)
    if isinstance(supply_cfg, (int, float)):
        return np.full(grid.nt + 1, float(supply_cfg), dtype=np.float64)
    if isinstance(supply_cfg, (list, tuple)):
        data = np.asarray(supply_cfg, dtype=np.float64)
        if data.size == grid.nt + 1:
            return data
        if data.size == 1:
            return np.full(grid.nt + 1, float(data[0]), dtype=np.float64)
        base_times = np.linspace(0.0, grid.T, data.size)
        return np.interp(grid.t, base_times, data).astype(np.float64)
    return np.zeros(grid.nt + 1, dtype=np.float64)


def _run_single_experiment(
    cfg: Dict[str, Any],
    artifacts_dir: pathlib.Path,
    *,
    compute_price: bool = False,
) -> Dict[str, Any]:
    grid = _build_grid(cfg["grid"])
    params = _build_params(cfg["params"])
    solver_cfg = cfg.get("solver", {})

    m0 = initial_density(grid, params)
    eta_callback = eta_from_m_alpha if solver_cfg.get("use_dynamic_eta", True) else None
    metrics_path = artifacts_dir / "metrics.json"

    cfl_limits = grid.suggest_cfl_limits(
        params.nu,
        solver_cfg.get("velocity_guess"),
    )
    diffusion_limit = cfl_limits.get("diffusion_dt")
    if diffusion_limit is not None and grid.dt > diffusion_limit:
        print(f"[warning] Time step dt={grid.dt:.3e} exceeds diffusion CFL â‰ˆ {diffusion_limit:.3e}.")
    advection_limit = cfl_limits.get("advection_dt")
    if advection_limit is not None and grid.dt > advection_limit:
        print(f"[warning] Time step dt={grid.dt:.3e} exceeds advection CFL â‰ˆ {advection_limit:.3e}.")

    U_all, M_all, alpha_all, errors, metrics = solve_mfg_picard(
        grid,
        params,
        max_iter=int(solver_cfg.get("max_iter", 200)),
        tol=float(solver_cfg.get("tol", 1e-8)),
        mix=float(solver_cfg.get("mix", 0.3)),
        relative_tol=float(solver_cfg["relative_tol"]) if "relative_tol" in solver_cfg else None,
        mix_min=float(solver_cfg.get("mix_min", 1e-4)),
        mix_decay=float(solver_cfg.get("mix_decay", 0.5)),
        stagnation_tol=float(solver_cfg.get("stagnation_tol", 0.02)),
        m0=m0,
        hjb_kwargs={"max_inner": solver_cfg.get("hjb_inner", 4), "tol": solver_cfg.get("hjb_tol", 1e-8)},
        fp_kwargs={},
        eta_callback=eta_callback,
        metrics_path=metrics_path,
    )

    _ensure_dir(artifacts_dir)
    _save_numpy_array(artifacts_dir / "U_all.npy", U_all)
    _save_numpy_array(artifacts_dir / "M_all.npy", M_all)
    _save_numpy_array(artifacts_dir / "alpha_all.npy", alpha_all)
    _write_errors_csv(
        artifacts_dir / "errors.csv",
        errors,
        relative=metrics.get("relative_errors"),
        mix=metrics.get("mix_history"),
    )

    with (artifacts_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    plot_cfg = PlotConfig()
    plot_value_time(U_all, grid, artifacts_dir / "value_function.png", plot_cfg)
    plot_density_time(M_all, grid, artifacts_dir / "density.png", plot_cfg)
    plot_alpha_cuts(
        alpha_all,
        grid,
        times=[0.0, grid.T / 2, grid.T],
        path=artifacts_dir / "alpha_cuts.png",
        cfg=PlotConfig(figsize=(10.0, 5.0)),
    )
    plot_convergence(errors, artifacts_dir / "convergence.png")

    price_mean: float | None = None
    price_std: float | None = None
    if compute_price:
        supply_schedule = _build_supply_schedule(solver_cfg, grid)
        sensitivity = float(solver_cfg.get("price_sensitivity", 1.0))
        bracket_cfg = solver_cfg.get("price_bracket", [-10.0, 10.0])
        if isinstance(bracket_cfg, (list, tuple)) and len(bracket_cfg) == 2:
            bracket = (float(bracket_cfg[0]), float(bracket_cfg[1]))
        else:
            bracket = (-10.0, 10.0)

        def alpha_field(idx: int, price: float) -> np.ndarray:
            return alpha_all[idx] - sensitivity * price

        prices = solve_price_clearing(
            alpha_field,
            M_all,
            supply_schedule,
            grid.dx,
            bracket=bracket,
        )
        price_mean = float(np.mean(prices))
        price_std = float(np.std(prices))

        np.savetxt(
            artifacts_dir / "price.csv",
            np.column_stack((grid.t, prices)),
            delimiter=",",
            header="time,price",
            comments="",
        )
        plot_price(grid.t, prices, artifacts_dir / "price.png")

    final_error = float(metrics.get("final_error", errors[-1] if errors else 0.0))
    iterations = int(metrics.get("iterations", len(errors)))

    return {
        "grid": grid,
        "params": params,
        "final_error": final_error,
        "iterations": iterations,
        "mean_abs_alpha": metrics["mean_abs_alpha"],
        "std_alpha": metrics["std_alpha"],
        "liquidity_proxy": metrics["liquidity_proxy"],
        "price_mean": price_mean,
        "price_std": price_std,
    }


def _parse_float_list(values: str) -> List[float]:
    if not values:
        return []
    return [float(item.strip()) for item in values.split(",") if item.strip()]


def _handle_run(config_path: pathlib.Path, endogenous_price: bool) -> None:
    cfg = _load_config(config_path)
    run_dir = pathlib.Path("artifacts") / f"run-{_timestamp()}"
    stats = _run_single_experiment(cfg, run_dir, compute_price=endogenous_price)

    print(f"Run completed. Artifacts stored in {run_dir}")
    message = (
        f"Iterations: {stats['iterations']}, final error: {stats['final_error']:.3e}, "
        f"mean |alpha|: {stats['mean_abs_alpha']:.3e}"
    )
    if stats.get("price_mean") is not None:
        message += f", mean price: {stats['price_mean']:.3f}"
    print(message)


def _handle_sweep(config_path: pathlib.Path, phi_values: str, gamma_values: str) -> None:
    cfg = _load_config(config_path)
    phi_list = _parse_float_list(phi_values)
    gamma_list = _parse_float_list(gamma_values)

    if not phi_list:
        phi_list = [float(cfg["params"]["phi"])]
    if not gamma_list:
        gamma_list = [float(cfg["params"]["gamma_T"])]

    sweep_dir = pathlib.Path("artifacts") / f"sweep-{_timestamp()}"
    _ensure_dir(sweep_dir)

    summary_rows: List[List[Any]] = []

    for phi, gamma in itertools.product(phi_list, gamma_list):
        cfg_variant = {
            **cfg,
            "params": {**cfg["params"], "phi": phi, "gamma_T": gamma},
        }
        folder_name = f"phi-{phi:g}_gamma-{gamma:g}"
        run_dir = sweep_dir / folder_name
        stats = _run_single_experiment(cfg_variant, run_dir, compute_price=False)

        summary_rows.append([
            phi,
            gamma,
            stats["iterations"],
            stats["final_error"],
            stats["mean_abs_alpha"],
            stats["std_alpha"],
            stats["liquidity_proxy"],
            stats.get("price_mean"),
            stats.get("price_std"),
        ])
        print(f"Completed sweep case phi={phi} gamma_T={gamma} -> final error {stats['final_error']:.3e}")

    with (sweep_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["phi", "gamma_T", "iterations", "final_error", "mean_abs_alpha", "std_alpha", "liquidity_proxy", "price_mean", "price_std"]
        )
        writer.writerows(summary_rows)

    print(f"Sweep finished. Artifacts stored in {sweep_dir}")


def main(argv: List[str] | None = None) -> int:
    """
    Execute the command-line interface.
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        _handle_run(args.config.expanduser().resolve(), args.endogenous_price)
        return 0
    if args.command == "sweep":
        _handle_sweep(args.config.expanduser().resolve(), args.phi, args.gamma_T)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
