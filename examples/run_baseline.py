# """
# Run the baseline Mean Field Game experiment and generate summary artefacts.
# """

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

import numpy as np
import yaml

from mfg_finance.grid import Grid1D
from mfg_finance.models.hft import HFTParams, eta_from_m_alpha, initial_density
from mfg_finance.solver import solve_mfg_picard
from mfg_finance.viz import PlotConfig, plot_alpha_cuts, plot_convergence, plot_density_time, plot_value_time


def _load_config() -> Dict[str, Any]:
    config_path = pathlib.Path(__file__).resolve().parents[1] / "mfg-finance" / "configs" / "baseline.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_grid(cfg: Dict[str, Any]) -> Grid1D:
    return Grid1D(
        x_min=float(cfg["x_min"]),
        x_max=float(cfg["x_max"]),
        nx=int(cfg["nx"]),
        T=float(cfg["T"]),
        nt=int(cfg["nt"]),
        bc=str(cfg.get("bc", "neumann")),
    )


def _build_params(cfg: Dict[str, Any]) -> HFTParams:
    return HFTParams(
        nu=float(cfg["nu"]),
        phi=float(cfg["phi"]),
        gamma_T=float(cfg["gamma_T"]),
        eta0=float(cfg["eta0"]),
        eta1=float(cfg["eta1"]),
        m0_mean=float(cfg.get("m0_mean", 0.0)),
        m0_std=float(cfg.get("m0_std", 1.0)),
    )


def main() -> None:
    cfg = _load_config()
    grid = _build_grid(cfg["grid"])
    params = _build_params(cfg["params"])
    solver_cfg = cfg.get("solver", {})

    output_dir = pathlib.Path("examples_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = initial_density(grid, params)

    U_all, M_all, alpha_all, errors, metrics = solve_mfg_picard(
        grid,
        params,
        max_iter=int(solver_cfg.get("max_iter", 200)),
        tol=float(solver_cfg.get("tol", 1.0e-7)),
        mix=float(solver_cfg.get("mix", 0.4)),
        m0=initial,
        hjb_kwargs={"max_inner": solver_cfg.get("hjb_inner", 4), "tol": solver_cfg.get("hjb_tol", 1e-8)},
        eta_callback=eta_from_m_alpha,
    )

    # Persist arrays for further inspection.
    np.save(output_dir / "U_all.npy", U_all)
    np.save(output_dir / "M_all.npy", M_all)
    np.save(output_dir / "alpha_all.npy", alpha_all)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    plot_cfg = PlotConfig(figsize=(10.0, 4.0))
    plot_density_time(M_all, grid, output_dir / "density.png", plot_cfg)
    plot_value_time(U_all, grid, output_dir / "value.png", plot_cfg)
    plot_alpha_cuts(
        alpha_all,
        grid,
        times=[0.0, 0.25 * grid.T, 0.5 * grid.T, 0.75 * grid.T, grid.T],
        path=output_dir / "alpha_cuts.png",
        cfg=PlotConfig(figsize=(10.0, 5.0)),
    )
    plot_convergence(errors, output_dir / "convergence.png")

    mass_final = float(np.sum(M_all[-1]) * grid.dx)
    print("Baseline run complete")
    print(f"Iterations: {len(errors)}")
    print(f"Final error: {errors[-1]:.3e}" if errors else "Final error: n/a")
    print(f"Density min/max: {np.min(M_all):.3e}/{np.max(M_all):.3e}")
    print(f"Final mass: {mass_final:.6f}")
    print(f"Mean |alpha|: {metrics['mean_abs_alpha']:.6f}")


if __name__ == "__main__":
    main()
