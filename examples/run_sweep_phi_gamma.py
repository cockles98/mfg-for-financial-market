# """
# Parameter sweep over (phi, gamma_T) to study liquidity regimes.
# """

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from mfg_finance.grid import Grid1D
from mfg_finance.models.hft import HFTParams, eta_from_m_alpha, initial_density
from mfg_finance.solver import compute_alpha_metrics, solve_mfg_picard


def _load_config() -> Dict[str, Any]:
    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "mfg-finance" / "configs" / "baseline.yaml"
    with cfg_path.open("r", encoding="utf-8") as handle:
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


def _build_params(cfg: Dict[str, Any], phi: float, gamma_T: float) -> HFTParams:
    return HFTParams(
        nu=float(cfg["nu"]),
        phi=phi,
        gamma_T=gamma_T,
        eta0=float(cfg["eta0"]),
        eta1=float(cfg["eta1"]),
        m0_mean=float(cfg.get("m0_mean", 0.0)),
        m0_std=float(cfg.get("m0_std", 1.0)),
    )


def _run_case(
    grid: Grid1D,
    params: HFTParams,
    solver_cfg: Dict[str, Any],
) -> Tuple[float, int, float]:
    initial = initial_density(grid, params)

    _, _, alpha_all, errors, metrics = solve_mfg_picard(
        grid,
        params,
        max_iter=int(solver_cfg.get("max_iter", 200)),
        tol=float(solver_cfg.get("tol", 1.0e-7)),
        mix=float(solver_cfg.get("mix", 0.4)),
        m0=initial,
        hjb_kwargs={"max_inner": solver_cfg.get("hjb_inner", 4), "tol": solver_cfg.get("hjb_tol", 1e-8)},
        eta_callback=eta_from_m_alpha,
    )

    return metrics["mean_abs_alpha"], len(errors), errors[-1] if errors else 0.0


def _grid(values: Iterable[float]) -> List[float]:
    return list(values)


def _plot_heatmap(
    phi_vals: List[float],
    gamma_vals: List[float],
    matrix: np.ndarray,
    path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    im = ax.imshow(
        matrix,
        origin="lower",
        cmap="viridis",
        extent=(min(phi_vals), max(phi_vals), min(gamma_vals), max(gamma_vals)),
        aspect="auto",
    )
    ax.set_xlabel("phi")
    ax.set_ylabel("gamma_T")
    ax.set_title("Mean |alpha| across (phi, gamma_T)")
    fig.colorbar(im, ax=ax, label="mean |alpha|")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = _load_config()
    grid = _build_grid(cfg["grid"])
    solver_cfg = cfg.get("solver", {})

    phi_values = _grid([0.05, 0.1, 0.2])
    gamma_values = _grid([0.5, 1.0, 2.0])

    output_dir = pathlib.Path("examples_output") / "sweep_phi_gamma"
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_alpha_matrix = np.zeros((len(gamma_values), len(phi_values)))
    summary_rows: List[List[Any]] = []

    for i, gamma in enumerate(gamma_values):
        for j, phi in enumerate(phi_values):
            params = _build_params(cfg["params"], phi=phi, gamma_T=gamma)
            mean_alpha, iterations, final_error = _run_case(grid, params, solver_cfg)
            mean_alpha_matrix[i, j] = mean_alpha
            summary_rows.append([phi, gamma, iterations, final_error, mean_alpha])
            print(f"phi={phi:.3f}, gamma_T={gamma:.3f} -> mean|alpha|={mean_alpha:.4f}, iterations={iterations}, error={final_error:.3e}")

    _plot_heatmap(phi_values, gamma_values, mean_alpha_matrix, output_dir / "heatmap_alpha_mean.png")

    with (output_dir / "sweep_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phi", "gamma_T", "iterations", "final_error", "mean_abs_alpha"])
        writer.writerows(summary_rows)

    np.save(output_dir / "phi_values.npy", np.array(phi_values))
    np.save(output_dir / "gamma_values.npy", np.array(gamma_values))
    np.save(output_dir / "mean_alpha_matrix.npy", mean_alpha_matrix)

    summary_stats = {
        "phi_values": phi_values,
        "gamma_values": gamma_values,
        "mean_alpha_matrix": mean_alpha_matrix.tolist(),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_stats, handle, indent=2)

    print(f"Sweep completed. Artefacts stored in {output_dir}")


if __name__ == "__main__":
    main()
