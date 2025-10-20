"""
Picard iteration driver coupling the HJB and Fokker-Planck solvers.
"""
from __future__ import annotations
import json
import pathlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np
from .fp import FPSolver, solve_fp_forward
from .grid import Grid1D
from .hamiltonian import EtaCallback, alpha_star
from .hjb import HJBSolver, solve_hjb_backward
from .models.hft import HFTParams
from .ops import project_positive_and_renormalize
__all__ = [
    "ConvergenceCallback",
    "compute_alpha_traj",
    "compute_alpha_metrics",
    "save_metrics",
    "solve_mfg_picard",
    "MeanFieldGameSolver",
]
ConvergenceCallback = Callable[[int, float], None]
def compute_alpha_traj(
    U_all: np.ndarray,
    M_all: np.ndarray,
    grid: Grid1D,
    params: HFTParams,
    eta_callback: EtaCallback | None = None,
) -> np.ndarray:
    """
    Compute the optimal control trajectory from the value and density paths.
    """
    if U_all.shape != M_all.shape:
        msg = "Value and density trajectories must share the same shape."
        raise ValueError(msg)
    alpha = np.zeros_like(U_all)
    for n in range(U_all.shape[0]):
        U_n = U_all[n]
        m_n = M_all[n]
        grad = np.empty_like(U_n)
        grad[1:-1] = (U_n[2:] - U_n[:-2]) / (2.0 * grid.dx)
        grad[0] = (U_n[1] - U_n[0]) / grid.dx
        grad[-1] = (U_n[-1] - U_n[-2]) / grid.dx
        alpha[n] = alpha_star(grad, m_n, params, mean_alpha=None, eta_callback=eta_callback)
    return alpha
def compute_alpha_metrics(alpha_all: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for the control trajectory.
    """
    abs_alpha = np.abs(alpha_all)
    mean_abs = float(np.mean(abs_alpha))
    std_alpha = float(np.std(alpha_all))
    liquidity_proxy = float(np.mean(np.exp(-abs_alpha)))
    return {
        "mean_abs_alpha": mean_abs,
        "std_alpha": std_alpha,
        "liquidity_proxy": liquidity_proxy,
    }
def save_metrics(metrics: Dict[str, float], path: pathlib.Path | str) -> None:
    """
    Persist metrics dictionary to disk as JSON.
    """
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
def solve_mfg_picard(
    grid: Grid1D,
   params: HFTParams,
   *,
   max_iter: int = 200,
   tol: float = 1e-8,
   mix: float = 0.3,
    relative_tol: float | None = None,
    mix_min: float = 1e-4,
    mix_decay: float = 0.5,
    stagnation_tol: float = 0.02,
    m0: np.ndarray | None = None,
    hjb_kwargs: Dict[str, float] | None = None,
    fp_kwargs: Dict[str, float] | None = None,
    callback: ConvergenceCallback | None = None,
    eta_callback: EtaCallback | None = None,
    metrics_path: pathlib.Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], Dict[str, float]]:
    """
    Run the Picard fixed-point iteration for the coupled MFG system.
    """
    hjb_kwargs = dict(hjb_kwargs or {})
    hjb_kwargs.setdefault("max_dissipation", 1.0)
    hjb_kwargs.setdefault("alpha_cap", 1.0)
    hjb_kwargs.setdefault("value_cap", 50.0)
    hjb_kwargs.setdefault("value_relaxation", 0.5)
    fp_kwargs = dict(fp_kwargs or {})
    if m0 is None:
        from .models.hft import initial_density
        m0 = initial_density(grid, params)
    m0 = project_positive_and_renormalize(np.asarray(m0, dtype=np.float64), dx=grid.dx)
    M_k = np.tile(m0, (grid.nt + 1, 1))
    errors: List[float] = []
    relative_errors: List[float] = []
    mix_history: List[float] = []
    U_all = np.zeros_like(M_k)
    current_mix = float(mix)
    eps = 1e-12
    stalled_flag = False
    for iteration in range(max_iter):
        U_all = solve_hjb_backward(
            M_k,
            grid,
            params,
            progress=False,
            eta_callback=eta_callback,
            **hjb_kwargs,
        )
        M_raw = solve_fp_forward(
            U_all,
            grid,
            params,
            m0,
            progress=False,
            **fp_kwargs,
        )
        prev_error = errors[-1] if errors else None
        candidate_mix = current_mix
        stalled = False

        while True:
            M_next = candidate_mix * M_raw + (1.0 - candidate_mix) * M_k
            diff = M_next - M_k
            error_abs = float(np.linalg.norm(diff.ravel()))
            norm_ref = float(np.linalg.norm(M_next.ravel())) + eps
            error_rel = error_abs / norm_ref
            if (
                prev_error is None
                or error_abs <= prev_error * (1.0 - stagnation_tol)
                or candidate_mix <= mix_min + eps
            ):
                break
            candidate_mix = max(candidate_mix * mix_decay, mix_min)

        if prev_error is not None and error_abs > prev_error and candidate_mix <= mix_min + eps:
            stalled = True
            stalled_flag = True

        if stalled:
            break

        current_mix = candidate_mix
        M_k = M_next
        errors.append(error_abs)
        relative_errors.append(error_rel)
        mix_history.append(float(current_mix))

        if callback is not None:
            callback(iteration, error_abs)

        absolute_check = error_abs < tol
        relative_check = relative_tol is not None and error_rel < relative_tol
        if absolute_check or relative_check:
            break
    alpha_all = compute_alpha_traj(U_all, M_k, grid, params, eta_callback=eta_callback)
    metrics = compute_alpha_metrics(alpha_all)
    final_error = errors[-1] if errors else 0.0
    final_error_rel = relative_errors[-1] if relative_errors else 0.0
    metrics.update(
        final_error=final_error,
        final_error_relative=final_error_rel,
        iterations=len(errors),
        mix_initial=float(mix),
        mix_final=float(current_mix),
        mix_min=float(mix_min),
        mix_history=mix_history,
        relative_errors=relative_errors,
        stalled=stalled_flag,
    )
    if metrics_path is not None:
        save_metrics(metrics, metrics_path)
    return U_all, M_k, alpha_all, errors, metrics
@dataclass(slots=True)
class MeanFieldGameSolver:
    """
    High-level convenience wrapper for the Picard solver.
    """
    grid: Grid1D
    params: HFTParams
    max_iter: int = 200
    tol: float = 1e-8
    mix: float = 0.3
    relative_tol: float | None = None
    mix_min: float = 1e-4
    mix_decay: float = 0.5
    stagnation_tol: float = 0.02
    hjb_kwargs: Dict[str, float] | None = None
    fp_kwargs: Dict[str, float] | None = None
    eta_callback: EtaCallback | None = None
    def run(
        self,
        initial_density: np.ndarray | None = None,
        metrics_path: pathlib.Path | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], Dict[str, float]]:
        """
        Execute the Picard iteration and return trajectories with logs.
        """
        return solve_mfg_picard(
            self.grid,
            self.params,
            max_iter=self.max_iter,
            tol=self.tol,
            mix=self.mix,
            relative_tol=self.relative_tol,
            mix_min=self.mix_min,
            mix_decay=self.mix_decay,
            stagnation_tol=self.stagnation_tol,
            m0=initial_density,
            hjb_kwargs=self.hjb_kwargs,
            fp_kwargs=self.fp_kwargs,
            eta_callback=self.eta_callback,
            metrics_path=metrics_path,
        )
