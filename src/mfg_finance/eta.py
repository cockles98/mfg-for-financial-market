from __future__ import annotations

import numpy as np

from .models.hft import HFTParams, eta_from_m_alpha

__all__ = ["bounded_eta_callback"]


def bounded_eta_callback(
    eta_min: float = 1e-4,
    eta_max: float = 0.1,
):
    eta_min = max(float(eta_min), 1e-12)
    eta_max = max(float(eta_max), eta_min * 1.01)

    def _callback(m: np.ndarray, alpha: np.ndarray, params: HFTParams) -> float:
        eta = eta_from_m_alpha(m, alpha, params)
        return float(np.clip(eta, eta_min, eta_max))

    return _callback
