"""
Mean Field Game solver components tailored to financial market models.

The package exposes building blocks for discretising Hamilton-Jacobi-Bellman
(HJB) and Fokker-Planck (FP) equations, along with orchestration utilities that
enable Picard iterations between both solvers.
"""

from __future__ import annotations


__all__ = ["__version__", "get_version"]

__version__ = "0.1.0"


def get_version() -> str:
    """
    Return the library version string.

    Returns
    -------
    str
        Semantic version identifier.
    """

    return __version__
