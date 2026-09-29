"""Iterative Retinex decomposition (Core Task 2 - owner: Dhinesh Babu C M).

Contract:
    retinex_decompose(v, alpha, beta, ...) -> (illumination, reflectance)
    v: float32 (H, W) in [0, 1]; outputs float32 (H, W).

Sobel terms come from retinex_utils.compute_sobel_terms.
"""
from __future__ import annotations

import numpy as np


def retinex_decompose(
    v: np.ndarray, alpha: float, beta: float, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """TODO (Dhinesh): iterative Retinex update, convergence check, config-driven parameters."""
    raise NotImplementedError("TODO: owned by Dhinesh Babu C M")
