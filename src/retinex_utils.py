"""Retinex helper utilities.

Contains only the shared helpers: mean V, adaptive alpha/beta initialization and
Sobel terms. The iterative Retinex update (T^(k+1), R^(k+1)) lives in retinex.py.
This module must not import retinex.py.
"""
from __future__ import annotations

import cv2
import numpy as np

# Adaptive alpha/beta formulas and clipping bounds, exactly as specified:
#   alpha = clip(0.001  + 0.002  * (0.5 - mean_v), 0.0001, 0.003)
#   beta  = clip(0.0001 + 0.0003 * (0.5 - mean_v), 0.0001, 0.0005)
_MIDPOINT = 0.5
_ALPHA_BASE, _ALPHA_SLOPE, _ALPHA_MIN, _ALPHA_MAX = 0.001, 0.002, 0.0001, 0.003
_BETA_BASE, _BETA_SLOPE, _BETA_MIN, _BETA_MAX = 0.0001, 0.0003, 0.0001, 0.0005

_RANGE_TOLERANCE = 1e-6


def _validate_2d_numeric(image: np.ndarray, name: str) -> None:
    """Shared check: non-empty 2-D numeric array with finite values."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(image).__name__}.")
    if not (
        np.issubdtype(image.dtype, np.integer) or np.issubdtype(image.dtype, np.floating)
    ):
        raise TypeError(f"{name} must have a numeric dtype, got {image.dtype}.")
    if image.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (H, W) array, got shape {image.shape}.")
    if image.size == 0:
        raise ValueError(f"{name} is empty (shape {image.shape}).")
    if not np.isfinite(image).all():
        raise ValueError(f"{name} contains NaN or Inf values.")


def calculate_mean_v(v: np.ndarray) -> float:
    """Return the mean of the V channel as a Python float.

    ``v`` must be a finite 2-D numeric array in [0, 1] (project convention).
    """
    _validate_2d_numeric(v, "v")
    if v.min() < -_RANGE_TOLERANCE or v.max() > 1.0 + _RANGE_TOLERANCE:
        raise ValueError(f"v must lie in [0, 1], got range [{v.min()}, {v.max()}].")
    return float(np.mean(v))


def initialize_retinex_parameters(mean_v: float) -> tuple[float, float]:
    """Compute the adaptive Retinex weights (alpha, beta) from the mean V.

    Darker images (smaller mean_v) get larger weights. Both values are clipped to
    the bounds required by the methodology. Returns Python floats.
    """
    if isinstance(mean_v, (bool, np.bool_)) or not isinstance(
        mean_v, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"mean_v must be a real number, got {type(mean_v).__name__}.")
    mean_v = float(mean_v)
    if not np.isfinite(mean_v):
        raise ValueError(f"mean_v must be finite, got {mean_v}.")
    if not (0.0 - _RANGE_TOLERANCE <= mean_v <= 1.0 + _RANGE_TOLERANCE):
        raise ValueError(f"mean_v must lie in [0, 1], got {mean_v}.")

    darkness = _MIDPOINT - mean_v
    alpha_raw = _ALPHA_BASE + _ALPHA_SLOPE * darkness
    beta_raw = _BETA_BASE + _BETA_SLOPE * darkness
    alpha = float(np.clip(alpha_raw, _ALPHA_MIN, _ALPHA_MAX))
    beta = float(np.clip(beta_raw, _BETA_MIN, _BETA_MAX))
    return alpha, beta


def compute_sobel_gradient(image: np.ndarray) -> np.ndarray:
    """Return the Sobel gradient magnitude of a 2-D image/estimate as float32.

    Uses a 3x3 Sobel kernel (unnormalized) in x and y: G = sqrt(gx^2 + gy^2).
    Generic on purpose: the Retinex loop can call it on whichever current estimate
    it needs, on each iteration.
    """
    _validate_2d_numeric(image, "image")
    img = np.ascontiguousarray(image, dtype=np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def compute_sobel_terms(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (G, D) for a 2-D image/estimate.

    G is the Sobel gradient magnitude and D = abs(G), as the methodology defines it.
    (A magnitude is already non-negative, so D equals G numerically; they are
    returned as separate arrays because the Retinex updates use them as distinct terms.)
    """
    gradient = compute_sobel_gradient(image)
    return gradient, np.abs(gradient)
