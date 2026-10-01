"""Final color reconstruction.

Report section L: recombine the denoised value channel with the original hue and
saturation, convert the HSV image back to RGB, and leave H and S unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np


def _validate_hsv_component(name: str, arr: np.ndarray, *, lower: float, upper: float) -> np.ndarray:
    """Validate a 2-D float32 channel against a numeric range."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(arr).__name__}.")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {arr.shape}.")
    if arr.dtype != np.float32:
        raise TypeError(f"{name} must be float32, got dtype {arr.dtype}.")
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    if arr.min() < lower or arr.max() > upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}], got range [{arr.min()}, {arr.max()}].")
    return arr.astype(np.float32, copy=True)


def reconstruct_image(h: np.ndarray, s: np.ndarray, denoised_v: np.ndarray) -> np.ndarray:
    """Merge original H and S with the denoised V and convert HSV -> RGB. Report section L."""
    if not isinstance(h, np.ndarray) or not isinstance(s, np.ndarray) or not isinstance(denoised_v, np.ndarray):
        raise TypeError("h, s and denoised_v must all be NumPy arrays.")
    if h.shape != s.shape or h.shape != denoised_v.shape:
        raise ValueError(
            f"h, s and denoised_v must have identical shapes, got {h.shape}, {s.shape}, {denoised_v.shape}."
        )

    h_arr = _validate_hsv_component("h", h, lower=0.0, upper=360.0)
    s_arr = _validate_hsv_component("s", s, lower=0.0, upper=1.0)
    v_arr = _validate_hsv_component("denoised_v", denoised_v, lower=0.0, upper=1.0)

    v_clipped = np.clip(v_arr, 0.0, 1.0).astype(np.float32, copy=True)
    hsv = np.stack((h_arr, s_arr, v_clipped), axis=-1).astype(np.float32, copy=False)
    rgb = cv2.cvtColor(np.ascontiguousarray(hsv), cv2.COLOR_HSV2RGB)

    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    if not np.isfinite(rgb).all():
        raise ValueError("reconstructed RGB contains NaN or Inf values.")
    if rgb.min() < -1e-8 or rgb.max() > 1.0 + 1e-8:
        raise ValueError(
            f"reconstructed RGB must lie in [0, 1], got range [{rgb.min()}, {rgb.max()}]."
        )
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
