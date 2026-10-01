"""Dark-region detection and selective Fast NLM denoising.

Report section K: create a dark-region mask on the enhanced value channel and apply
Fast NLM only inside that mask while leaving brighter regions unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

_DEFAULT_DARK_THRESHOLD = 0.3
_DEFAULT_H = 10.0
_DEFAULT_TEMPLATE_WINDOW_SIZE = 7
_DEFAULT_SEARCH_WINDOW_SIZE = 21


def _validate_enhanced_v(enhanced_v: np.ndarray) -> np.ndarray:
    """Validate a 2-D numeric brightness image in [0, 1]."""
    if not isinstance(enhanced_v, np.ndarray):
        raise TypeError(
            f"enhanced_v must be a NumPy array, got {type(enhanced_v).__name__}."
        )
    if enhanced_v.ndim != 2:
        raise ValueError(
            f"enhanced_v must be a 2-D array, got shape {enhanced_v.shape}."
        )
    if enhanced_v.size == 0:
        raise ValueError("enhanced_v is empty.")
    if not np.issubdtype(enhanced_v.dtype, np.number):
        raise TypeError(
            f"enhanced_v must be numeric, got dtype {enhanced_v.dtype}."
        )
    if not np.isfinite(enhanced_v).all():
        raise ValueError("enhanced_v contains NaN or Inf values.")
    arr = np.asarray(enhanced_v, dtype=np.float32)
    if arr.min() < -1e-8 or arr.max() > 1.0 + 1e-8:
        raise ValueError(
            f"enhanced_v must lie in [0, 1], got range [{arr.min()}, {arr.max()}]."
        )
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=True)


def _validate_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Validate the boolean mask used to select noisy dark regions."""
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a NumPy array, got {type(mask).__name__}.")
    if mask.shape != shape:
        raise ValueError(f"mask must have shape {shape}, got {mask.shape}.")
    if mask.dtype != np.bool_:
        raise TypeError(f"mask must be boolean, got dtype {mask.dtype}.")
    return mask.astype(bool, copy=False)


def _validate_positive_param(name: str, value: float, *, allow_zero: bool = False) -> float:
    """Validate positivity constraints for the NLM parameters."""
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    if allow_zero:
        if numeric < 0.0:
            raise ValueError(f"{name} must be >= 0.")
    elif numeric <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return numeric


def _float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a float32 [0, 1] image to uint8 [0, 255]."""
    clipped = np.clip(image.astype(np.float32, copy=True), 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8)


def _uint8_to_float(image_uint8: np.ndarray) -> np.ndarray:
    """Convert a uint8 image back to float32 in [0, 1]."""
    return (image_uint8.astype(np.float32) / 255.0).astype(np.float32, copy=False)


def detect_dark_regions(enhanced_v: np.ndarray, threshold: float = _DEFAULT_DARK_THRESHOLD) -> np.ndarray:
    """Create a boolean dark-region mask from the enhanced brightness. Report section K."""
    validated = _validate_enhanced_v(enhanced_v)
    if not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise TypeError(f"threshold must be a real number, got {type(threshold).__name__}.")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite.")
    if threshold_value < 0.0 or threshold_value > 1.0:
        raise ValueError(
            f"threshold must lie in [0, 1], got {threshold_value}."
        )
    return validated < threshold_value


def denoise_dark_regions(
    enhanced_v: np.ndarray,
    mask: np.ndarray,
    **params,
) -> np.ndarray:
    """Apply Fast NLM only inside the dark-region mask. Report section K."""
    validated_v = _validate_enhanced_v(enhanced_v)
    validated_mask = _validate_mask(mask, validated_v.shape)

    h = params.get("h", _DEFAULT_H)
    template_window_size = params.get("template_window_size", _DEFAULT_TEMPLATE_WINDOW_SIZE)
    search_window_size = params.get("search_window_size", _DEFAULT_SEARCH_WINDOW_SIZE)

    _validate_positive_param("h", h)
    _validate_positive_param("template_window_size", template_window_size)
    _validate_positive_param("search_window_size", search_window_size)

    template = int(template_window_size)
    search = int(search_window_size)
    if template % 2 == 0:
        raise ValueError("template_window_size must be odd.")
    if search % 2 == 0:
        raise ValueError("search_window_size must be odd.")

    uint8_v = _float_to_uint8(validated_v)
    denoised_uint8 = cv2.fastNlMeansDenoising(
        uint8_v,
        h=float(h),
        templateWindowSize=template,
        searchWindowSize=search,
    )
    denoised_v = _uint8_to_float(denoised_uint8).astype(np.float32, copy=False)

    result = np.where(validated_mask, denoised_v, validated_v)
    result = result.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("denoised output contains NaN or Inf values.")
    if result.min() < -1e-8 or result.max() > 1.0 + 1e-8:
        raise ValueError(
            f"denoised output must lie in [0, 1], got range [{result.min()}, {result.max()}]."
        )
    return np.clip(result, 0.0, 1.0).astype(np.float32)
