"""Illumination enhancement using adaptive gamma correction and CLAHE.

The public pipeline accepts a float32 illumination image in [0, 1] and
returns a float32 enhanced illumination image in [0, 1].
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _validate_illumination(illumination: np.ndarray) -> None:
    """Validate the illumination image."""
    if not isinstance(illumination, np.ndarray):
        raise TypeError("illumination must be a NumPy array")

    if illumination.ndim != 2:
        raise ValueError("illumination must be a 2-D array")

    if not np.all(np.isfinite(illumination)):
        raise ValueError("illumination must not contain NaN or Inf")

    if np.any(illumination < 0.0) or np.any(illumination > 1.0):
        raise ValueError("illumination values must be in [0, 1]")


def compute_adaptive_gamma(illumination: np.ndarray) -> float:
    """Compute adaptive gamma from mean illumination.

    Implements the adaptive gamma method described in report sections I and J.
    """
    _validate_illumination(illumination)

    mean_illumination = float(np.mean(illumination))

    if mean_illumination < 0.5:
        return 1.0 + 2.0 * (0.5 - mean_illumination)

    return 1.0


def apply_gamma_correction(
    illumination: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Apply gamma correction while preserving the [0, 1] range.

    Sections I and J require gamma > 1 to produce stronger brightening.
    Therefore, the exponent 1 / gamma is used instead of gamma.
    """
    _validate_illumination(illumination)

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite positive value")

    corrected = np.power(
        illumination.astype(np.float32),
        1.0 / gamma,
    )

    return np.clip(corrected, 0.0, 1.0).astype(np.float32)


def apply_clahe(
    v: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to normalized illumination.

    Implements the CLAHE stage described in report sections I and J.
    """
    _validate_illumination(v)

    if not np.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("clip_limit must be a positive finite value")

    if len(tile_grid_size) != 2:
        raise ValueError("tile_grid_size must contain exactly two values")

    rows, cols = tile_grid_size

    if not isinstance(rows, (int, np.integer)) or not isinstance(
        cols, (int, np.integer)
    ):
        raise TypeError("tile_grid_size values must be integers")

    if rows <= 0 or cols <= 0:
        raise ValueError("tile_grid_size values must be positive")

    # The report does not specify bit depth. Use uint8 as a conventional
    # OpenCV representation for normalized [0, 1] illumination.
    image_uint8 = np.round(v * 255.0).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(rows), int(cols)),
    )

    enhanced_uint8 = clahe.apply(image_uint8)

    return (
        enhanced_uint8.astype(np.float32) / 255.0
    ).astype(np.float32)


def enhance_illumination(
    illumination: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance illumination using adaptive gamma followed by CLAHE.

    Implements the enhancement pipeline described in report sections I and J.
    """
    _validate_illumination(illumination)

    gamma = compute_adaptive_gamma(illumination)

    gamma_corrected = apply_gamma_correction(
        illumination,
        gamma,
    )

    enhanced_v = apply_clahe(
        gamma_corrected,
        clip_limit=clahe_clip_limit,
        tile_grid_size=clahe_tile_grid_size,
    )

    return np.clip(
        enhanced_v,
        0.0,
        1.0,
    ).astype(np.float32)