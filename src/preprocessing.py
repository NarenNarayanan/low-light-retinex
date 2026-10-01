"""Input preprocessing.

Responsibilities:
    * image validation
    * optional initial denoising hook (the source does not specify an algorithm)
    * BGR -> RGB conversion and float32 [0, 1] normalization
    * RGB -> HSV conversion and H/S/V extraction

Working convention: RGB is float32, shape (H, W, 3), values in [0, 1].

HSV convention (OpenCV, float32 input): H is in degrees [0, 360), S and V are in [0, 1].
H is deliberately NOT rescaled, so cv2.cvtColor(..., cv2.COLOR_HSV2RGB) in
reconstruction.py round-trips without any extra conversion.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Tolerance for float inputs that are meant to lie in [0, 1].
_RANGE_TOLERANCE = 1e-6


@dataclass
class PreprocessedImage:
    """Carries RGB, H, S and V arrays safely between pipeline stages.

    rgb: float32, shape (H, W, 3), values in [0, 1]
    h:   float32, shape (H, W), hue in degrees [0, 360)  (OpenCV float HSV)
    s:   float32, shape (H, W), values in [0, 1]
    v:   float32, shape (H, W), values in [0, 1]
    """

    rgb: np.ndarray
    h: np.ndarray
    s: np.ndarray
    v: np.ndarray


def validate_image(image_bgr: np.ndarray) -> None:
    """Check that ``image_bgr`` is a usable H x W x 3 numeric image with finite values.

    Raises:
        TypeError: input is None, not a NumPy array, or has a non-numeric dtype.
        ValueError: input is not H x W x 3, is empty, or contains NaN/Inf.
    """
    if image_bgr is None:
        raise TypeError(
            "image_bgr is None (cv2.imread returns None when a file cannot be read)."
        )
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError(f"image_bgr must be a NumPy array, got {type(image_bgr).__name__}.")
    if not (
        np.issubdtype(image_bgr.dtype, np.integer)
        or np.issubdtype(image_bgr.dtype, np.floating)
    ):
        raise TypeError(f"image_bgr must have a numeric dtype, got {image_bgr.dtype}.")
    if image_bgr.ndim != 3:
        raise ValueError(
            f"image_bgr must be a 3-D H x W x 3 array, got {image_bgr.ndim}-D "
            f"with shape {image_bgr.shape}."
        )
    if image_bgr.shape[2] != 3:
        raise ValueError(
            f"image_bgr must have exactly 3 channels, got {image_bgr.shape[2]} "
            f"(shape {image_bgr.shape})."
        )
    if image_bgr.size == 0:
        raise ValueError(f"image_bgr is empty (shape {image_bgr.shape}).")
    if not np.isfinite(image_bgr).all():
        raise ValueError("image_bgr contains NaN or Inf values.")


def initial_denoise(image_rgb: np.ndarray) -> np.ndarray:
    """Optional initial denoising hook - currently returns the image unchanged.

    The project report lists an initial denoising stage in the architecture but does
    not specify which algorithm it uses. Rather than invent one (Gaussian, median,
    bilateral, NLM, ...), this hook is a no-op that keeps the stage visible in the
    pipeline. Replace the body only if the course specification names a filter.

    The input array itself is returned (no copy, no modification).
    """
    return image_rgb


def prepare_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to float32 RGB in [0, 1].

    Expects an image that already passed :func:`validate_image`.
    Supported dtypes: uint8 (divided by 255), uint16 (divided by 65535), and
    float32/float64, which must already lie in [0, 1]. Dimensions are unchanged.

    Raises:
        TypeError: unsupported integer dtype.
        ValueError: float values outside [0, 1].
    """
    dtype = image_bgr.dtype
    if dtype == np.uint8:
        bgr = image_bgr.astype(np.float32) / 255.0
    elif dtype == np.uint16:
        bgr = image_bgr.astype(np.float32) / 65535.0
    elif np.issubdtype(dtype, np.floating):
        low, high = float(image_bgr.min()), float(image_bgr.max())
        if low < -_RANGE_TOLERANCE or high > 1.0 + _RANGE_TOLERANCE:
            raise ValueError(
                f"Float images must lie in [0, 1], got range [{low}, {high}]."
            )
        bgr = np.clip(image_bgr, 0.0, 1.0).astype(np.float32)
    else:
        raise TypeError(
            f"Unsupported dtype {dtype}; use uint8, uint16, float32 or float64."
        )
    return cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)


def rgb_to_hsv_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert float32 RGB in [0, 1] to separate H, S, V arrays using OpenCV.

    Returns (h, s, v), each float32 with shape (H, W). H is in degrees [0, 360);
    S and V are in [0, 1]. H and S are returned as-is and must not be modified
    downstream: they are reused unchanged for the final reconstruction.
    """
    if not isinstance(rgb, np.ndarray):
        raise TypeError(f"rgb must be a NumPy array, got {type(rgb).__name__}.")
    if rgb.dtype != np.float32:
        raise TypeError(f"rgb must be float32 in [0, 1], got dtype {rgb.dtype}.")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}.")
    if not np.isfinite(rgb).all():
        raise ValueError("rgb contains NaN or Inf values.")
    if rgb.min() < -_RANGE_TOLERANCE or rgb.max() > 1.0 + _RANGE_TOLERANCE:
        raise ValueError(
            f"rgb must lie in [0, 1], got range [{rgb.min()}, {rgb.max()}]."
        )

    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    return h, s, v


def preprocess_image(image_bgr: np.ndarray) -> PreprocessedImage:
    """Run preprocessing: validate -> BGR->RGB float32 -> initial denoise hook -> HSV split.

    No Retinex or enhancement happens here. The input array is not modified.
    """
    validate_image(image_bgr)
    rgb = prepare_rgb(image_bgr)
    rgb = initial_denoise(rgb)
    h, s, v = rgb_to_hsv_channels(rgb)
    return PreprocessedImage(rgb=rgb, h=h, s=s, v=v)
