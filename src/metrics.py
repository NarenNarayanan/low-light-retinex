"""Image-quality metrics: Average Brightness (AB), Discrete Entropy (DE) and NIQE.

All metrics are computed on the 8-bit grayscale version of an image so that they are
directly comparable between the input and the enhanced result.

* AB  - mean 8-bit gray level, 0..255 (standard definition).
* DE  - Shannon entropy of the 256-bin gray-level histogram, ``-sum(p * log2 p)``, 0..8 bits.
* NIQE - Natural Image Quality Evaluator (Mittal et al., 2013). Lower means the image statistics are
  closer to those of pristine natural images.

How to read the numbers (none of them is a "score out of ten"):

* NIQE: lower generally indicates better natural-image statistical quality.
* AB: an appropriate value is wanted, not the largest one; a very high AB can mean over-exposure.
* DE: a higher value can indicate richer information, but amplified noise also raises it.
* Runtime: lower means less computational cost. It is measured by the pipeline, not here.

NIQE implementation: the feature extraction (MSCN coefficients, asymmetric generalised Gaussian fits,
paired-product features at two scales on 96x96 blocks) and the final multivariate-Gaussian distance follow
the published algorithm and the authors' MATLAB reference. The pre-trained pristine model is loaded from
``src/data/niqe_pris_params.npz`` (see ``src/data/NIQE_PARAMETERS.md``). NIQE values from different
implementations can differ slightly, so compare scores produced by this code with each other.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.special import gamma as _gamma

_NIQE_PARAMS_PATH = Path(__file__).resolve().parent / "data" / "niqe_pris_params.npz"
_BLOCK_SIZE = 96  # block size recommended by the NIQE authors


class NiqeUnavailable(ValueError):
    """NIQE is not defined for this image (too small, or too uniform to fit the model)."""


# --------------------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------------------
def _check_image(image: np.ndarray) -> None:
    """Accept uint8 or float [0, 1] images shaped (H, W) or (H, W, 3); reject anything else."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a NumPy array, got {type(image).__name__}.")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError(f"image must have shape (H, W) or (H, W, 3), got {image.shape}.")
    if image.size == 0:
        raise ValueError("image is empty.")
    if image.dtype == np.uint8:
        return
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("image contains NaN or Inf values.")
        if image.min() < 0.0 or image.max() > 1.0:
            raise ValueError("float images must lie in [0, 1].")
        return
    raise TypeError(f"image dtype must be uint8 or float in [0, 1], got {image.dtype}.")


def to_gray_uint8(image: np.ndarray, color_order: str = "bgr") -> np.ndarray:
    """Return the 8-bit grayscale version of an image (ITU-R BT.601 luma, as used by OpenCV).

    Accepts uint8 or float [0, 1] images; a 3-channel image is read as BGR (OpenCV order) unless
    ``color_order="rgb"``. Float input is scaled to 0..255 and rounded.
    """
    if color_order not in ("bgr", "rgb"):
        raise ValueError("color_order must be 'bgr' or 'rgb'.")
    _check_image(image)
    if image.dtype != np.uint8:
        image = np.rint(image.astype(np.float64) * 255.0).astype(np.uint8)
    if image.ndim == 2:
        return image
    code = cv2.COLOR_BGR2GRAY if color_order == "bgr" else cv2.COLOR_RGB2GRAY
    return cv2.cvtColor(np.ascontiguousarray(image), code)


# --------------------------------------------------------------------------------------
# AB and DE
# --------------------------------------------------------------------------------------
def average_brightness(image: np.ndarray, color_order: str = "bgr") -> float:
    """Average Brightness: mean 8-bit gray level (0..255)."""
    return float(np.mean(to_gray_uint8(image, color_order), dtype=np.float64))


def discrete_entropy(image: np.ndarray, color_order: str = "bgr") -> float:
    """Discrete Entropy in bits: ``-sum(p(i) * log2 p(i))`` over the 256 gray levels (0..8).

    Empty histogram bins are skipped (``0 * log 0`` is taken as 0).
    """
    gray = to_gray_uint8(image, color_order)
    counts = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p))) + 0.0  # + 0.0 turns -0.0 (constant image) into 0.0


# --------------------------------------------------------------------------------------
# NIQE
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_niqe_model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the pristine-image mean, covariance and 7x7 Gaussian window (read once)."""
    if not _NIQE_PARAMS_PATH.is_file():
        raise FileNotFoundError(f"NIQE model parameters not found: {_NIQE_PARAMS_PATH}")
    params = np.load(_NIQE_PARAMS_PATH)
    return (
        np.asarray(params["mu_pris_param"], dtype=np.float64).reshape(-1),
        np.asarray(params["cov_pris_param"], dtype=np.float64),
        np.asarray(params["gaussian_window"], dtype=np.float64),
    )


@lru_cache(maxsize=1)
def _aggd_lookup() -> tuple[np.ndarray, np.ndarray]:
    """Shape-parameter grid and the matching generalised-Gaussian ratio function r(gamma)."""
    shapes = np.arange(0.2, 10.001, 0.001)
    inv = 1.0 / shapes
    ratio = _gamma(2.0 * inv) ** 2 / (_gamma(inv) * _gamma(3.0 * inv))
    return shapes, ratio


def _fit_aggd(values: np.ndarray) -> tuple[float, float, float]:
    """Fit an asymmetric generalised Gaussian; return (shape, left scale, right scale)."""
    values = values.ravel()
    shapes, ratio = _aggd_lookup()
    left = values[values < 0]
    right = values[values > 0]
    left_std = math.sqrt(np.mean(left ** 2)) if left.size else 0.0
    right_std = math.sqrt(np.mean(right ** 2)) if right.size else 0.0
    if left_std == 0.0 or right_std == 0.0:
        return float("nan"), float("nan"), float("nan")  # degenerate block, dropped later
    gamma_hat = left_std / right_std
    r_hat = np.mean(np.abs(values)) ** 2 / np.mean(values ** 2)
    r_hat_norm = r_hat * (gamma_hat ** 3 + 1.0) * (gamma_hat + 1.0) / (gamma_hat ** 2 + 1.0) ** 2
    alpha = float(shapes[np.argmin((ratio - r_hat_norm) ** 2)])
    scale = math.sqrt(_gamma(1.0 / alpha) / _gamma(3.0 / alpha))
    return alpha, left_std * scale, right_std * scale


def _block_features(block: np.ndarray) -> list[float]:
    """18 features of one block of MSCN coefficients (2 from the coefficients, 4x4 from products)."""
    alpha, beta_l, beta_r = _fit_aggd(block)
    features = [alpha, (beta_l + beta_r) / 2.0]
    # Products of neighbouring coefficients (horizontal, vertical, two diagonals) capture structure.
    for shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        product = block * np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = _fit_aggd(product)
        mean = (beta_r - beta_l) * (_gamma(2.0 / alpha) / _gamma(1.0 / alpha))
        features.extend([alpha, mean, beta_l, beta_r])
    return features


def _mscn(image: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Mean-subtracted, contrast-normalised coefficients (Eq. 1 of the NIQE paper)."""
    mu = cv2.filter2D(image, -1, window, borderType=cv2.BORDER_REPLICATE)
    var = cv2.filter2D(image * image, -1, window, borderType=cv2.BORDER_REPLICATE) - mu * mu
    return (image - mu) / (np.sqrt(np.abs(var)) + 1.0)


@lru_cache(maxsize=None)
def _bicubic_half_matrix(length: int) -> np.ndarray:
    """Weights that halve a length-``length`` signal like MATLAB ``imresize`` (bicubic, antialiased)."""
    scale = 0.5
    out_length = length // 2
    kernel_width = 4.0 / scale  # the cubic kernel is stretched by 1/scale when shrinking
    x = np.arange(1, out_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(u - kernel_width / 2.0)
    taps = int(math.ceil(kernel_width)) + 2
    indices = left[:, None] + np.arange(taps)[None, :]
    distance = (u[:, None] - indices) * scale
    absd = np.abs(distance)
    weights = scale * np.where(
        absd <= 1.0,
        1.5 * absd ** 3 - 2.5 * absd ** 2 + 1.0,
        np.where(absd <= 2.0, -0.5 * absd ** 3 + 2.5 * absd ** 2 - 4.0 * absd + 2.0, 0.0),
    )
    weights /= weights.sum(axis=1, keepdims=True)
    matrix = np.zeros((out_length, length), dtype=np.float64)
    rows = np.repeat(np.arange(out_length), taps)
    cols = indices.ravel().astype(np.int64) - 1  # to 0-based
    # Symmetric (mirror) padding at the borders, as MATLAB does.
    cols = np.where(cols < 0, -cols - 1, cols)
    cols = np.where(cols >= length, 2 * length - 1 - cols, cols)
    np.add.at(matrix, (rows, cols), weights.ravel())
    return matrix


def _downscale_half(image: np.ndarray) -> np.ndarray:
    """Halve a 2-D image with MATLAB-style antialiased bicubic interpolation."""
    height, width = image.shape
    return _bicubic_half_matrix(height) @ image @ _bicubic_half_matrix(width).T


def niqe_from_gray(gray: np.ndarray) -> float:
    """NIQE of a 2-D grayscale image given in 0..255 (float or integer values).

    The image is cropped to a whole number of 96x96 blocks. At least two blocks are required, because
    the method fits a covariance matrix to the per-block features.

    Raises:
        ValueError: the image is not 2-D, or is too small to contain two 96x96 blocks.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("NIQE needs a 2-D grayscale array.")
    rows, cols = gray.shape[0] // _BLOCK_SIZE, gray.shape[1] // _BLOCK_SIZE
    if rows * cols < 2:
        raise NiqeUnavailable(
            f"Image is too small for NIQE: need at least two {_BLOCK_SIZE}x{_BLOCK_SIZE} blocks, "
            f"got {gray.shape[1]}x{gray.shape[0]} pixels."
        )
    mu_pris, cov_pris, window = _load_niqe_model()
    img = gray.astype(np.float64)[: rows * _BLOCK_SIZE, : cols * _BLOCK_SIZE]

    per_scale = []
    for scale in (1, 2):
        mscn = _mscn(img, window)
        block = _BLOCK_SIZE // scale
        feats = [
            _block_features(mscn[r * block:(r + 1) * block, c * block:(c + 1) * block])
            for c in range(cols) for r in range(rows)
        ]
        per_scale.append(np.array(feats))
        if scale == 1:
            img = _downscale_half(img / 255.0) * 255.0
    features = np.concatenate(per_scale, axis=1)  # (blocks, 36)

    features = features[~np.isnan(features).any(axis=1)]  # drop degenerate (flat) blocks
    if features.shape[0] < 2:
        raise NiqeUnavailable("Image has too few textured blocks for NIQE (it is almost uniform).")
    mu_dist = features.mean(axis=0)
    cov_dist = np.cov(features, rowvar=False)

    # Distance between the pristine model and the image's fitted Gaussian (Eq. 10 of the paper).
    diff = mu_pris - mu_dist
    inv_cov = np.linalg.pinv((cov_pris + cov_dist) / 2.0)
    return float(np.sqrt(diff @ inv_cov @ diff))


def niqe(image: np.ndarray, color_order: str = "bgr") -> float:
    """NIQE of an image (lower is better). Computed on the 8-bit grayscale version."""
    return niqe_from_gray(to_gray_uint8(image, color_order).astype(np.float64))


# --------------------------------------------------------------------------------------
# Combined record
# --------------------------------------------------------------------------------------
def evaluate_image(image: np.ndarray, runtime_s: float | None = None, color_order: str = "bgr") -> dict:
    """Return ``{"niqe", "ab", "de", "runtime_s"}`` for one image.

    ``runtime_s`` is the processing time measured by the pipeline (not re-measured here). ``niqe`` is
    ``None`` when the image is too small or too uniform for NIQE to be defined; AB and DE are always
    returned.
    """
    if runtime_s is not None and (not np.isfinite(runtime_s) or runtime_s < 0):
        raise ValueError(f"runtime_s must be a finite non-negative number, got {runtime_s}.")
    try:
        niqe_value: float | None = niqe(image, color_order)
    except NiqeUnavailable:
        niqe_value = None  # AB and DE are still valid; invalid input errors are not swallowed
    return {
        "niqe": niqe_value,
        "ab": average_brightness(image, color_order),
        "de": discrete_entropy(image, color_order),
        "runtime_s": None if runtime_s is None else float(runtime_s),
    }
