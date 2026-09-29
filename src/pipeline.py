"""End-to-end pipeline orchestration (owner: Naren Narayanan P S).

Intended flow:
    load -> preprocess -> mean V + alpha/beta -> Retinex -> gamma + CLAHE
         -> dark mask -> selective Fast NLM -> HSV->RGB -> metrics

Current state: a skeleton that runs Task 1 (preprocessing, mean V, alpha/beta) and
stops cleanly before the modules owned by other members.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .preprocessing import PreprocessedImage, preprocess_image
from .retinex_utils import calculate_mean_v, initialize_retinex_parameters

logger = logging.getLogger(__name__)


def load_image(input_path: str | Path) -> np.ndarray:
    """Read an image with OpenCV and return it as a BGR uint8 array.

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the file exists but OpenCV cannot decode it as a colour image.
    """
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read an image from: {path}")
    return image_bgr


def run_pipeline(input_path: str | Path) -> dict:
    """Run the currently implemented stages and return their hand-off values.

    Returns a dict with:
        preprocessed: PreprocessedImage (rgb, h, s, v)
        mean_v:       float
        alpha, beta:  float (adaptive Retinex weights)
    """
    start = time.perf_counter()
    logger.info("Input path: %s", input_path)

    image_bgr = load_image(input_path)
    logger.info("Image shape: %s", image_bgr.shape)

    preprocessed: PreprocessedImage = preprocess_image(image_bgr)

    mean_v = calculate_mean_v(preprocessed.v)
    alpha, beta = initialize_retinex_parameters(mean_v)

    # Future member modules (not implemented here - owned by other members):
    # illumination, reflectance = retinex_decompose(preprocessed.v, alpha, beta, ...)  # Dhinesh
    # enhanced_v = enhance_illumination(illumination, ...)                             # Sarvesh
    # dark_mask = detect_dark_regions(enhanced_v, ...); denoised_v = denoise_dark_regions(enhanced_v, dark_mask, ...)  # Syam
    # enhanced_rgb = reconstruct_image(preprocessed.h, preprocessed.s, denoised_v)     # Syam

    logger.info(
        "Stage 1 done in %.3fs (mean_v=%.4f, alpha=%.6f, beta=%.6f)",
        time.perf_counter() - start, mean_v, alpha, beta,
    )
    return {
        "preprocessed": preprocessed,
        "mean_v": mean_v,
        "alpha": alpha,
        "beta": beta,
    }
