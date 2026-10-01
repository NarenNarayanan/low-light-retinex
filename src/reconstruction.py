"""Final color reconstruction.

Contract:
    reconstruct_image(h, s, denoised_v) -> enhanced_rgb
    Recombine enhanced V with the ORIGINAL H and S, then HSV -> RGB.
    Output: float32 (H, W, 3) in [0, 1].
"""
from __future__ import annotations

import numpy as np


def reconstruct_image(h: np.ndarray, s: np.ndarray, denoised_v: np.ndarray) -> np.ndarray:
    """TODO: HSV recombination and conversion back to RGB."""
    raise NotImplementedError("Not implemented yet")
