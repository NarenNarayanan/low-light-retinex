"""Dark-region detection and selective Fast NLM denoising.

Contract:
    denoise_dark_regions(enhanced_v, mask, ...) -> denoised_v
    enhanced_v: float32 (H, W); mask: bool (H, W); denoised_v: float32 (H, W).
"""
from __future__ import annotations

import numpy as np


def detect_dark_regions(enhanced_v: np.ndarray, **kwargs) -> np.ndarray:
    """TODO: boolean dark-region mask on the enhanced brightness."""
    raise NotImplementedError("Not implemented yet")


def denoise_dark_regions(enhanced_v: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
    """TODO: apply Fast NLM selectively inside the mask."""
    raise NotImplementedError("Not implemented yet")
