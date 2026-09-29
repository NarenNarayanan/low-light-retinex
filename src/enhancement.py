"""Illumination enhancement (Task 3 - owner: Sarvesh S).

Contract:
    enhance_illumination(illumination, ...) -> enhanced_v
    illumination: float32 (H, W); enhanced_v: float32 (H, W) in [0, 1].

Scope: adaptive gamma correction followed by CLAHE.
"""
from __future__ import annotations

import numpy as np


def enhance_illumination(illumination: np.ndarray, **kwargs) -> np.ndarray:
    """TODO (Sarvesh): adaptive gamma correction + CLAHE."""
    raise NotImplementedError("TODO: owned by Sarvesh S")
