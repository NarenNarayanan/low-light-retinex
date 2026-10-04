"""Shared fixtures: deterministic synthetic images and a repository-root import path."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_dark_bgr(height: int = 120, width: int = 160, seed: int = 0, scale: float = 0.18) -> np.ndarray:
    """A coloured, textured, noisy low-light BGR uint8 image (deterministic)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    red = 0.35 + 0.30 * (xx / width)
    green = 0.30 + 0.30 * (yy / height)
    blue = 0.40 + 0.20 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    rgb = np.stack([red, green, blue], axis=-1)
    rgb[height // 3: 2 * height // 3, width // 4: width // 2] *= 1.6  # a brighter patch
    rgb = rgb * scale + rng.normal(0.0, 0.004, rgb.shape)
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.ascontiguousarray((rgb[..., ::-1] * 255).round().astype(np.uint8))


def make_textured_bgr(size: int = 200, seed: int = 0) -> np.ndarray:
    """A mid-brightness textured image large enough for NIQE (>= 2 blocks of 96x96)."""
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.random((size, size, 3)).astype(np.float32), (0, 0), 2.0)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return (noise * 200 + 20).astype(np.uint8)


@pytest.fixture
def dark_bgr() -> np.ndarray:
    return make_dark_bgr()


@pytest.fixture
def dark_png(tmp_path, dark_bgr) -> Path:
    path = tmp_path / "dark_scene.png"
    assert cv2.imwrite(str(path), dark_bgr)
    return path
