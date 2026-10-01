"""Tests for src/reconstruction.py."""
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import preprocess_image  # noqa: E402
from src.reconstruction import reconstruct_image  # noqa: E402


def test_module_imports_cleanly():
    importlib.import_module("src.reconstruction")


def test_round_trip_matches_original_rgb_within_tolerance():
    rng = np.random.default_rng(42)
    bgr = rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)
    pre = preprocess_image(bgr)
    rgb = reconstruct_image(pre.h, pre.s, pre.v)
    assert rgb.dtype == np.float32
    assert rgb.shape == pre.rgb.shape
    assert np.allclose(rgb, pre.rgb, atol=1e-5)


def test_hue_and_saturation_are_preserved_when_v_changes():
    rng = np.random.default_rng(13)
    bgr = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    pre = preprocess_image(bgr)
    v = np.clip(pre.v * 0.5, 0.0, 1.0)
    rgb = reconstruct_image(pre.h.copy(), pre.s.copy(), v)
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    h2, s2, _ = cv2.split(hsv)
    mask = (pre.s > 1e-3) & (pre.v > 1e-3)
    assert np.allclose(h2[mask], pre.h[mask], atol=1e-3)
    assert np.allclose(s2[mask], pre.s[mask], atol=1e-3)


def test_output_contract_and_constant_inputs_are_finite():
    h = np.full((8, 8), 120.0, dtype=np.float32)
    s = np.full((8, 8), 0.8, dtype=np.float32)
    v = np.full((8, 8), 0.2, dtype=np.float32)
    rgb = reconstruct_image(h, s, v)
    assert rgb.dtype == np.float32
    assert rgb.shape == (8, 8, 3)
    assert np.isfinite(rgb).all()
    assert np.all((rgb >= 0.0) & (rgb <= 1.0))

    black = reconstruct_image(np.zeros((5, 5), dtype=np.float32), np.zeros((5, 5), dtype=np.float32), np.zeros((5, 5), dtype=np.float32))
    assert np.isfinite(black).all()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        reconstruct_image(np.zeros((4, 4), dtype=np.float32), np.zeros((4, 3), dtype=np.float32), np.zeros((4, 4), dtype=np.float32))


def test_wrong_dtype_nan_and_range_raise():
    with pytest.raises(TypeError):
        reconstruct_image(np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        reconstruct_image(np.full((4, 4), np.nan, dtype=np.float32), np.ones((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        reconstruct_image(np.ones((4, 4), dtype=np.float32), np.full((4, 4), 2.0, dtype=np.float32), np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        reconstruct_image(np.ones((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32), np.full((4, 4), 1.5, dtype=np.float32))


def test_inputs_are_unchanged_after_call():
    h = np.array([[0.0, 120.0], [240.0, 300.0]], dtype=np.float32)
    s = np.array([[0.3, 0.6], [0.5, 0.8]], dtype=np.float32)
    v = np.array([[0.1, 0.2], [0.4, 0.9]], dtype=np.float32)
    h_before = h.copy(); s_before = s.copy(); v_before = v.copy()
    reconstruct_image(h, s, v)
    assert np.array_equal(h, h_before)
    assert np.array_equal(s, s_before)
    assert np.array_equal(v, v_before)
