"""Tests for src/preprocessing.py."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import preprocessing  # noqa: E402
from src.preprocessing import (  # noqa: E402
    PreprocessedImage,
    initial_denoise,
    prepare_rgb,
    preprocess_image,
    rgb_to_hsv_channels,
    validate_image,
)


def _random_bgr(h=12, w=16, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---- validate_image ----
def test_valid_color_image_passes():
    validate_image(_random_bgr())


def test_none_raises():
    with pytest.raises(TypeError):
        validate_image(None)


def test_non_array_raises():
    with pytest.raises(TypeError):
        validate_image([[[0, 0, 0]]])


def test_grayscale_raises():
    with pytest.raises(ValueError):
        validate_image(np.zeros((8, 8), dtype=np.uint8))


def test_four_channel_raises():
    with pytest.raises(ValueError):
        validate_image(np.zeros((8, 8, 4), dtype=np.uint8))


def test_non_finite_raises():
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[1, 1, 1] = np.nan
    with pytest.raises(ValueError):
        validate_image(img)
    img[1, 1, 1] = np.inf
    with pytest.raises(ValueError):
        validate_image(img)


# ---- prepare_rgb ----
def test_prepared_rgb_is_float32_in_unit_range():
    rgb = prepare_rgb(_random_bgr())
    assert rgb.dtype == np.float32
    assert rgb.shape == (12, 16, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_prepare_rgb_swaps_bgr_to_rgb():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # pure blue in BGR
    rgb = prepare_rgb(bgr)
    assert np.allclose(rgb[..., 2], 1.0) and np.allclose(rgb[..., :2], 0.0)


def test_prepare_rgb_uint16_and_float_inputs():
    u16 = np.full((2, 2, 3), 65535, dtype=np.uint16)
    assert np.allclose(prepare_rgb(u16), 1.0)
    f64 = np.full((2, 2, 3), 0.25, dtype=np.float64)
    out = prepare_rgb(f64)
    assert out.dtype == np.float32 and np.allclose(out, 0.25)


def test_prepare_rgb_rejects_out_of_range_float_and_bad_int_dtype():
    with pytest.raises(ValueError):
        prepare_rgb(np.full((2, 2, 3), 200.0, dtype=np.float32))
    with pytest.raises(TypeError):
        prepare_rgb(np.zeros((2, 2, 3), dtype=np.int32))


# ---- initial_denoise ----
def test_initial_denoise_is_identity_and_does_not_mutate():
    rgb = prepare_rgb(_random_bgr())
    before = rgb.copy()
    out = initial_denoise(rgb)
    assert np.array_equal(out, before)
    assert np.array_equal(rgb, before)


# ---- rgb_to_hsv_channels ----
def test_hsv_channels_share_shape_and_dtype():
    rgb = prepare_rgb(_random_bgr(10, 7))
    h, s, v = rgb_to_hsv_channels(rgb)
    for ch in (h, s, v):
        assert ch.shape == (10, 7)
        assert ch.dtype == np.float32
    assert 0.0 <= s.min() and s.max() <= 1.0
    assert 0.0 <= v.min() and v.max() <= 1.0


def test_hsv_known_colors_use_opencv_degree_hue():
    rgb = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float32)
    h, s, v = rgb_to_hsv_channels(rgb)
    assert np.allclose(h[0], [0, 120, 240])
    assert np.allclose(s, 1.0) and np.allclose(v, 1.0)


def test_rgb_to_hsv_rejects_bad_input():
    with pytest.raises(TypeError):
        rgb_to_hsv_channels(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        rgb_to_hsv_channels(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        rgb_to_hsv_channels(np.full((2, 2, 3), 2.0, dtype=np.float32))


# ---- preprocess_image ----
@pytest.mark.parametrize("value", [0, 128, 255])
def test_constant_image_has_no_nan_or_inf(value):
    pre = preprocess_image(np.full((6, 9, 3), value, dtype=np.uint8))
    for arr in (pre.rgb, pre.h, pre.s, pre.v):
        assert np.isfinite(arr).all()


def test_preprocess_returns_consistent_dataclass():
    bgr = _random_bgr(11, 13)
    before = bgr.copy()
    pre = preprocess_image(bgr)
    assert isinstance(pre, PreprocessedImage)
    assert pre.rgb.shape == (11, 13, 3)
    assert pre.h.shape == pre.s.shape == pre.v.shape == (11, 13)
    assert np.allclose(pre.v, pre.rgb.max(axis=2), atol=1e-6)  # V = max(R, G, B)
    assert np.array_equal(bgr, before)  # input untouched


def test_preprocess_rejects_invalid_input():
    with pytest.raises(TypeError):
        preprocess_image(None)
    with pytest.raises(ValueError):
        preprocess_image(np.zeros((4, 4), dtype=np.uint8))


def test_module_imports_cleanly_without_side_effects(capsys):
    importlib.reload(preprocessing)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
