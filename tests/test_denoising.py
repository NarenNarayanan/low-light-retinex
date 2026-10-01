"""Tests for src/denoising.py."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import denoising  # noqa: E402


@pytest.fixture
def sample_image():
    image = np.linspace(0.05, 0.95, 256, dtype=np.float32).reshape(16, 16)
    return image


def test_module_imports_cleanly():
    importlib.import_module("src.denoising")


def test_detect_dark_regions_shape_and_threshold_behavior():
    image = np.array(
        [[0.1, 0.2, 0.4], [0.6, 0.7, 0.9]],
        dtype=np.float32,
    )
    mask = denoising.detect_dark_regions(image, threshold=0.5)
    assert mask.dtype == np.bool_
    assert mask.shape == image.shape
    assert np.array_equal(mask, np.array([[True, True, True], [False, False, False]], dtype=bool))


def test_selective_denoising_keeps_pixels_outside_mask_identical(sample_image):
    mask = sample_image < 0.5
    original = sample_image.copy()
    out = denoising.denoise_dark_regions(sample_image, mask)
    assert np.array_equal(out[~mask], original[~mask])


def test_denoising_reduces_noise_inside_dark_mask():
    rng = np.random.default_rng(7)
    base = np.full((64, 64), 0.2, dtype=np.float32)
    noisy = np.clip(base + rng.normal(0.0, 0.05, size=(64, 64)).astype(np.float32), 0.0, 1.0)
    mask = noisy < 0.5
    original_std = float(np.std(noisy[mask] - 0.2))
    denoised = denoising.denoise_dark_regions(noisy, mask)
    denoised_std = float(np.std(denoised[mask] - 0.2))
    assert denoised_std < original_std


def test_all_false_mask_returns_input_unchanged():
    image = np.linspace(0.1, 0.9, 100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros_like(image, dtype=bool)
    out = denoising.denoise_dark_regions(image, mask)
    assert np.array_equal(out, image)


def test_all_true_mask_denoises_whole_image():
    image = np.full((32, 32), 0.2, dtype=np.float32)
    image[0:16, :] += 0.05
    mask = np.ones_like(image, dtype=bool)
    out = denoising.denoise_dark_regions(image, mask)
    assert out.dtype == np.float32
    assert out.shape == image.shape
    assert np.isfinite(out).all()
    assert np.all((out >= 0.0) & (out <= 1.0))
    assert not np.array_equal(out, image)


def test_output_contract_and_reproducibility(sample_image):
    mask = sample_image < 0.5
    out1 = denoising.denoise_dark_regions(sample_image, mask)
    out2 = denoising.denoise_dark_regions(sample_image, mask)
    assert out1.dtype == np.float32
    assert out1.shape == sample_image.shape
    assert np.isfinite(out1).all()
    assert (out1 >= 0.0).all() and (out1 <= 1.0).all()
    assert np.array_equal(out1, out2)
    assert np.array_equal(sample_image, sample_image.copy())


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.array([[True, False]], dtype=np.uint8),
        np.array([[True, False, True]], dtype=bool),
        np.zeros((2, 2, 2), dtype=bool),
    ],
)
def test_invalid_mask_values_raise(sample_image, bad_mask):
    with pytest.raises((TypeError, ValueError)):
        denoising.denoise_dark_regions(sample_image, bad_mask)


@pytest.mark.parametrize(
    "bad_v,bad_mask",
    [
        (np.full((4, 4), np.nan, dtype=np.float32), np.zeros((4, 4), dtype=bool)),
        (np.full((4, 4), 1.5, dtype=np.float32), np.zeros((4, 4), dtype=bool)),
        (np.zeros((4, 4, 1), dtype=np.float32), np.zeros((4, 4, 1), dtype=bool)),
    ],
)
def test_invalid_input_values_raise(bad_v, bad_mask):
    with pytest.raises((TypeError, ValueError)):
        denoising.denoise_dark_regions(bad_v, bad_mask)


@pytest.mark.parametrize("bad_param", [0, -1, 0.0])
def test_non_positive_nlm_parameters_raise(sample_image, bad_param):
    with pytest.raises(ValueError):
        denoising.denoise_dark_regions(sample_image, sample_image < 0.5, h=bad_param)
    with pytest.raises(ValueError):
        denoising.denoise_dark_regions(sample_image, sample_image < 0.5, template_window_size=bad_param)
    with pytest.raises(ValueError):
        denoising.denoise_dark_regions(sample_image, sample_image < 0.5, search_window_size=bad_param)
