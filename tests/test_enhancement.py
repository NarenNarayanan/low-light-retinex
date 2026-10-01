"""Tests for illumination enhancement."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_module_imports_cleanly():
    importlib.import_module("src.enhancement")


def test_adaptive_gamma_dark_image():
    from src.enhancement import compute_adaptive_gamma

    illumination = np.full(
        (10, 10),
        0.25,
        dtype=np.float32,
    )

    gamma = compute_adaptive_gamma(illumination)

    assert gamma == pytest.approx(1.5)


def test_adaptive_gamma_very_dark_image():
    from src.enhancement import compute_adaptive_gamma

    illumination = np.zeros(
        (10, 10),
        dtype=np.float32,
    )

    gamma = compute_adaptive_gamma(illumination)

    assert gamma == pytest.approx(2.0)


def test_adaptive_gamma_midpoint():
    from src.enhancement import compute_adaptive_gamma

    illumination = np.full(
        (10, 10),
        0.5,
        dtype=np.float32,
    )

    gamma = compute_adaptive_gamma(illumination)

    assert gamma == pytest.approx(1.0)


def test_adaptive_gamma_bright_image():
    from src.enhancement import compute_adaptive_gamma

    illumination = np.full(
        (10, 10),
        0.8,
        dtype=np.float32,
    )

    gamma = compute_adaptive_gamma(illumination)

    assert gamma == pytest.approx(1.0)


def test_gamma_direction_brightens_dark_pixel():
    from src.enhancement import (
        apply_gamma_correction,
        compute_adaptive_gamma,
    )

    illumination = np.full(
        (10, 10),
        0.1,
        dtype=np.float32,
    )

    gamma = compute_adaptive_gamma(illumination)

    # The naive T ** gamma would darken a value below 1.
    naive = illumination ** gamma

    assert float(naive.mean()) < float(illumination.mean())

    # The required direction is gamma > 1 -> stronger brightening.
    corrected = apply_gamma_correction(
        illumination,
        gamma,
    )

    assert float(corrected.mean()) > float(illumination.mean())


def test_gamma_identity_when_gamma_is_one():
    from src.enhancement import apply_gamma_correction

    illumination = np.array(
        [[0.1, 0.3, 0.5, 0.8]],
        dtype=np.float32,
    )

    corrected = apply_gamma_correction(
        illumination,
        gamma=1.0,
    )

    np.testing.assert_allclose(
        corrected,
        illumination,
        atol=1e-6,
    )


def test_bright_map_is_unchanged_by_gamma():
    from src.enhancement import (
        apply_gamma_correction,
        compute_adaptive_gamma,
    )

    illumination = np.full(
        (10, 10),
        0.8,
        dtype=np.float32,
    )

    gamma = compute_adaptive_gamma(illumination)

    corrected = apply_gamma_correction(
        illumination,
        gamma,
    )

    np.testing.assert_allclose(
        corrected,
        illumination,
        atol=1e-6,
    )


def test_gamma_output_range():
    from src.enhancement import apply_gamma_correction

    illumination = np.linspace(
        0.0,
        1.0,
        100,
        dtype=np.float32,
    ).reshape(10, 10)

    corrected = apply_gamma_correction(
        illumination,
        gamma=2.0,
    )

    assert corrected.dtype == np.float32
    assert corrected.min() >= 0.0
    assert corrected.max() <= 1.0


def test_gamma_is_monotonic():
    from src.enhancement import apply_gamma_correction

    illumination = np.array(
        [[0.1, 0.2, 0.4, 0.7, 1.0]],
        dtype=np.float32,
    )

    corrected = apply_gamma_correction(
        illumination,
        gamma=2.0,
    )

    assert np.all(np.diff(corrected[0]) >= 0)


def test_clahe_output_shape_dtype_and_range():
    from src.enhancement import apply_clahe

    illumination = np.full(
        (16, 16),
        0.2,
        dtype=np.float32,
    )

    enhanced = apply_clahe(
        illumination,
        clip_limit=2.0,
        tile_grid_size=(8, 8),
    )

    assert enhanced.shape == illumination.shape
    assert enhanced.dtype == np.float32
    assert enhanced.min() >= 0.0
    assert enhanced.max() <= 1.0


def test_clahe_improves_low_contrast_image():
    from src.enhancement import apply_clahe

    illumination = np.linspace(
        0.40,
        0.45,
        256,
        dtype=np.float32,
    ).reshape(16, 16)

    enhanced = apply_clahe(
        illumination,
        clip_limit=2.0,
        tile_grid_size=(8, 8),
    )

    assert float(enhanced.std()) > float(illumination.std())


def test_constant_image_is_supported():
    from src.enhancement import enhance_illumination

    illumination = np.full(
        (16, 16),
        0.1,
        dtype=np.float32,
    )

    enhanced = enhance_illumination(illumination)

    assert enhanced.shape == illumination.shape
    assert enhanced.dtype == np.float32
    assert np.all(np.isfinite(enhanced))
    assert enhanced.min() >= 0.0
    assert enhanced.max() <= 1.0


def test_small_image_with_default_tile_grid():
    from src.enhancement import enhance_illumination

    illumination = np.full(
        (16, 16),
        0.2,
        dtype=np.float32,
    )

    enhanced = enhance_illumination(illumination)

    assert enhanced.shape == (16, 16)
    assert enhanced.dtype == np.float32


def test_complete_enhancement_brightens_dark_image():
    from src.enhancement import enhance_illumination

    illumination = np.full(
        (32, 32),
        0.1,
        dtype=np.float32,
    )

    enhanced = enhance_illumination(illumination)

    assert float(enhanced.mean()) > float(illumination.mean())


def test_input_is_not_modified():
    from src.enhancement import enhance_illumination

    illumination = np.random.default_rng(42).random(
        (32, 32),
        dtype=np.float32,
    )

    original = illumination.copy()

    enhance_illumination(illumination)

    np.testing.assert_array_equal(
        illumination,
        original,
    )


def test_rejects_three_dimensional_input():
    from src.enhancement import enhance_illumination

    illumination = np.zeros(
        (16, 16, 3),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        enhance_illumination(illumination)


def test_rejects_nan_input():
    from src.enhancement import enhance_illumination

    illumination = np.zeros(
        (16, 16),
        dtype=np.float32,
    )

    illumination[0, 0] = np.nan

    with pytest.raises(ValueError):
        enhance_illumination(illumination)


def test_rejects_infinite_input():
    from src.enhancement import enhance_illumination

    illumination = np.zeros(
        (16, 16),
        dtype=np.float32,
    )

    illumination[0, 0] = np.inf

    with pytest.raises(ValueError):
        enhance_illumination(illumination)


def test_rejects_out_of_range_input():
    from src.enhancement import enhance_illumination

    illumination = np.full(
        (16, 16),
        1.5,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        enhance_illumination(illumination)


def test_rejects_invalid_clahe_clip_limit():
    from src.enhancement import apply_clahe

    illumination = np.full(
        (16, 16),
        0.2,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        apply_clahe(
            illumination,
            clip_limit=0,
        )


def test_rejects_invalid_tile_grid_size():
    from src.enhancement import apply_clahe

    illumination = np.full(
        (16, 16),
        0.2,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        apply_clahe(
            illumination,
            clip_limit=2.0,
            tile_grid_size=(0, 8),
        )


def test_rejects_negative_tile_grid_size():
    from src.enhancement import apply_clahe

    illumination = np.full(
        (16, 16),
        0.2,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        apply_clahe(
            illumination,
            clip_limit=2.0,
            tile_grid_size=(-1, 8),
        )