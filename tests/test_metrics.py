"""Tests for src/metrics.py (AB, DE, NIQE and the combined record)."""
import numpy as np
import pytest

from src import metrics
from src.metrics import (
    NiqeUnavailable, average_brightness, discrete_entropy, evaluate_image, niqe, niqe_from_gray,
    to_gray_uint8,
)
from tests.conftest import make_textured_bgr


# ---- average brightness ----
def test_ab_of_constant_images():
    assert average_brightness(np.full((8, 8), 128, np.uint8)) == pytest.approx(128.0)
    assert average_brightness(np.zeros((8, 8, 3), np.uint8)) == 0.0
    assert average_brightness(np.full((8, 8, 3), 255, np.uint8)) == pytest.approx(255.0)


def test_ab_float_and_uint8_agree():
    u8 = (np.random.default_rng(0).random((20, 30, 3)) * 255).astype(np.uint8)
    assert average_brightness(u8.astype(np.float32) / 255.0) == pytest.approx(average_brightness(u8), abs=0.5)


def test_color_order_changes_luma_as_expected():
    red_in_bgr = np.zeros((4, 4, 3), np.uint8)
    red_in_bgr[..., 2] = 255
    # Pure red has luma about 0.299 * 255 when read as BGR, but pure blue (about 29) when read as RGB.
    assert average_brightness(red_in_bgr, "bgr") == pytest.approx(76, abs=1)
    assert average_brightness(red_in_bgr, "rgb") == pytest.approx(29, abs=1)
    with pytest.raises(ValueError):
        to_gray_uint8(red_in_bgr, "xyz")


# ---- discrete entropy ----
def test_de_constant_image_is_zero():
    value = discrete_entropy(np.full((10, 10), 77, np.uint8))
    assert value == 0.0 and np.signbit(value) == False  # noqa: E712 (no negative zero)


def test_de_uniform_histogram_is_eight_bits():
    image = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    assert discrete_entropy(image) == pytest.approx(8.0, abs=1e-9)


def test_de_two_equal_levels_is_one_bit():
    image = np.concatenate([np.zeros(50, np.uint8), np.full(50, 200, np.uint8)]).reshape(10, 10)
    assert discrete_entropy(image) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "bad, exc",
    [
        (None, TypeError),
        ([[1, 2], [3, 4]], TypeError),
        (np.zeros((4,), np.uint8), ValueError),
        (np.zeros((4, 4, 4), np.uint8), ValueError),
        (np.zeros((0, 4), np.uint8), ValueError),
        (np.zeros((4, 4), np.int32), TypeError),
        (np.array([[0.5, np.nan]], np.float32), ValueError),
        (np.array([[0.5, 1.5]], np.float32), ValueError),
    ],
)
def test_invalid_images_are_rejected(bad, exc):
    with pytest.raises(exc):
        average_brightness(bad)
    with pytest.raises(exc):
        discrete_entropy(bad)


# ---- NIQE ----
def test_niqe_model_parameters_have_expected_shapes():
    mu, cov, window = metrics._load_niqe_model()
    assert mu.shape == (36,) and cov.shape == (36, 36) and window.shape == (7, 7)
    assert np.isfinite(mu).all() and np.isfinite(cov).all()
    np.testing.assert_allclose(cov, cov.T, atol=1e-9)


def test_halving_matrix_preserves_constants_and_ramps():
    for length in (8, 96, 200):
        matrix = metrics._bicubic_half_matrix(length)
        assert matrix.shape == (length // 2, length)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-12)  # a constant stays constant
    ramp = np.arange(96, dtype=np.float64)
    halved = metrics._bicubic_half_matrix(96) @ ramp
    np.testing.assert_allclose(halved[3:-3], ramp[::2][3:-3] + 0.5, atol=1e-6)  # ramp stays linear


def test_niqe_is_finite_deterministic_and_positive():
    image = make_textured_bgr(200)
    first, second = niqe(image), niqe(image)
    assert np.isfinite(first) and first > 0.0
    assert first == second


def test_niqe_is_worse_for_a_noisier_image():
    clean = make_textured_bgr(200)
    noisy = np.clip(clean + np.random.default_rng(1).normal(0, 30, clean.shape), 0, 255).astype(np.uint8)
    assert niqe(noisy) > niqe(clean)


def test_niqe_rejects_small_and_uniform_images():
    with pytest.raises(NiqeUnavailable):
        niqe(np.zeros((95, 300, 3), np.uint8))          # fewer than two blocks
    with pytest.raises(NiqeUnavailable):
        niqe(np.full((300, 300, 3), 90, np.uint8))      # uniform: nothing to fit
    with pytest.raises(ValueError):
        niqe_from_gray(np.zeros((300, 300, 3)))


# ---- combined record ----
def test_evaluate_image_record():
    image = make_textured_bgr(200)
    record = evaluate_image(image, runtime_s=0.25)
    assert set(record) == {"niqe", "ab", "de", "runtime_s"}
    assert record["runtime_s"] == 0.25
    assert np.isfinite(record["niqe"]) and 0 < record["ab"] < 255 and 0 < record["de"] <= 8


def test_evaluate_image_reports_missing_niqe_for_small_images_but_keeps_ab_de():
    record = evaluate_image(np.full((50, 50, 3), 100, np.uint8))
    assert record["niqe"] is None
    assert record["ab"] == pytest.approx(100.0) and record["de"] == 0.0 and record["runtime_s"] is None


def test_evaluate_image_validates_inputs():
    with pytest.raises(ValueError):
        evaluate_image(np.zeros((8, 8, 3), np.uint8), runtime_s=-1.0)
    with pytest.raises(ValueError):
        evaluate_image(np.zeros((8, 8, 3), np.uint8), runtime_s=float("nan"))
    with pytest.raises(TypeError):
        evaluate_image(None)
