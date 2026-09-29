"""Tests for src/retinex_utils.py (mean V, alpha/beta, Sobel terms)."""
import ast
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import retinex_utils  # noqa: E402
from src.retinex_utils import (  # noqa: E402
    calculate_mean_v,
    compute_sobel_gradient,
    compute_sobel_terms,
    initialize_retinex_parameters,
)


# ---- calculate_mean_v ----
def test_mean_v_known_matrix():
    v = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    result = calculate_mean_v(v)
    assert isinstance(result, float)
    assert result == pytest.approx(0.5)


def test_mean_v_rejects_bad_input():
    with pytest.raises(ValueError):
        calculate_mean_v(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        calculate_mean_v(np.array([[0.1, np.nan]], dtype=np.float32))
    with pytest.raises(ValueError):
        calculate_mean_v(np.array([[0.1, 1.5]], dtype=np.float32))
    with pytest.raises(TypeError):
        calculate_mean_v([[0.1, 0.2]])


# ---- initialize_retinex_parameters ----
@pytest.mark.parametrize("mean_v", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_alpha_beta_stay_within_bounds(mean_v):
    alpha, beta = initialize_retinex_parameters(mean_v)
    assert isinstance(alpha, float) and isinstance(beta, float)
    assert 0.0001 <= alpha <= 0.003
    assert 0.0001 <= beta <= 0.0005


@pytest.mark.parametrize(
    "mean_v, alpha, beta",
    [
        (0.0, 0.002, 0.00025),
        (0.25, 0.0015, 0.000175),
        (0.5, 0.001, 0.0001),
        (0.75, 0.0005, 0.0001),  # beta_raw = 0.000025 -> clipped up to 0.0001
        (1.0, 0.0001, 0.0001),   # alpha_raw = 0 and beta_raw < 0 -> both clipped up
    ],
)
def test_alpha_beta_known_values(mean_v, alpha, beta):
    got_alpha, got_beta = initialize_retinex_parameters(mean_v)
    assert got_alpha == pytest.approx(alpha, rel=1e-9)
    assert got_beta == pytest.approx(beta, rel=1e-9)


def test_darker_images_get_larger_weights():
    dark = initialize_retinex_parameters(0.1)
    bright = initialize_retinex_parameters(0.6)
    assert dark[0] > bright[0] and dark[1] > bright[1]


def test_alpha_beta_reject_bad_mean_v():
    for bad in (np.nan, np.inf, -0.1, 1.5):
        with pytest.raises(ValueError):
            initialize_retinex_parameters(bad)
    with pytest.raises(TypeError):
        initialize_retinex_parameters("0.5")


# ---- Sobel ----
def _left_right_edge(h=10, w=12):
    img = np.zeros((h, w), dtype=np.float32)
    img[:, w // 2:] = 1.0
    return img


def test_sobel_preserves_shape_and_is_finite():
    img = np.random.default_rng(1).random((9, 14)).astype(np.float32)
    g, d = compute_sobel_terms(img)
    assert g.shape == d.shape == img.shape
    assert g.dtype == np.float32
    assert np.isfinite(g).all() and np.isfinite(d).all()


def test_sobel_responds_at_a_vertical_edge_only():
    img = _left_right_edge()
    g = compute_sobel_gradient(img)
    edge_cols = [img.shape[1] // 2 - 1, img.shape[1] // 2]
    assert (g[:, edge_cols] > 0).all()
    assert np.allclose(g[:, :3], 0.0) and np.allclose(g[:, -3:], 0.0)


def test_sobel_constant_image_has_zero_gradient():
    g = compute_sobel_gradient(np.full((8, 8), 0.4, dtype=np.float32))
    assert np.allclose(g, 0.0)


def test_d_equals_abs_g():
    img = np.random.default_rng(2).random((7, 7)).astype(np.float32)
    g, d = compute_sobel_terms(img)
    assert np.allclose(d, np.abs(g), atol=1e-6)


def test_sobel_does_not_mutate_input_and_rejects_bad_input():
    img = _left_right_edge()
    before = img.copy()
    compute_sobel_terms(img)
    assert np.array_equal(img, before)
    with pytest.raises(ValueError):
        compute_sobel_terms(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        compute_sobel_terms(np.full((4, 4), np.nan, dtype=np.float32))


# ---- integration rules ----
def test_retinex_utils_does_not_import_retinex_module():
    tree = ast.parse(Path(retinex_utils.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(("." * node.level) + (node.module or ""))
    assert not any(name.split(".")[-1] == "retinex" for name in imported)


def test_module_imports_cleanly_without_side_effects(capsys):
    importlib.reload(retinex_utils)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
