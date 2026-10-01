"""Tests for src/retinex.py (core adaptive Retinex decomposition)."""
import ast
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import retinex  # noqa: E402
from src.retinex import retinex_decompose, retinex_decompose_with_info  # noqa: E402

ALPHA, BETA = 0.001, 0.0001  # typical values from initialize_retinex_parameters


def _random_v(h=8, w=10, seed=0, scale=1.0):
    return (np.random.default_rng(seed).random((h, w)) * scale).astype(np.float32)


def _reference_sobel_terms(image):
    """Independent NumPy Sobel (3x3, unnormalized, reflect-101 border) used only in tests."""
    img = image.astype(np.float64)
    p = np.pad(img, 1, mode="reflect")  # numpy 'reflect' == OpenCV BORDER_REFLECT_101
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    g = np.sqrt(gx ** 2 + gy ** 2)
    return g, np.abs(g)


def _reference_iterations(v, alpha, beta, epsilon, n):
    """Plain float64 implementation of the report equations, written separately from src."""
    v = v.astype(np.float64)
    g, d = _reference_sobel_terms(v)
    r = np.ones_like(v)  # documented initialization: T(0) = V, R(0) = 1
    t = v.copy()
    for _ in range(n):
        t = (v * r) / (r + alpha * g + epsilon)
        r = (v * t) / (t + beta * d + epsilon)
    return t, r


# ---- TEST 1 / 2: shape and dtype ----
def test_output_shape_matches_input():
    v = _random_v(7, 11)
    t, r = retinex_decompose(v, ALPHA, BETA)
    assert t.shape == v.shape and r.shape == v.shape


def test_output_dtype_is_float32():
    t, r = retinex_decompose(_random_v(), ALPHA, BETA)
    assert t.dtype == np.float32 and r.dtype == np.float32


def test_float64_input_is_accepted_and_returns_float32():
    t, r = retinex_decompose(_random_v().astype(np.float64), ALPHA, BETA)
    assert t.dtype == np.float32 and r.dtype == np.float32


# ---- TEST 3: determinism ----
def test_deterministic_output():
    v = _random_v()
    t1, r1, i1 = retinex_decompose_with_info(v, ALPHA, BETA)
    t2, r2, i2 = retinex_decompose_with_info(v, ALPHA, BETA)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(r1, r2)
    assert i1 == i2


# ---- TEST 4: input untouched ----
def test_input_is_not_modified():
    v = _random_v()
    original = v.copy()
    retinex_decompose(v, ALPHA, BETA)
    assert np.array_equal(v, original)


def test_outputs_do_not_share_memory_with_input():
    v = _random_v()
    t, r = retinex_decompose(v, ALPHA, BETA)
    assert not np.shares_memory(t, v) and not np.shares_memory(r, v)


# ---- TEST 5 / 6 / 7: special inputs ----
def test_zero_input_is_finite_and_zero():
    v = np.zeros((6, 6), dtype=np.float32)
    t, r, info = retinex_decompose_with_info(v, ALPHA, BETA)
    assert np.isfinite(t).all() and np.isfinite(r).all()
    assert t.shape == r.shape == v.shape
    np.testing.assert_array_equal(t, 0.0)
    np.testing.assert_array_equal(r, 0.0)
    assert info["converged"]


@pytest.mark.parametrize("value", [0.2, 0.5, 1.0])
def test_constant_input_is_finite_uniform_and_deterministic(value):
    v = np.full((9, 7), value, dtype=np.float32)
    t1, r1 = retinex_decompose(v, ALPHA, BETA)
    t2, r2 = retinex_decompose(v, ALPHA, BETA)
    assert np.isfinite(t1).all() and np.isfinite(r1).all()
    assert t1.shape == v.shape
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(r1, r2)
    # A constant image has zero gradient, so the estimates must be spatially constant.
    assert np.ptp(t1) == pytest.approx(0.0, abs=1e-6)
    assert np.ptp(r1) == pytest.approx(0.0, abs=1e-6)


def test_very_dark_image_terminates_with_finite_float32_outputs():
    v = _random_v(seed=3, scale=1e-4)
    t, r, info = retinex_decompose_with_info(v, 0.002, 0.00025)
    assert t.dtype == r.dtype == np.float32
    assert np.isfinite(t).all() and np.isfinite(r).all()
    assert 1 <= info["iterations"] <= 50


# ---- TEST 8: maximum-iteration termination ----
def test_single_iteration_limit_is_respected():
    # After one iteration R has moved from 1 to about V, so delta is far above this tolerance.
    _, _, info = retinex_decompose_with_info(
        _random_v(), ALPHA, BETA, max_iterations=1, tolerance=1e-12
    )
    assert info["iterations"] == 1
    assert info["converged"] is False


def test_never_exceeds_max_iterations_with_extremely_strict_tolerance():
    _, _, info = retinex_decompose_with_info(
        _random_v(), ALPHA, BETA, max_iterations=4, tolerance=1e-30
    )
    assert 1 <= info["iterations"] <= 4


# ---- TEST 9: info dictionary ----
def test_info_contents_are_sensible():
    _, _, info = retinex_decompose_with_info(_random_v(), ALPHA, BETA)
    assert {"iterations", "converged", "final_delta"} <= set(info)
    assert isinstance(info["iterations"], int) and info["iterations"] >= 1
    assert isinstance(info["converged"], bool)
    assert isinstance(info["final_delta"], float)
    assert np.isfinite(info["final_delta"]) and info["final_delta"] >= 0.0
    if info["converged"]:
        assert info["final_delta"] < 1e-4  # the default tolerance


def test_constant_image_converges_before_the_limit():
    _, _, info = retinex_decompose_with_info(np.full((5, 5), 0.3, np.float32), ALPHA, BETA)
    assert info["converged"] is True
    assert info["iterations"] < 50


def test_simple_function_matches_detailed_function():
    v = _random_v()
    t, r = retinex_decompose(v, ALPHA, BETA)
    t2, r2, _ = retinex_decompose_with_info(v, ALPHA, BETA)
    np.testing.assert_array_equal(t, t2)
    np.testing.assert_array_equal(r, r2)


# ---- TEST 10: invalid input ----
GOOD_V = np.full((4, 4), 0.5, dtype=np.float32)


@pytest.mark.parametrize(
    "bad_v, exc",
    [
        (None, TypeError),
        ([[0.1, 0.2], [0.3, 0.4]], TypeError),
        (np.full((4, 4), 1, dtype=np.uint8), TypeError),
        (np.zeros(5, dtype=np.float32), ValueError),
        (np.zeros((2, 4, 4), dtype=np.float32), ValueError),
        (np.zeros((0, 4), dtype=np.float32), ValueError),
        (np.array([[0.1, np.nan]], dtype=np.float32), ValueError),
        (np.array([[0.1, np.inf]], dtype=np.float32), ValueError),
        (np.array([[0.1, -0.01]], dtype=np.float32), ValueError),
        (np.array([[0.1, 1.01]], dtype=np.float32), ValueError),
    ],
)
def test_invalid_v_raises(bad_v, exc):
    with pytest.raises(exc):
        retinex_decompose(bad_v, ALPHA, BETA)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"alpha": 0.0}, ValueError),
        ({"alpha": -0.001}, ValueError),
        ({"alpha": np.nan}, ValueError),
        ({"alpha": np.inf}, ValueError),
        ({"alpha": "0.001"}, TypeError),
        ({"alpha": True}, TypeError),
        ({"beta": 0.0}, ValueError),
        ({"beta": -0.0001}, ValueError),
        ({"beta": np.nan}, ValueError),
        ({"beta": None}, TypeError),
        ({"epsilon": 0.0}, ValueError),
        ({"epsilon": -1e-3}, ValueError),
        ({"epsilon": np.inf}, ValueError),
        ({"max_iterations": 0}, ValueError),
        ({"max_iterations": -3}, ValueError),
        ({"max_iterations": 2.5}, TypeError),
        ({"max_iterations": True}, TypeError),
        ({"max_iterations": "5"}, TypeError),
        ({"tolerance": 0.0}, ValueError),
        ({"tolerance": -1e-4}, ValueError),
        ({"tolerance": np.nan}, ValueError),
        ({"tolerance": np.inf}, ValueError),
    ],
)
def test_invalid_parameters_raise(kwargs, exc):
    params = {"alpha": ALPHA, "beta": BETA}
    params.update(kwargs)
    with pytest.raises(exc):
        retinex_decompose(GOOD_V, **params)


# ---- TEST 11: documented numerical invariant ----
def test_outputs_stay_between_zero_and_v():
    """Documented invariant: with V in [0, 1] and R(0) = 1, both T and R lie in [0, V]."""
    for seed in range(5):
        v = _random_v(12, 12, seed=seed)
        t, r = retinex_decompose(v, ALPHA, BETA)
        assert t.min() >= 0.0 and r.min() >= 0.0
        assert (t <= v + 1e-6).all() and (r <= v + 1e-6).all()
        assert t.max() <= 1.0 and r.max() <= 1.0


def test_no_numerical_explosion_on_extreme_alpha_beta_and_tiny_epsilon():
    v = _random_v(10, 10, seed=7, scale=0.05)
    t, r = retinex_decompose(v, 0.003, 0.0005, epsilon=1e-8)
    assert np.isfinite(t).all() and np.isfinite(r).all()
    assert t.max() <= 1.0 + 1e-6 and r.max() <= 1.0 + 1e-6


# ---- TEST 12: equation correctness ----
def test_first_iteration_hand_calculated_on_constant_image():
    """Constant V has G = D = 0, so T1 = c / (1 + eps) and R1 = c * T1 / (T1 + eps)."""
    c, eps = 0.25, 1e-3
    t1 = c * 1.0 / (1.0 + eps)
    r1 = c * t1 / (t1 + eps)
    v = np.full((4, 4), c, dtype=np.float32)
    t, r, info = retinex_decompose_with_info(v, ALPHA, BETA, epsilon=eps, max_iterations=1)
    assert info["iterations"] == 1
    np.testing.assert_allclose(t, t1, rtol=1e-6)
    np.testing.assert_allclose(r, r1, rtol=1e-6)


def test_first_iteration_matches_independent_equations_with_edges():
    v = np.array(
        [[0.05, 0.05, 0.60, 0.60],
         [0.05, 0.10, 0.60, 0.55],
         [0.08, 0.12, 0.50, 0.50],
         [0.06, 0.10, 0.40, 0.70]],
        dtype=np.float32,
    )
    eps = 1e-3
    g, d = _reference_sobel_terms(v)
    assert g.max() > 1.0  # the image has real edges, so the G and D terms matter
    t_ref, r_ref = _reference_iterations(v, 0.002, 0.00025, eps, n=1)
    t, r, info = retinex_decompose_with_info(
        v, 0.002, 0.00025, epsilon=eps, max_iterations=1, tolerance=1e-12
    )
    assert info["iterations"] == 1
    np.testing.assert_allclose(t, t_ref, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(r, r_ref, rtol=1e-5, atol=1e-7)


def test_G_term_actually_changes_the_illumination_at_edges():
    """A large alpha must reduce T at edge pixels (G > 0) but not in flat areas (G = 0)."""
    v = np.full((6, 6), 0.2, dtype=np.float32)
    v[:, 3:] = 0.8
    t_small, _ = retinex_decompose(v, 1e-9, 1e-9, max_iterations=1)
    t_large, _ = retinex_decompose(v, 0.1, 1e-9, max_iterations=1)
    assert (t_large[:, 2:4] < t_small[:, 2:4] - 0.03).all()
    np.testing.assert_allclose(t_large[:, :1], t_small[:, :1], rtol=1e-6)  # flat area unaffected


def test_D_term_actually_changes_the_reflectance_at_edges():
    """A large beta must reduce R at edge pixels (D > 0) but not in flat areas (D = 0)."""
    v = np.full((6, 6), 0.2, dtype=np.float32)
    v[:, 3:] = 0.8
    _, r_small = retinex_decompose(v, 1e-9, 1e-9, max_iterations=1)
    _, r_large = retinex_decompose(v, 1e-9, 0.1, max_iterations=1)
    assert (r_large[:, 2:4] < r_small[:, 2:4] - 0.05).all()
    np.testing.assert_allclose(r_large[:, :1], r_small[:, :1], rtol=1e-6)


def test_several_iterations_match_independent_equations():
    v = _random_v(6, 6, seed=11, scale=0.4)
    eps = 1e-3
    for n in (2, 3):
        t_ref, r_ref = _reference_iterations(v, 0.0015, 0.0002, eps, n=n)
        t, r, info = retinex_decompose_with_info(
            v, 0.0015, 0.0002, epsilon=eps, max_iterations=n, tolerance=1e-30
        )
        assert info["iterations"] == n
        np.testing.assert_allclose(t, t_ref, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(r, r_ref, rtol=1e-4, atol=1e-6)


def test_update_order_uses_new_illumination_in_reflectance_update():
    """R(1) must be computed from the NEW T(1), not from T(0) = V."""
    v = np.full((6, 6), 0.2, dtype=np.float32)
    v[:, 3:] = 0.8
    # Large alpha and epsilon make T(1) small compared with V and comparable to epsilon,
    # which is what makes the two update orders give visibly different R(1).
    alpha, beta, eps = 3.0, 1e-9, 0.05
    g, _ = _reference_sobel_terms(v)
    t1 = (v * 1.0) / (1.0 + alpha * g + eps)                 # hand-written T(1)
    r_new_order = (v * t1) / (t1 + beta * g + eps)           # correct: uses T(1)
    r_old_order = (v * v) / (v + beta * g + eps)             # wrong: would use T(0) = V
    assert np.abs(r_new_order - r_old_order).max() > 0.05    # the two orders are distinguishable
    _, r = retinex_decompose(v, alpha, beta, epsilon=eps, max_iterations=1)
    np.testing.assert_allclose(r, r_new_order, rtol=1e-4, atol=1e-6)
    assert np.abs(r - r_old_order).max() > 0.05


# ---- project rules ----
def test_config_retinex_section_matches_function_defaults():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    cfg = yaml.safe_load(config_path.read_text())["retinex"]
    assert cfg["epsilon"] == retinex.DEFAULT_EPSILON
    assert cfg["max_iterations"] == retinex.DEFAULT_MAX_ITERATIONS
    assert cfg["tolerance"] == retinex.DEFAULT_TOLERANCE



def test_retinex_reuses_sobel_utility_and_has_no_duplicate_sobel():
    path = Path(retinex.__file__)
    tree = ast.parse(path.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "compute_sobel_terms" in imported
    assert "cv2" not in imported  # Sobel (and all OpenCV use) lives in retinex_utils


def test_module_imports_cleanly_without_side_effects(capsys):
    importlib.reload(retinex)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
