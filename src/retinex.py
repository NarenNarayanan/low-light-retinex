"""Adaptive Retinex decomposition.

Contract:
    retinex_decompose(v, alpha, beta, ...) -> (illumination, reflectance)
    retinex_decompose_with_info(v, alpha, beta, ...) -> (illumination, reflectance, info)

    v:                 float32 (H, W), values in [0, 1]  (the HSV V channel)
    illumination (T):  float32 (H, W)
    reflectance  (R):  float32 (H, W)
    info:              {"iterations": int, "converged": bool, "final_delta": float}

Retinex model (report sections F-H): the observed V is explained by V = R * T,
and T and R are estimated jointly and iteratively:

    T(k+1) = (V * R(k))   / (R(k)   + alpha * G + epsilon)
    R(k+1) = (V * T(k+1)) / (T(k+1) + beta  * D + epsilon)

The update order matters: T(k+1) is computed first from R(k), and the NEW T(k+1)
is then used to compute R(k+1).

Implementation decisions. The report does NOT specify items 1-3, so they are
implementation assumptions, not statements from the report:

1. Initialization: T(0) = V and R(0) = 1 (all ones). This is the simplest
   deterministic start, and it satisfies the model V = R * T exactly.
2. Sobel terms: G is the Sobel gradient magnitude of V and D = |G| (from
   retinex_utils.compute_sobel_terms). V never changes during the iteration, so G
   and D are computed ONCE before the loop and reused every iteration.
3. Convergence: delta = max(max|T(k+1) - T(k)|, max|R(k+1) - R(k)|). Iteration stops
   when delta < tolerance, or after max_iterations. The default epsilon,
   max_iterations and tolerance are also assumptions (see configs/default.yaml).

Numerical stability. alpha, beta, G, D, V are all >= 0 and R(0) = 1, so by induction
T and R stay >= 0 and every denominator is >= epsilon > 0: no division by zero, NaN
or Inf, even for all-zero or constant images. The same induction gives
T(k+1) <= V * R(k) / (R(k) + epsilon) <= V, and likewise R(k+1) <= V, so both
outputs lie in [0, V] (subset of [0, 1]) without any clipping. The bound is exact in real
arithmetic; in float32 it holds up to rounding (about 1e-7).
"""
from __future__ import annotations

import numpy as np

from .retinex_utils import compute_sobel_terms

# Implementation assumptions (the report gives no values); mirrored in configs/default.yaml.
DEFAULT_EPSILON = 1e-3
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_TOLERANCE = 1e-4


def _validate_v(v: np.ndarray) -> None:
    """V must be a finite 2-D float array with values in [0, 1].

    The range check is strict: non-negative V is what guarantees every denominator
    in the update equations stays >= epsilon.
    """
    if not isinstance(v, np.ndarray):
        raise TypeError(f"v must be a NumPy array, got {type(v).__name__}.")
    if not np.issubdtype(v.dtype, np.floating):
        raise TypeError(f"v must be a floating-point array in [0, 1], got dtype {v.dtype}.")
    if v.ndim != 2:
        raise ValueError(f"v must be a 2-D (H, W) array, got shape {v.shape}.")
    if v.size == 0:
        raise ValueError(f"v is empty (shape {v.shape}).")
    if not np.isfinite(v).all():
        raise ValueError("v contains NaN or Inf values.")
    if v.min() < 0.0 or v.max() > 1.0:
        raise ValueError(f"v must lie in [0, 1], got range [{v.min()}, {v.max()}].")


def _positive_finite(value, name: str) -> float:
    """Return value as a Python float after checking it is a finite real number > 0."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    return value


def _positive_int(value, name: str) -> int:
    """Return value as int after checking it is a positive integer (bool is rejected)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return int(value)


def retinex_decompose_with_info(
    v: np.ndarray,
    alpha: float,
    beta: float,
    epsilon: float = DEFAULT_EPSILON,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Iteratively decompose V into illumination T and reflectance R.

    Args:
        v: float (H, W) array in [0, 1]. Not modified.
        alpha: illumination regularization weight (> 0), from initialize_retinex_parameters.
        beta: reflectance regularization weight (> 0), from initialize_retinex_parameters.
        epsilon: small positive constant that keeps every denominator non-zero.
        max_iterations: upper limit on the number of iterations (>= 1).
        tolerance: stop when the largest change between successive estimates is below this.

    Returns:
        (illumination, reflectance, info), where the arrays are float32 with the shape of v and
        info = {"iterations": int performed, "converged": bool, "final_delta": float}.
        "converged" is True whenever the last measured delta was below tolerance (even if that
        happened on the final allowed iteration).
    """
    _validate_v(v)
    alpha = _positive_finite(alpha, "alpha")
    beta = _positive_finite(beta, "beta")
    epsilon = _positive_finite(epsilon, "epsilon")
    max_iterations = _positive_int(max_iterations, "max_iterations")
    tolerance = _positive_finite(tolerance, "tolerance")

    v32 = np.asarray(v, dtype=np.float32)  # working copy; the caller's array is never written to

    # Structural priors: fixed because V is fixed, so they are computed once (assumption 2).
    g, d = compute_sobel_terms(v32)
    alpha_g = alpha * g  # alpha * G in the T update
    beta_d = beta * d    # beta  * D in the R update

    # Initialization (assumption 1): T(0) = V, R(0) = 1, so R(0) * T(0) = V.
    t = v32.copy()
    r = np.ones_like(v32)

    delta = float("inf")
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        t_new = v32 * r / (r + alpha_g + epsilon)        # T(k+1) from R(k)
        r_new = v32 * t_new / (t_new + beta_d + epsilon)  # R(k+1) from the NEW T(k+1)

        delta = float(max(np.abs(t_new - t).max(), np.abs(r_new - r).max()))
        t, r = t_new, r_new
        if delta < tolerance:
            converged = True
            break

    info = {"iterations": iterations, "converged": converged, "final_delta": delta}
    return t.astype(np.float32, copy=False), r.astype(np.float32, copy=False), info


def retinex_decompose(
    v: np.ndarray,
    alpha: float,
    beta: float,
    epsilon: float = DEFAULT_EPSILON,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (illumination, reflectance) for V. See retinex_decompose_with_info for details."""
    illumination, reflectance, _ = retinex_decompose_with_info(
        v, alpha, beta, epsilon=epsilon, max_iterations=max_iterations, tolerance=tolerance
    )
    return illumination, reflectance
