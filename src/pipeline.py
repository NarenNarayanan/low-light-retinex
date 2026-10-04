"""End-to-end low-light enhancement pipeline.

Flow (one source of truth, used by the command line, the web UI, the evaluation script and the tests):

    BGR image
      -> validation, BGR -> RGB, float32 [0, 1], RGB -> HSV            (preprocessing)
      -> mean V -> adaptive alpha and beta                              (retinex_utils)
      -> iterative Retinex decomposition of V -> illumination T, reflectance R   (retinex)
      -> adaptive gamma correction of T -> CLAHE -> enhanced V          (enhancement)
      -> dark-region mask on the enhanced V -> selective Fast NLM       (denoising)
      -> denoised V + ORIGINAL H and S -> HSV -> RGB                    (reconstruction)
      -> float32 RGB in [0, 1]; saved as an 8-bit image by ``save_image``.

All parameters come from ``configs/default.yaml``. The configuration is loaded here, once, and handed to
the stage functions as explicit arguments; the stage modules never read the file themselves.

Conventions: the RGB working image is float32 (H, W, 3) in [0, 1]; H is float32 in degrees [0, 360)
(OpenCV float-HSV; it is not rescaled so that HSV -> RGB round-trips exactly), S and V are float32
in [0, 1]. Inputs are never modified in place.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .denoising import denoise_dark_regions, detect_dark_regions
from .enhancement import apply_clahe, apply_gamma_correction, compute_adaptive_gamma
from .preprocessing import preprocess_image
from .reconstruction import reconstruct_image
from .retinex import retinex_decompose_with_info
from .retinex_utils import calculate_mean_v, initialize_retinex_parameters

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
_OUTPUT_FORMATS = ("png", "bmp", "tiff")  # lossless formats only

# Keys every configuration must provide, grouped by section.
_REQUIRED_KEYS = {
    "pipeline": ("save_intermediates", "output_dir", "output_format"),
    "retinex": ("epsilon", "max_iterations", "tolerance"),
    "enhancement": ("clahe_clip_limit", "clahe_tile_grid_size"),
    "denoising": ("dark_threshold", "nlm_h", "nlm_template_window_size", "nlm_search_window_size"),
}


class ConfigError(ValueError):
    """The configuration file is missing, unreadable, or lacks a required value."""


@dataclass
class EnhancementResult:
    """Everything produced by one run of the pipeline.

    enhanced_rgb:  float32 (H, W, 3) in [0, 1] - the final image.
    info:          parameters and statistics (mean V, alpha, beta, Retinex iterations, gamma, ...).
    timings:       seconds per stage plus ``total`` (processing only: no file I/O, no metrics).
    intermediates: images for inspection; float32 arrays in [0, 1] (RGB or 2-D), mask is bool.
    output_path:   where the image was saved, when produced by ``run_pipeline``.
    """

    enhanced_rgb: np.ndarray
    info: dict
    timings: dict
    intermediates: dict = field(default_factory=dict)
    output_path: Path | None = None
    intermediates_dir: Path | None = None


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> dict:
    """Read and validate the YAML configuration (``configs/default.yaml`` when ``path`` is None)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Configuration file is not valid YAML ({config_path}): {exc}") from exc
    return validate_config(config)


def validate_config(config: object) -> dict:
    """Check that every required section and key is present and not null; return the config."""
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a mapping of sections.")
    for section, keys in _REQUIRED_KEYS.items():
        values = config.get(section)
        if not isinstance(values, dict):
            raise ConfigError(f"Configuration section '{section}' is missing.")
        for key in keys:
            if values.get(key) is None:
                raise ConfigError(f"Configuration value '{section}.{key}' is missing or null.")
    if not isinstance(config["pipeline"]["save_intermediates"], bool):
        raise ConfigError("'pipeline.save_intermediates' must be true or false.")
    if config["pipeline"]["output_format"] not in _OUTPUT_FORMATS:
        raise ConfigError(f"'pipeline.output_format' must be one of {_OUTPUT_FORMATS}.")
    grid = config["enhancement"]["clahe_tile_grid_size"]
    if not (isinstance(grid, (list, tuple)) and len(grid) == 2):
        raise ConfigError("'enhancement.clahe_tile_grid_size' must be a list of two integers.")
    return config


# --------------------------------------------------------------------------------------
# Image input / output
# --------------------------------------------------------------------------------------
def load_image(input_path: str | Path) -> np.ndarray:
    """Read an image file as a BGR uint8 array (OpenCV order).

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the file is empty, or OpenCV cannot decode it as an image.
    """
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input image file is empty: {path}")
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unsupported or corrupt image file (OpenCV could not decode it): {path}")
    return image_bgr


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes (for example an uploaded file) to a BGR uint8 array.

    ``cv2.imdecode`` returns BGR, which is the order the preprocessing stage expects, so no
    channel swap is needed here.
    """
    if not data:
        raise ValueError("The uploaded file is empty.")
    image_bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Unsupported or corrupt image data: it could not be decoded as an image.")
    return image_bgr


def rgb_float_to_bgr_uint8(rgb: np.ndarray) -> np.ndarray:
    """Explicit float RGB [0, 1] -> 8-bit BGR conversion (round to the nearest level, then reorder).

    Writing a float array straight to disk would store values in 0..1 as almost-black 8-bit pixels,
    so the scaling to 0..255 is done here on purpose.
    """
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB array of shape (H, W, 3).")
    if not np.issubdtype(rgb.dtype, np.floating) or not np.isfinite(rgb).all():
        raise ValueError("Expected a finite floating-point RGB array.")
    if rgb.min() < 0.0 or rgb.max() > 1.0:
        raise ValueError("RGB values must lie in [0, 1].")
    rgb_u8 = np.rint(rgb * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)


def rgb_float_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Float RGB [0, 1] -> 8-bit RGB (for display in the UI)."""
    return cv2.cvtColor(rgb_float_to_bgr_uint8(rgb), cv2.COLOR_BGR2RGB)


def bgr_to_display_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (OpenCV order) -> RGB uint8, the order web/PIL image viewers expect."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_image(rgb: np.ndarray, path: str | Path) -> Path:
    """Save a float RGB [0, 1] image as an 8-bit file; creates missing directories."""
    path = Path(path)
    data = rgb_float_to_bgr_uint8(rgb)  # validate before touching the file system
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = cv2.imwrite(str(path), data)
    except cv2.error as exc:  # OpenCV raises (instead of returning False) for unknown extensions
        raise OSError(f"Could not write the image to {path}: unsupported file extension.") from exc
    if not written:
        raise OSError(f"Could not write the image to {path} (unwritable path).")
    return path


def encode_png(rgb: np.ndarray) -> bytes:
    """PNG bytes of a float RGB [0, 1] image, for downloads."""
    ok, buffer = cv2.imencode(".png", rgb_float_to_bgr_uint8(rgb))
    if not ok:
        raise OSError("PNG encoding failed.")
    return buffer.tobytes()


def _gray_float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a 2-D float [0, 1] (or bool) image to uint8 for visualisation."""
    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_intermediates(result: EnhancementResult, directory: str | Path, stem: str) -> Path:
    """Write the intermediate images as PNGs named ``<stem>_<name>.png`` and return the directory.

    The hue channel is not written: H is an angle in degrees [0, 360), not a brightness, so a raw
    grayscale dump would be misleading. It is preserved unchanged inside the final reconstruction.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, image in result.intermediates.items():
        if image.ndim == 3:
            data = rgb_float_to_bgr_uint8(image)
        else:
            data = _gray_float_to_uint8(image)
        target = directory / f"{stem}_{name}.png"
        if not cv2.imwrite(str(target), data):
            raise OSError(f"Could not write {target}")
    return directory


# --------------------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------------------
def process_image(image_bgr: np.ndarray, config: dict | None = None) -> EnhancementResult:
    """Enhance one BGR uint8 image and return the result with its statistics and timings.

    This is the single implementation of the algorithm. ``config`` is a loaded configuration
    (``load_config()`` is used when omitted). The input array is not modified.
    """
    config = validate_config(config) if config is not None else load_config()
    retinex_cfg, enh_cfg, den_cfg = config["retinex"], config["enhancement"], config["denoising"]

    timings: dict[str, float] = {}
    start = time.perf_counter()

    def mark(stage: str, since: float) -> float:
        now = time.perf_counter()
        timings[stage] = now - since
        return now

    t = start
    # H, S, V are extracted once; H and S stay untouched until the final reconstruction.
    pre = preprocess_image(image_bgr)
    t = mark("preprocessing", t)

    # Darker images (small mean V) receive larger regularisation weights.
    mean_v = calculate_mean_v(pre.v)
    alpha, beta = initialize_retinex_parameters(mean_v)
    t = mark("parameters", t)

    illumination, reflectance, retinex_info = retinex_decompose_with_info(
        pre.v, alpha, beta,
        epsilon=retinex_cfg["epsilon"],
        max_iterations=retinex_cfg["max_iterations"],
        tolerance=retinex_cfg["tolerance"],
    )
    t = mark("retinex", t)

    # Adaptive gamma (brightening direction) on the illumination, then CLAHE on the result.
    gamma = compute_adaptive_gamma(illumination)
    gamma_corrected = apply_gamma_correction(illumination, gamma)
    t = mark("gamma", t)
    enhanced_v = apply_clahe(
        gamma_corrected,
        clip_limit=enh_cfg["clahe_clip_limit"],
        tile_grid_size=tuple(enh_cfg["clahe_tile_grid_size"]),
    )
    t = mark("clahe", t)

    # Noise is amplified mostly in dark regions, so only those are denoised; the mask is built on the
    # enhanced V and every pixel outside it is left exactly as it was.
    dark_mask = detect_dark_regions(enhanced_v, threshold=den_cfg["dark_threshold"])
    t = mark("dark_mask", t)
    denoised_v = denoise_dark_regions(
        enhanced_v, dark_mask,
        h=den_cfg["nlm_h"],
        template_window_size=den_cfg["nlm_template_window_size"],
        search_window_size=den_cfg["nlm_search_window_size"],
    )
    t = mark("denoising", t)

    # Recombine with the ORIGINAL H and S so that colours are not shifted by the brightness change.
    enhanced_rgb = reconstruct_image(pre.h, pre.s, denoised_v)
    t = mark("reconstruction", t)
    timings["total"] = t - start

    info = {
        "image_height": int(image_bgr.shape[0]),
        "image_width": int(image_bgr.shape[1]),
        "mean_v": mean_v,
        "alpha": alpha,
        "beta": beta,
        "retinex_iterations": retinex_info["iterations"],
        "retinex_converged": retinex_info["converged"],
        "retinex_final_delta": retinex_info["final_delta"],
        "gamma": gamma,
        "dark_region_fraction": float(dark_mask.mean()),
        "mean_v_enhanced": float(denoised_v.mean()),
        "mean_abs_v_minus_illumination": float(np.abs(pre.v - illumination).mean()),
    }
    intermediates = {
        "input": pre.rgb,
        "v_channel": pre.v,
        "illumination": illumination,
        "reflectance": reflectance,
        "gamma_corrected": gamma_corrected,
        "clahe_enhanced": enhanced_v,
        "dark_mask": dark_mask,
        "denoised_v": denoised_v,
        "final": enhanced_rgb,
    }
    return EnhancementResult(enhanced_rgb, info, timings, intermediates)


def default_output_path(input_path: str | Path, config: dict) -> Path:
    """``<output_dir>/<input stem>_enhanced.<format>``, e.g. data/output/test_enhanced.png."""
    pipeline_cfg = config["pipeline"]
    stem = Path(input_path).stem
    return Path(pipeline_cfg["output_dir"]) / f"{stem}_enhanced.{pipeline_cfg['output_format']}"


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
    config: dict | None = None,
    save_intermediate_images: bool | None = None,
) -> EnhancementResult:
    """Load an image file, enhance it, save the result and return it.

    ``output_path`` defaults to ``default_output_path``. Intermediates are written next to the output
    (in ``<stem>_intermediates``) when ``save_intermediate_images`` is true, or when it is None and
    ``pipeline.save_intermediates`` is true in the configuration.
    """
    config = validate_config(config) if config is not None else load_config()
    logger.info("Input path: %s", input_path)
    image_bgr = load_image(input_path)
    logger.info("Image shape: %s", image_bgr.shape)

    result = process_image(image_bgr, config)

    target = Path(output_path) if output_path is not None else default_output_path(input_path, config)
    result.output_path = save_image(result.enhanced_rgb, target)

    if save_intermediate_images is None:
        save_intermediate_images = config["pipeline"]["save_intermediates"]
    if save_intermediate_images:
        stem = Path(input_path).stem
        result.intermediates_dir = save_intermediates(
            result, result.output_path.parent / f"{stem}_intermediates", stem
        )
    logger.info("Processed in %.3fs; saved to %s", result.timings["total"], result.output_path)
    return result
