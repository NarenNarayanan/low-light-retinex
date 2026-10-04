"""Tests for src/pipeline.py: configuration, image I/O, the complete pipeline and the end-to-end flow."""
import copy
import importlib

import cv2
import numpy as np
import pytest
import yaml

from src import denoising, enhancement, pipeline, retinex
from src.enhancement import enhance_illumination
from src.pipeline import (
    ConfigError, bgr_to_display_rgb, decode_image, default_output_path, encode_png, load_config, load_image,
    process_image, rgb_float_to_bgr_uint8, rgb_float_to_uint8, run_pipeline, save_image, validate_config,
)
from tests.conftest import make_dark_bgr

STAGES = ("preprocessing", "parameters", "retinex", "gamma", "clahe", "dark_mask", "denoising", "reconstruction")


@pytest.fixture
def config(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["pipeline"]["output_dir"] = str(tmp_path / "out")
    return cfg


def gray_mean(bgr_u8):
    return float(cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY).mean())


# ---------------------------------------------------------------- configuration
def test_default_config_loads_and_is_complete():
    cfg = load_config()
    assert cfg["denoising"]["dark_threshold"] == 0.3
    assert cfg["enhancement"]["clahe_tile_grid_size"] == [8, 8]
    assert cfg["pipeline"]["output_format"] == "png"


def test_config_values_match_stage_function_defaults():
    cfg = load_config()
    assert cfg["retinex"] == {
        "epsilon": retinex.DEFAULT_EPSILON,
        "max_iterations": retinex.DEFAULT_MAX_ITERATIONS,
        "tolerance": retinex.DEFAULT_TOLERANCE,
    }
    assert cfg["enhancement"]["clahe_clip_limit"] == 2.0
    assert tuple(cfg["enhancement"]["clahe_tile_grid_size"]) == (8, 8)
    assert cfg["denoising"]["dark_threshold"] == denoising._DEFAULT_DARK_THRESHOLD
    assert cfg["denoising"]["nlm_h"] == denoising._DEFAULT_H
    assert cfg["denoising"]["nlm_template_window_size"] == denoising._DEFAULT_TEMPLATE_WINDOW_SIZE
    assert cfg["denoising"]["nlm_search_window_size"] == denoising._DEFAULT_SEARCH_WINDOW_SIZE


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("pipeline: [unclosed")
    with pytest.raises(ConfigError):
        load_config(bad)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.pop("retinex"),
        lambda c: c["retinex"].pop("epsilon"),
        lambda c: c["denoising"].__setitem__("nlm_h", None),
        lambda c: c["pipeline"].__setitem__("output_format", "gif"),
        lambda c: c["pipeline"].__setitem__("save_intermediates", "yes"),
        lambda c: c["enhancement"].__setitem__("clahe_tile_grid_size", [8]),
    ],
)
def test_invalid_configuration_is_rejected(mutate):
    cfg = copy.deepcopy(load_config())
    mutate(cfg)
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_load_config_from_custom_file(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["denoising"]["dark_threshold"] = 0.2
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert load_config(path)["denoising"]["dark_threshold"] == 0.2


# ---------------------------------------------------------------- image input / output helpers
def test_float_to_uint8_conversion_is_explicit_and_correct():
    rgb = np.zeros((2, 2, 3), np.float32)
    rgb[0, 0] = (1.0, 0.0, 0.2)
    bgr = rgb_float_to_bgr_uint8(rgb)
    assert bgr.dtype == np.uint8
    assert tuple(bgr[0, 0]) == (51, 0, 255)            # B, G, R order; 0.2 -> 51; 1.0 -> 255 (not 1)
    assert tuple(rgb_float_to_uint8(rgb)[0, 0]) == (255, 0, 51)
    assert tuple(bgr_to_display_rgb(bgr)[0, 0]) == (255, 0, 51)


@pytest.mark.parametrize(
    "bad",
    [np.zeros((4, 4), np.float32), np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 1.5, np.float32),
     np.full((4, 4, 3), np.nan, np.float32)],
)
def test_float_to_uint8_rejects_invalid_arrays(bad):
    with pytest.raises(ValueError):
        rgb_float_to_bgr_uint8(bad)


def test_save_image_creates_directories_and_round_trips(tmp_path):
    rgb = np.random.default_rng(0).random((20, 30, 3)).astype(np.float32)
    path = save_image(rgb, tmp_path / "a" / "b" / "result.png")
    assert path.is_file()
    loaded = cv2.imread(str(path))
    assert loaded.shape == (20, 30, 3) and loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, rgb_float_to_bgr_uint8(rgb))
    assert loaded.max() > 50  # a float array written as if it were uint8 would be almost black


def test_save_image_unsupported_extension_raises(tmp_path):
    with pytest.raises(OSError):
        save_image(np.zeros((4, 4, 3), np.float32), tmp_path / "x.unknownext")


def test_load_and_decode_errors(tmp_path, dark_png):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        load_image(empty)
    garbage = tmp_path / "garbage.png"
    garbage.write_text("this is not an image")
    with pytest.raises(ValueError):
        load_image(garbage)
    with pytest.raises(ValueError):
        decode_image(b"")
    with pytest.raises(ValueError):
        decode_image(b"not an image at all")
    assert load_image(dark_png).dtype == np.uint8


def test_decode_image_is_bgr_and_encode_png_round_trips(dark_bgr):
    data = cv2.imencode(".png", dark_bgr)[1].tobytes()
    np.testing.assert_array_equal(decode_image(data), dark_bgr)  # lossless, still BGR
    rgb = np.random.default_rng(2).random((16, 16, 3)).astype(np.float32)
    np.testing.assert_array_equal(decode_image(encode_png(rgb)), rgb_float_to_bgr_uint8(rgb))


def test_decode_image_handles_grayscale_and_alpha_files():
    gray = np.full((10, 12), 90, np.uint8)
    assert decode_image(cv2.imencode(".png", gray)[1].tobytes()).shape == (10, 12, 3)
    rgba = np.zeros((10, 12, 4), np.uint8)
    assert decode_image(cv2.imencode(".png", rgba)[1].tobytes()).shape == (10, 12, 3)


# ---------------------------------------------------------------- complete pipeline
def test_process_image_output_is_valid_and_every_stage_ran(dark_bgr, config):
    result = process_image(dark_bgr, config)
    out = result.enhanced_rgb
    assert out.shape == (*dark_bgr.shape[:2], 3) and out.dtype == np.float32
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0
    for stage in STAGES:
        assert result.timings[stage] > 0.0, stage
    assert result.timings["total"] >= sum(result.timings[s] for s in STAGES) * 0.99
    assert set(result.intermediates) == {
        "input", "v_channel", "illumination", "reflectance", "gamma_corrected", "clahe_enhanced",
        "dark_mask", "denoised_v", "final",
    }
    assert result.intermediates["final"] is out


def test_output_is_actually_brighter_not_black_and_not_unchanged(dark_bgr, config):
    result = process_image(dark_bgr, config)
    out_u8 = rgb_float_to_bgr_uint8(result.enhanced_rgb)
    assert gray_mean(out_u8) > 1.5 * gray_mean(dark_bgr)
    assert out_u8.max() > 80                                  # not a black image
    assert np.abs(out_u8.astype(int) - dark_bgr.astype(int)).mean() > 10  # not the input again
    assert result.info["mean_v_enhanced"] > 1.5 * result.info["mean_v"]
    assert result.info["gamma"] > 1.0                          # dark input -> brightening gamma


def test_info_contains_the_reported_quantities(dark_bgr, config):
    info = process_image(dark_bgr, config).info
    assert (info["image_height"], info["image_width"]) == dark_bgr.shape[:2]
    assert 0.0001 <= info["alpha"] <= 0.003 and 0.0001 <= info["beta"] <= 0.0005
    assert isinstance(info["retinex_iterations"], int) and info["retinex_iterations"] >= 1
    assert isinstance(info["retinex_converged"], bool)
    assert 0.0 <= info["dark_region_fraction"] <= 1.0
    assert info["mean_abs_v_minus_illumination"] >= 0.0


def test_input_is_not_mutated(dark_bgr, config):
    before = dark_bgr.copy()
    process_image(dark_bgr, config)
    np.testing.assert_array_equal(dark_bgr, before)


def test_pipeline_is_deterministic(dark_bgr, config):
    first = process_image(dark_bgr, config).enhanced_rgb
    second = process_image(dark_bgr, config).enhanced_rgb
    np.testing.assert_array_equal(first, second)


def test_modules_are_integrated_with_matching_data_flow(dark_bgr, config):
    r = process_image(dark_bgr, config)
    inter = r.intermediates
    # The step-by-step orchestration equals the public one-call enhancement function.
    np.testing.assert_array_equal(enhance_illumination(inter["illumination"]), inter["clahe_enhanced"])
    # Dark mask is built on the enhanced V with the configured threshold.
    np.testing.assert_array_equal(inter["dark_mask"], inter["clahe_enhanced"] < config["denoising"]["dark_threshold"])
    assert inter["dark_mask"].dtype == bool
    # Selective denoising: outside the mask the values are exactly the enhanced V.
    outside = ~inter["dark_mask"]
    np.testing.assert_array_equal(inter["denoised_v"][outside], inter["clahe_enhanced"][outside])
    assert not np.array_equal(inter["denoised_v"][inter["dark_mask"]], inter["clahe_enhanced"][inter["dark_mask"]])
    # Retinex on V gives T and R with the documented bound; stage ranges are valid.
    assert (inter["illumination"] <= inter["v_channel"] + 1e-6).all()
    for name in ("v_channel", "illumination", "reflectance", "gamma_corrected", "clahe_enhanced", "denoised_v"):
        assert inter[name].dtype == np.float32 and inter[name].min() >= 0.0 and inter[name].max() <= 1.0


def test_hue_and_saturation_are_preserved(dark_bgr, config):
    result = process_image(dark_bgr, config)
    original_rgb = result.intermediates["input"]
    hsv_in = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    hsv_out = cv2.cvtColor(result.enhanced_rgb, cv2.COLOR_RGB2HSV)
    well_defined = (hsv_in[..., 1] > 0.25) & (hsv_in[..., 2] > 0.1) & (hsv_out[..., 2] > 0.1)
    assert well_defined.sum() > 100
    hue_error = np.abs(((hsv_out[..., 0] - hsv_in[..., 0] + 180.0) % 360.0) - 180.0)[well_defined]
    assert hue_error.max() < 1.5                                   # degrees
    assert np.abs(hsv_out[..., 1] - hsv_in[..., 1])[well_defined].max() < 0.02


@pytest.mark.parametrize("channel, name", [(2, "red"), (0, "blue"), (1, "green")])
def test_no_red_blue_channel_swap(config, channel, name):
    bgr = np.full((64, 64, 3), 4, np.uint8)
    bgr[..., channel] = 40  # a dark image dominated by one colour (OpenCV BGR order)
    bgr = bgr + np.random.default_rng(0).integers(0, 3, bgr.shape).astype(np.uint8)
    out = rgb_float_to_bgr_uint8(process_image(bgr, config).enhanced_rgb)
    means = out.reshape(-1, 3).mean(axis=0)
    assert int(np.argmax(means)) == channel, f"{name}-dominant input must stay {name}-dominant"


def test_configuration_is_respected(dark_bgr, config):
    base = process_image(dark_bgr, config)
    no_dark = copy.deepcopy(config)
    no_dark["denoising"]["dark_threshold"] = 0.0
    result = process_image(dark_bgr, no_dark)
    assert result.info["dark_region_fraction"] == 0.0
    np.testing.assert_array_equal(result.intermediates["denoised_v"], result.intermediates["clahe_enhanced"])

    stronger = copy.deepcopy(config)
    stronger["enhancement"]["clahe_clip_limit"] = 8.0
    assert not np.array_equal(process_image(dark_bgr, stronger).enhanced_rgb, base.enhanced_rgb)

    capped = copy.deepcopy(config)
    capped["retinex"]["max_iterations"] = 1
    assert process_image(dark_bgr, capped).info["retinex_iterations"] == 1


def test_brighter_input_gets_gamma_of_one_and_valid_output(config):
    bright = np.full((40, 40, 3), 200, np.uint8) + np.random.default_rng(0).integers(0, 20, (40, 40, 3)).astype(np.uint8)
    result = process_image(bright, config)
    assert result.info["gamma"] == 1.0
    assert np.isfinite(result.enhanced_rgb).all()


@pytest.mark.parametrize("bad, exc", [(None, TypeError), (np.zeros((8, 8), np.uint8), ValueError),
                                      (np.zeros((8, 8, 4), np.uint8), ValueError)])
def test_invalid_image_arrays_are_rejected(bad, exc, config):
    with pytest.raises(exc):
        process_image(bad, config)


def test_process_image_rejects_invalid_config(dark_bgr):
    with pytest.raises(ConfigError):
        process_image(dark_bgr, {"pipeline": {}})


# ---------------------------------------------------------------- end to end (file -> file)
def test_end_to_end_file_in_file_out(dark_png, dark_bgr, config, tmp_path):
    result = run_pipeline(dark_png, config=config)
    expected = tmp_path / "out" / "dark_scene_enhanced.png"
    assert result.output_path == expected and expected.is_file()
    assert default_output_path(dark_png, config) == expected

    reopened = cv2.imread(str(expected), cv2.IMREAD_UNCHANGED)
    assert reopened is not None and reopened.dtype == np.uint8
    assert reopened.shape == dark_bgr.shape                      # same height, width and 3 channels
    np.testing.assert_array_equal(reopened, rgb_float_to_bgr_uint8(result.enhanced_rgb))  # file == pipeline result
    assert gray_mean(reopened) > 1.5 * gray_mean(dark_bgr)
    assert reopened.std() > 5 and reopened.max() > 80 and np.abs(reopened.astype(int) - dark_bgr).mean() > 10
    assert not (tmp_path / "out" / "dark_scene_intermediates").exists()   # nothing extra by default


def test_explicit_output_path_and_intermediates_flag(dark_png, config, tmp_path):
    target = tmp_path / "elsewhere" / "result.png"
    result = run_pipeline(dark_png, output_path=target, config=config, save_intermediate_images=True)
    assert target.is_file() and result.output_path == target
    folder = target.parent / "dark_scene_intermediates"
    assert result.intermediates_dir == folder
    saved = sorted(p.name for p in folder.iterdir())
    assert saved == sorted(f"dark_scene_{n}.png" for n in result.intermediates)
    mask = cv2.imread(str(folder / "dark_scene_dark_mask.png"), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(mask)) <= {0, 255}
    assert cv2.imread(str(folder / "dark_scene_illumination.png"), cv2.IMREAD_UNCHANGED).ndim == 2


def test_save_intermediates_from_config(dark_png, config, tmp_path):
    config["pipeline"]["save_intermediates"] = True
    result = run_pipeline(dark_png, config=config)
    assert result.intermediates_dir is not None and result.intermediates_dir.is_dir()


def test_run_pipeline_errors(tmp_path, config):
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / "missing.png", config=config)
    bad = tmp_path / "broken.png"
    bad.write_text("not an image")
    with pytest.raises(ValueError):
        run_pipeline(bad, config=config)
    assert not (tmp_path / "out").exists()                       # no output on failure


def test_run_pipeline_works_with_a_smaller_and_odd_sized_image(config, tmp_path):
    odd = make_dark_bgr(37, 53, seed=3)
    path = tmp_path / "odd.png"
    cv2.imwrite(str(path), odd)
    result = run_pipeline(path, config=config)
    assert cv2.imread(str(result.output_path)).shape == odd.shape


def test_module_imports_cleanly_without_side_effects(capsys):
    importlib.reload(pipeline)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
