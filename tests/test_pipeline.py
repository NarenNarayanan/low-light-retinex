"""Tests for the pipeline skeleton in src/pipeline.py."""
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import pipeline  # noqa: E402
from src.pipeline import load_image, run_pipeline  # noqa: E402


@pytest.fixture
def synthetic_image(tmp_path):
    rng = np.random.default_rng(0)
    img = (rng.random((20, 30, 3)) * 60).astype(np.uint8)  # dark image
    path = tmp_path / "dark.png"
    assert cv2.imwrite(str(path), img)
    return path


def test_run_pipeline_hands_off_stage_one_outputs(synthetic_image):
    result = run_pipeline(synthetic_image)
    assert set(result) == {"preprocessed", "mean_v", "alpha", "beta"}
    pre = result["preprocessed"]
    assert pre.rgb.shape == (20, 30, 3)
    assert pre.v.shape == (20, 30)
    assert result["mean_v"] == pytest.approx(float(pre.v.mean()))
    assert 0.0001 <= result["alpha"] <= 0.003
    assert 0.0001 <= result["beta"] <= 0.0005
    assert result["mean_v"] < 0.5  # dark input -> alpha above the 0.001 midpoint value
    assert result["alpha"] > 0.001


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")


def test_load_image_unreadable_file_raises(tmp_path):
    bad = tmp_path / "not_an_image.png"
    bad.write_text("this is not an image")
    with pytest.raises(ValueError):
        load_image(bad)


def test_module_imports_cleanly_without_side_effects(capsys):
    importlib.reload(pipeline)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
