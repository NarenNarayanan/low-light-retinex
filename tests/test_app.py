"""Tests for the Streamlit UI (app.py), driven with Streamlit's AppTest runner and a real upload."""
import ast
import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.pipeline import load_config, process_image
from tests.conftest import make_dark_bgr

APP = Path(__file__).resolve().parent.parent / "app.py"


def new_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=180).run()


def png_bytes(image) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


def test_app_starts_without_error_and_asks_for_an_upload():
    at = new_app()
    assert not at.exception
    assert at.title[0].value == "Low-Light Image Enhancement"
    assert at.file_uploader and any("Choose an image" in i.value for i in at.info)
    assert len(at.image) == 0


def test_upload_shows_original_then_enhance_shows_real_pipeline_output():
    image = make_dark_bgr(140, 180, seed=5)
    at = new_app()
    at.file_uploader[0].upload("night.png", png_bytes(image), "image/png").run()
    assert not at.exception and len(at.image) == 1                      # original only, nothing fake yet
    assert [b.label for b in at.button] == ["Enhance image"]

    at.button[0].click().run()
    assert not at.exception and not at.error
    assert len(at.image) == 2 and len(at.get("download_button")) == 1
    assert at.get("download_button")[0].proto.label == "Download enhanced image"

    # The numbers on screen must be the ones the real pipeline returns for this exact image.
    expected = process_image(image, load_config())
    shown = {m.label: m.value for m in at.metric}
    assert shown["Mean V (input)"] == f"{expected.info['mean_v']:.4f}"
    assert shown["Mean V (enhanced)"] == f"{expected.info['mean_v_enhanced']:.4f}"
    assert shown["Alpha"] == f"{expected.info['alpha']:.6f}"
    assert shown["Beta"] == f"{expected.info['beta']:.6f}"
    assert shown["Adaptive gamma"] == f"{expected.info['gamma']:.4f}"
    assert shown["Retinex iterations"] == str(expected.info["retinex_iterations"])
    assert shown["Dark-region coverage"] == f"{expected.info['dark_region_fraction'] * 100:.1f}%"
    assert "Processing time" in shown


def test_optional_metrics_and_intermediates_panels():
    at = new_app()
    at.file_uploader[0].upload("night.png", png_bytes(make_dark_bgr(200, 220, seed=6)), "image/png").run()
    at.checkbox[0].check()
    at.checkbox[1].check().run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert len(at.table) == 1 and "NIQE" in at.table[0].value.iloc[0, 0]
    assert len(at.image) == 2 + 7                                       # original, enhanced, 7 intermediates


@pytest.mark.parametrize("name, content, message", [
    ("empty.png", b"", "empty"),
    ("broken.png", b"this is not an image", "Unsupported or corrupt"),
])
def test_bad_uploads_show_a_readable_error_not_a_traceback(name, content, message):
    at = new_app()
    at.file_uploader[0].upload(name, content, "image/png").run()
    assert not at.exception
    assert len(at.error) == 1 and message in at.error[0].value
    assert len(at.image) == 0 and not at.button


def test_uploading_a_different_file_discards_the_previous_result():
    at = new_app()
    at.file_uploader[0].upload("a.png", png_bytes(make_dark_bgr(100, 120, seed=1)), "image/png").run()
    at.button[0].click().run()
    assert len(at.image) == 2
    at.file_uploader[0].upload("b.png", png_bytes(make_dark_bgr(90, 110, seed=2)), "image/png").run()
    assert not at.exception


def test_app_contains_no_processing_logic_and_uses_the_shared_pipeline():
    tree = ast.parse(APP.read_text())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "cv2" not in imported                                        # no image processing in the UI
    from_pipeline = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "src.pipeline"
                     for a in n.names}
    assert "process_image" in from_pipeline
    source = APP.read_text()
    for forbidden in ("fastNlMeans", "createCLAHE", "retinex_decompose", "cvtColor"):
        assert forbidden not in source
