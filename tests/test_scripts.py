"""Tests for the command-line tools: scripts/run_pipeline.py and scripts/evaluate.py (run as subprocesses)."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pipeline import rgb_float_to_bgr_uint8  # noqa: F401  (import check only)
from tests.conftest import make_dark_bgr

ROOT = Path(__file__).resolve().parent.parent
RUN = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py")]
EVALUATE = [sys.executable, str(ROOT / "scripts" / "evaluate.py")]


def run(cmd, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=300)


# ---------------------------------------------------------------- run_pipeline.py
def test_cli_runs_the_complete_pipeline_and_saves_the_image(dark_png, dark_bgr, tmp_path):
    output = tmp_path / "result" / "enhanced.png"
    done = run(RUN + ["--input", str(dark_png), "--output", str(output)])
    assert done.returncode == 0, done.stderr
    assert output.is_file()
    reopened = cv2.imread(str(output))
    assert reopened.shape == dark_bgr.shape
    assert cv2.cvtColor(reopened, cv2.COLOR_BGR2GRAY).mean() > 1.5 * cv2.cvtColor(dark_bgr, cv2.COLOR_BGR2GRAY).mean()
    text = done.stdout
    for expected in ("Mean V (input)", "Alpha / Beta", "Retinex iterations", "Adaptive gamma",
                     "Dark-region coverage", "Total processing time", f"Enhanced image saved  : {output}"):
        assert expected in text
    assert "implemented" not in text.lower() and "stops" not in text.lower()


def test_cli_default_output_location_and_intermediates(dark_png, tmp_path):
    done = run(RUN + ["--input", str(dark_png), "--save-intermediates"], cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "data" / "output" / "dark_scene_enhanced.png").is_file()
    assert (tmp_path / "data" / "output" / "dark_scene_intermediates" / "dark_scene_illumination.png").is_file()


def test_cli_accepts_a_custom_config(dark_png, tmp_path):
    cfg = (ROOT / "configs" / "default.yaml").read_text().replace("dark_threshold: 0.3", "dark_threshold: 0.0")
    cfg_path = tmp_path / "custom.yaml"
    cfg_path.write_text(cfg)
    done = run(RUN + ["--input", str(dark_png), "--output", str(tmp_path / "o.png"), "--config", str(cfg_path)])
    assert done.returncode == 0, done.stderr
    assert "Dark-region coverage  : 0.0%" in done.stdout


@pytest.mark.parametrize("case", ["missing", "garbage", "empty", "bad_config", "bad_extension"])
def test_cli_fails_cleanly_with_nonzero_status(case, dark_png, tmp_path):
    args = ["--input", str(dark_png), "--output", str(tmp_path / "o.png")]
    if case == "missing":
        args[1] = str(tmp_path / "missing.png")
    elif case == "garbage":
        bad = tmp_path / "garbage.png"
        bad.write_text("not an image")
        args[1] = str(bad)
    elif case == "empty":
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        args[1] = str(empty)
    elif case == "bad_config":
        args += ["--config", str(tmp_path / "nope.yaml")]
    elif case == "bad_extension":
        args[3] = str(tmp_path / "o.unknownext")
    done = run(RUN + args)
    assert done.returncode == 1
    assert done.stderr.startswith("Error:") and "Traceback" not in done.stderr
    assert not (tmp_path / "o.png").exists()


# ---------------------------------------------------------------- evaluate.py
@pytest.fixture
def image_folder(tmp_path):
    folder = tmp_path / "TESTSET"
    folder.mkdir()
    cv2.imwrite(str(folder / "a.png"), make_dark_bgr(200, 220, seed=1))
    cv2.imwrite(str(folder / "b.png"), make_dark_bgr(210, 200, seed=2, scale=0.25))
    cv2.imwrite(str(folder / "small.png"), make_dark_bgr(60, 60, seed=3))
    (folder / "notes.txt").write_text("not an image file and not collected")
    (folder / "broken.png").write_text("not an image")
    return folder


def test_evaluate_writes_real_metrics(image_folder, tmp_path):
    results, out = tmp_path / "results", tmp_path / "enhanced"
    done = run(EVALUATE + ["--input", str(image_folder), "--dataset", "TESTSET", "--results-dir", str(results),
                           "--output-dir", str(out), "--save-comparisons", "--repeats", "2"])
    assert done.returncode == 0, done.stderr
    assert "Processed 3 image(s), skipped 1." in done.stdout

    rows = list(csv.DictReader(open(results / "quantitative" / "TESTSET_metrics.csv")))
    assert sorted(r["image"] for r in rows) == ["a.png", "b.png", "small.png"]
    by_name = {r["image"]: r for r in rows}
    for name in ("a.png", "b.png"):
        row = by_name[name]
        assert float(row["niqe"]) > 0 and float(row["input_niqe"]) > 0
        assert float(row["ab"]) > float(row["input_ab"])               # measured, brighter than the input
        assert 0 < float(row["de"]) <= 8 and float(row["runtime_s"]) > 0
    assert by_name["small.png"]["niqe"] == ""                           # NIQE undefined for tiny images: blank
    assert float(by_name["small.png"]["ab"]) > 0                        # but AB/DE are still reported

    summary = json.loads((results / "quantitative" / "TESTSET_summary.json").read_text())
    assert summary["images_processed"] == 3 and summary["runtime_repeats"] == 2
    assert len(summary["images_skipped"]) == 1 and "broken.png" in summary["images_skipped"][0]
    assert summary["metrics"]["niqe"]["count"] == 2
    assert summary["metrics"]["ab"]["mean"] == pytest.approx(np.mean([float(r["ab"]) for r in rows]))

    assert (out / "a_enhanced.png").is_file() and (out / "b_enhanced.png").is_file()
    comparison = cv2.imread(str(results / "qualitative" / "TESTSET" / "a_comparison.png"))
    assert comparison.shape == (200, 440, 3)                            # input | enhanced, side by side


@pytest.mark.parametrize("case", ["missing_dir", "empty_dir", "bad_repeats"])
def test_evaluate_fails_cleanly(case, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    args = {"missing_dir": ["--input", str(tmp_path / "nope")],
            "empty_dir": ["--input", str(empty)],
            "bad_repeats": ["--input", str(empty), "--repeats", "0"]}[case]
    done = run(EVALUATE + args + ["--results-dir", str(tmp_path / "r")])
    assert done.returncode == 1 and "Error:" in done.stderr and "Traceback" not in done.stderr
    assert not list((tmp_path / "r").glob("**/*.csv"))                  # no result files, nothing fabricated
