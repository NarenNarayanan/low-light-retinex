"""Placeholder tests for src/reconstruction.py. Real tests are added together with the implementation."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_module_imports_cleanly():
    importlib.import_module("src.reconstruction")
