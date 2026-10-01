"""CLI entry point for the low-light enhancement pipeline.

Usage:
    python scripts/run_pipeline.py --input data/input/example.png

Currently runs only the implemented stages (preprocessing, mean V, alpha/beta) and
prints their values; the Retinex, enhancement, denoising and reconstruction stages
are not wired in yet.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Low-light image enhancement (Retinex + adaptive gamma + dark-region "
        "denoising). Currently runs the preprocessing stage only."
    )
    parser.add_argument("--input", required=True, help="Path to a low-light input image")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = run_pipeline(args.input)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        f"mean_v={result['mean_v']:.4f}  alpha={result['alpha']:.6f}  beta={result['beta']:.6f}"
    )
    print("Pipeline stops here: Retinex and later stages are not implemented yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
