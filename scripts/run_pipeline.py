"""Command-line entry point: enhance one low-light image with the complete pipeline.

Usage:
    python scripts/run_pipeline.py --input data/input/test.png
    python scripts/run_pipeline.py --input photo.jpg --output results/photo_out.png --config configs/default.yaml
    python scripts/run_pipeline.py --input photo.jpg --save-intermediates

The enhanced image is written to ``data/output/<name>_enhanced.png`` unless ``--output`` is given.
The exit status is 0 on success and 1 when the input, configuration or processing is invalid.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import load_config, run_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Low-light image enhancement using Retinex with adaptive gamma and dark-region denoising."
    )
    parser.add_argument("--input", required=True, help="Path to a low-light input image")
    parser.add_argument("--output", help="Path for the enhanced image (default: <output_dir>/<name>_enhanced.png)")
    parser.add_argument("--config", help="Path to a YAML configuration (default: configs/default.yaml)")
    parser.add_argument(
        "--save-intermediates", action="store_true",
        help="Also save intermediate images (illumination, gamma, CLAHE, mask, ...)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show per-stage log messages")
    return parser


def print_summary(result) -> None:
    info, timings = result.info, result.timings
    print(f"Input size            : {info['image_width']} x {info['image_height']}")
    print(f"Mean V (input)        : {info['mean_v']:.4f}")
    print(f"Alpha / Beta          : {info['alpha']:.6f} / {info['beta']:.6f}")
    print(
        f"Retinex iterations    : {info['retinex_iterations']} "
        f"({'converged' if info['retinex_converged'] else 'iteration limit reached'}, "
        f"final change {info['retinex_final_delta']:.2e})"
    )
    print(f"Adaptive gamma        : {info['gamma']:.4f}")
    print(f"Dark-region coverage  : {info['dark_region_fraction'] * 100:.1f}% of pixels")
    print(f"Mean V (enhanced)     : {info['mean_v_enhanced']:.4f}")
    stages = ", ".join(f"{name} {seconds:.3f}s" for name, seconds in timings.items() if name != "total")
    print(f"Stage timings         : {stages}")
    print(f"Total processing time : {timings['total']:.3f} s")
    print(f"Enhanced image saved  : {result.output_path}")
    if result.intermediates_dir:
        print(f"Intermediate images   : {result.intermediates_dir}")


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s"
    )
    try:
        config = load_config(args.config)
        result = run_pipeline(
            args.input,
            output_path=args.output,
            config=config,
            save_intermediate_images=True if args.save_intermediates else None,
        )
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
