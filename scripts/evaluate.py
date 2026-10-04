"""Evaluate the pipeline on a directory of images and write metric tables.

Usage:
    python scripts/evaluate.py --input data/input/LIME --dataset LIME
    python scripts/evaluate.py --input data/input/NPE --dataset NPE --repeats 10 --save-comparisons

For every image the full pipeline is run and NIQE, Average Brightness (AB) and Discrete Entropy (DE)
are computed on the input and on the enhanced result. Runtime is the pipeline's processing time
(file reading, saving and metric computation are excluded); with ``--repeats N`` it is the mean over N
runs. Nothing is simulated: every number written comes from running the code on the images found.

Outputs:
    results/quantitative/<dataset>_metrics.csv   one row per image
    results/quantitative/<dataset>_summary.json  per-dataset mean and standard deviation
    data/output/<dataset>/<name>_enhanced.png    enhanced images
    results/qualitative/<dataset>/<name>_comparison.png   input | enhanced (with --save-comparisons)
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import evaluate_image  # noqa: E402
from src.pipeline import (  # noqa: E402
    load_config, load_image, process_image, rgb_float_to_bgr_uint8, save_image,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
METRIC_FIELDS = ("niqe", "ab", "de")
logger = logging.getLogger("evaluate")


def find_images(directory: Path) -> list[Path]:
    """Image files directly inside ``directory`` (sorted for reproducible ordering)."""
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def evaluate_one(path: Path, config: dict, repeats: int):
    """Process one image ``repeats`` times; return (metric row, input BGR, enhanced BGR, last result)."""
    image_bgr = load_image(path)
    runtimes = []
    result = None
    for _ in range(repeats):
        result = process_image(image_bgr, config)
        runtimes.append(result.timings["total"])
    runtime = float(np.mean(runtimes))
    enhanced_bgr = rgb_float_to_bgr_uint8(result.enhanced_rgb)

    before = evaluate_image(image_bgr)
    after = evaluate_image(enhanced_bgr, runtime_s=runtime)
    row = {
        "image": path.name,
        "width": image_bgr.shape[1],
        "height": image_bgr.shape[0],
        "input_niqe": before["niqe"], "input_ab": before["ab"], "input_de": before["de"],
        "niqe": after["niqe"], "ab": after["ab"], "de": after["de"],
        "runtime_s": runtime,
        "mean_v": result.info["mean_v"],
        "alpha": result.info["alpha"], "beta": result.info["beta"],
        "retinex_iterations": result.info["retinex_iterations"],
        "gamma": result.info["gamma"],
        "dark_region_fraction": result.info["dark_region_fraction"],
    }
    return row, image_bgr, enhanced_bgr, result


def summarise(rows: list[dict], dataset: str, repeats: int, skipped: list[str]) -> dict:
    """Mean and standard deviation of each metric over the processed images."""
    summary = {"dataset": dataset, "images_processed": len(rows), "runtime_repeats": repeats,
               "images_skipped": skipped, "metrics": {}}
    for key in ("input_niqe", "input_ab", "input_de", "niqe", "ab", "de", "runtime_s"):
        values = [r[key] for r in rows if r[key] is not None]
        summary["metrics"][key] = (
            {"mean": float(np.mean(values)), "std": float(np.std(values)), "count": len(values)}
            if values else None
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the pipeline and metrics on a directory of images.")
    parser.add_argument("--input", required=True, help="Directory containing the images")
    parser.add_argument("--dataset", help="Dataset label used in file names (default: directory name)")
    parser.add_argument("--config", help="YAML configuration (default: configs/default.yaml)")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per image for the runtime average")
    parser.add_argument("--results-dir", default="results", help="Base directory for results")
    parser.add_argument("--output-dir", help="Where enhanced images go (default: <output_dir>/<dataset>)")
    parser.add_argument("--save-comparisons", action="store_true", help="Save input | enhanced images")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    directory = Path(args.input)
    if not directory.is_dir():
        print(f"Error: input directory not found: {directory}", file=sys.stderr)
        return 1
    if args.repeats < 1:
        print("Error: --repeats must be at least 1.", file=sys.stderr)
        return 1
    images = find_images(directory)
    if not images:
        print(f"Error: no image files found in {directory}", file=sys.stderr)
        return 1

    try:
        config = load_config(args.config)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    dataset = args.dataset or directory.name
    enhanced_dir = Path(args.output_dir) if args.output_dir else Path(config["pipeline"]["output_dir"]) / dataset
    quantitative = Path(args.results_dir) / "quantitative"
    qualitative = Path(args.results_dir) / "qualitative" / dataset
    quantitative.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    for index, path in enumerate(images, start=1):
        try:
            row, input_bgr, enhanced_bgr, result = evaluate_one(path, config, args.repeats)
        except (ValueError, TypeError, OSError) as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            skipped.append(f"{path.name}: {exc}")
            continue
        save_image(result.enhanced_rgb, enhanced_dir / f"{path.stem}_enhanced.png")
        if args.save_comparisons:
            qualitative.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(qualitative / f"{path.stem}_comparison.png"), np.hstack([input_bgr, enhanced_bgr]))
        rows.append(row)
        niqe_text = "n/a" if row["niqe"] is None else f"{row['niqe']:.3f}"
        logger.info("[%d/%d] %s  NIQE %s  AB %.1f  DE %.3f  %.3fs",
                    index, len(images), path.name, niqe_text, row["ab"], row["de"], row["runtime_s"])

    if not rows:
        print("Error: no image could be processed.", file=sys.stderr)
        return 1

    csv_path = quantitative / f"{dataset}_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows, dataset, args.repeats, skipped)
    json_path = quantitative / f"{dataset}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Processed {len(rows)} image(s), skipped {len(skipped)}.")
    print(f"Per-image metrics : {csv_path}")
    print(f"Dataset summary   : {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
