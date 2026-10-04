# Low-Light Image Enhancement Using Retinex with Adaptive Gamma and Dark Region Denoising

A classical, interpretable image-processing system that brightens low-light photographs while preserving
colour and suppressing the noise that brightening amplifies. Every stage is a
documented mathematical operation built on NumPy, SciPy and OpenCV. It ships with a command-line tool, a
Streamlit web interface, an evaluation script and a test suite.


## Contents

1. [How it works](#how-it-works)
2. [Project structure](#project-structure)
3. [Installation](#installation)
4. [Command-line usage](#command-line-usage)
5. [Web interface](#web-interface)
6. [Using the pipeline from Python](#using-the-pipeline-from-python)
7. [Configuration](#configuration)
8. [Output locations](#output-locations)
9. [Evaluation and metrics](#evaluation-and-metrics)
10. [Testing](#testing)
11. [Assumptions](#assumptions)
12. [Limitations](#limitations)
13. [References](#references)

---

## How it works

Only the **Value (V) channel** of the HSV colour space is enhanced. Hue (H) and Saturation (S) are carried through
untouched and recombined at the end, so colours are not shifted by the brightness change. Denoising is applied
**selectively to dark regions**, where noise is most visible, so fine detail in brighter regions is kept exactly.

```
Low-light image (BGR, 8-bit)
      |
      v
Validation  ->  BGR to RGB  ->  float32 in [0, 1]  ->  RGB to HSV  ->  H, S (kept) and V
      |
      v
Mean of V  ->  adaptive alpha and beta
      |
      v
Iterative Retinex decomposition of V   ->   illumination T     reflectance R
      |
      v
Adaptive gamma correction of T
      |
      v
CLAHE                                   ->   enhanced V
      |
      v
Dark-region mask (on enhanced V)  ->  Fast Non-Local Means inside the mask only   ->   denoised V
      |
      v
Denoised V + ORIGINAL H + ORIGINAL S  ->  HSV to RGB  ->  final enhanced image
```

### Stages

1. **Preprocessing** (`src/preprocessing.py`). The image is validated (3 channels, numeric, finite), converted from
   OpenCV's BGR to RGB, scaled to `float32` in [0, 1] and converted to HSV. An initial-denoising hook exists in the
   pipeline; the source method does not name an algorithm for it, so it currently returns the image unchanged.
2. **Adaptive parameters** (`src/retinex_utils.py`). The mean of V decides how strongly the two Retinex
   regularisation terms act; darker images get larger weights:
   ```
   alpha = clip(0.001  + 0.002  * (0.5 - mean_V), 0.0001, 0.003)
   beta  = clip(0.0001 + 0.0003 * (0.5 - mean_V), 0.0001, 0.0005)
   ```
3. **Retinex decomposition** (`src/retinex.py`). V is modelled as `V = R * T` (reflectance times illumination)
   and both are estimated jointly. The updates are applied in this order, the new `T` being used for the `R` update:
   ```
   T(k+1) = ( V * R(k)   ) / ( R(k)   + alpha * G + epsilon )
   R(k+1) = ( V * T(k+1) ) / ( T(k+1) + beta  * D + epsilon )
   ```
   `G` is the Sobel gradient magnitude of V (a structural prior), `D = |G|`, and `epsilon` keeps every denominator
   positive. Iteration stops when the largest change in `T` and `R` falls below a tolerance, or at an iteration cap.
4. **Adaptive gamma correction** (`src/enhancement.py`). With `mu_T` the mean illumination,
   `gamma = 1 + 2 * (0.5 - mu_T)` when `mu_T < 0.5`, otherwise `1`. A larger gamma brightens more; the correction is
   applied as `T ** (1 / gamma)`, which brightens for gamma > 1.
5. **CLAHE** (`src/enhancement.py`). Contrast-Limited Adaptive Histogram Equalisation on the gamma-corrected
   brightness reveals local detail in dark areas while limiting over-amplification. The result is the enhanced V.
6. **Dark-region denoising** (`src/denoising.py`). A boolean mask marks pixels whose enhanced V is below a
   threshold. Fast Non-Local Means is run and its result is used **only inside the mask**; every pixel outside the
   mask keeps exactly its previous value.
7. **Reconstruction** (`src/reconstruction.py`). The denoised V is combined with the original H and S and converted
   back to RGB. H is kept in OpenCV's float-HSV degrees (0 to 360) so that the conversion round-trips exactly.

`src/pipeline.py` orchestrates these stages and is the **only** place where they are chained. The command line, the
web interface, the evaluation script and the tests all call it, so there is one implementation of the algorithm.

## Project structure

```
.
├── app.py                    # Streamlit web interface
├── configs/
│   └── default.yaml          # All stage parameters
├── data/
│   ├── input/                # Put your images here (contents are git-ignored)
│   └── output/               # Enhanced images are written here (git-ignored)
├── results/
│   ├── qualitative/          # Input | enhanced comparisons from scripts/evaluate.py
│   └── quantitative/         # Metric tables (CSV) and summaries (JSON)
├── scripts/
│   ├── run_pipeline.py       # Enhance one image from the command line
│   └── evaluate.py           # Run the pipeline and metrics over a directory of images
├── src/
│   ├── preprocessing.py      # Validation, BGR -> RGB, float32 [0, 1], HSV split
│   ├── retinex_utils.py      # Mean V, adaptive alpha/beta, Sobel terms
│   ├── retinex.py            # Iterative Retinex decomposition
│   ├── enhancement.py        # Adaptive gamma correction and CLAHE
│   ├── denoising.py          # Dark-region mask and selective Fast NLM
│   ├── reconstruction.py     # Recombination with H and S, HSV -> RGB
│   ├── metrics.py            # NIQE, Average Brightness, Discrete Entropy
│   ├── pipeline.py           # Orchestration, configuration, image input/output
│   └── data/                 # NIQE pre-trained model parameters (see NIQE_PARAMETERS.md)
├── tests/                    # pytest suite
├── requirements.txt
└── README.md
```

### Data conventions

| Data | Type | Shape | Range |
|---|---|---|---|
| RGB working image | `float32` | `(H, W, 3)` | [0, 1] |
| H (hue) | `float32` | `(H, W)` | degrees, [0, 360) |
| S, V | `float32` | `(H, W)` | [0, 1] |
| Illumination T, reflectance R | `float32` | `(H, W)` | [0, 1] |
| Dark-region mask | `bool` | `(H, W)` | – |

No stage modifies its input arrays in place. Images are decoded and written with OpenCV, so file I/O is BGR and
8-bit; the conversion between that and the internal float RGB is done in explicit, tested functions.

## Installation

Python 3 is required (developed and tested with Python 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: NumPy, OpenCV (`opencv-python`), SciPy (used by NIQE), PyYAML, Streamlit (web interface) and pytest.

## Command-line usage

```bash
python scripts/run_pipeline.py --input data/input/test.png
```

This runs the complete pipeline, saves `data/output/test_enhanced.png` and prints the statistics: input size,
mean V, alpha and beta, Retinex iterations and convergence, gamma, dark-region coverage, mean V after enhancement,
per-stage timings and the total processing time.

| Option | Meaning |
|---|---|
| `--input PATH` | Input image (required) |
| `--output PATH` | Where to write the result (default `data/output/<name>_enhanced.png`) |
| `--config PATH` | Alternative YAML configuration (default `configs/default.yaml`) |
| `--save-intermediates` | Also save the intermediate images (see below) |
| `--verbose` | Print per-stage log messages |

The exit status is `0` on success and `1` with a one-line `Error:` message for a missing, empty, corrupt or
unsupported image, an invalid configuration, or an unwritable output path.

## Web interface

```bash
streamlit run app.py
```

Open the address Streamlit prints (usually http://localhost:8501), then:

1. Upload a low-light image (PNG, JPEG, BMP, TIFF or WebP).
2. The original is shown immediately. Press **Enhance image**.
3. The enhanced image appears beside it, together with the processing information: mean V, alpha, beta,
   Retinex iterations, adaptive gamma, dark-region coverage, mean V after enhancement and processing time.
4. Press **Download enhanced image** to save the result as a PNG.
5. Optional check boxes also show the quality metrics (NIQE, AB, DE for the input and the result) and the
   intermediate images (V channel, illumination, reflectance, gamma, CLAHE, mask, denoised V).

The interface contains no image-processing code: it calls `src.pipeline.process_image` and displays what that
returns. Empty or undecodable uploads produce a readable error message.

## Using the pipeline from Python

```python
from src.pipeline import load_config, load_image, process_image, run_pipeline, save_image

config = load_config()                                   # configs/default.yaml

# Array in, result out (no files involved). image_bgr is an 8-bit BGR array, as cv2.imread returns.
result = process_image(load_image("data/input/test.png"), config)
result.enhanced_rgb        # float32 (H, W, 3) in [0, 1]
result.info                # mean_v, alpha, beta, retinex_iterations, gamma, dark_region_fraction, ...
result.timings             # seconds per stage and "total"
result.intermediates       # illumination, reflectance, gamma_corrected, clahe_enhanced, dark_mask, ...

# File in, file out
result = run_pipeline("data/input/test.png")             # writes data/output/test_enhanced.png
print(result.output_path)
```

## Configuration

`configs/default.yaml` is the single source of parameters. `src/pipeline.py` loads it once, validates it and passes
the values to each stage as explicit arguments; the stage modules never read the file themselves.

| Section | Key | Default | Meaning |
|---|---|---|---|
| `pipeline` | `save_intermediates` | `false` | Also write intermediate images |
| `pipeline` | `output_dir` | `data/output` | Output directory (created if missing) |
| `pipeline` | `output_format` | `png` | Lossless output format: `png`, `bmp` or `tiff` |
| `retinex` | `epsilon` | `0.001` | Constant that keeps every denominator positive |
| `retinex` | `max_iterations` | `50` | Iteration cap |
| `retinex` | `tolerance` | `0.0001` | Stop when the largest change in T and R is below this |
| `enhancement` | `clahe_clip_limit` | `2.0` | CLAHE contrast limit |
| `enhancement` | `clahe_tile_grid_size` | `[8, 8]` | CLAHE tiles as `[rows, columns]` |
| `denoising` | `dark_threshold` | `0.3` | Pixels with enhanced V below this are denoised |
| `denoising` | `nlm_h` | `10.0` | Fast NLM filter strength |
| `denoising` | `nlm_template_window_size` | `7` | NLM template window |
| `denoising` | `nlm_search_window_size` | `21` | NLM search window |

`alpha` and `beta` are not configured: they are computed from each image's mean V. Key names are translated to
function arguments in the pipeline (`nlm_h` -> `h`, `dark_threshold` -> `threshold`, and so on). A missing section,
a `null` value or an invalid value stops the run with a clear message.

## Output locations

| What | Where |
|---|---|
| Enhanced image | `data/output/<name>_enhanced.png` (or `--output`) |
| Intermediate images (optional) | `data/output/<name>_intermediates/<name>_<stage>.png` |
| Evaluation: enhanced images | `data/output/<dataset>/<name>_enhanced.png` |
| Evaluation: metric tables | `results/quantitative/<dataset>_metrics.csv` and `<dataset>_summary.json` |
| Evaluation: comparisons | `results/qualitative/<dataset>/<name>_comparison.png` |

Intermediate images are the V channel, illumination, reflectance, the gamma and CLAHE results, the dark-region mask,
the denoised V and the final image, all converted to 8-bit for viewing. The hue channel is not written on its own:
it is an angle in degrees rather than a brightness, so a grayscale dump of it would be misleading.

## Evaluation and metrics

```bash
python scripts/evaluate.py --input data/input/LIME --dataset LIME --repeats 10 --save-comparisons
```

For every image in the folder the full pipeline is run and the metrics are computed on both the input and the
enhanced result. The script writes a per-image CSV and a JSON summary (mean and standard deviation per metric), and
lists any file it had to skip. Datasets such as **LIME**, **NPE** and **MEF** are not included in this repository;
download them and place them under `data/input/`. Every number in the output comes from running the code on the
images that are present.

| Metric | Definition | How to read it |
|---|---|---|
| **NIQE** | Natural Image Quality Evaluator (Mittal et al., 2013) on the 8-bit grayscale image | Lower generally means the image statistics are closer to natural images |
| **AB** | Average brightness: mean 8-bit gray level, 0 to 255 | Should be appropriate, not simply maximised |
| **DE** | Discrete entropy of the 256-bin gray-level histogram, `-sum p(i) log2 p(i)`, 0 to 8 bits | Higher can mean richer detail, but amplified noise also raises it |
| **Runtime** | Pipeline processing time from the loaded array to the final RGB image, via `time.perf_counter()` | Lower means less computation; file reading, saving and metrics are excluded |

With `--repeats N`, the runtime is the mean over `N` runs per image (the first run includes warm-up). No metric is
turned into a score or a verdict.

**About NIQE.** scikit-image does not provide NIQE, so it is implemented in `src/metrics.py` from the published
algorithm (MSCN coefficients, asymmetric generalised Gaussian fits and paired-product features at two scales on
96x96 blocks, compared with a pre-trained pristine-image model). The pristine model is stored in
`src/data/niqe_pris_params.npz`; its origin and the points to check before redistributing it are in
`src/data/NIQE_PARAMETERS.md`. NIQE needs at least two 96x96 blocks, so for smaller or perfectly uniform images
the value is reported as unavailable (blank in the CSV) while AB and DE are still produced. Different NIQE
implementations can give slightly different values, so compare scores produced by this code with each other.

## Testing

```bash
pytest -q
```

The suite covers every module (validation, ranges, dtypes, determinism, non-mutation of inputs, special inputs),
the Retinex equations against an independent hand-written implementation, the metrics against known values, the
configuration loader, the complete pipeline (stages, timings, selective denoising, hue and saturation
preservation, absence of a red/blue channel swap), file-in/file-out behaviour with the saved image reopened and
checked, both command-line tools, and the web interface driven with a real upload.

## Assumptions

The source method fixes the update equations, the formulas for alpha, beta and gamma, and the order of the stages.
It does not fix the following, which are therefore implementation choices:

- **Retinex initialisation and stopping:** `T(0) = V`, `R(0) = 1`; stop when `max(max|dT|, max|dR|) < tolerance`.
  `G` and `D` are computed once from V. `epsilon`, the iteration cap and the tolerance take the values above.
- **Illumination enhancement:** the enhanced V is derived from the illumination `T` alone, as the method's data flow
  describes; the reflectance `R` is returned and shown but is not recombined into the result.
- **CLAHE:** clip limit 2.0 and an 8x8 tile grid; the image is quantised to 8 bits for OpenCV's CLAHE.
- **Dark-region denoising:** a fixed threshold of 0.3 on the enhanced V, and Fast NLM on an 8-bit copy with the
  parameters above.
- **Initial denoising:** no algorithm is specified, so that hook is the identity.
- **Metrics:** all are computed on the 8-bit grayscale image (ITU-R BT.601 luma). The method does not give equations
  for AB and DE, so the standard definitions are used.

## Limitations

- **Illumination stays close to V.** With the update equations as specified, both `T` and `R` settle very close to
  `V` within a few iterations (typically 3 to 5), so `R * T` is close to `V^2` rather than `V`. This follows from the
  equations and has not been altered; it means the Retinex step itself changes the image only slightly, and most of
  the visible enhancement comes from the gamma correction and CLAHE that follow.
- **Colour noise is brightened with the image.** H and S are preserved by design, and only V is denoised. In very
  dark, noisy regions the original hue and saturation are themselves noisy, so brightening can reveal coloured
  speckle (and occasional colour casts, for example a greenish tint on a dark wall). Raising `dark_threshold` or
  `nlm_h` smooths luminance noise further but does not remove this.
- **Regions brightened above the threshold are not denoised.** A dim but textured area that CLAHE lifts above
  `dark_threshold` keeps its noise.
- **Not every image improves.** On an already-dark image full of fine, faint structure (a night map in the tests) the
  CLAHE step amplified detail and noise, and the measured NIQE became worse. Brightness and entropy rise on every
  image tried, but that is not the same as better quality.
- **Constant or nearly constant images** are shifted in brightness by CLAHE (a pure black image becomes dark gray).
- **Speed.** Fast NLM dominates the runtime: roughly one second for a 1039x789 image on the development machine,
  growing with the number of pixels. Very large photographs will take correspondingly longer.
- **Input:** 8-bit colour images; grayscale and alpha-channel files are read as 3-channel colour.

## References

1. D. Gupta, Naveen, and S. D. Pawar, "Enhancing Low-Light Images Using Retinex with Adaptive Gamma and Dark
   Region Denoising," 2025 IEEE International Conference on Computer Vision and Machine Intelligence (CVMI), 2025,
   doi: 10.1109/CVMI66673.2025.11337939.
2. E. H. Land, "The retinex theory of color vision," Scientific American, vol. 237, no. 6, pp. 108–129, 1977.
3. D. J. Jobson, Z.-u. Rahman, and G. A. Woodell, "A multiscale retinex for bridging the gap between color images
   and the human observation of scenes," IEEE Transactions on Image Processing, vol. 6, no. 7, pp. 965–976, 1997.
4. J. Xu et al., "STAR: A structure and texture aware Retinex model," IEEE Transactions on Image Processing,
   vol. 29, pp. 5022–5037, 2020.
5. A. Mittal, R. Soundararajan, and A. C. Bovik, "Making a 'completely blind' image quality analyzer," IEEE Signal
   Processing Letters, vol. 20, no. 3, pp. 209–212, 2013.
