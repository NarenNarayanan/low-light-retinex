# Low-Light Image Enhancement Using Retinex with Adaptive Gamma and Dark Region Denoising

A classical, interpretable image-processing pipeline that brightens low-light photographs while preserving
colour and suppressing the noise that brightening amplifies. It uses no machine learning: every stage is
a documented mathematical operation built on NumPy and OpenCV.

Developed for **CS23045 – Image Processing**.

---

## Table of contents

1. [Overview](#overview)
2. [Method](#method)
3. [Implementation status](#implementation-status)
4. [Repository structure](#repository-structure)
5. [Data conventions](#data-conventions)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Configuration](#configuration)
9. [Testing](#testing)
10. [Evaluation](#evaluation)
11. [Notes on the Retinex implementation](#notes-on-the-retinex-implementation)
12. [References](#references)

---

## Overview

Low-light images are dark, low in contrast and noisy. Simply scaling brightness washes out colour and amplifies
sensor noise. This project follows a Retinex-based approach that works on the **Value (V) channel** of the HSV
colour space:

- **Hue (H) and Saturation (S) are preserved untouched**, so colours are not shifted.
- Only the brightness channel is decomposed, enhanced and denoised.
- Denoising is applied **selectively to dark regions**, where noise is most visible, so fine detail in
  brighter regions is kept.

```
Low-light RGB image
      |
      v
Input validation, BGR -> RGB, float32 [0, 1]
      |
      v
RGB -> HSV            (H and S are kept; V is enhanced)
      |
      v
Mean V -> adaptive alpha and beta
      |
      v
Iterative Retinex decomposition of V  ->  illumination T, reflectance R
      |
      v
Adaptive gamma correction (on T)
      |
      v
CLAHE
      |
      v
Enhanced V
      |
      v
Dark-region mask -> selective Fast Non-Local Means denoising
      |
      v
Enhanced V + ORIGINAL H and S -> HSV -> RGB
      |
      v
Final enhanced image
```

## Method

### 1. Preprocessing
The input is validated (a 3-channel numeric image with finite values), converted from OpenCV's BGR order to RGB,
and normalised to `float32` in [0, 1]. An initial denoising hook is present in the pipeline, but the report does
not specify an algorithm for it, so it currently returns the image unchanged. The RGB image is then converted
to HSV and split into H, S and V.

### 2. Adaptive Retinex parameters
The mean of V decides how strongly the two regularisation terms act. Darker images get larger weights:

```
alpha = clip(0.001  + 0.002  * (0.5 - mean_V), 0.0001, 0.003)
beta  = clip(0.0001 + 0.0003 * (0.5 - mean_V), 0.0001, 0.0005)
```

### 3. Iterative Retinex decomposition
The Value channel is modelled as `V = R * T`, where `T` is the illumination and `R` the reflectance. Both are
estimated jointly with the updates below, in this order (the new `T` is used for the `R` update):

```
T(k+1) = ( V * R(k)   ) / ( R(k)   + alpha * G + epsilon )
R(k+1) = ( V * T(k+1) ) / ( T(k+1) + beta  * D + epsilon )
```

- `G` is the Sobel gradient magnitude (a structural prior) and `D = |G|`.
- `epsilon` is a small constant that keeps every denominator positive.
- Iteration stops when the largest change in `T` and `R` between iterations falls below a tolerance, or after a
  maximum number of iterations.

### 4. Adaptive gamma correction
Computed from the mean illumination `mu_T`:

```
gamma = 1 + 2 * (0.5 - mu_T)    if mu_T < 0.5
gamma = 1                       otherwise
```

Darker illumination gives a larger gamma and therefore stronger brightening.

### 5. CLAHE
Contrast-Limited Adaptive Histogram Equalisation is applied to the gamma-corrected brightness to bring out local
detail in dark regions while limiting over-amplification.

### 6. Dark-region denoising
A dark-region mask is computed on the enhanced brightness. Fast Non-Local Means denoising is then applied
**only inside that mask**; pixels outside it are left exactly as they were.

### 7. Reconstruction
The enhanced, denoised V is recombined with the **original** H and S, and the HSV image is converted back to RGB.

## Implementation status

| Stage | Module | Status |
|---|---|---|
| Input validation, RGB/HSV conversion, channel split | `src/preprocessing.py` | Implemented and tested |
| Mean V, adaptive alpha/beta, Sobel terms | `src/retinex_utils.py` | Implemented and tested |
| Iterative Retinex decomposition | `src/retinex.py` | Implemented and tested |
| Loading and the first pipeline stages | `src/pipeline.py` | Partial: runs preprocessing, mean V and alpha/beta |
| Adaptive gamma correction + CLAHE | `src/enhancement.py` | Not implemented yet |
| Dark-region mask + selective Fast NLM | `src/denoising.py` | Not implemented yet |
| HSV -> RGB reconstruction | `src/reconstruction.py` | Not implemented yet |
| NIQE, AB, DE, runtime | `src/metrics.py` | Not implemented yet |

Modules marked "Not implemented yet" raise `NotImplementedError` and have placeholder tests that only check
they import. Running the command-line tool therefore currently stops after the stages marked as implemented.

## Repository structure

```
.
├── configs/
│   └── default.yaml          # Parameters for the pipeline stages
├── data/
│   ├── input/                # Place input images here (git-ignored)
│   └── output/               # Generated images (git-ignored)
├── results/
│   ├── qualitative/          # Visual comparisons
│   └── quantitative/         # Metric tables
├── scripts/
│   └── run_pipeline.py       # Command-line entry point
├── src/
│   ├── preprocessing.py      # Validation, BGR -> RGB, float32 [0, 1], HSV split
│   ├── retinex_utils.py      # Mean V, adaptive alpha/beta, Sobel gradient terms
│   ├── retinex.py            # Iterative Retinex decomposition
│   ├── enhancement.py        # Adaptive gamma correction + CLAHE
│   ├── denoising.py          # Dark-region mask + selective Fast NLM
│   ├── reconstruction.py     # Recombination with H and S, HSV -> RGB
│   ├── metrics.py            # NIQE, AB, DE, runtime
│   └── pipeline.py           # Orchestration of the stages
├── tests/                    # pytest suite, one file per module
├── requirements.txt
└── README.md
```

## Data conventions

| Data | Type | Shape | Range |
|---|---|---|---|
| RGB working image | `float32` | `(H, W, 3)` | [0, 1] |
| H (hue) | `float32` | `(H, W)` | degrees, [0, 360) (OpenCV float HSV) |
| S, V | `float32` | `(H, W)` | [0, 1] |
| Illumination T, reflectance R | `float32` | `(H, W)` | [0, 1] |
| Dark-region mask | `bool` | `(H, W)` | — |

Hue is deliberately kept in degrees so that `cv2.cvtColor(..., cv2.COLOR_HSV2RGB)` round-trips without any
rescaling. No stage modifies its input arrays in place.

## Installation

Requires Python 3 (developed and tested with Python 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies (`requirements.txt`): NumPy, OpenCV (`opencv-python`), scikit-image, PyYAML, pytest.
scikit-image is listed for the evaluation stage and is not used by the current code.

## Usage

### Command line

```bash
python scripts/run_pipeline.py --input data/input/example.png
```

The tool loads the image, runs the implemented stages, logs progress and prints the mean V together with the
adaptive `alpha` and `beta`. It exits with a non-zero code and a clear message if the file is missing or cannot be
decoded. The later stages (enhancement, denoising, reconstruction) are not wired into this command yet.

### From Python

```python
import cv2
from src.preprocessing import preprocess_image
from src.retinex_utils import calculate_mean_v, initialize_retinex_parameters
from src.retinex import retinex_decompose_with_info

image_bgr = cv2.imread("data/input/example.png")          # uint8 BGR
pre = preprocess_image(image_bgr)                         # pre.rgb, pre.h, pre.s, pre.v

mean_v = calculate_mean_v(pre.v)
alpha, beta = initialize_retinex_parameters(mean_v)

illumination, reflectance, info = retinex_decompose_with_info(pre.v, alpha, beta)
print(info)   # {"iterations": ..., "converged": ..., "final_delta": ...}
```

`retinex_decompose(v, alpha, beta, ...)` returns just `(illumination, reflectance)`.

## Configuration

`configs/default.yaml` holds the parameters for each stage. Values that are still `null` belong to stages that
are not implemented yet.

| Section | Key | Value | Meaning |
|---|---|---|---|
| `retinex` | `epsilon` | `0.001` | Constant that keeps every denominator positive |
| `retinex` | `max_iterations` | `50` | Iteration cap |
| `retinex` | `tolerance` | `0.0001` | Stop when the largest change in T and R is below this |
| `pipeline` | `save_intermediates` | `false` | Whether to save intermediate images |
| `pipeline` | `output_dir` / `output_format` | `data/output` / `png` | Where and how outputs are written |

The report does not give values for `epsilon`, `max_iterations` or `tolerance`; the values above are
implementation choices. `alpha` and `beta` are not configured here: they are computed from each image's mean V.
The functions take all parameters as explicit arguments, and their defaults match this file.

## Testing

```bash
pytest -q
```

The suite covers input validation, value ranges and dtypes, determinism, non-mutation of inputs, special inputs
(all-zero, constant, very dark, edge images) and, for the Retinex module, a comparison of the first iterations against an
independent hand-written implementation of the update equations.

## Evaluation

Planned evaluation on the public low-light datasets **LIME**, **NPE** and **MEF** (not included in this
repository; place them under `data/input/`). Metrics:

| Metric | Meaning | Interpretation |
|---|---|---|
| NIQE | Natural image quality evaluator | Lower is better |
| AB | Average brightness (mean 8-bit grayscale intensity, 0–255) | Should be appropriate, not simply maximised |
| DE | Discrete entropy, `-sum p(i) log2 p(i)` over gray levels | Higher can mean richer detail, but noise also raises it |
| Runtime | Processing time per image | Lower is better |

AB and DE use the standard definitions; they are not equations taken from the source paper. Results are
written to `results/quantitative/` once the evaluation stage exists, and are only ever produced by actually
running the code.

## Notes on the Retinex implementation

The report specifies the update equations but not the starting estimates, the choice of image for the
Sobel terms, the stopping rule, or numerical values for the constants. The implementation chooses:

- **Initialisation:** `T(0) = V` and `R(0) = 1`, so that `R(0) * T(0) = V`.
- **Sobel terms:** computed once from V, since V does not change during the iteration.
- **Stopping rule:** `max(max|T(k+1) - T(k)|, max|R(k+1) - R(k)|) < tolerance`, or the iteration cap.
- **Numerical safety:** with V in [0, 1], every denominator is at least `epsilon`, and both `T` and `R` stay in
  [0, V] without any clipping.

With the equations as specified, both estimates settle very close to V after only a few iterations on typical
inputs, so the product `R * T` is close to `V²` rather than `V`. This follows directly from the specified updates and has
not been altered.

## References

1. D. Gupta, Naveen, and S. D. Pawar, "Enhancing Low-Light Images Using Retinex with Adaptive Gamma and Dark
   Region Denoising," 2025 IEEE International Conference on Computer Vision and Machine Intelligence (CVMI), 2025,
   doi: 10.1109/CVMI66673.2025.11337939.
2. E. H. Land, "The retinex theory of color vision," Scientific American, vol. 237, no. 6, pp. 108–129, 1977.
3. D. J. Jobson, Z.-u. Rahman, and G. A. Woodell, "A multiscale retinex for bridging the gap between color images
   and the human observation of scenes," IEEE Transactions on Image Processing, vol. 6, no. 7, pp. 965–976, 1997.
4. J. Xu et al., "STAR: A structure and texture aware Retinex model," IEEE Transactions on Image Processing,
   vol. 29, pp. 5022–5037, 2020.
