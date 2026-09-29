````markdown
# Low-Light Image Enhancement Using Retinex with Adaptive Gamma and Dark Region Denoising

**CS23045 – Image Processing**

A classical, interpretable low-light image enhancement pipeline working primarily on the **HSV Value (V) channel** to improve brightness, local contrast, visibility, and image quality while preserving color information and reducing noise in dark regions.

## Overview

Low-light images often suffer from poor visibility, reduced dynamic range, loss of structural detail, color distortion, and amplified sensor noise.

This project implements a classical image-processing approach based on **Retinex theory**, which models an observed image as the product of illumination and reflectance components.

The pipeline performs adaptive Retinex decomposition using iterative least-squares optimization and Sobel-based gradient information. The estimated illumination is then enhanced using adaptive gamma correction and CLAHE. Finally, noise is selectively suppressed in dark regions using Fast Non-Local Means denoising before reconstructing the enhanced RGB image.

## Pipeline

```text
Low-light RGB Image
        ↓
Initial Denoising Hook
        ↓
RGB → HSV
        ↓
H, S preserved     V extracted
                       ↓
                 Mean V Calculation
                       ↓
              Adaptive α / β Initialization
                       ↓
             Iterative Retinex Decomposition
                       │
                       └── Sobel G / D Terms
                       ↓
                  Convergence Check
                       ↓
             Adaptive Gamma Correction
                       ↓
                       CLAHE
                       ↓
              Dark-Region Detection
                       ↓
        Selective Fast Non-Local Means
                       ↓
        Enhanced V + Original H, S
                       ↓
                    HSV → RGB
                       ↓
              Enhanced RGB Image
                       ↓
          NIQE / AB / DE / Runtime
````

## Methodology

### 1. Input Preprocessing

The input image is validated and prepared for processing.

The image is converted from BGR to RGB and represented as `float32` in the range `[0, 1]`.

An initial denoising stage is provided as a hook for preprocessing.

### 2. RGB to HSV Conversion

The RGB image is converted into the HSV color space:

```text
H = Hue
S = Saturation
V = Value / Brightness
```

The original **H and S channels are preserved**, while the **V channel** is used for low-light enhancement.

### 3. Adaptive Retinex Decomposition

The V channel is decomposed using the Retinex model:

```text
V = R · T
```

where:

```text
R = Reflectance
T = Illumination
```

The adaptive regularization parameters are initialized using the mean intensity of the V channel:

```text
α = clip(0.001 + 0.002 · (0.5 − μV), 0.0001, 0.003)

β = clip(0.0001 + 0.0003 · (0.5 − μV), 0.0001, 0.0005)
```

The decomposition is refined iteratively using Sobel-derived gradient information to help preserve important edges and structural details.

### 4. Adaptive Gamma Correction

After Retinex decomposition converges, the estimated illumination is enhanced using an adaptive gamma value based on its mean intensity.

The enhancement strength is adjusted according to the brightness of the input rather than using a single fixed correction value.

### 5. CLAHE

**Contrast Limited Adaptive Histogram Equalization (CLAHE)** is applied after gamma correction.

CLAHE improves local contrast while limiting excessive contrast amplification in individual regions.

### 6. Targeted Dark-Region Denoising

Low-light enhancement can amplify noise, particularly in regions that were originally very dark.

A dark-region mask is generated and **Fast Non-Local Means (NLM)** denoising is applied selectively to those regions.

Brighter regions are preserved to avoid unnecessary smoothing of textures and edges.

### 7. Color Reconstruction

The enhanced V channel is recombined with the original H and S channels:

```text
Original H
     +
Original S
     +
Enhanced V
     ↓
    HSV
     ↓
    RGB
```

The result is the final enhanced RGB image.

## Data Conventions

```text
RGB image         : float32, shape (H, W, 3), range [0, 1]

H channel         : float32, shape (H, W), range [0, 360)
S channel         : float32, shape (H, W), range [0, 1]
V channel         : float32, shape (H, W), range [0, 1]

Illumination      : float32, shape (H, W)
Reflectance       : float32, shape (H, W)

Dark-region mask  : bool, shape (H, W)
```

## Project Structure

```text
low-light-retinex/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── default.yaml
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── retinex_utils.py
│   ├── retinex.py
│   ├── enhancement.py
│   ├── denoising.py
│   ├── reconstruction.py
│   ├── metrics.py
│   └── pipeline.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_retinex_utils.py
│   ├── test_retinex.py
│   ├── test_enhancement.py
│   ├── test_denoising.py
│   ├── test_reconstruction.py
│   ├── test_metrics.py
│   └── test_pipeline.py
│
├── scripts/
│   └── run_pipeline.py
│
├── data/
│   ├── input/
│   └── output/
│
└── results/
    ├── qualitative/
    └── quantitative/
```

## Setup

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Environment

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running Tests

Run the complete test suite:

```bash
pytest -q
```

## Running the Pipeline

```bash
python scripts/run_pipeline.py --input data/input/example.png
```

## Expected Outputs

The complete pipeline should produce the following intermediate and final outputs:

```text
Original Low-Light Image
        ↓
Preprocessed Image
        ↓
V Channel
        ↓
Estimated Illumination
        ↓
Estimated Reflectance
        ↓
Gamma-Enhanced Image
        ↓
CLAHE-Enhanced Image
        ↓
Dark-Region Mask
        ↓
Denoised Enhanced Image
        ↓
Final Enhanced RGB Image
```

The system should also produce quantitative evaluation results:

```text
NIQE
Average Brightness (AB)
Discrete Entropy (DE)
Runtime
```

Intermediate outputs can be stored for visualization, debugging, and qualitative comparison.

## Performance Evaluation

### NIQE

**Natural Image Quality Evaluator (NIQE)** is a no-reference image-quality metric based on natural image statistics.

A lower NIQE value generally indicates better naturalness and perceptual quality.

### Average Brightness (AB)

Average Brightness measures the mean intensity of the image.

The objective is to obtain an appropriately bright image without excessive over-brightening.

### Discrete Entropy (DE)

Discrete Entropy measures the distribution of intensity values and provides an indication of information variation and image detail.

Higher entropy can indicate richer information, but increased noise can also raise entropy, so DE should be interpreted together with other quality measures.

### Runtime

Runtime is measured to evaluate the computational efficiency of the enhancement pipeline.

## Current Status

The project is implemented as a modular image-processing pipeline.

The preprocessing and Retinex utility stages provide image validation, RGB preparation, HSV conversion, V-channel extraction, adaptive parameter initialization, and Sobel-based gradient utilities.

The complete pipeline is being developed incrementally, with Retinex decomposition, adaptive enhancement, targeted denoising, reconstruction, and quantitative evaluation integrated as separate processing stages.

## References

The methodology is based on classical Retinex theory and the selected low-light image enhancement literature used for this project.

```
```
