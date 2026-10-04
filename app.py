"""Streamlit web UI for the low-light image enhancement pipeline.

Run with:
    streamlit run app.py

The UI contains no image-processing code. It decodes the uploaded file, calls the same
``src.pipeline.process_image`` used by the command line, and displays what that function returns.
"""
from pathlib import Path

import streamlit as st

from src.metrics import evaluate_image
from src.pipeline import (
    ConfigError,
    bgr_to_display_rgb,
    decode_image,
    encode_png,
    load_config,
    process_image,
    rgb_float_to_bgr_uint8,
    rgb_float_to_uint8,
)

UPLOAD_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]


@st.cache_resource
def get_config() -> dict:
    """Load configs/default.yaml once per server session."""
    return load_config()


def display_gray(image):
    """Float [0, 1] 2-D array (or boolean mask) -> uint8 for display."""
    return (image.astype("uint8") * 255) if image.dtype == bool else (image.clip(0, 1) * 255).round().astype("uint8")


def show_processing_information(result) -> None:
    info, timings = result.info, result.timings
    st.subheader("Processing information")
    left, middle, right = st.columns(3)
    left.metric("Mean V (input)", f"{info['mean_v']:.4f}")
    middle.metric("Alpha", f"{info['alpha']:.6f}")
    right.metric("Beta", f"{info['beta']:.6f}")
    left, middle, right = st.columns(3)
    left.metric("Retinex iterations", f"{info['retinex_iterations']}")
    middle.metric("Adaptive gamma", f"{info['gamma']:.4f}")
    right.metric("Dark-region coverage", f"{info['dark_region_fraction'] * 100:.1f}%")
    left, middle, right = st.columns(3)
    left.metric("Mean V (enhanced)", f"{info['mean_v_enhanced']:.4f}")
    middle.metric("Processing time", f"{timings['total']:.3f} s")
    right.metric("Retinex converged", "yes" if info["retinex_converged"] else "no (iteration limit)")


def show_metrics(input_bgr, result) -> None:
    """NIQE / AB / DE for the input and the enhanced image, shown as plain numbers."""
    before = evaluate_image(input_bgr)
    after = evaluate_image(rgb_float_to_bgr_uint8(result.enhanced_rgb), runtime_s=result.timings["total"])

    def fmt(value, digits=3):
        return "n/a (image too small or uniform)" if value is None else f"{value:.{digits}f}"

    st.table({
        "Metric": ["NIQE (lower is better)", "Average brightness (0-255)", "Discrete entropy (bits)"],
        "Input": [fmt(before["niqe"]), fmt(before["ab"], 1), fmt(before["de"])],
        "Enhanced": [fmt(after["niqe"]), fmt(after["ab"], 1), fmt(after["de"])],
    })
    st.caption(
        "These are raw measurements, not a quality score: average brightness should be appropriate rather "
        "than maximal, and entropy can also rise when noise is amplified."
    )


def show_intermediates(result) -> None:
    names = {
        "v_channel": "V channel (input)",
        "illumination": "Illumination T",
        "reflectance": "Reflectance R",
        "gamma_corrected": "After adaptive gamma",
        "clahe_enhanced": "After CLAHE (enhanced V)",
        "dark_mask": "Dark-region mask",
        "denoised_v": "Denoised V",
    }
    columns = st.columns(4)
    for index, (key, title) in enumerate(names.items()):
        columns[index % 4].image(display_gray(result.intermediates[key]), caption=title)


def main() -> None:
    st.set_page_config(page_title="Low-Light Image Enhancement", layout="wide")
    st.title("Low-Light Image Enhancement")
    st.caption("Retinex decomposition with adaptive gamma correction, CLAHE and dark-region denoising.")

    try:
        config = get_config()
    except ConfigError as exc:
        st.error(f"Configuration problem: {exc}")
        return

    uploaded = st.file_uploader("Upload a low-light image", type=UPLOAD_TYPES)
    if uploaded is None:
        st.info("Choose an image to begin.")
        return

    data = uploaded.getvalue()
    try:
        input_bgr = decode_image(data)
    except ValueError as exc:
        st.error(f"Could not read this file: {exc}")
        return

    # Forget a previous result when a different file is uploaded.
    upload_key = (uploaded.name, len(data))
    if st.session_state.get("upload_key") != upload_key:
        st.session_state["upload_key"] = upload_key
        st.session_state.pop("result", None)

    show_metrics_box = st.checkbox("Also compute quality metrics (NIQE, AB, DE)")
    show_steps_box = st.checkbox("Show intermediate images")
    if st.button("Enhance image", type="primary"):
        try:
            with st.spinner("Enhancing..."):
                st.session_state["result"] = process_image(input_bgr, config)
        except (ValueError, TypeError) as exc:
            st.error(f"Processing failed: {exc}")
            return

    original_column, enhanced_column = st.columns(2)
    with original_column:
        st.subheader("Original image")
        st.image(bgr_to_display_rgb(input_bgr), output_format="PNG")
    result = st.session_state.get("result")
    with enhanced_column:
        st.subheader("Enhanced image")
        if result is None:
            st.write("Press **Enhance image** to process the upload.")
        else:
            st.image(rgb_float_to_uint8(result.enhanced_rgb), output_format="PNG")

    if result is not None:
        st.download_button(
            "Download enhanced image",
            data=encode_png(result.enhanced_rgb),
            file_name=f"{Path(uploaded.name).stem}_enhanced.png",
            mime="image/png",
        )
        show_processing_information(result)
        if show_metrics_box:
            show_metrics(input_bgr, result)
        if show_steps_box:
            st.subheader("Intermediate images")
            show_intermediates(result)


main()
