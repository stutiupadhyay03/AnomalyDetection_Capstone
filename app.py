"""
Real-Time Anomaly Detection & Crowd Monitoring — Streamlit dashboard.

Entry point: `streamlit run app.py`

This file is intentionally a thin UI layer.  All model loading and inference
logic lives in the `src/` package so it can be tested and reused independently.
"""

# st.set_page_config must be the very first Streamlit call.
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Surveillance Dashboard",
    page_icon="📹",
)

# ---------------------------------------------------------------------------
# Environment & logging — configure before heavy imports
# ---------------------------------------------------------------------------
import logging
import os
import warnings

from src.config import THREAD_ENV_VARS
from src.utils import setup_logging

for _k, _v in THREAD_ENV_VARS.items():
    os.environ.setdefault(_k, _v)

warnings.filterwarnings("ignore")
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application imports
# ---------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from src.inference import (
        annotate_video,
        get_alert_level,
        infer_crowd_image,
        iter_crowd_video_frames,
        load_models,
    )
    from src.utils import temp_video_file, validate_upload
except Exception as _import_exc:
    st.error(f"**Startup import failed:** {_import_exc}")
    logger.exception("Import-time failure")
    st.stop()


def safe_file_uploader(label: str, type: list, key: str, **kwargs):
    """Wrapper around st.file_uploader that clears stale session state on type mismatch."""
    try:
        return st.file_uploader(label, type=type, key=key, **kwargs)
    except st.errors.StreamlitAPIException:
        st.session_state.pop(key, None)
        st.rerun()

matplotlib.use("Agg")   # headless backend — safe for cloud deployments

# ---------------------------------------------------------------------------
# Load models (cached across reruns)
# ---------------------------------------------------------------------------
try:
    csrnet_model, anomaly_model, yolo_model = load_models()
    _models_ready = True
except FileNotFoundError as exc:
    st.error(f"**Model files missing.** {exc}")
    logger.error("Model loading failed: %s", exc)
    _models_ready = False
except Exception as exc:
    st.error(f"**Unexpected error loading models:** {exc}")
    logger.exception("Unexpected model loading error")
    _models_ready = False

# ---------------------------------------------------------------------------
# Shared styles
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] { background-color: #e7f5ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Real-Time Anomaly Detection & Crowd Monitoring")
st.markdown(
    "Powered by **CSRNet** (crowd density) and **ResNet-34** (anomaly classification). "
    "Upload a video or image to analyse instantly."
)

if not _models_ready:
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Select a Module")
    selected_tab = st.radio(
        "Choose a tab:",
        ["ShanghaiTech Crowd Monitoring", "Avenue Anomaly Detection"],
    )

# ===========================================================================
# Module 1 — ShanghaiTech Crowd Monitoring
# ===========================================================================
if selected_tab == "ShanghaiTech Crowd Monitoring":
    st.subheader("Crowd Monitoring")
    mode = st.radio("Select Input Type", ["Image", "Video"], key="shang_mode")

    # -----------------------------------------------------------------------
    # Image mode
    # -----------------------------------------------------------------------
    if mode == "Image":
        image_file = safe_file_uploader(
            "Upload a crowd image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            key="shang_image",
        )

        if image_file is not None:
            if not validate_upload(image_file):
                st.stop()

            try:
                img = Image.open(image_file).convert("RGB")
            except Exception as exc:
                st.error(f"Cannot read image: {exc}")
                st.stop()

            st.image(img, caption="Uploaded image", use_container_width=True)

            with st.spinner("Running crowd density estimation…"):
                try:
                    density_map, pred_count = infer_crowd_image(
                        img, csrnet_model, yolo_model
                    )
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")
                    logger.exception("CSRNet inference error")
                    st.stop()

            alert_text, alert_color = get_alert_level(pred_count)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"### Alert: "
                    f"<span style='color:{alert_color}'>{alert_text}</span>",
                    unsafe_allow_html=True,
                )
                st.image(
                    img,
                    caption=f"Estimated crowd count: {int(pred_count)}",
                    use_container_width=True,
                )
            with col2:
                fig, ax = plt.subplots()
                ax.imshow(density_map, cmap="jet")
                ax.set_title("Density Map")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

    # -----------------------------------------------------------------------
    # Video mode
    # -----------------------------------------------------------------------
    elif mode == "Video":
        video_file = safe_file_uploader(
            "Upload a video (MP4 / AVI)",
            type=["mp4", "avi"],
            key="shang_video",
        )

        if video_file is not None:
            if not validate_upload(video_file):
                st.stop()

            frame_placeholder = st.empty()
            progress = st.progress(0.0, text="Processing frames…")

            try:
                with temp_video_file(video_file, suffix=".mp4") as tmp_path:
                    frames = list(iter_crowd_video_frames(
                        tmp_path, csrnet_model, yolo_model
                    ))

                total = len(frames)
                for idx, (rgb, density_map, count, alert_text, alert_color) in enumerate(frames):
                    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

                    axs[0].imshow(rgb)
                    axs[0].set_title(f"Alert: {alert_text}", color=alert_color, fontsize=11)
                    axs[0].axis("off")

                    axs[1].imshow(density_map, cmap="jet")
                    axs[1].set_title(f"Density Map  |  Count: {int(count)}", fontsize=11)
                    axs[1].axis("off")

                    frame_placeholder.pyplot(fig)
                    plt.close(fig)
                    progress.progress((idx + 1) / total)

                progress.empty()

            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
                logger.exception("Crowd video processing error")

# ===========================================================================
# Module 2 — Avenue Anomaly Detection
# ===========================================================================
elif selected_tab == "Avenue Anomaly Detection":
    st.subheader("Anomaly Detection")

    video_file = safe_file_uploader(
        "Upload a surveillance video (MP4 / AVI)",
        type=["mp4", "avi"],
        key="avenue_video",
    )

    if video_file is not None:
        if not validate_upload(video_file):
            st.stop()

        st.video(video_file)
        st.write("Annotating video — this may take a moment…")

        progress = st.progress(0.0, text="Classifying frames…")
        annotated_path = None

        try:
            with temp_video_file(video_file, suffix=".mp4") as tmp_path:
                annotated_path = annotate_video(
                    tmp_path,
                    anomaly_model,
                    progress_bar=progress,
                )
        except Exception as exc:
            st.error(f"Annotation failed: {exc}")
            logger.exception("Avenue annotation error")

        if annotated_path and annotated_path.exists():
            progress.empty()
            st.success("Annotated video ready.")

            with open(annotated_path, "rb") as fh:
                st.download_button(
                    label="Download Annotated Video",
                    data=fh,
                    file_name="anomaly_labeled_avenue.mp4",
                    mime="video/mp4",
                )

            st.video(str(annotated_path))
            # Clean up output file after it has been served
            try:
                os.unlink(annotated_path)
            except OSError:
                logger.warning("Could not delete output file: %s", annotated_path)
