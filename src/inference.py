"""
Model loading and inference logic.

All heavy computation lives here so app.py stays a thin UI layer.
Heavy third-party libraries (torch, cv2, ultralytics) are imported lazily
inside functions so the app can start without loading the full ML stack.
"""

import logging
from pathlib import Path
from typing import Any, Tuple

import streamlit as st

from src.config import (
    ALERT_THRESHOLDS,
    ANOMALY_INPUT_SIZE,
    ANOMALY_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    CSRNET_FALLBACK_THRESHOLD,
    CSRNET_MODEL_PATH,
    IDX2LABEL,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_COLORS_BGR,
    MAX_VIDEO_FRAMES,
    UNCERTAIN_COLOR_BGR,
    UNCERTAIN_LABEL,
    YOLO_MODEL_PATH,
)
from src.utils import unique_output_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading (cached so Streamlit reruns don't reload weights each time)
# ---------------------------------------------------------------------------

def _load_state_dict_safe(path: Path, map_location="cpu"):
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        logger.warning("weights_only=True unavailable; falling back for %s", path.name)
        return torch.load(path, map_location=map_location)


def _validate_model_paths() -> None:
    missing = [
        p for p in (CSRNET_MODEL_PATH, ANOMALY_MODEL_PATH, YOLO_MODEL_PATH)
        if not p.exists()
    ]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Model file(s) not found: {names}. "
            f"Place them in '{CSRNET_MODEL_PATH.parent}' or set MODEL_DIR."
        )


@st.cache_resource(show_spinner="Loading models…")
def load_models() -> Tuple[Any, Any, Any]:
    """Load all three models once and cache them for the session."""
    # Validate paths first — avoids importing torch/ultralytics (~400 MB) when
    # model files are absent (e.g. on Streamlit Cloud without LFS / model storage).
    _validate_model_paths()

    import torch
    from ultralytics import YOLO
    from src.models import AnomalyClassifier, CSRNet

    csrnet = CSRNet()
    csrnet.load_state_dict(_load_state_dict_safe(CSRNET_MODEL_PATH))
    csrnet.eval()
    logger.info("CSRNet loaded from %s", CSRNET_MODEL_PATH)

    anomaly_clf = AnomalyClassifier(num_classes=3)
    anomaly_clf.load_state_dict(_load_state_dict_safe(ANOMALY_MODEL_PATH))
    anomaly_clf.eval()
    logger.info("AnomalyClassifier loaded from %s", ANOMALY_MODEL_PATH)

    yolo = YOLO(str(YOLO_MODEL_PATH))
    logger.info("YOLOv8 loaded from %s", YOLO_MODEL_PATH)

    return csrnet, anomaly_clf, yolo


# ---------------------------------------------------------------------------
# Crowd density (CSRNet)
# ---------------------------------------------------------------------------

def get_alert_level(count: float) -> Tuple[str, str]:
    """Map a crowd count to (alert_label, hex_color)."""
    for threshold, label, color in ALERT_THRESHOLDS:
        if count < threshold:
            return label, color
    return ALERT_THRESHOLDS[-1][1], ALERT_THRESHOLDS[-1][2]


def infer_crowd_image(img: Any, csrnet: Any, yolo: Any) -> Tuple[Any, float]:
    """
    Run CSRNet crowd density inference on a PIL image.

    Falls back to YOLOv8 person-count when CSRNet output is below threshold.
    """
    import numpy as np
    import torch
    from torchvision import transforms

    csr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor = csr_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = csrnet(tensor)
    density_map: Any = output.squeeze().cpu().numpy()
    count = float(max(0.0, density_map.sum()))

    if count < CSRNET_FALLBACK_THRESHOLD:
        count = _yolo_person_count(np.array(img), yolo)
        logger.debug("CSRNet count below threshold; YOLO fallback count=%d", count)

    return density_map, count


def _yolo_person_count(img_rgb: Any, yolo: Any) -> int:
    """Count detected persons (class 0) in an RGB image using YOLOv8."""
    results = yolo(img_rgb, verbose=False)
    classes = results[0].boxes.cls.tolist()
    return int(classes.count(0))


# ---------------------------------------------------------------------------
# Anomaly detection (ResNet-34)
# ---------------------------------------------------------------------------

def _classify_frame(frame_bgr: Any, anomaly_clf: Any) -> Tuple[str, float]:
    """
    Classify a single BGR frame.

    Returns:
        label: predicted class label (or UNCERTAIN_LABEL if below threshold).
        confidence: softmax confidence of the top prediction.
    """
    import cv2
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    anomaly_transform = transforms.Compose([
        transforms.Resize(ANOMALY_INPUT_SIZE),
        transforms.ToTensor(),
    ])

    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor = anomaly_transform(pil).unsqueeze(0)

    with torch.no_grad():
        logits = anomaly_clf(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return UNCERTAIN_LABEL, confidence

    return IDX2LABEL[pred_idx], confidence


def annotate_video(
    video_path: Path,
    anomaly_clf: Any,
    progress_bar=None,
) -> Path:
    """
    Annotate each frame of *video_path* with the predicted anomaly label.

    Returns:
        Path to the annotated output video (caller is responsible for cleanup).

    Raises:
        RuntimeError: if the video cannot be opened or the writer fails.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_count = min(total_frames, MAX_VIDEO_FRAMES)

    out_path = unique_output_path(".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to initialise VideoWriter.")

    try:
        for i in range(process_count):
            ret, frame = cap.read()
            if not ret:
                break

            label, confidence = _classify_frame(frame, anomaly_clf)
            text = f"{label} ({confidence:.2f})"
            color = LABEL_COLORS_BGR.get(label, UNCERTAIN_COLOR_BGR)

            cv2.putText(
                frame, text,
                org=(10, 35),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            writer.write(frame)

            if progress_bar is not None:
                progress_bar.progress((i + 1) / process_count)

        logger.info("Annotated %d/%d frames → %s", i + 1, process_count, out_path)
    finally:
        cap.release()
        writer.release()

    return out_path


# ---------------------------------------------------------------------------
# Crowd video streaming helper
# ---------------------------------------------------------------------------

def iter_crowd_video_frames(video_path: Path, csrnet: Any, yolo: Any):
    """
    Generator yielding (rgb_frame, density_map, count, alert_text, alert_color)
    for up to MAX_VIDEO_FRAMES frames of *video_path*.
    """
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_count = min(total, MAX_VIDEO_FRAMES)

    try:
        for _ in range(process_count):
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            density_map, count = infer_crowd_image(pil, csrnet, yolo)
            alert_text, alert_color = get_alert_level(count)

            yield rgb, density_map, count, alert_text, alert_color
    finally:
        cap.release()
