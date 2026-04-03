"""
Centralized configuration for the Anomaly Detection & Crowd Monitoring System.
All tuneable parameters and paths are defined here. Override via environment variables.
"""

import os
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "notebooks")))

CSRNET_MODEL_PATH = Path(
    os.getenv("CSRNET_MODEL_PATH", str(MODEL_DIR / "csrnet_shanghai.pt"))
)
ANOMALY_MODEL_PATH = Path(
    os.getenv("ANOMALY_MODEL_PATH", str(MODEL_DIR / "final_op_avenue_model.pt"))
)
YOLO_MODEL_PATH = Path(
    os.getenv("YOLO_MODEL_PATH", str(MODEL_DIR / "yolov8m.pt"))
)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
# Maximum frames to process per video (prevents OOM on long videos)
MAX_VIDEO_FRAMES: int = int(os.getenv("MAX_VIDEO_FRAMES", "150"))

# Minimum softmax confidence to display a label; below this shows "uncertain"
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))

# CSRNet crowd count below this triggers YOLOv8 fallback
CSRNET_FALLBACK_THRESHOLD: float = float(os.getenv("CSRNET_FALLBACK_THRESHOLD", "1.0"))

# Maximum upload size in MB (enforced in UI layer)
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))

# ---------------------------------------------------------------------------
# Crowd alert thresholds  (upper_bound_exclusive, label, hex_color)
# ---------------------------------------------------------------------------
ALERT_THRESHOLDS: List[Tuple[float, str, str]] = [
    (5.0,        "Normal",                 "#4dabf7"),
    (10.0,       "Can lead to overcrowding", "#228be6"),
    (20.0,       "Possible overcrowding",  "#1864ab"),
    (float("inf"), "Overcrowding",         "#103d60"),
]

# ---------------------------------------------------------------------------
# Class mappings for anomaly detection
# ---------------------------------------------------------------------------
IDX2LABEL: dict[int, str] = {
    0: "normal",
    1: "unusual action",
    2: "abnormal object",
}

# BGR colors used when rendering OpenCV annotations
LABEL_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "normal":          (50,  205,  50),   # green
    "unusual action":  (0,   165, 255),   # orange
    "abnormal object": (0,    0,  220),   # red
}

UNCERTAIN_LABEL = "uncertain"
UNCERTAIN_COLOR_BGR = (160, 160, 160)   # grey

# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]

ANOMALY_INPUT_SIZE: Tuple[int, int] = (224, 224)

# ---------------------------------------------------------------------------
# Threading (must be set before numpy/torch are imported)
# ---------------------------------------------------------------------------
THREAD_ENV_VARS = {
    "OMP_NUM_THREADS":         "1",
    "OPENBLAS_NUM_THREADS":    "1",
    "MKL_NUM_THREADS":         "1",
    "VECLIB_MAXIMUM_THREADS":  "1",
    "NUMEXPR_NUM_THREADS":     "1",
}
