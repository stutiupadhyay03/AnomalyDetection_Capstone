"""
Utility helpers: logging setup, file validation, temp-file management.
"""

import logging
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import streamlit as st

from src.config import MAX_UPLOAD_SIZE_MB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Temp file helpers
# ---------------------------------------------------------------------------

@contextmanager
def temp_video_file(uploaded_file, suffix: str = ".mp4") -> Generator[Path, None, None]:
    """
    Write an uploaded Streamlit file to a temporary path, then clean up on exit.

    Usage::

        with temp_video_file(uploaded_file) as path:
            cap = cv2.VideoCapture(str(path))
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()
        yield Path(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            logger.warning("Could not delete temp file: %s", tmp.name)


def unique_output_path(suffix: str = ".mp4") -> Path:
    """Return a unique path in the system temp directory for output files."""
    return Path(tempfile.gettempdir()) / f"annotated_{uuid.uuid4().hex}{suffix}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_upload(uploaded_file, max_mb: Optional[int] = None) -> bool:
    """
    Return True if *uploaded_file* passes size validation.
    Display an st.error and return False otherwise.
    """
    limit = max_mb or MAX_UPLOAD_SIZE_MB
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > limit:
        st.error(
            f"File is {size_mb:.1f} MB — exceeds the {limit} MB limit. "
            "Please upload a smaller file."
        )
        logger.warning(
            "Rejected upload '%s': %.1f MB > %d MB limit",
            uploaded_file.name, size_mb, limit,
        )
        return False
    return True
