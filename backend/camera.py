"""Cross-platform webcam open/read helpers with graceful fallback."""
from __future__ import annotations

import logging
import platform
from typing import Optional

import cv2

logger = logging.getLogger(__name__)


def _candidate_backends() -> list:
    system = platform.system()
    if system == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if system == "Linux":
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_camera(index: int, width: int, height: int) -> Optional[cv2.VideoCapture]:
    """Try each platform-appropriate backend until one opens the camera.

    Returns None (never raises) if no backend can open it, so callers can
    surface a clean error instead of crashing.
    """
    for backend in _candidate_backends():
        cap = None
        try:
            cap = cv2.VideoCapture(int(index), backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                return cap
            cap.release()
        except cv2.error:
            logger.exception("Backend %s failed to open camera %s", backend, index)
            if cap is not None:
                cap.release()
    return None


def read_frame(cap: cv2.VideoCapture):
    """Read one frame, swallowing OpenCV errors as a failed read instead of a crash."""
    if cap is None:
        return False, None
    try:
        return cap.read()
    except cv2.error:
        logger.exception("cap.read() raised")
        return False, None


def release_camera(cap: Optional[cv2.VideoCapture]) -> None:
    if cap is None:
        return
    try:
        cap.release()
    except cv2.error:
        logger.exception("cap.release() raised")
