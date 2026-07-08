"""Background camera-capture + detection service.

The capture loop runs in its own thread, independent of any HTTP request, so
the frontend can call /stop, change settings, or set a reference at any time
without waiting on a blocking loop (the bug in the original single-process
Streamlit app, where the Stop button could never be read until the capture
`while` loop happened to exit).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np

from camera import open_camera, read_frame, release_camera
from detection import DetectionSettings, find_blue_hex, px_to_cm_estimate

logger = logging.getLogger(__name__)

SAMPLE_PERIOD = 5.0  # seconds between history samples
MAX_HISTORY = 500
MAX_CONSECUTIVE_READ_FAILURES = 20


@dataclass
class CameraSettings:
    index: int = 0
    width: int = 640
    height: int = 480
    processing_fps: float = 10.0
    known_side_cm: float = 29.0


@dataclass
class HistoryRecord:
    t: float
    dx_cm: Optional[float]
    dy_cm: Optional[float]
    distance_cm: Optional[float]
    angle_drift_deg: Optional[float]


class TrackerService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.detection_settings = DetectionSettings()
        self.camera_settings = CameraSettings()

        self._ref_point: Optional[tuple] = None
        self._ref_angle: Optional[float] = None
        self._last_sample_time: Optional[float] = None
        self._history: Deque[HistoryRecord] = deque(maxlen=MAX_HISTORY)

        self._running = False
        self._error: Optional[str] = None
        self._latest_jpeg: Optional[bytes] = None
        self._frame_ready = threading.Event()

        self._status: Dict = {
            "detected": False,
            "centroid": None,
            "angle_deg": None,
            "dx_cm": None,
            "dy_cm": None,
            "distance_cm": None,
            "angle_drift_deg": None,
        }

    # ---------------------------------------------------------------- public

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._error = None
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._running = True
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        with self._lock:
            self._running = False
            self._thread = None

    def update_detection_settings(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if v is not None and hasattr(self.detection_settings, k):
                    setattr(self.detection_settings, k, v)

    def update_camera_settings(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if v is not None and hasattr(self.camera_settings, k):
                    setattr(self.camera_settings, k, v)

    def set_reference(self) -> bool:
        with self._lock:
            centroid = self._status.get("centroid")
            angle = self._status.get("angle_deg")
            if centroid is None or angle is None:
                return False
            self._ref_point = tuple(centroid)
            self._ref_angle = angle
            self._last_sample_time = None
            return True

    def clear_reference(self) -> None:
        with self._lock:
            self._ref_point = None
            self._ref_angle = None
            self._last_sample_time = None
            self._history.clear()

    def get_status(self) -> Dict:
        with self._lock:
            return {
                "running": self._running,
                "error": self._error,
                "ref_set": self._ref_point is not None,
                "ref_point": self._ref_point,
                "ref_angle": self._ref_angle,
                **self._status,
            }

    def get_history(self) -> List[Dict]:
        with self._lock:
            return [asdict(r) for r in self._history]

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def wait_for_frame(self, timeout: float = 2.0) -> Optional[bytes]:
        if self._frame_ready.wait(timeout=timeout):
            return self.get_latest_jpeg()
        return None

    def test_camera(self, index: int, width: int, height: int):
        """Open the camera, grab one frame, and release it. Returns (ok, jpeg_bytes_or_None, error_or_None)."""
        cap = open_camera(index, width, height)
        if cap is None:
            return False, None, f"Could not open camera index {index}."
        try:
            ok, frame = read_frame(cap)
            if not ok or frame is None:
                return False, None, "Camera opened but did not return a frame."
            success, buf = cv2.imencode(".jpg", frame)
            if not success:
                return False, None, "Failed to encode test frame."
            return True, buf.tobytes(), None
        finally:
            release_camera(cap)

    # --------------------------------------------------------------- internal

    def _run_loop(self) -> None:
        with self._lock:
            cam_settings = CameraSettings(**asdict(self.camera_settings))

        cap = open_camera(cam_settings.index, cam_settings.width, cam_settings.height)
        if cap is None:
            with self._lock:
                self._error = (
                    f"Could not open camera index {cam_settings.index}. "
                    "Try a different index or close other apps using the webcam."
                )
                self._running = False
            return

        consecutive_failures = 0
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    fps = max(0.1, self.camera_settings.processing_fps)
                frame_interval = 1.0 / fps
                loop_start = time.time()

                ok, frame = read_frame(cap)
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                        with self._lock:
                            self._error = "Repeated frame grab failures; stopped."
                        break
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0

                self._process_frame(frame)

                elapsed = time.time() - loop_start
                remaining = frame_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        except Exception:
            logger.exception("Unexpected error in capture loop")
            with self._lock:
                self._error = "Unexpected internal error; capture loop stopped. See server logs."
        finally:
            release_camera(cap)
            with self._lock:
                self._running = False

    def _process_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            det_settings = DetectionSettings(**asdict(self.detection_settings))
            known_side = self.camera_settings.known_side_cm
            ref_point = self._ref_point
            ref_angle = self._ref_angle

        try:
            result = find_blue_hex(frame, det_settings)
        except Exception:
            logger.exception("Detection failed for a frame; skipping annotation")
            result = None

        if result is None:
            annotated = frame
            centroid = None
            angle = None
            contour = None
        else:
            annotated = result.annotated
            centroid = result.centroid
            angle = result.angle_deg
            contour = result.contour

        dx_cm = dy_cm = distance_cm = angle_drift = None
        scale = px_to_cm_estimate(contour, known_side)

        if centroid is not None and ref_point is not None:
            rx, ry = ref_point
            dx = float(centroid[0] - rx)
            dy = float(centroid[1] - ry)
            dist = float(np.hypot(dx, dy))
            if ref_angle is not None and angle is not None:
                angle_drift = float(angle - ref_angle)
            if scale:
                dx_cm, dy_cm, distance_cm = dx * scale, dy * scale, dist * scale

            self._draw_reference_overlay(annotated, rx, ry, centroid, distance_cm, dist)

            ts = time.time()
            with self._lock:
                should_sample = distance_cm is not None and (
                    self._last_sample_time is None
                    or ts - self._last_sample_time >= SAMPLE_PERIOD
                )
                if should_sample:
                    self._history.append(HistoryRecord(ts, dx_cm, dy_cm, distance_cm, angle_drift))
                    self._last_sample_time = ts

        if centroid is not None:
            angle_text = f"Angle: {angle:.1f} deg" if angle is not None else "Angle: -"
            cv2.putText(annotated, angle_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        success, buf = cv2.imencode(".jpg", annotated)
        jpeg_bytes = buf.tobytes() if success else None

        with self._lock:
            self._status = {
                "detected": centroid is not None,
                "centroid": centroid,
                "angle_deg": angle,
                "dx_cm": dx_cm,
                "dy_cm": dy_cm,
                "distance_cm": distance_cm,
                "angle_drift_deg": angle_drift,
            }
            if jpeg_bytes is not None:
                self._latest_jpeg = jpeg_bytes
                self._frame_ready.set()

    @staticmethod
    def _draw_reference_overlay(annotated, rx, ry, centroid, distance_cm, dist_px) -> None:
        try:
            cv2.circle(annotated, (int(rx), int(ry)), 8, (0, 255, 255), -1)
            cv2.putText(annotated, "REF", (int(rx) + 10, int(ry)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.line(annotated, (int(rx), int(ry)), centroid, (0, 255, 255), 2)
            mid_point = ((int(rx) + centroid[0]) // 2, (int(ry) + centroid[1]) // 2)
            dist_text = f"{distance_cm:.1f} cm" if distance_cm is not None else f"{dist_px:.1f} px"
            cv2.putText(annotated, dist_text, (mid_point[0] + 10, mid_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if distance_cm is not None:
                label = f"dist={distance_cm:.1f} cm"
                cv2.putText(annotated, label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        except cv2.error:
            logger.exception("Failed to draw reference overlay")


tracker_service = TrackerService()
