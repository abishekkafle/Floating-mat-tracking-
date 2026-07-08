"""Pure image-processing helpers for the blue hex mat tracker.

No camera or UI state lives here so this module can be unit tested in isolation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionSettings:
    h_min: int = 90
    s_min: int = 80
    v_min: int = 80
    h_max: int = 140
    s_max: int = 255
    v_max: int = 255
    area_min: int = 8000
    poly_eps_pct: int = 3  # percent, matches original slider (1-10)
    morph_k: int = 5
    debug: bool = False

    def clamped(self) -> "DetectionSettings":
        """Return a copy with all fields coerced into their valid, non-crashing ranges."""
        h_min = int(np.clip(self.h_min, 0, 179))
        h_max = int(np.clip(self.h_max, 0, 179))
        s_min = int(np.clip(self.s_min, 0, 255))
        s_max = int(np.clip(self.s_max, 0, 255))
        v_min = int(np.clip(self.v_min, 0, 255))
        v_max = int(np.clip(self.v_max, 0, 255))
        # Auto-correct swapped min/max so a bad slider combo never yields an empty mask forever.
        if h_min > h_max:
            h_min, h_max = h_max, h_min
        if s_min > s_max:
            s_min, s_max = s_max, s_min
        if v_min > v_max:
            v_min, v_max = v_max, v_min
        return DetectionSettings(
            h_min=h_min, s_min=s_min, v_min=v_min,
            h_max=h_max, s_max=s_max, v_max=v_max,
            area_min=max(1, int(self.area_min)),
            poly_eps_pct=int(np.clip(self.poly_eps_pct, 1, 10)),
            morph_k=max(1, int(self.morph_k)),
            debug=bool(self.debug),
        )

    def lower(self) -> np.ndarray:
        return np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)

    def upper(self) -> np.ndarray:
        return np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)


@dataclass
class DetectionResult:
    annotated: np.ndarray
    centroid: Optional[tuple] = None
    angle_deg: Optional[float] = None
    contour: Optional[np.ndarray] = None
    contour_count: int = 0


def find_blue_hex(frame_bgr: np.ndarray, settings: DetectionSettings) -> DetectionResult:
    """Detect the largest 6-sided blue contour in a frame and annotate it.

    Never raises: any internal failure degrades to "nothing detected" so a single
    bad frame can't take down the capture loop.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("frame_bgr is empty")

    s = settings.clamped()
    frame = frame_bgr.copy()

    try:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, s.lower(), s.upper())
        kernel = np.ones((s.morph_k, s.morph_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error:
        logger.exception("OpenCV error during mask/contour extraction")
        return DetectionResult(annotated=frame)

    best = None
    best_area = 0.0
    best_poly = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < s.area_min:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        eps = (s.poly_eps_pct / 100.0) * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) == 6 and area > best_area:
            best = cnt
            best_area = area
            best_poly = approx

    centroid = None
    angle_deg = None

    if best is not None:
        try:
            cv2.drawContours(frame, [best_poly], -1, (0, 255, 0), 2)

            M = cv2.moments(best)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)
                cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

            data_pts = best.reshape(-1, 2).astype(np.float32)
            mean = np.empty((0))
            _, eigenvectors, _ = cv2.PCACompute2(data_pts, mean)
            angle_rad = float(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
            angle_deg = float(np.degrees(angle_rad))

            if centroid is not None:
                length = 80
                p2 = (
                    int(centroid[0] + length * np.cos(angle_rad)),
                    int(centroid[1] + length * np.sin(angle_rad)),
                )
                cv2.line(frame, centroid, p2, (255, 0, 0), 2)

            cv2.putText(frame, f"Hex area: {int(best_area)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if s.debug:
                cv2.putText(frame, f"Contours: {len(contours)}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Area: {int(best_area)}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except (cv2.error, ZeroDivisionError, ValueError):
            logger.exception("Failed to annotate detected hex; continuing without it")
            centroid = None
            angle_deg = None

    try:
        small_mask = cv2.resize(mask, (0, 0), fx=0.3, fy=0.3)
        small_mask_bgr = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
        h, w = small_mask_bgr.shape[:2]
        frame[0:h, 0:w] = cv2.addWeighted(frame[0:h, 0:w], 0.3, small_mask_bgr, 0.7, 0)
    except cv2.error:
        logger.exception("Failed to render mask inset")

    return DetectionResult(
        annotated=frame,
        centroid=centroid,
        angle_deg=angle_deg,
        contour=best,
        contour_count=len(contours),
    )


def px_to_cm_estimate(contour, known_side_cm: float) -> Optional[float]:
    """Rough pixel->cm scale from the hex's min enclosing circle.

    For a regular hex of side a, circumscribed radius R = a, so diameter = 2a.
    """
    if contour is None or known_side_cm is None or known_side_cm <= 0:
        return None
    try:
        (_, _), radius = cv2.minEnclosingCircle(contour)
    except cv2.error:
        logger.exception("minEnclosingCircle failed")
        return None
    px_diam = 2 * radius
    if px_diam <= 0:
        return None
    cm_diam = 2 * known_side_cm
    return cm_diam / px_diam
