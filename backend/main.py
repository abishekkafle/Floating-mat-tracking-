"""FastAPI backend for the Blue Hex Mat Tracker.

Owns the camera and detection loop in a background thread so control
requests (start/stop/settings/reference) are always handled immediately,
regardless of what the capture loop is doing. Run with:

    uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from tracker_service import tracker_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Blue Hex Mat Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectionSettingsIn(BaseModel):
    h_min: Optional[int] = Field(None, ge=0, le=179)
    s_min: Optional[int] = Field(None, ge=0, le=255)
    v_min: Optional[int] = Field(None, ge=0, le=255)
    h_max: Optional[int] = Field(None, ge=0, le=179)
    s_max: Optional[int] = Field(None, ge=0, le=255)
    v_max: Optional[int] = Field(None, ge=0, le=255)
    area_min: Optional[int] = Field(None, ge=1, le=1_000_000)
    poly_eps_pct: Optional[int] = Field(None, ge=1, le=10)
    morph_k: Optional[int] = Field(None, ge=1, le=51)
    debug: Optional[bool] = None


class CameraSettingsIn(BaseModel):
    index: Optional[int] = Field(None, ge=0)
    width: Optional[int] = Field(None, gt=0)
    height: Optional[int] = Field(None, gt=0)
    processing_fps: Optional[float] = Field(None, gt=0, le=60)
    known_side_cm: Optional[float] = Field(None, ge=0)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/status")
def get_status():
    return tracker_service.get_status()


@app.get("/api/history")
def get_history():
    return tracker_service.get_history()


@app.post("/api/settings/detection")
def update_detection_settings(settings: DetectionSettingsIn):
    tracker_service.update_detection_settings(**settings.model_dump(exclude_none=True))
    return tracker_service.get_status()


@app.post("/api/settings/camera")
def update_camera_settings(settings: CameraSettingsIn):
    tracker_service.update_camera_settings(**settings.model_dump(exclude_none=True))
    return tracker_service.get_status()


@app.post("/api/start")
def start():
    tracker_service.start()
    return tracker_service.get_status()


@app.post("/api/stop")
def stop():
    tracker_service.stop()
    return tracker_service.get_status()


@app.post("/api/reference/set")
def set_reference():
    ok = tracker_service.set_reference()
    if not ok:
        raise HTTPException(status_code=409, detail="No mat currently detected; cannot set reference.")
    return tracker_service.get_status()


@app.post("/api/reference/clear")
def clear_reference():
    tracker_service.clear_reference()
    return tracker_service.get_status()


@app.get("/api/frame")
def get_frame():
    """Single latest annotated JPEG frame (useful for clients that can't do MJPEG)."""
    jpeg = tracker_service.get_latest_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=404, detail="No frame available yet.")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/video_feed")
def video_feed():
    """MJPEG stream for direct <img src="..."> embedding in the frontend."""

    def generator():
        boundary = b"--frame"
        while True:
            jpeg = tracker_service.wait_for_frame(timeout=2.0)
            if jpeg is None:
                if not tracker_service.get_status()["running"]:
                    break
                continue
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )

    return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/camera/test")
def test_camera(
    index: int = Query(0, ge=0),
    width: int = Query(640, gt=0),
    height: int = Query(480, gt=0),
):
    ok, jpeg, error = tracker_service.test_camera(index, width, height)
    if not ok:
        raise HTTPException(status_code=502, detail=error or "Camera test failed.")
    return Response(content=jpeg, media_type="image/jpeg")


@app.on_event("shutdown")
def on_shutdown():
    logger.info("Shutting down: stopping capture loop and releasing camera.")
    tracker_service.stop()
