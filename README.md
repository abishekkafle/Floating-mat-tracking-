Blue Hex Mat Tracker
A real-time computer vision application for tracking the position and orientation of a blue hexagonal mat using a USB webcam. Built with Python, OpenCV, FastAPI, and a static HTML/CSS/JS frontend.

Overview
This application detects a blue, 6-sided floating mat in real-time video and tracks its deviation from an initial reference position. Perfect for motion analysis, physical therapy monitoring, balance training assessment, or any application requiring precise 2D object tracking.

Architecture
The app is split into two independent processes:

- `backend/` — a FastAPI service that owns the webcam and the detection loop, running it in a background thread. It exposes a small REST API (start/stop/settings/reference/history) plus an MJPEG video endpoint.
- `frontend_html/` — a plain static HTML/CSS/JS page (no build step, no framework) that renders the controls and talks to the backend purely over HTTP (`fetch` for control/data, an `<img>` tag for the MJPEG video). It holds no camera state itself.

This split fixes the main robustness issue in the original single-file version: there, the capture loop ran *inside* the Streamlit script as a blocking `while` loop, so the Stop button (and any setting change) could only be read once that loop happened to exit. Now the capture loop runs independently in the backend, so control requests are handled immediately, and the backend keeps running (and can be reused by other clients) regardless of what the frontend is doing.

The original single-file script (`blue_hex_mat_tracker.py`) is kept for reference but is no longer the recommended way to run the app. A Streamlit version of the frontend also previously existed under `frontend/`; `frontend_html/` is now the primary, recommended frontend.

Features
- Real-time Detection: Tracks blue hexagonal objects at adjustable frame rates (1-30 FPS)
- Position Tracking: Measures X/Y displacement and total distance from reference point
- Orientation Tracking: Calculates angular drift using PCA-based orientation analysis
- Calibration Support: Convert pixel measurements to centimeters using known mat dimensions
- Visual Feedback: Live annotated MJPEG video stream with reference markers and displacement vectors
- Data Logging: Records tracking data every 5 seconds with CSV export capability
- Interactive Charts: Real-time plots of distance and angle drift over time
- Customizable Detection: Adjustable HSV thresholds, morphological operations, and polygon approximation parameters
- Resilient capture loop: a bad/dropped frame or a detection error is logged and skipped rather than crashing the service; the camera is always released on stop/shutdown/error

Requirements
python >= 3.9

Backend: fastapi, uvicorn, opencv-python, numpy
Frontend: none — a browser is all you need. `frontend_html/` is plain HTML/CSS/JS with zero dependencies.

Installation
Clone the repository, then install the backend's dependencies:

    cd backend
    pip install -r requirements.txt

Usage
1. Start the backend (owns the webcam):

       cd backend
       uvicorn main:app --host 0.0.0.0 --port 8000

2. In a second terminal, serve the static frontend (any static file server works; Python's built-in one needs no install):

       cd frontend_html
       python -m http.server 8501

3. Open http://localhost:8501 in a browser.
4. If the backend runs on a different host/port, update the "Backend URL" field at the top of the page.
5. Connect your USB webcam and adjust settings in the sidebar:
   - HSV Thresholds: Fine-tune color detection for your specific mat
   - Camera Settings: Select camera index and resolution
   - Detection Parameters: Adjust minimum area, polygon approximation, and morphological operations
6. Click ▶️ Start to begin video capture.
7. Tune HSV sliders until only the mat is highlighted (inset mask at top-left).
8. Position the mat at your desired reference point and click 📍 Set reference (current).
9. Move the mat and observe real-time tracking data and visualizations.

Configuration
HSV Presets
- Default Blue: (H: 90-140, S: 80-255, V: 80-255)
- Light Blue: (H: 100-140, S: 50-255, V: 50-255)
- Dark Blue: (H: 90-140, S: 100-255, V: 50-200)

Calibration
Enter your mat's side length in centimeters for accurate distance measurements. The application uses the mat's circumscribed circle diameter for pixel-to-cm conversion.

Processing Control
Adjust processing FPS to balance between tracking accuracy and CPU usage.

Output Data
The application records and exports:
- time_s: Elapsed time in seconds
- dx_cm: X-axis displacement (cm)
- dy_cm: Y-axis displacement (cm)
- distance_cm: Total displacement from reference (cm)
- angle_drift_deg: Angular rotation from reference (degrees)

Backend API (for reference / other clients)
- GET  /api/health
- GET  /api/status
- GET  /api/history
- POST /api/settings/detection
- POST /api/settings/camera
- POST /api/start
- POST /api/stop
- POST /api/reference/set
- POST /api/reference/clear
- GET  /api/frame          — single latest annotated JPEG
- GET  /api/video_feed     — MJPEG stream
- GET  /api/camera/test?index=&width=&height=

Troubleshooting
Camera not detected:
- Try different camera indices (0, 1, 2...)
- Use the "Test Camera" button to verify camera access
- Close other applications using the webcam

Mat not detected:
- Ensure good, even lighting
- Remove other blue objects from view
- Adjust HSV thresholds using the sliders
- Increase morphological kernel size for noisy environments

Inaccurate measurements:
- Verify the mat's side length is correctly entered
- Keep camera height and angle consistent
- Ensure the mat is fully visible and not occluded

Cross-Platform Support
The backend automatically tries multiple OpenCV backends for camera access:
- Windows: DirectShow (CAP_DSHOW), then Media Foundation (CAP_MSMF)
- Linux: Video4Linux2 (CAP_V4L2)
- Fallback: Generic (CAP_ANY)

Tips
- Use a mat with 6 clear, distinct edges for best results
- Avoid shadows and reflections on the mat surface
- For precise measurements, mount the camera directly above the tracking area
- The polygon approximation epsilon parameter affects edge detection sensitivity
- Debug mode shows additional detection information for troubleshooting

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Acknowledgments
Built with:
- OpenCV - Computer vision processing
- FastAPI - Backend API and background capture service
- Vanilla HTML/CSS/JS - Frontend, no build step or framework
- NumPy - Numerical computations
