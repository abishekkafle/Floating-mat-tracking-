"""Streamlit frontend — thin client for the Blue Hex Mat Tracker backend.

This process holds no camera handle and runs no detection code: it only
renders controls and polls the FastAPI backend (see ../backend/main.py) over
HTTP. That split is what makes Start/Stop and setting changes responsive —
the previous single-process version ran the capture loop *inside* the
Streamlit script, so the Stop button could never be read until that blocking
loop happened to exit.

Run the backend first:
    cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
Then this app:
    cd frontend && streamlit run app.py
"""
from __future__ import annotations

import time

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Blue Hex Mat Tracker", layout="wide")

st.title("\U0001F537 Blue Hex Mat Tracker — Real-time USB Webcam")
st.caption(
    "Detect a blue, 6-sided floating mat; set an initial reference; and track "
    "deviation from its original position in real time."
)

DEFAULT_BACKEND_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 5

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL

backend_url = st.sidebar.text_input("Backend URL", value=st.session_state.backend_url).rstrip("/")
st.session_state.backend_url = backend_url


def api_get(path: str):
    try:
        r = requests.get(f"{backend_url}{path}", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def api_post(path: str, payload: dict | None = None):
    try:
        r = requests.post(f"{backend_url}{path}", json=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            st.sidebar.error(f"Backend: {detail}")
            return None
        return r.json()
    except requests.RequestException as e:
        st.sidebar.error(f"Could not reach backend at {backend_url}: {e}")
        return None


status = api_get("/api/status")
backend_ok = status is not None

if not backend_ok:
    st.error(
        f"⚠️ Cannot reach backend at **{backend_url}**. Start it with "
        "`uvicorn main:app --host 0.0.0.0 --port 8000` from the `backend/` folder."
    )
    status = {
        "running": False, "error": None, "ref_set": False, "ref_point": None,
        "ref_angle": None, "detected": False, "centroid": None, "angle_deg": None,
        "dx_cm": None, "dy_cm": None, "distance_cm": None, "angle_drift_deg": None,
    }

# ----------------------- SIDEBAR: DETECTOR SETTINGS -----------------------
st.sidebar.header("\U0001F39B️ Detector Settings")

presets = {
    "Default Blue": {"h_min": 90, "s_min": 80, "v_min": 80, "h_max": 140, "s_max": 255, "v_max": 255},
    "Light Blue": {"h_min": 100, "s_min": 50, "v_min": 50, "h_max": 140, "s_max": 255, "v_max": 255},
    "Dark Blue": {"h_min": 90, "s_min": 100, "v_min": 50, "h_max": 140, "s_max": 255, "v_max": 200},
}


def apply_preset():
    preset = presets[st.session_state.preset_select]
    for k, v in preset.items():
        st.session_state[k] = v


st.sidebar.selectbox("HSV Presets", list(presets.keys()), key="preset_select", on_change=apply_preset)

h_min = st.sidebar.slider("H min", 0, 179, 90, key="h_min")
s_min = st.sidebar.slider("S min", 0, 255, 80, key="s_min")
v_min = st.sidebar.slider("V min", 0, 255, 80, key="v_min")
h_max = st.sidebar.slider("H max", 0, 179, 140, key="h_max")
s_max = st.sidebar.slider("S max", 0, 255, 255, key="s_max")
v_max = st.sidebar.slider("V max", 0, 255, 255, key="v_max")

area_min = st.sidebar.slider("Min area (px^2)", 1000, 300000, 8000, step=500)
poly_eps = st.sidebar.slider("Polygon approx ε (%)", 1, 10, 3)
morph_k = st.sidebar.slider("Morph kernel (px)", 1, 15, 5, step=2)
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# ----------------------- SIDEBAR: CAMERA SETTINGS -----------------------
cam_index = st.sidebar.number_input(
    "Camera index", value=0, step=1, min_value=0,
    help="0 is the default webcam. Change if you have multiple cameras.",
)
frame_width = st.sidebar.selectbox("Frame width", [640, 800, 960, 1280], index=0)
frame_height = st.sidebar.selectbox("Frame height", [480, 600, 720, 900], index=0)
processing_fps = st.sidebar.slider("Processing FPS", 1, 30, 10, help="Lower values reduce CPU usage")

st.sidebar.header("\U0001F4D0 Calibration (optional)")
known_side = st.sidebar.number_input(
    "Known mat side length (cm)", value=29.0, step=1.0, min_value=0.0,
    help="If provided, displacement is also estimated in cm using the detected "
         "hexagon's circumscribed circle diameter.",
)

if st.sidebar.button("Test Camera"):
    try:
        r = requests.get(
            f"{backend_url}/api/camera/test",
            params={"index": int(cam_index), "width": int(frame_width), "height": int(frame_height)},
            timeout=10,
        )
        if r.status_code == 200:
            st.sidebar.image(r.content, caption="Camera Test")
        else:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            st.sidebar.error(detail)
    except requests.RequestException as e:
        st.sidebar.error(f"Could not reach backend: {e}")

st.sidebar.divider()
auto_refresh = st.sidebar.checkbox("Live updates", value=True, help="Auto-refresh status/metrics/history.")
refresh_interval = st.sidebar.slider("Refresh interval (s)", 0.5, 5.0, 1.0, step=0.5)

# Push current settings to the backend every run; cheap JSON POST regardless of run cause.
if backend_ok:
    api_post("/api/settings/detection", {
        "h_min": h_min, "s_min": s_min, "v_min": v_min,
        "h_max": h_max, "s_max": s_max, "v_max": v_max,
        "area_min": area_min, "poly_eps_pct": poly_eps, "morph_k": morph_k,
        "debug": debug_mode,
    })
    api_post("/api/settings/camera", {
        "index": int(cam_index), "width": int(frame_width), "height": int(frame_height),
        "processing_fps": processing_fps, "known_side_cm": known_side,
    })

# ----------------------- CONTROLS -----------------------
cols = st.columns([1, 1, 1, 2])
with cols[0]:
    if st.button("▶️ Start", type="primary", disabled=not backend_ok):
        api_post("/api/start")
        st.rerun()
with cols[1]:
    if st.button("⏹️ Stop", disabled=not backend_ok):
        api_post("/api/stop")
        st.rerun()
with cols[2]:
    if st.button("\U0001F4CD Set reference (current)", disabled=not backend_ok):
        api_post("/api/reference/set")
        st.rerun()
with cols[3]:
    if st.button("\U0001F9F9 Clear reference & history", disabled=not backend_ok):
        api_post("/api/reference/clear")
        st.rerun()

if status.get("error"):
    st.warning(f"⚠️ Backend reported: {status['error']}")

# ----------------------- STATUS -----------------------
status_cols = st.columns(3)
with status_cols[0]:
    icon = "\U0001F7E2" if status["running"] else "\U0001F534"
    st.markdown(f"{icon} **Status:** {'Running' if status['running'] else 'Stopped'}")
with status_cols[1]:
    st.markdown(f"\U0001F4CD **Reference:** {'Set' if status['ref_set'] else 'Not Set'}")
with status_cols[2]:
    st.markdown(f"\U0001F537 **Mat:** {'Detected' if status['detected'] else 'Not Detected'}")

# ----------------------- LAYOUT -----------------------
left, right = st.columns([2, 1])
with left:
    if status["running"]:
        st.markdown(
            f'<img src="{backend_url}/api/video_feed" style="width:100%; border-radius:8px;">',
            unsafe_allow_html=True,
        )
    else:
        st.info("Click ▶️ Start to begin the video stream.")

with right:
    m1, m2 = st.columns(2)
    m1.metric("Δ distance (cm)", value=f"{status['distance_cm']:.1f}" if status.get("distance_cm") is not None else "—")
    m2.metric("Angle drift (deg)", value=f"{status['angle_drift_deg']:.1f}" if status.get("angle_drift_deg") is not None else "—")

# ----------------------- HISTORY & EXPORT -----------------------
st.subheader("\U0001F4C8 Deviation over time (cm)")
history = api_get("/api/history") if backend_ok else None

if history:
    df = pd.DataFrame(history)
    df["time_s"] = df["t"] - df["t"].iloc[0]

    tab1, tab2 = st.tabs(["Distance (cm)", "Angle drift (deg)"])
    with tab1:
        st.line_chart(df.set_index("time_s")["distance_cm"], height=240)
    with tab2:
        st.line_chart(df.set_index("time_s")["angle_drift_deg"], height=240)

    df_clean = df[["time_s", "dx_cm", "dy_cm", "distance_cm", "angle_drift_deg"]]
    st.download_button(
        label="\U0001F4BE Download CSV (cm)",
        data=df_clean.to_csv(index=False).encode("utf-8"),
        file_name="hex_mat_tracking_cm.csv",
        mime="text/csv",
    )
else:
    st.info("Start the stream and set a reference to populate tracking history.")

# ----------------------- QUICK START -----------------------
st.markdown(
    """
**Quick start**
1. Start the backend (`uvicorn main:app` from `backend/`), then this app.
2. Connect a USB webcam.
3. Adjust *Camera index* if needed and click **Start**.
4. Tune HSV sliders until only the mat is highlighted (inset mask at top-left of the video).
5. Hold the mat steady and click **Set reference (current)**.
6. Move the mat; the app shows distance and angle drift (and cm if you entered the mat's side length).

**Tips**
- Ensure good lighting and avoid other blue objects in view.
- Increase *Min area* to ignore small false detections; tweak *Polygon approx ε* if edges
  aren't being approximated to 6 vertices.
- For accurate cm conversion, provide the true side length and keep camera height roughly constant.
"""
)

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
