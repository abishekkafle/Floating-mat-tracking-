import cv2
import numpy as np
import streamlit as st
import pandas as pd
import time
from collections import deque
import platform

# ----------------------- UI CONFIG -----------------------
st.set_page_config(
    page_title="Blue Hex Mat Tracker",
    layout="wide",
)

st.title("🔷 Blue Hex Mat Tracker — Real‑time USB Webcam")
st.caption(
    "Detect a blue, 6‑sided floating mat; set an initial reference; and track deviation from its original position in real time."
)

# Sample rate for history recording (seconds)
SAMPLE_PERIOD = 5.0

# ----------------------- SESSION STATE -----------------------
DEFAULTS = {
    "ref_point": None,          # (x, y)
    "ref_angle": None,          # degrees
    "history": deque(maxlen=500),  # limit records
    "running": False,
    "last_sample_time": None,   # timestamp of last recorded sample
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------- SIDEBAR CONTROLS -----------------------
st.sidebar.header("🎛️ Detector Settings")

# HSV thresholds for "blue"; tune with sliders as needed
h_min = st.sidebar.slider("H min", 0, 179, 90)
s_min = st.sidebar.slider("S min", 0, 255, 80)
v_min = st.sidebar.slider("V min", 0, 255, 80)

h_max = st.sidebar.slider("H max", 0, 179, 140)
s_max = st.sidebar.slider("S max", 0, 255, 255)
v_max = st.sidebar.slider("V max", 0, 255, 255)

area_min = st.sidebar.slider("Min area (px^2)", 1000, 300000, 8000, step=500)
poly_eps = st.sidebar.slider("Polygon approx ε (%)", 1, 10, 3)
morph_k = st.sidebar.slider("Morph kernel (px)", 1, 15, 5, step=2)

# HSV presets
presets = {
    "Default Blue": ((90, 80, 80), (140, 255, 255)),
    "Light Blue": ((100, 50, 50), (140, 255, 255)),
    "Dark Blue": ((90, 100, 50), (140, 255, 200)),
}

preset_names = list(presets.keys())
selected_preset = st.sidebar.selectbox("HSV Presets", preset_names)
if selected_preset:
    lower_preset, upper_preset = presets[selected_preset]
    # Note: This is just for display, the actual values are set by the sliders

# Camera settings
cam_index = st.sidebar.number_input("Camera index", value=0, step=1, help="0 is the default webcam. Change if you have multiple cameras.")
frame_width = st.sidebar.selectbox("Frame width", [640, 800, 960, 1280], index=0)
frame_height = st.sidebar.selectbox("Frame height", [480, 600, 720, 720], index=0)

# Processing control
processing_fps = st.sidebar.slider("Processing FPS", 1, 30, 10, 
                                  help="Lower values reduce CPU usage")

st.sidebar.header("📐 Calibration (optional)")
known_side = st.sidebar.number_input(
    "Known mat side length (cm)", value=29.0, step=1.0,
    help="If provided, displacement will also be estimated in cm using a simple pixel‑to‑cm scale based on the detected hexagon's circumscribed circle diameter."
)

# Debug mode
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Camera test button
if st.sidebar.button("Test Camera"):
    # Try different backends for cross-platform compatibility
    backends = [
        cv2.CAP_DSHOW,  # Windows
        cv2.CAP_V4L2,   # Linux
        cv2.CAP_ANY     # Fallback
    ]
    
    test_cap = None
    for backend in backends:
        test_cap = cv2.VideoCapture(int(cam_index), backend)
        if test_cap.isOpened():
            break
    
    if test_cap and test_cap.isOpened():
        test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(frame_width))
        test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(frame_height))
        ret, test_frame = test_cap.read()
        if ret:
            st.sidebar.image(test_frame, channels="BGR", caption="Camera Test")
        else:
            st.sidebar.error("Could not read frame from camera")
        test_cap.release()
    else:
        st.sidebar.error("Could not open camera")

# Controls
cols = st.columns([1,1,1,2])
with cols[0]:
    run_btn = st.button("▶️ Start", type="primary")
with cols[1]:
    stop_btn = st.button("⏹️ Stop")
with cols[2]:
    set_ref_btn = st.button("📍 Set reference (current)")
with cols[3]:
    clear_btn = st.button("🧹 Clear reference & history")

if run_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False
if clear_btn:
    st.session_state.ref_point = None
    st.session_state.ref_angle = None
    st.session_state.history.clear()
    st.session_state.last_sample_time = None


# ----------------------- HELPERS -----------------------

def find_blue_hex(frame_bgr, lower, upper, area_min=8000, poly_eps_ratio=0.03, morph_k=5, debug=False):
    """Return annotated frame, centroid (x,y), angle_deg, contour for the largest 6‑gon that matches blue mask."""
    frame = frame_bgr.copy()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((morph_k, morph_k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    best_poly = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_min:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = poly_eps_ratio * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) == 6 and area > best_area:
            best = cnt
            best_area = area
            best_poly = approx

    centroid = None
    angle_deg = None

    if best is not None:
        # Draw
        cv2.drawContours(frame, [best_poly], -1, (0, 255, 0), 2)

        # Centroid
        M = cv2.moments(best)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

        # Orientation via PCA
        data_pts = best.reshape(-1, 2).astype(np.float32)
        mean = np.empty((0))
        try:
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
            # Principal axis angle (in image coords: x right, y down)
            angle_rad = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])
            angle_deg = float(np.degrees(angle_rad))

            # Draw a direction line from centroid along principal axis
            if centroid is not None:
                length = 80
                p2 = (
                    int(centroid[0] + length * np.cos(angle_rad)),
                    int(centroid[1] + length * np.sin(angle_rad))
                )
                cv2.line(frame, centroid, p2, (255, 0, 0), 2)
        except Exception as e:
            angle_deg = None

        # Label
        cv2.putText(frame, f"Hex area: {int(best_area)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Debug info
        if debug:
            cv2.putText(frame, f"Contours: {len(contours)}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Area: {int(best_area)}", (10, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Blend mask preview as small inset
    small_mask = cv2.resize(mask, (0,0), fx=0.3, fy=0.3)
    small_mask_bgr = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
    h, w = small_mask_bgr.shape[:2]
    frame[0:h, 0:w] = cv2.addWeighted(frame[0:h, 0:w], 0.3, small_mask_bgr, 0.7, 0)

    return frame, centroid, angle_deg, best


def px_to_cm_estimate(contour, known_side_cm):
    """Rough pixel->cm scaling from hex geometry, using circumscribed circle diameter.
    For a regular hex of side a, circ radius R = a. Diameter = 2a. We estimate pixel diameter from contour's min enclosing circle.
    """
    if contour is None or known_side_cm <= 0:
        return None
    (x, y), radius = cv2.minEnclosingCircle(contour)
    px_diam = 2 * radius
    cm_diam = 2 * known_side_cm  # since R=a
    if px_diam <= 0:
        return None
    return cm_diam / px_diam  # cm per pixel

# ----------------------- LAYOUT -----------------------
left, right = st.columns([2, 1])
video_area = left.empty()

# Status indicators
status_cols = st.columns(3)
with status_cols[0]:
    status_icon = "🟢" if st.session_state.running else "🔴"
    st.markdown(f"{status_icon} **Status:** {'Running' if st.session_state.running else 'Stopped'}")
with status_cols[1]:
    ref_status = "Set" if st.session_state.ref_point else "Not Set"
    st.markdown(f"📍 **Reference:** {ref_status}")
with status_cols[2]:
    detection_status = "Detected" if st.session_state.running else "Not Running"
    st.markdown(f"🔷 **Mat:** {detection_status}")

metrics_area = right.container()
chart_area = right.container()

def append_history(ts, dx_cm, dy_cm, dist_cm, dang_deg):
    st.session_state.history.append({
        "t": ts,
        "dx_cm": dx_cm,
        "dy_cm": dy_cm,
        "distance_cm": dist_cm,
        "angle_drift_deg": dang_deg,
    })


# ----------------------- MAIN LOOP -----------------------

lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

cap = None
if st.session_state.running:
    # Try different backends for cross-platform compatibility
    backends = [
        cv2.CAP_DSHOW,  # Windows
        cv2.CAP_V4L2,   # Linux
        cv2.CAP_ANY     # Fallback
    ]
    
    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(int(cam_index), backend)
        if cap.isOpened():
            break

    if cap is None or not cap.isOpened():
        st.error("❌ Could not open camera. Try a different index or close other apps using the webcam.")
        st.session_state.running = False
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(frame_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(frame_height))

last_time = time.time()
frame_interval = 1.0 / processing_fps

while st.session_state.running:
    # Control frame rate
    current_time = time.time()
    if current_time - last_time < frame_interval:
        time.sleep(0.001)
        continue
    last_time = current_time
    
    ok, frame = cap.read()
    if not ok:
        st.warning("⚠️ Frame grab failed. Stopping.")
        st.session_state.running = False
        break

    annotated, centroid, ang, contour = find_blue_hex(
        frame, lower, upper, area_min=area_min, 
        poly_eps_ratio=poly_eps/100.0, morph_k=morph_k, debug=debug_mode
    )

    # Update detection status
    detection_status = "Detected" if centroid is not None else "Not Detected"
    status_cols[2].markdown(f"🔷 **Mat:** {detection_status}")

    # Handle reference setting on demand
    if set_ref_btn and centroid is not None and ang is not None:
        st.session_state.ref_point = centroid
        st.session_state.ref_angle = ang
        st.session_state.last_sample_time = None
        status_cols[1].markdown("📍 **Reference:** Set")

    # Compute deviations (in cm only)
    dx = dy = dist = dang = None
    scale_cm_per_px = px_to_cm_estimate(contour, known_side)

    if centroid is not None and st.session_state.ref_point is not None:
        rx, ry = st.session_state.ref_point
        dx = float(centroid[0] - rx)
        dy = float(centroid[1] - ry)
        dist = float(np.hypot(dx, dy))
        if st.session_state.ref_angle is not None and ang is not None:
            dang = float(ang - st.session_state.ref_angle)

        # Convert to cm if we have scale
        if scale_cm_per_px:
            dx_cm = dx * scale_cm_per_px
            dy_cm = dy * scale_cm_per_px
            dist_cm = dist * scale_cm_per_px
        else:
            dx_cm = dy_cm = dist_cm = None

        # Enhanced visualization
        # Draw reference point
        cv2.circle(annotated, (int(rx), int(ry)), 8, (0, 255, 255), -1)
        cv2.putText(annotated, "REF", (int(rx)+10, int(ry)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw line from reference to current position
        cv2.line(annotated, (int(rx), int(ry)), centroid, (0, 255, 255), 2)
        
        # Draw distance text
        mid_point = ((int(rx) + centroid[0])//2, (int(ry) + centroid[1])//2)
        if scale_cm_per_px and dist is not None:
            dist_text = f"{dist_cm:.1f} cm"
        else:
            dist_text = f"{dist:.1f} px"
        cv2.putText(annotated, dist_text, 
                   (mid_point[0]+10, mid_point[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Annotate with measurements
        if dist_cm is not None:
            label = f"Δx={dx_cm:.1f} cm Δy={dy_cm:.1f} cm | dist={dist_cm:.1f} cm"
            cv2.putText(annotated, label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Log history at fixed period (every 5 seconds)
        ts = time.time()
        if dist_cm is not None and ((st.session_state.last_sample_time is None) or (ts - st.session_state.last_sample_time >= SAMPLE_PERIOD)):
            append_history(ts, dx_cm, dy_cm, dist_cm, dang)
            st.session_state.last_sample_time = ts

    # Visuals
    if centroid is not None:
        cv2.putText(annotated, f"Angle: {ang:.1f} deg" if ang is not None else "Angle: —", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Stream to UI
    video_area.image(annotated, channels="BGR")

    # Update metrics (cm)
    with metrics_area:
        c1, c2 = st.columns(2)
        with c1:
            if 'dist_cm' in locals() and dist_cm is not None:
                st.metric("Δ distance (cm)", value=f"{dist_cm:.1f}")
            else:
                st.metric("Δ distance (cm)", value="—")
        with c2:
            st.metric("Angle drift (deg)", value=f"{dang:.1f}" if dang is not None else "—")

# Release camera when stopped
if cap is not None:
    cap.release()

# ----------------------- HISTORY & EXPORT -----------------------

st.subheader("📈 Deviation over time (cm)")
if len(st.session_state.history) > 0:
    df = pd.DataFrame(list(st.session_state.history))
    # relative time
    df["time_s"] = df["t"] - df["t"].iloc[0]

    # Plot distance and angle drift (two tabs)
    tab1, tab2 = st.tabs(["Distance (cm)", "Angle drift (deg)"])
    with tab1:
        st.line_chart(df.set_index("time_s")["distance_cm"], height=240)
    with tab2:
        st.line_chart(df.set_index("time_s")["angle_drift_deg"], height=240)

    # Clean CSV only with cm and angle
    df_clean = df[["time_s", "dx_cm", "dy_cm", "distance_cm", "angle_drift_deg"]]
    st.download_button(
        label="💾 Download CSV (cm)",
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
1. Connect a USB webcam.
2. Adjust *Camera index* if needed and click **Start**.
3. Tune HSV sliders until only the mat is highlighted (inset mask at top-left).
4. Hold the mat steady and click **Set reference (current)**.
5. Move the mat; the app shows Δx, Δy, pixel distance (and cm if you entered the mat's side length).

**Tips**
- Ensure good lighting and avoid other blue objects in view.
- If the mat isn't perfectly regular, the 6‑sided approximation will still work as long as it has 6 clear edges.
- Increase *Min area* to ignore small false detections; tweak *Polygon approx ε* if edges aren't being approximated to 6 vertices.
- For accurate cm conversion, provide the true side length and keep camera height roughly constant.
"""
)