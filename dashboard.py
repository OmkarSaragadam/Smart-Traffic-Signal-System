"""
dashboard.py
------------
Streamlit dashboard for the full 4-road Smart Traffic AI system.

Run with:
    streamlit run dashboard.py

Upload up to 4 videos (one per road) or leave a road empty to simulate
it with synthetic random traffic (handy for demoing the signal-timing
and analytics logic without needing four camera feeds).
"""

import time

import cv2
import streamlit as st

import config
from analytics import Analytics
from counter import LineCounter
from database import Database
from detector import VehicleDetector, EmergencyClassifier
from signal_controller import IntersectionController
from tracker import TrackManager
from utils import get_logger, FPSCounter, classify_density

logger = get_logger(__name__)

st.set_page_config(
    page_title="Smart Traffic AI Dashboard",
    page_icon="🚦",
    layout="wide",
)

# ----------------------------------------------------------------------
# DARK THEME / CUSTOM CSS
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #E6E6E6; }
    .metric-card {
        background: linear-gradient(135deg, #1B1F2A 0%, #262B3A 100%);
        border-radius: 12px; padding: 16px; text-align: center;
        border: 1px solid #333849;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #4CAF50; }
    .metric-label { font-size: 13px; color: #9AA0AC; text-transform: uppercase; }
    .signal-dot { height: 22px; width: 22px; border-radius: 50%; display: inline-block; margin-right: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚦 AI-Based Smart Traffic Signal Control System")
st.caption("Real-time vehicle detection (YOLOv8) + dynamic signal timing for a 4-way intersection")


# ----------------------------------------------------------------------
# SESSION STATE INIT
# ----------------------------------------------------------------------
def init_state():
    if "controller" not in st.session_state:
        st.session_state.controller = IntersectionController()
    if "db" not in st.session_state:
        st.session_state.db = Database()
    if "detector" not in st.session_state:
        st.session_state.detector = VehicleDetector()
    if "emergency_clf" not in st.session_state:
        st.session_state.emergency_clf = EmergencyClassifier()
    if "road_runtime" not in st.session_state:
        # Per-road: TrackManager, LineCounter, FPSCounter, cv2.VideoCapture
        st.session_state.road_runtime = {}
    if "running" not in st.session_state:
        st.session_state.running = False


init_state()


# ----------------------------------------------------------------------
# SIDEBAR: video sources + controls
# ----------------------------------------------------------------------
st.sidebar.header("Video Sources")
uploaded = {}
for road in config.ROADS:
    uploaded[road] = st.sidebar.file_uploader(f"{road} camera feed", type=["mp4", "avi", "mov"], key=f"upl_{road}")

run_col1, run_col2 = st.sidebar.columns(2)
start_clicked = run_col1.button("▶ Start", use_container_width=True)
stop_clicked = run_col2.button("⏹ Stop", use_container_width=True)

if start_clicked:
    st.session_state.running = True
if stop_clicked:
    st.session_state.running = False

st.sidebar.markdown("---")
st.sidebar.subheader("Reports")
if st.sidebar.button("Export CSV reports"):
    p1 = st.session_state.db.export_csv("traffic_log", f"{config.REPORTS_DIR}/traffic_log_report.csv")
    p2 = st.session_state.db.export_csv("vehicle_events", f"{config.REPORTS_DIR}/vehicle_events_report.csv")
    st.sidebar.success(f"Exported to:\n{p1}\n{p2}")


def get_runtime_for_road(road, video_file):
    """Lazily create per-road runtime objects (video capture, counter, tracker)."""
    rt = st.session_state.road_runtime.get(road)
    if rt is not None:
        return rt

    line_start = (0, int(config.FRAME_HEIGHT * 0.6))
    line_end = (config.FRAME_WIDTH, int(config.FRAME_HEIGHT * 0.6))

    cap = None
    if video_file is not None:
        # Persist the uploaded file to disk so cv2.VideoCapture can read it
        tmp_path = f"{config.VIDEOS_DIR}/_upload_{road}.mp4"
        with open(tmp_path, "wb") as f:
            f.write(video_file.getbuffer())
        cap = cv2.VideoCapture(tmp_path)

    rt = {
        "cap": cap,
        "track_manager": TrackManager(),
        "counter": LineCounter(line_start, line_end, road_name=road),
        "fps_counter": FPSCounter(),
        "last_db_log": 0.0,
        "simulated_count": 0,
    }
    st.session_state.road_runtime[road] = rt
    return rt


# ----------------------------------------------------------------------
# LAYOUT PLACEHOLDERS
# ----------------------------------------------------------------------
video_cols = st.columns(4)
video_placeholders = {road: video_cols[i].empty() for i, road in enumerate(config.ROADS)}
video_captions = {road: video_cols[i].empty() for i, road in enumerate(config.ROADS)}

st.markdown("### Intersection Status")
signal_cols = st.columns(4)
signal_placeholders = {road: signal_cols[i].empty() for i, road in enumerate(config.ROADS)}

st.markdown("### Live Metrics")
metric_cols = st.columns(4)
metric_placeholders = {
    "total": metric_cols[0].empty(),
    "fps": metric_cols[1].empty(),
    "active_green": metric_cols[2].empty(),
    "emergency": metric_cols[3].empty(),
}

analytics_expander = st.expander("📊 Analytics & Reports", expanded=False)


SIGNAL_COLORS = {"Red": "#D32F2F", "Yellow": "#FBC02D", "Green": "#4CAF50"}


def render_signal_card(placeholder, road, state):
    color = SIGNAL_COLORS[state["state"]]
    placeholder.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:16px; font-weight:600;">{road}</div>
            <div style="margin:8px 0;">
                <span class="signal-dot" style="background:{color};"></span>
                <span style="font-size:18px; font-weight:700;">{state['state']}</span>
            </div>
            <div class="metric-label">Time Remaining</div>
            <div class="metric-value">{state['time_remaining']}s</div>
            <div class="metric-label">Density</div>
            <div style="font-weight:600;">{state['density']}</div>
            <div class="metric-label">Vehicles</div>
            <div style="font-weight:600;">{state['vehicle_count']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analytics():
    with analytics_expander:
        analytics = Analytics(st.session_state.db)
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            st.pyplot(analytics.chart_hourly_bar())
            st.pyplot(analytics.chart_density_pie())
        with a_col2:
            st.pyplot(analytics.chart_daily_line())
            st.pyplot(analytics.chart_utilization_bar())

        st.markdown("**Key Stats**")
        peak = analytics.peak_traffic()
        st.write(f"- Peak traffic hour: **{peak['hour']}** with **{peak['total_vehicles']}** vehicles")
        st.write(f"- Average green time granted: **{analytics.average_green_time():.1f}s**")
        st.write(f"- Estimated average waiting time per road: **{analytics.average_waiting_time():.1f}s**")


# ----------------------------------------------------------------------
# MAIN PROCESSING LOOP
# ----------------------------------------------------------------------
def process_road_frame(road, video_file):
    rt = get_runtime_for_road(road, video_file)
    controller = st.session_state.controller
    detector = st.session_state.detector
    emergency_clf = st.session_state.emergency_clf
    db = st.session_state.db

    emergency_flag = False

    if rt["cap"] is not None and rt["cap"].isOpened():
        ok, frame = rt["cap"].read()
        if not ok:  # loop the video for a continuous demo
            rt["cap"].set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = rt["cap"].read()
        if ok:
            frame, detections = detector.detect_and_track(frame)
            rt["track_manager"].update(detections)
            newly_counted = rt["counter"].update(detections, rt["track_manager"])
            for det in newly_counted:
                db.log_vehicle_event(road, det.track_id, det.class_name)

            emergency_flag = any(
                emergency_clf.is_emergency_vehicle(frame, det) for det in detections
            )

            # draw boxes + counting line
            cv2.line(frame, rt["counter"].line_start, rt["counter"].line_end, (0, 255, 255), 2)
            for det in detections:
                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 200, 0), 2)
                cv2.putText(frame, f"{det.class_name} #{det.track_id}", (det.x1, max(0, det.y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)

            video_placeholders[road].image(frame, channels="BGR", use_container_width=True)
            vehicle_count = rt["counter"].total_count
            fps = rt["fps_counter"].tick()
        else:
            vehicle_count = 0
            fps = 0.0
    else:
        # No video uploaded for this road -> lightweight synthetic simulation
        # so the signal-timing / analytics pipeline is still fully demonstrable.
        import random
        rt["simulated_count"] = max(0, rt["simulated_count"] + random.randint(-2, 3))
        vehicle_count = rt["simulated_count"]
        fps = 0.0
        video_placeholders[road].info(f"No feed uploaded for {road} — simulating traffic.")

    video_captions[road].caption(f"{road} Road Feed")

    if emergency_flag:
        controller.trigger_emergency(road)

    controller.update_counts(road, vehicle_count)

    if time.time() - rt["last_db_log"] > 10:
        state = controller.snapshot()[road]
        db.log_signal_cycle(road, vehicle_count, state["density"], state["green_time"], emergency=emergency_flag)
        rt["last_db_log"] = time.time()

    return vehicle_count, fps, emergency_flag


def main_loop():
    controller = st.session_state.controller
    total_emergency = False
    total_fps = 0.0

    for road in config.ROADS:
        _, fps, emergency_flag = process_road_frame(road, uploaded[road])
        total_fps += fps
        total_emergency = total_emergency or emergency_flag

    controller.tick()
    snapshot = controller.snapshot()

    for road in config.ROADS:
        render_signal_card(signal_placeholders[road], road, snapshot[road])

    total_vehicles = sum(s["vehicle_count"] for s in snapshot.values())
    active_green = next((r for r, s in snapshot.items() if s["state"] == "Green"), "-")

    metric_placeholders["total"].markdown(
        f'<div class="metric-card"><div class="metric-label">Total Vehicles</div>'
        f'<div class="metric-value">{total_vehicles}</div></div>', unsafe_allow_html=True)
    metric_placeholders["fps"].markdown(
        f'<div class="metric-card"><div class="metric-label">Avg FPS</div>'
        f'<div class="metric-value">{total_fps / 4:.1f}</div></div>', unsafe_allow_html=True)
    metric_placeholders["active_green"].markdown(
        f'<div class="metric-card"><div class="metric-label">Active Green</div>'
        f'<div class="metric-value">{active_green}</div></div>', unsafe_allow_html=True)
    metric_placeholders["emergency"].markdown(
        f'<div class="metric-card"><div class="metric-label">Emergency</div>'
        f'<div class="metric-value">{"YES" if total_emergency else "No"}</div></div>',
        unsafe_allow_html=True)


if st.session_state.running:
    render_analytics()
    loop_placeholder = st.empty()
    # Streamlit reruns top-to-bottom; a bounded loop here keeps one frame
    # cycle live-updating without needing st.experimental_rerun spam.
    for _ in range(100000):
        if not st.session_state.running:
            break
        main_loop()
        time.sleep(0.03)
else:
    st.info("Upload video feeds for one or more roads in the sidebar (or leave blank to simulate), then click ▶ Start.")
    render_analytics()
