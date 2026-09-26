# 🚦 AI-Based Smart Traffic Signal Control System

Real-time vehicle detection (YOLOv8) + dynamic signal timing for a 4-way
intersection, with a Streamlit dashboard, SQLite logging, and analytics.

---

## 1. Project Objective

Replace fixed-timer traffic signals with an AI system that watches live
video from each approach road, counts vehicles with computer vision, and
recomputes each road's green-light duration in real time based on actual
queue length — while still giving instant priority to emergency vehicles.

---

## 2. Technology Stack

| Layer            | Choice                                  |
|-------------------|------------------------------------------|
| Language          | Python 3.11+                             |
| Detection         | YOLOv8 (Ultralytics)                     |
| Tracking          | ByteTrack (bundled with Ultralytics)     |
| CV / Image utils  | OpenCV                                   |
| Data              | NumPy, Pandas                            |
| Storage           | SQLite                                   |
| Dashboard         | Streamlit                                |
| Charts            | Matplotlib                               |
| IDE               | PyCharm                                  |
| OS                | Windows (cross-platform in practice)     |

---

## 3. Folder Structure

```
smart_traffic_ai/
├── app.py                 # Launcher (runs the Streamlit dashboard)
├── main.py                 # CLI runner for a single road/video (no UI)
├── detector.py             # YOLOv8 wrapper + emergency-vehicle heuristic
├── tracker.py               # Per-track position history / stationary check
├── counter.py                # Virtual line-crossing vehicle counter
├── signal_controller.py       # 4-road signal state machine + green-time formula
├── database.py                 # SQLite read/write + CSV export
├── dashboard.py                  # Streamlit UI (4 feeds, live signals, analytics)
├── analytics.py                   # Hourly/daily/peak stats + charts
├── config.py                       # All tunable constants
├── utils.py                         # Logging, geometry, density, night-mode helpers
├── requirements.txt
├── README.md
├── models/                          # YOLO weights (auto-downloaded if missing)
├── videos/                           # Input video files
├── database/                          # traffic.db (SQLite)
├── outputs/                            # Saved annotated frames/videos
└── reports/                              # Auto-generated CSV reports
```

---

## 4. Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) place a YOLOv8 weight file in models/, or let it auto-download
```

**Run the dashboard:**
```bash
python app.py
# or directly:
streamlit run dashboard.py
```

**Run a single-road CLI test (webcam or video file):**
```bash
python main.py --source videos/sample.mp4 --road North
python main.py --source 0 --road North     # webcam
```

---

## 5. Core Algorithm

### 5.1 Dynamic Green Time Formula

```
green_time = clamp( MIN_GREEN_TIME + vehicle_count * 0.5,
                     MIN_GREEN_TIME,
                     MAX_GREEN_TIME )

MIN_GREEN_TIME = 15 sec
MAX_GREEN_TIME = 45 sec
```

### 5.2 Traffic Density Classification

| Vehicle Count | Density   |
|----------------|-----------|
| 0 – 10          | Low       |
| 11 – 25          | Medium    |
| 26 – 45           | High      |
| 46+                | Very High |

### 5.3 Pseudocode — Main Loop (per road, per frame)

```
for each frame:
    frame = resize(frame)
    if is_low_light(frame):
        frame = enhance_with_CLAHE(frame)

    detections = YOLOv8.track(frame)               # + ByteTrack IDs
    track_history.update(detections)

    for detection in detections:
        if detection.crossed_virtual_line() and not detection.already_counted:
            vehicle_count += 1
            mark_as_counted(detection)

    if any(detection.is_emergency_vehicle() for detection in detections):
        signal_controller.force_green(this_road)
        display("Emergency Vehicle Detected")

    if track_history.stationary_for(> 60s):
        display("Possible Accident")

    density = classify_density(vehicle_count)
    green_time = clamp(15 + vehicle_count * 0.5, 15, 45)

    signal_controller.update(this_road, vehicle_count, green_time)
    signal_controller.tick()   # advances Red -> Green -> Yellow -> Red state machine

    log_to_database(...)
    render_dashboard(...)
```

### 5.4 Signal State Machine (4-Way Intersection)

```
        ┌────────────┐   timer expires   ┌─────────────┐
        │   GREEN    │ ─────────────────▶│   YELLOW    │
        │ (this road)│                    │ (this road) │
        └────────────┘                    └─────────────┘
              ▲                                  │ timer expires
              │                                  ▼
        ┌────────────┐   round-robin      ┌─────────────┐
        │ next road  │ ◀───────────────── │     RED     │
        │  GREEN     │                    │ (this road) │
        └────────────┘                    └─────────────┘

Only one road is GREEN at any instant.
Emergency override: force ALL-RED -> instant GREEN for the emergency road.
```

---

## 6. System Architecture (high level)

```
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│  Camera/  │───▶│ detector. │───▶│ tracker.py│───▶│ counter.py│
│  Video In │    │ py (YOLO) │    │ (history) │    │ (line-x)  │
└───────────┘    └───────────┘    └───────────┘    └─────┬─────┘
                                                          │ vehicle_count
                                                          ▼
                                          ┌────────────────────────────┐
                                          │ signal_controller.py        │
                                          │ (density + green-time calc, │
                                          │  4-road state machine)      │
                                          └───────────────┬────────────┘
                                                          │
                    ┌─────────────────────────────────────┼───────────────┐
                    ▼                                     ▼               ▼
             ┌─────────────┐                     ┌────────────────┐ ┌───────────┐
             │ database.py │                     │  dashboard.py  │ │ analytics │
             │  (SQLite)   │◀────────────────────│  (Streamlit)   │▶│    .py    │
             └─────────────┘                     └────────────────┘ └───────────┘
```

### Data Flow (DFD, level 1)

```
Video Frame → [Preprocess: resize + low-light enhance]
            → [YOLOv8 Detect + ByteTrack IDs]
            → [Track History Update]
            → [Line-Crossing Count]
            → [Density Classification] → [Green-Time Calculation]
            → [Signal State Machine]
            → [SQLite Log] → [Analytics / CSV Reports]
            → [Streamlit Dashboard Render]
```

### Class Diagram (textual)

```
VehicleDetector          EmergencyClassifier
  - model                  - red/blue HSV ranges
  - detect_and_track()      - is_emergency_vehicle()

TrackManager              LineCounter
  - tracks: {id: TrackState} - line_start/end
  - update()                 - update() -> newly_counted
  - stationary_track_ids()   - summary()

RoadSignal                IntersectionController
  - state, green_time        - signals: {road: RoadSignal}
  - time_remaining()          - update_counts()
                                - trigger_emergency()
                                - tick()

Database                  Analytics
  - log_signal_cycle()       - hourly_traffic(), daily_traffic()
  - log_vehicle_event()       - peak_traffic(), signal_utilization()
  - export_csv()               - chart_*() (matplotlib figures)
```

### Use-Case Diagram (textual)

```
Actor: Traffic Authority / Operator
 - Start/Stop monitoring
 - View live dashboard
 - Export CSV reports
 - View analytics (hourly/daily/peak/utilization)

Actor: System (automated)
 - Detect + track + count vehicles
 - Classify density, compute green time
 - Detect emergency vehicles → override signal
 - Detect stationary traffic → flag possible accident
 - Log every cycle to SQLite
```

### ER Diagram (textual)

```
traffic_log                         vehicle_events
------------                        ---------------
id (PK)                             id (PK)
date                                 date
time                                  time
road                                   road
vehicle_count                           track_id
density                                  vehicle_class
signal_time
emergency
```

---

## 7. Feature Notes & Honest Limitations

- **Emergency vehicle detection**: the stock YOLOv8/COCO model has no
  ambulance/fire-truck/police classes, so `EmergencyClassifier` uses a
  siren-light-bar color heuristic as a working stand-in. For production
  accuracy, fine-tune a YOLOv8 model on a labeled emergency-vehicle
  dataset and point `config.YOLO_MODEL_NAME` at it — the rest of the
  pipeline needs no changes.
- **Auto-rickshaw**: not a native COCO class either; the hook exists in
  `config.VEHICLE_CLASSES` for a custom-trained model.
- **Average waiting time**: computed analytically from signal-cycle
  logs (how long other roads' phases take), not from true per-vehicle
  dwell-time tracking, which would require persisting each vehicle's
  "time since it entered the frame" until it crosses the line.
- **Night/weather handling**: implemented via CLAHE contrast
  enhancement + denoising, a lightweight and genuinely effective
  approach, but not a substitute for an IR/thermal camera in true
  zero-visibility conditions.

---

## 8. Advantages

- Signal timing responds to real traffic instead of a fixed clock.
- Emergency vehicles get priority automatically.
- Historical data enables city-planning decisions (peak hours, road
  utilization).
- Modular codebase — swap YOLO weights, tracker, or add a road without
  rewriting the system.

## 9. Applications

- Smart-city traffic management.
- Campus / gated-community intersection control.
- Traffic research and simulation for urban planning.

## 10. Future Enhancements

- Fine-tuned emergency-vehicle and auto-rickshaw detection model.
- Multi-camera calibration for true multi-lane counting.
- Reinforcement-learning-based signal optimization across the whole
  city grid (not just one intersection).
- Mobile app / SMS alerts for accidents.

## 11. Conclusion

This system demonstrates, end-to-end, how a modern object-detection and
tracking pipeline can replace static traffic-signal timers with a
data-driven controller — while remaining transparent about where a
production deployment would need a custom-trained model instead of the
stock heuristics used here for demonstrability.

## 12. References

- Ultralytics YOLOv8 documentation — https://docs.ultralytics.com
- ByteTrack: Zhang et al., "ByteTrack: Multi-Object Tracking by
  Associating Every Detection Box"
- OpenCV documentation — https://docs.opencv.org
- Streamlit documentation — https://docs.streamlit.io
