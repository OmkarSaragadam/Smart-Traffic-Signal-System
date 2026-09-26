"""
main.py
-------
Standalone CLI runner (no Streamlit needed) — useful for quickly testing
the detection/tracking/counting/signal pipeline on one video or webcam
before wiring it into the full 4-road dashboard.

Usage:
    python main.py --source videos/sample.mp4 --road North
    python main.py --source 0 --road North          (webcam)

Press 'q' in the preview window to quit.
"""

import argparse
import time

import cv2

import config
from analytics import Analytics
from counter import LineCounter
from database import Database
from detector import VehicleDetector, EmergencyClassifier
from signal_controller import IntersectionController
from tracker import TrackManager
from utils import get_logger, FPSCounter

logger = get_logger(__name__)


def draw_overlay(frame, detections, counter, signal_snapshot, road, fps, emergency_flag):
    # Draw the virtual counting line
    cv2.line(frame, counter.line_start, counter.line_end, (0, 255, 255), 2)

    # Draw detections
    for det in detections:
        color = (0, 200, 0)
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        label = f"{det.class_name} #{det.track_id} {det.confidence:.2f}"
        cv2.putText(frame, label, (det.x1, max(0, det.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # HUD text
    road_state = signal_snapshot[road]
    hud_lines = [
        f"Road: {road}",
        f"Signal: {road_state['state']}  ({road_state['time_remaining']}s)",
        f"Vehicles counted: {counter.total_count}",
        f"Density: {road_state['density']}",
        f"Green time allotted: {road_state['green_time']}s",
        f"FPS: {fps:.1f}",
    ]
    if emergency_flag:
        hud_lines.append("EMERGENCY VEHICLE DETECTED")

    y = 25
    for line in hud_lines:
        color = (0, 0, 255) if "EMERGENCY" in line else (255, 255, 255)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 25

    return frame


def run(source, road_name):
    detector = VehicleDetector()
    emergency_classifier = EmergencyClassifier()
    track_manager = TrackManager()
    db = Database()
    fps_counter = FPSCounter()

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {source}")
        return

    # Default counting line: horizontal line at 60% frame height
    line_start = (0, int(config.FRAME_HEIGHT * 0.6))
    line_end = (config.FRAME_WIDTH, int(config.FRAME_HEIGHT * 0.6))
    counter = LineCounter(line_start, line_end, road_name=road_name)

    controller = IntersectionController()
    last_db_log = time.time()

    logger.info(f"Starting pipeline for road={road_name}, source={source}")

    while True:
        ok, frame = cap.read()
        if not ok:
            logger.info("End of stream / cannot read frame.")
            break

        frame, detections = detector.detect_and_track(frame)
        track_manager.update(detections)
        newly_counted = counter.update(detections, track_manager)

        for det in newly_counted:
            db.log_vehicle_event(road_name, det.track_id, det.class_name)

        # Emergency vehicle check
        emergency_flag = any(
            emergency_classifier.is_emergency_vehicle(frame, det) for det in detections
        )
        if emergency_flag:
            controller.trigger_emergency(road_name)

        # Accident / stationary check
        stationary_ids = track_manager.stationary_track_ids(fps_counter.tick() or 15)
        if stationary_ids:
            logger.warning(f"Possible Accident on {road_name}: stationary tracks {stationary_ids}")

        controller.update_counts(road_name, counter.total_count)
        controller.tick()

        fps = fps_counter.tick()
        snapshot = controller.snapshot()
        frame = draw_overlay(frame, detections, counter, snapshot, road_name, fps, emergency_flag)

        cv2.imshow("Smart Traffic AI - " + road_name, frame)

        # Log a snapshot row to DB every 10 seconds
        if time.time() - last_db_log > 10:
            state = snapshot[road_name]
            db.log_signal_cycle(
                road_name, counter.total_count, state["density"],
                state["green_time"], emergency=emergency_flag,
            )
            last_db_log = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Auto-generate a CSV report on exit
    Database().export_csv("traffic_log", f"{config.REPORTS_DIR}/traffic_log_report.csv")
    Database().export_csv("vehicle_events", f"{config.REPORTS_DIR}/vehicle_events_report.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Traffic AI - single road pipeline runner")
    parser.add_argument("--source", default="0", help="Video file path or webcam index (default: 0)")
    parser.add_argument("--road", default="North", choices=config.ROADS, help="Which road this feed represents")
    args = parser.parse_args()
    run(args.source, args.road)
