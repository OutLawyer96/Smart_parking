import cv2
import time
from perception.camera import CameraStream
from perception.tracker import VehicleTracker
from spatial.calibration import CalibrationMap
from spatial.zone_map import ZoneMap
from reasoning.layout_engine import LayoutEngine
from reasoning.occupancy import OccupancyEngine
from reasoning.exit_path import filter_slots_by_exit_path
from reasoning.preparked import StoppedCarMonitor

# ── Session config ────────────────────────────────────────────────────────────
# Default parking mode. Press M at runtime to cycle modes.
PARKING_TYPE  = "open"   # "open" | "parallel" | "angled"
PARKING_MODES = ["open", "parallel", "angled"]

# Toggle overlays
SHOW_ZONES = True
SHOW_SLOTS = True

# Debug overlay: shows inflated slot outlines + overlap % per slot.
# Press D at runtime to toggle.
DEBUG_OCCUPANCY = False

# Vehicle dimensions (cm). Used by LayoutEngine when calibration is active.
# When None, falls back to module-level defaults in layout_engine.py.
# In a live backend integration, populate this dict from the backend response.
VEHICLE_DIMS: dict | None = None   # e.g. {"length_cm": 8.7, "width_cm": 3.6}
# ─────────────────────────────────────────────────────────────────────────────


def draw_detection(frame, bbox, label, conf, track_id):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label}  {conf*100:.0f}%  ID:{track_id}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_w, text_h = text_size
    cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), (0, 255, 0), -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def draw_hud(frame, fps, detections, occupancy_summary, parking_type):
    free     = occupancy_summary.get("free", "?")
    occupied = occupancy_summary.get("occupied", "?")
    total    = occupancy_summary.get("total", "?")

    lines = [
        (f"FPS: {fps:.1f}",                       (0,   0, 255)),
        (f"Cars detected: {len(detections)}",      (0, 200, 255)),
        (f"Mode: {parking_type}",                  (200, 200, 0)),
        (f"Slots: {free} free / {occupied} occupied / {total} total", (180, 180, 180)),
    ]
    y = 36
    for text, color in lines:
        cv2.putText(frame, text, (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        y += 32


def filter_restricted(detections: list, zones: ZoneMap) -> list:
    """Drop any detection whose centroid falls inside a restricted zone."""
    allowed = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        zone = zones.get_zone_at((cx, cy))
        if zone and zone["type"] == "restricted":
            continue
        allowed.append(det)
    return allowed


def rebuild_layout(parking_type, zones, calibration, vehicle_dims=None,
                   pinned_slots=None):
    """Rebuild slot layout and occupancy engine for the given mode."""
    engine = LayoutEngine(
        parking_type=parking_type,
        calibrated=calibration.is_calibrated,
        vehicle_dims=vehicle_dims,
    )
    # The engine generates obstacle-aware layouts:
    #   open mode   → tries all orientations, filters overlaps with pinned
    #                  cars, picks orientation with most non-overlapping slots
    #   par / angled → shifts slots along each row around pinned anchors
    generated = engine.generate_all(zones, calibration,
                                    pinned_slots=pinned_slots)

    # Combine generated + pinned before exit-path filter
    all_slots = generated + (pinned_slots or [])

    # Car width for exit-path width check — from backend dims or fallback
    car_width_px = None
    if vehicle_dims and "width_px" in vehicle_dims:
        car_width_px = vehicle_dims["width_px"]

    all_slots, warnings = filter_slots_by_exit_path(
        all_slots, zones, car_width_px=car_width_px
    )
    for w in warnings:
        print(w)
    return OccupancyEngine(all_slots)


def main():
    # ── Load spatial config ───────────────────
    calibration = CalibrationMap()
    calibration.load()                           # graceful no-op if not calibrated yet

    zones = ZoneMap()
    zones.load()

    # ── Perception ───────────────────────────
    tracker = VehicleTracker()

    # ── Continuous stopped-car monitor ────────
    stopped_monitor = StoppedCarMonitor(zones)

    parking_type    = PARKING_TYPE
    debug_occupancy = DEBUG_OCCUPANCY

    with CameraStream() as cam:
        # ── Create fullscreen window before the loop ──
        cv2.namedWindow("Smart Parking System", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Smart Parking System",
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        # ── Initial layout (no pinned slots yet — monitor needs time) ──
        occupancy = rebuild_layout(parking_type, zones, calibration,
                                   VEHICLE_DIMS)

        while True:
            start_time = time.time()

            frame = cam.read()
            if frame is None:
                break

            # ── Zone overlay ──────────────────
            if SHOW_ZONES:
                zones.draw(frame, show_labels=True)

            # ── Detection + tracking ──────────
            detections = tracker.track(frame)

            # ── Drop detections in restricted zones ──
            detections = filter_restricted(detections, zones)

            # ── Stopped-car monitor (every frame) ──
            pins_changed = stopped_monitor.update(detections)
            if pins_changed:
                pinned = stopped_monitor.pinned_slots()
                print(f"[Main] Pinned slots changed ({len(pinned)} active) "
                      f"→ rebuilding layout")
                occupancy = rebuild_layout(parking_type, zones, calibration,
                                           VEHICLE_DIMS, pinned)

            # ── Occupancy update ──────────────
            occupancy.update(detections)

            # ── Slot overlay ──────────────────
            if SHOW_SLOTS:
                occupancy.draw(frame, debug=debug_occupancy)

            # ── Detection boxes ───────────────
            for det in detections:
                draw_detection(frame, det["bbox"], det["label"],
                               det["confidence"], det["track_id"])

            # ── HUD ───────────────────────────
            fps = 1 / (time.time() - start_time)
            draw_hud(frame, fps, detections, occupancy.summary(), parking_type)

            cv2.imshow("Smart Parking System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m") or key == ord("M"):
                # Cycle to next parking mode
                idx = PARKING_MODES.index(parking_type)
                parking_type = PARKING_MODES[(idx + 1) % len(PARKING_MODES)]
                print(f"[Main] Switching parking mode → {parking_type}")
                occupancy = rebuild_layout(
                    parking_type, zones, calibration, VEHICLE_DIMS,
                    stopped_monitor.pinned_slots())
            elif key == ord("d") or key == ord("D"):
                debug_occupancy = not debug_occupancy
                print(f"[Main] Debug overlay {'ON' if debug_occupancy else 'OFF'}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()