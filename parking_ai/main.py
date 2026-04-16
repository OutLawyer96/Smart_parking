import cv2
import time
import threading
import os
from datetime import datetime, timezone
from perception.camera import CameraStream
from perception.tracker import VehicleTracker
from spatial.calibration import CalibrationMap
from spatial.zone_map import ZoneMap
from reasoning.layout_engine import LayoutEngine
from reasoning.occupancy import OccupancyEngine
from reasoning.exit_path import filter_slots_by_exit_path
from reasoning.preparked import StoppedCarMonitor
from reasoning.slot_recommender import SlotRecommender
from api.state import ParkingState
from api.route_planner import RoutePlanner

# ── Session config ────────────────────────────────────────────────────────────
# Default parking mode. Press M at runtime to cycle modes.
PARKING_TYPE  = "open"   # "open" | "parallel" | "angled"
PARKING_MODES = ["open", "parallel", "angled"]

# Toggle overlays
SHOW_ZONES = True
SHOW_SLOTS = True
SHOW_ZONE_LABELS = os.getenv("SHOW_ZONE_LABELS", "0") == "1"
SHOW_SLOT_LABELS = os.getenv("SHOW_SLOT_LABELS", "0") == "1"

# Debug overlay: shows inflated slot outlines + overlap % per slot.
# Press D at runtime to toggle.
DEBUG_OCCUPANCY = False

# Vehicle dimensions (cm). Used by LayoutEngine when calibration is active.
# You can override with env vars VEHICLE_LENGTH_CM and VEHICLE_WIDTH_CM.
_env_len = os.getenv("VEHICLE_LENGTH_CM")
_env_wid = os.getenv("VEHICLE_WIDTH_CM")
VEHICLE_DIMS: dict | None = (
    {
        "length_cm": float(_env_len),
        "width_cm": float(_env_wid),
    }
    if (_env_len is not None and _env_wid is not None)
    else None
)

# Run full detection/tracking every Nth frame and reuse the last result in between.
DETECTION_STRIDE = max(1, int(os.getenv("DETECTION_STRIDE", "1")))
FPS_EMA_ALPHA = 0.2
LAYOUT_REBUILD_COOLDOWN_S = float(os.getenv("LAYOUT_REBUILD_COOLDOWN_S", "0.8"))
# ─────────────────────────────────────────────────────────────────────────────


class _IdentityCalibration:
    """Pixel-space mapping used by uncalibrated layout fallback."""

    @staticmethod
    def to_world(pixel_pt: tuple) -> tuple:
        return (float(pixel_pt[0]), float(pixel_pt[1]))

    @staticmethod
    def to_pixel(world_pt: tuple) -> tuple:
        return (int(round(world_pt[0])), int(round(world_pt[1])))

    def polygon_to_world(self, pixel_polygon: list) -> list:
        return [self.to_world(pt) for pt in pixel_polygon]

    def polygon_to_pixel(self, world_polygon: list) -> list:
        return [self.to_pixel(pt) for pt in world_polygon]


def _resolve_main_camera_source() -> str | int:
    """Main must use MJPEG stream and never /capture snapshot."""
    def _to_stream_only(src: str) -> str:
        if "/capture" in src:
            return src.replace("/capture", "/stream")
        if src.startswith("http://") or src.startswith("https://"):
            # If user passed only host or host/, force /stream endpoint.
            if src.endswith("/"):
                return src + "stream"
            tail = src.rsplit("/", 1)[-1]
            if "." in tail or ":" in tail:
                return src + "/stream"
        return src

    direct = os.getenv("MAIN_CAMERA_SOURCE") or os.getenv("CAMERA_SOURCE")
    if direct:
        if direct.isdigit():
            return int(direct)
        return _to_stream_only(direct)

    stream = os.getenv("ESP32_STREAM_URL")
    if stream:
        return _to_stream_only(stream)

    capture = os.getenv("ESP32_CAPTURE_URL") or os.getenv("ESP32_CAM_URL")
    if capture:
        return _to_stream_only(capture)

    return "http://10.54.215.196/stream"


def draw_detection(frame, bbox, label, conf, track_id):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    scale = 0.45 if w <= 360 else 0.6
    thick = 1 if w <= 360 else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thick)
    text = f"{label} {conf*100:.0f}% ID:{track_id}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    text_w, text_h = text_size
    y_top = max(0, y1 - text_h - 6)
    cv2.rectangle(frame, (x1, y_top), (x1 + text_w + 4, y1), (0, 255, 0), -1)
    cv2.putText(frame, text, (x1 + 2, max(10, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick)


def draw_hud(frame, fps, detections, occupancy_summary, parking_type):
    free     = occupancy_summary.get("free", "?")
    occupied = occupancy_summary.get("occupied", "?")
    total    = occupancy_summary.get("total", "?")

    h, w = frame.shape[:2]
    scale = 0.5 if w <= 360 else 0.75
    thick = 1 if w <= 360 else 2
    y_step = 20 if w <= 360 else 32
    y = 20 if w <= 360 else 36
    x = 8 if w <= 360 else 16

    lines = [
        (f"FPS: {fps:.1f}",                       (0,   0, 255)),
        (f"Cars: {len(detections)}",              (0, 200, 255)),
        (f"Mode: {parking_type}",                 (200, 200, 0)),
        (f"Slots: {free}F/{occupied}O/{total}T",  (180, 180, 180)),
    ]
    for text, color in lines:
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
        y += y_step


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
                   pinned_slots=None, use_calibrated_layout=True):
    """Rebuild slot layout and occupancy engine for the given mode."""
    layout_map = calibration if use_calibrated_layout else _IdentityCalibration()
    engine = LayoutEngine(
        parking_type=parking_type,
        calibrated=use_calibrated_layout,
        vehicle_dims=vehicle_dims,
    )
    # The engine generates obstacle-aware layouts:
    #   open mode   → tries all orientations, filters overlaps with pinned
    #                  cars, picks orientation with most non-overlapping slots
    #   par / angled → shifts slots along each row around pinned anchors
    generated = engine.generate_all(zones, layout_map,
                                    pinned_slots=pinned_slots)

    # Guardrail: if calibrated geometry produces too few slots (often due to
    # imperfect homography), retry in pixel-space and keep the better result.
    min_expected = int(os.getenv("LAYOUT_MIN_EXPECTED_SLOTS", "3"))
    pixel_map = _IdentityCalibration()
    if use_calibrated_layout and len(generated) < min_expected:
        fallback_engine = LayoutEngine(
            parking_type=parking_type,
            calibrated=False,
            vehicle_dims=vehicle_dims,
        )
        fallback_generated = fallback_engine.generate_all(
            zones, pixel_map, pinned_slots=pinned_slots
        )
        if len(fallback_generated) > len(generated):
            print(
                "[Main] Calibrated layout produced too few slots "
                f"({len(generated)}<{min_expected}); using pixel fallback "
                f"({len(fallback_generated)} slots)."
            )
            generated = fallback_generated

    # Combine generated + pinned before exit-path filter
    all_slots = generated + (pinned_slots or [])

    # Car width for exit-path width check — from backend dims or fallback
    car_width_px = None
    if vehicle_dims and "width_px" in vehicle_dims:
        car_width_px = vehicle_dims["width_px"]

    all_slots, warnings = filter_slots_by_exit_path(
        all_slots, zones, car_width_px=car_width_px
    )

    # Second guardrail: calibrated geometry can look valid before path
    # filtering but collapse to too few slots after exit-width constraints.
    if use_calibrated_layout and len(all_slots) < min_expected:
        fallback_engine = LayoutEngine(
            parking_type=parking_type,
            calibrated=False,
            vehicle_dims=vehicle_dims,
        )
        fb_generated = fallback_engine.generate_all(
            zones, pixel_map, pinned_slots=pinned_slots
        )
        fb_all_slots = fb_generated + (pinned_slots or [])
        fb_all_slots, fb_warnings = filter_slots_by_exit_path(
            fb_all_slots, zones, car_width_px=car_width_px
        )
        if len(fb_all_slots) > len(all_slots):
            print(
                "[Main] Post-filter calibrated layout too sparse "
                f"({len(all_slots)}<{min_expected}); using pixel fallback "
                f"({len(fb_all_slots)} slots)."
            )
            all_slots = fb_all_slots
            warnings = fb_warnings

    for w in warnings:
        print(w)
    return OccupancyEngine(all_slots)


def _start_api_server():
    """Run the FastAPI server in a background thread."""
    import uvicorn
    from api.server import app
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def _check_parking_events(state, occupancy, detections):
    """
    For each assigned car, check if it has parked (on the correct or
    wrong slot) or stopped, and emit parking_event messages.
    """
    det_map = {d["track_id"]: d for d in detections}

    for tid, assignment in list(state.assignments.items()):
        if assignment["status"].startswith("parked"):
            continue  # already finalized

        assigned_slot_id = assignment["slot"]["slot_id"]

        # Find which slot this car is currently occupying (if any)
        actual_slot = None
        for slot in occupancy.slots:
            if slot.get("assigned_track_id") == tid and slot["status"] == "occupied":
                actual_slot = slot
                break

        if actual_slot is None:
            # Check if car has stopped (not occupying any slot)
            det = det_map.get(tid)
            if det is None:
                continue
            # Car is still moving or not on a slot yet
            assignment["status"] = "moving"
            continue

        now = time.time()
        if actual_slot["slot_id"] == assigned_slot_id:
            assignment["status"] = "parked_correct"
        else:
            assignment["status"] = "parked_incorrect"

        assignment["parked_at"] = now

        state.add_parking_event({
            "event": "parking_event",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tracking_id": tid,
            "status": assignment["status"],
            "assigned_slot": assigned_slot_id,
            "actual_slot": actual_slot["slot_id"],
            "parked_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "duration_seconds": 0,
        })


def main():
    # ── Load spatial config ───────────────────
    calibration = CalibrationMap()
    calibration.load()                           # graceful no-op if not calibrated yet

    zones = ZoneMap()
    zones.load()

    if calibration.is_calibrated and calibration.source_frame_size is None:
        zsrc = zones.source_frame_size
        if zsrc is not None:
            if calibration.adopt_source_frame_size(zsrc[0], zsrc[1]):
                print(
                    "[Main] Legacy calibration missing source_frame_size; "
                    f"adopted zones source size {zsrc[0]}x{zsrc[1]} for scaling."
                )

    layout_use_calibrated = calibration.is_calibrated
    if calibration.is_calibrated and calibration.source_frame_size is None:
        layout_use_calibrated = False
        print(
            "[Main] Calibration loaded but source_frame_size is missing; "
            "disabling calibrated slot sizing to avoid incorrect scaling. "
            "Re-run calibration to restore real-world slot sizing."
        )

    # ── Perception ───────────────────────────
    tracker = VehicleTracker()

    # ── Continuous stopped-car monitor ────────
    stopped_monitor = StoppedCarMonitor(zones)

    parking_type    = PARKING_TYPE
    debug_occupancy = DEBUG_OCCUPANCY

    # ── Shared state for API ─────────────────
    state = ParkingState()
    state.zone_map = zones
    state.calibration = calibration
    state.route_planner = RoutePlanner(zones)
    state.slot_recommender = SlotRecommender(zones, calibration)

    # ── Start API server in background thread ─
    api_thread = threading.Thread(target=_start_api_server, daemon=True)
    api_thread.start()

    cam_source = _resolve_main_camera_source()
    with CameraStream(source=cam_source) as cam:
        if cam.mode != "stream":
            raise RuntimeError(
                f"Main requires MJPEG stream source, got mode={cam.mode} "
                f"for source={cam_source}"
            )
        # ── Create window sized for current frame resolution ──
        cv2.namedWindow("Smart Parking System", cv2.WINDOW_NORMAL)

        # ── Initial layout (no pinned slots yet — monitor needs time) ──
        occupancy = rebuild_layout(parking_type, zones, calibration,
                       VEHICLE_DIMS,
                       use_calibrated_layout=layout_use_calibrated)
        state.set_occupancy(occupancy)
        print(f"[Main] Camera source mode: {cam.mode}")
        if cam.mode == "snapshot":
            print(
                "[Main] Snapshot config: "
                f"interval={getattr(cam, '_snapshot_interval_s', 'n/a')}s, "
                f"http_timeout={getattr(cam, '_http_timeout_s', 'n/a')}s"
            )
        frame_idx = 0
        runtime_geometry_applied = False
        fps_ema = 0.0
        cached_detections = []
        last_layout_rebuild_ts = time.time()
        pending_layout_rebuild = False

        while True:
            start_time = time.time()

            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Keep a clean copy for inference so visual overlays never affect
            # detector/tracker predictions.
            inference_frame = frame.copy()

            # Keep display close to camera resolution for low-res sources.
            if frame_idx == 0:
                h, w = frame.shape[:2]
                cv2.resizeWindow("Smart Parking System", w, h)

            # One-time adaptation: scale calibration/zones from setup resolution
            # (e.g. XGA) to runtime frame resolution (e.g. QVGA).
            if not runtime_geometry_applied:
                h, w = frame.shape[:2]
                cal_changed = calibration.set_runtime_frame_size(w, h)
                zones_changed = zones.set_runtime_frame_size(w, h)
                runtime_geometry_applied = True

                if cal_changed or zones_changed:
                    print(
                        f"[Main] Applied runtime geometry adaptation to {w}x{h} "
                        f"(calibration_changed={cal_changed}, zones_changed={zones_changed})"
                    )
                    # Recreate geometry-dependent components after scaling.
                    stopped_monitor = StoppedCarMonitor(zones)
                    occupancy = rebuild_layout(parking_type, zones, calibration,
                                               VEHICLE_DIMS,
                                               use_calibrated_layout=layout_use_calibrated)
                    state.zone_map = zones
                    state.calibration = calibration
                    state.route_planner = RoutePlanner(zones)
                    state.slot_recommender = SlotRecommender(zones, calibration)
                    state.set_occupancy(occupancy)
                    cached_detections = []

            # ── Detection + tracking ──────────
            frame_idx += 1
            # Apply stride in all modes. In snapshot mode this avoids wasting
            # expensive inference cycles on repeated stale frames during
            # network jitter and improves visible FPS stability.
            should_infer = (frame_idx % DETECTION_STRIDE == 0) or not cached_detections
            if should_infer:
                detections = tracker.track(inference_frame)

                # ── Drop detections in restricted zones ──
                detections = filter_restricted(detections, zones)
                cached_detections = detections
            else:
                detections = cached_detections

            # ── Update shared state for API ───
            state.update_detections(detections)
            state.frame_timestamp = datetime.now(timezone.utc).isoformat()

            # ── Stopped-car monitor (every frame) ──
            if should_infer:
                pins_changed = stopped_monitor.update(detections)
                if pins_changed:
                    pending_layout_rebuild = True

            if pending_layout_rebuild:
                now_ts = time.time()
                if (now_ts - last_layout_rebuild_ts) >= LAYOUT_REBUILD_COOLDOWN_S:
                    pinned = stopped_monitor.pinned_slots()
                    print(
                        f"[Main] Pinned slots changed ({len(pinned)} active) "
                        f"→ rebuilding layout (cooldown={LAYOUT_REBUILD_COOLDOWN_S:.1f}s)"
                    )
                    occupancy = rebuild_layout(parking_type, zones, calibration,
                                               VEHICLE_DIMS, pinned,
                                               use_calibrated_layout=layout_use_calibrated)
                    state.set_occupancy(occupancy)
                    last_layout_rebuild_ts = now_ts
                    pending_layout_rebuild = False

            # ── Occupancy update ──────────────
            if should_infer:
                occupancy.update(detections)

            # ── Check parking events for assigned cars ──
            if should_infer:
                _check_parking_events(state, occupancy, detections)

            # ── Zone overlay (draw only after inference) ───────────────
            if SHOW_ZONES:
                zones.draw(frame, show_labels=SHOW_ZONE_LABELS)

            # ── Slot overlay ──────────────────
            if SHOW_SLOTS:
                occupancy.draw(frame, show_id=SHOW_SLOT_LABELS,
                               debug=debug_occupancy)

            # ── Detection boxes ───────────────
            for det in detections:
                draw_detection(frame, det["bbox"], det["label"],
                               det["confidence"], det["track_id"])

            # ── HUD ───────────────────────────
            loop_fps = 1 / max(time.time() - start_time, 1e-6)
            if fps_ema == 0.0:
                fps_ema = loop_fps
            else:
                fps_ema = (FPS_EMA_ALPHA * loop_fps) + ((1 - FPS_EMA_ALPHA) * fps_ema)
            draw_hud(frame, fps_ema, detections, occupancy.summary(), parking_type)

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
                    stopped_monitor.pinned_slots(),
                    use_calibrated_layout=layout_use_calibrated)
                state.set_occupancy(occupancy)
            elif key == ord("d") or key == ord("D"):
                debug_occupancy = not debug_occupancy
                print(f"[Main] Debug overlay {'ON' if debug_occupancy else 'OFF'}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()