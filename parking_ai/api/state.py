"""
Parking State — shared mutable state between the vision loop and the API.

Thread-safe singleton holding current detections, occupancy, zone map,
calibration, and active car assignments.  The vision loop writes to it
every frame; the API reads from it on request.
"""

import threading
import time
from typing import Optional


class ParkingState:
    """Thread-safe container for live parking data."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._data_lock = threading.Lock()

        # Set by the vision loop
        self.zone_map = None
        self.calibration = None
        self.occupancy_engine = None
        self.route_planner = None
        self.slot_recommender = None

        # Current frame detections (from tracker)
        self.detections: list[dict] = []

        # Active assignments: tracking_id -> assignment dict
        self.assignments: dict[int, dict] = {}

        # Parking events (state transitions) — consumed by WS broadcast
        self._pending_events: list[dict] = []

        # Slot status changes — consumed by WS broadcast
        self._pending_slot_updates: list[dict] = []

        # Previous slot states for change detection
        self._prev_slot_status: dict[str, str] = {}

        self.frame_timestamp: str = ""

    def update_detections(self, detections: list[dict]):
        with self._data_lock:
            self.detections = list(detections)

    def get_detections(self) -> list[dict]:
        with self._data_lock:
            return list(self.detections)

    def set_occupancy(self, engine):
        with self._data_lock:
            self.occupancy_engine = engine
            # Detect slot status changes
            if engine:
                for slot in engine.slots:
                    sid = slot["slot_id"]
                    status = slot["status"]
                    prev = self._prev_slot_status.get(sid)
                    if prev and prev != status:
                        self._pending_slot_updates.append({
                            "event": "slot_update",
                            "slot_id": sid,
                            "status": status,
                            "tracking_id": slot.get("assigned_track_id"),
                        })
                    self._prev_slot_status[sid] = status

    def add_assignment(self, tracking_id: int, slot: dict,
                       route: list[dict], vehicle_dims: dict):
        with self._data_lock:
            self.assignments[tracking_id] = {
                "tracking_id": tracking_id,
                "slot": slot,
                "route": route,
                "vehicle_dims": vehicle_dims,
                "assigned_at": time.time(),
                "status": "navigating",
            }

    def get_assignment(self, tracking_id: int) -> Optional[dict]:
        with self._data_lock:
            return self.assignments.get(tracking_id)

    def add_parking_event(self, event: dict):
        with self._data_lock:
            self._pending_events.append(event)

    def drain_events(self) -> list[dict]:
        with self._data_lock:
            events = self._pending_events
            self._pending_events = []
            return events

    def drain_slot_updates(self) -> list[dict]:
        with self._data_lock:
            updates = self._pending_slot_updates
            self._pending_slot_updates = []
            return updates

    def build_world_state(self) -> dict:
        """Build the world_state WS message for all tracked cars."""
        with self._data_lock:
            cars = []
            for det in self.detections:
                x1, y1, x2, y2 = det["bbox"]
                tid = det["track_id"]
                assignment = self.assignments.get(tid)

                # Car polygon (4 corner points of bounding box)
                car_polygon = [
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ]

                car_data = {
                    "tracking_id": tid,
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                    "polygon": car_polygon,
                    "heading_deg": 0.0,
                    "status": "moving",
                    "assigned_slot": None,
                }

                if assignment:
                    car_data["assigned_slot"] = assignment["slot"].get("slot_id")
                    car_data["status"] = assignment["status"]

                    # Check if parked
                    if assignment["status"].startswith("parked"):
                        elapsed = time.time() - assignment.get("parked_at",
                                                               assignment["assigned_at"])
                        car_data["parked_duration_seconds"] = round(elapsed, 1)

                cars.append(car_data)

            return {
                "event": "world_state",
                "timestamp": self.frame_timestamp,
                "cars": cars,
            }

    def build_map_snapshot(self) -> dict:
        """Build the full map JSON for GET /map."""
        with self._data_lock:
            if not self.zone_map or not self.occupancy_engine:
                return {"map": None}

            # Determine map dimensions from all zone points
            all_pts = []
            for z in self.zone_map.all_zones():
                all_pts.extend(z["points"])
            if not all_pts:
                return {"map": None}

            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            width_px = max(xs) + 20
            height_px = max(ys) + 20

            scale = 0.0
            if self.calibration and self.calibration.is_calibrated:
                # Approximate scale: use two points 100px apart
                p1 = self.calibration.to_world((0, 0))
                p2 = self.calibration.to_world((100, 0))
                import math
                dist_cm = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                if dist_cm > 0:
                    scale = (dist_cm / 100.0) / 100.0  # m per px

            # Slots
            parking_slots = []
            for slot in self.occupancy_engine.slots:
                cx = int(sum(p[0] for p in slot["polygon_px"]) / len(slot["polygon_px"]))
                cy = int(sum(p[1] for p in slot["polygon_px"]) / len(slot["polygon_px"]))
                parking_slots.append({
                    "slot_id": slot["slot_id"],
                    "cx": cx,
                    "cy": cy,
                    "polygon": [[int(p[0]), int(p[1])] for p in slot["polygon_px"]],
                    "status": slot["status"],
                })

            # Restricted zones
            restricted_zones = []
            for z in self.zone_map.zones_by_type("restricted"):
                restricted_zones.append({
                    "id": z["id"],
                    "type": "restricted",
                    "polygon": [[int(p[0]), int(p[1])] for p in z["points"]],
                })

            # Entry/exit
            entry_exit = []
            for z in self.zone_map.zones_by_type("exit"):
                entry_exit.append({
                    "id": z["id"],
                    "type": "exit",
                    "polygon": [[int(p[0]), int(p[1])] for p in z["points"]],
                })

            # Driveways
            driveways = []
            for z in self.zone_map.zones_by_type("drive"):
                driveways.append({
                    "id": z["id"],
                    "polygon": [[int(p[0]), int(p[1])] for p in z["points"]],
                })

            return {
                "map": {
                    "width_px": width_px,
                    "height_px": height_px,
                    "scale_m_per_px": round(scale, 6),
                    "layers": {
                        "parking_slots": parking_slots,
                        "restricted_zones": restricted_zones,
                        "entry_exit": entry_exit,
                        "driveways": driveways,
                    }
                }
            }

    def build_slots_summary(self) -> dict:
        """Build lightweight slot status list for GET /slots."""
        with self._data_lock:
            if not self.occupancy_engine:
                return {"slots": []}
            slots = []
            for s in self.occupancy_engine.slots:
                slots.append({
                    "slot_id": s["slot_id"],
                    "status": s["status"],
                    "tracking_id": s.get("assigned_track_id"),
                })
            return {"slots": slots}
