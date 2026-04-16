"""
ZoneMap — runtime zone loader and query utility.

Loads zones saved by ZoneEditor and provides:
  - point-in-zone lookups  (cv2.pointPolygonTest)
  - zone overlay drawing on frames
  - organized access by zone type

Usage:
    from spatial.zone_map import ZoneMap

    zones = ZoneMap()
    zones.load("config/zones.json")

    zone = zones.get_zone_at((cx, cy))   # None or zone dict
    is_parking = zones.is_in_type((cx, cy), "parking")

    zones.draw(frame)
"""

import cv2
import json
import numpy as np
import os
from typing import Optional

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(_ROOT_DIR, "config", "zones.json")

ZONE_COLORS = {
    "parking":    (0,   210,  60),
    "drive":      (30,  160, 255),
    "restricted": (0,    50, 220),
    "exit":       (255, 150,  50),
}

FILL_ALPHA = 0.20


class ZoneMap:
    def __init__(self, config_path: str = CONFIG_PATH):
        self._zones_raw: list[dict] = []
        self._zones: list[dict] = []
        self._polygons: list[np.ndarray] = []  # parallel to _zones
        self.config_path = config_path
        self.unlabeled_is_restricted: bool = True  # default: unlabeled = restricted
        self._source_frame_size: Optional[tuple[int, int]] = None  # (w, h)
        self._runtime_frame_size: Optional[tuple[int, int]] = None  # (w, h)

    def _rebuild_polygons(self):
        self._polygons = [
            np.array(z["points"], dtype=np.int32) for z in self._zones
        ]

    def _apply_runtime_scaling(self):
        if not self._source_frame_size or not self._runtime_frame_size:
            self._zones = [
                {
                    "id": z["id"],
                    "type": z["type"],
                    "name": z["name"],
                    "points": [list(p) for p in z["points"]],
                }
                for z in self._zones_raw
            ]
            self._rebuild_polygons()
            return

        src_w, src_h = self._source_frame_size
        run_w, run_h = self._runtime_frame_size
        sx = float(run_w) / float(src_w)
        sy = float(run_h) / float(src_h)

        scaled = []
        for z in self._zones_raw:
            pts = []
            for p in z["points"]:
                x = int(round(float(p[0]) * sx))
                y = int(round(float(p[1]) * sy))
                pts.append([x, y])
            scaled.append(
                {
                    "id": z["id"],
                    "type": z["type"],
                    "name": z["name"],
                    "points": pts,
                }
            )

        self._zones = scaled
        self._rebuild_polygons()

    # ── load ─────────────────────────────────

    def load(self, path: Optional[str] = None):
        path = path or self.config_path
        if not os.path.exists(path):
            print(f"[ZoneMap] No zone config found at {path}. Zones will be empty.")
            return

        with open(path) as f:
            data = json.load(f)

        self._zones_raw = data.get("zones", [])

        src = data.get("source_frame_size")
        if isinstance(src, dict) and "width" in src and "height" in src:
            self._source_frame_size = (int(src["width"]), int(src["height"]))
        else:
            self._source_frame_size = None

        self._apply_runtime_scaling()
        self.unlabeled_is_restricted: bool = data.get("unlabeled_is_restricted", True)
        print(f"[ZoneMap] Loaded {len(self._zones)} zones from {path}  "
              f"(unlabeled={'restricted' if self.unlabeled_is_restricted else 'ignored'})")

    def set_runtime_frame_size(self, width: int, height: int) -> bool:
        runtime = (int(width), int(height))
        if self._runtime_frame_size == runtime:
            return False
        self._runtime_frame_size = runtime
        self._apply_runtime_scaling()
        return True

    # ── query ─────────────────────────────────

    # Synthetic zone returned for points outside all defined zones
    _UNLABELED_RESTRICTED_ZONE = {
        "id":   "unlabeled",
        "type": "restricted",
        "name": "unlabeled",
    }

    def get_zone_at(self, point: tuple) -> Optional[dict]:
        """
        Returns the zone whose polygon contains `point`.
        Zones are checked in reverse order so that later (top-drawn)
        zones take priority over earlier ones.
        If the point is outside all defined zones and unlabeled_is_restricted
        is True (default), returns a synthetic restricted zone instead of None.
        `point` is (x, y) pixel coordinates.
        """
        for zone, poly in zip(reversed(self._zones), reversed(self._polygons)):
            result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
            if result >= 0:
                return zone
        if self.unlabeled_is_restricted:
            return self._UNLABELED_RESTRICTED_ZONE
        return None

    def is_in_type(self, point: tuple, zone_type: str) -> bool:
        """True if the highest-priority zone at point matches zone_type."""
        zone = self.get_zone_at(point)
        if zone is None:
            return False
        return zone["type"] == zone_type

    def zones_by_type(self, zone_type: str) -> list[dict]:
        """Return all zones of a given type."""
        return [z for z in self._zones if z["type"] == zone_type]

    def all_zones(self) -> list[dict]:
        return list(self._zones)

    # ── draw ──────────────────────────────────

    def draw(self, frame: np.ndarray, show_labels: bool = True) -> np.ndarray:
        """
        Draw all zones onto the frame (semi-transparent fill + crisp border).
        Modifies in place and also returns the frame.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Keep zone tint subtle so parking-area overlays do not wash out the scene.
        alpha = 0.16 if w <= 360 else FILL_ALPHA
        # Increase line thickness at low resolution for visibility
        thickness = 2 if w <= 360 else 2
        label_scale = 0.35 if w <= 360 else 0.5

        for zone, poly in zip(self._zones, self._polygons):
            # Parking usually spans most of the frame; drawing it filled causes
            # full-screen green tint and visual blinking with slot overlays.
            if zone.get("type") == "parking":
                continue
            color = ZONE_COLORS.get(zone["type"], (200, 200, 200))
            cv2.fillPoly(overlay, [poly], color)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        for zone, poly in zip(self._zones, self._polygons):
            color = ZONE_COLORS.get(zone["type"], (200, 200, 200))
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=thickness)

            if show_labels:
                cx = int(np.mean(poly[:, 0]))
                cy = int(np.mean(poly[:, 1]))
                _put_label(frame, zone["name"], (cx, cy), color, scale=label_scale)

        return frame

    def __len__(self):
        return len(self._zones)

    def __bool__(self):
        return len(self._zones) > 0

    @property
    def source_frame_size(self) -> Optional[tuple[int, int]]:
        return self._source_frame_size


# ─── utility ──────────────────────────────────

def _put_label(frame, text, center, color, scale=0.5):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cx, cy = center
    cv2.rectangle(frame,
                  (cx - tw // 2 - 3, cy - th - 3),
                  (cx + tw // 2 + 3, cy + 3),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
