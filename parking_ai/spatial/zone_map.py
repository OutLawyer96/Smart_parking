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

CONFIG_PATH = os.path.join("config", "zones.json")

ZONE_COLORS = {
    "parking":    (0,   210,  60),
    "drive":      (30,  160, 255),
    "restricted": (0,    50, 220),
}

FILL_ALPHA = 0.20


class ZoneMap:
    def __init__(self, config_path: str = CONFIG_PATH):
        self._zones: list[dict] = []
        self._polygons: list[np.ndarray] = []  # parallel to _zones
        self.config_path = config_path
        self.unlabeled_is_restricted: bool = True  # default: unlabeled = restricted

    # ── load ─────────────────────────────────

    def load(self, path: Optional[str] = None):
        path = path or self.config_path
        if not os.path.exists(path):
            print(f"[ZoneMap] No zone config found at {path}. Zones will be empty.")
            return

        with open(path) as f:
            data = json.load(f)

        self._zones = data.get("zones", [])
        self._polygons = [
            np.array(z["points"], dtype=np.int32) for z in self._zones
        ]
        self.unlabeled_is_restricted: bool = data.get("unlabeled_is_restricted", True)
        print(f"[ZoneMap] Loaded {len(self._zones)} zones from {path}  "
              f"(unlabeled={'restricted' if self.unlabeled_is_restricted else 'ignored'})")

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
        If the point is outside all defined zones and unlabeled_is_restricted
        is True (default), returns a synthetic restricted zone instead of None.
        `point` is (x, y) pixel coordinates.
        """
        for zone, poly in zip(self._zones, self._polygons):
            result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
            if result >= 0:
                return zone
        if self.unlabeled_is_restricted:
            return self._UNLABELED_RESTRICTED_ZONE
        return None

    def is_in_type(self, point: tuple, zone_type: str) -> bool:
        """True if point lies inside any zone of the given type."""
        for zone, poly in zip(self._zones, self._polygons):
            if zone["type"] != zone_type:
                continue
            result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
            if result >= 0:
                return True
        # unlabeled area counts as restricted
        if zone_type == "restricted" and self.unlabeled_is_restricted:
            return True
        return False

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

        for zone, poly in zip(self._zones, self._polygons):
            color = ZONE_COLORS.get(zone["type"], (200, 200, 200))
            cv2.fillPoly(overlay, [poly], color)

        cv2.addWeighted(overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0, frame)

        for zone, poly in zip(self._zones, self._polygons):
            color = ZONE_COLORS.get(zone["type"], (200, 200, 200))
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

            if show_labels:
                cx = int(np.mean(poly[:, 0]))
                cy = int(np.mean(poly[:, 1]))
                _put_label(frame, zone["name"], (cx, cy), color)

        return frame

    def __len__(self):
        return len(self._zones)

    def __bool__(self):
        return len(self._zones) > 0


# ─── utility ──────────────────────────────────

def _put_label(frame, text, center, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cx, cy = center
    cv2.rectangle(frame,
                  (cx - tw // 2 - 3, cy - th - 3),
                  (cx + tw // 2 + 3, cy + 3),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
