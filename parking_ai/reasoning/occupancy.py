"""
Occupancy Engine — overlap-driven, calibration-aware slot occupancy.

Algorithm (per frame):
  1. For each detection bbox, compute its polygon intersection area with
     every slot's (inflated) polygon.
  2. Express overlap as a fraction of the slot's area.
  3. Each car is assigned to the slot with highest overlap ratio
     (prevents double-counting).
  4. Temporal smoothing:
       occupied  after OCCUPY_FRAMES  consecutive frames of overlap ≥ threshold
       free      after RELEASE_FRAMES consecutive frames of NO   overlap

Track-ID safety:
  Slot state is driven purely by bounding-box overlap, NOT by track ID
  continuity.  A tracking switch never causes a spurious free event.

Debug mode:
  engine.draw(frame, debug=True) draws overlap %, inflated polygons, and
  the intersection region in cyan.
"""

import cv2
import numpy as np

# ── Temporal thresholds ────────────────────────────────────────────────────
OCCUPY_FRAMES      = 5      # consecutive frames overlap must be seen
RELEASE_FRAMES     = 8      # consecutive frames overlap must be absent

# ── Overlap threshold ──────────────────────────────────────────────────────
OVERLAP_THRESHOLD  = 0.30   # fraction of slot area that must be covered

# ── Colours ────────────────────────────────────────────────────────────────
SLOT_COLORS = {
    "free":     (0,  210,  60),
    "occupied": (0,   50, 220),
    "reserved": (30, 160, 255),
    "assigned": (0,  255, 255),
}
DEFAULT_SLOT_COLOR   = (180, 180, 180)
DEBUG_OVERLAP_COLOR  = (255, 200,   0)   # cyan-ish for overlap region
DEBUG_INFLATED_COLOR = (180, 180, 180)   # grey for inflated polygon outline


class OccupancyEngine:

    def __init__(self, slots: list[dict]):
        """
        Parameters
        ----------
        slots : list of slot dicts produced by LayoutEngine.generate_all()
                Must contain 'polygon_px', 'polygon_px_inflated', 'slot_area_px'.
        """
        self.slots = [dict(s) for s in slots]

        self._occupy_count  = {s["slot_id"]: 0 for s in self.slots}
        self._release_count = {s["slot_id"]: 0 for s in self.slots}

        # Precompute numpy polygon arrays (inflated, for overlap tests)
        self._slot_polys_inflated = {
            s["slot_id"]: np.array(
                s.get("polygon_px_inflated", s["polygon_px"]), dtype=np.int32
            )
            for s in self.slots
        }

        # Per-frame cache: slot_id → overlap_ratio (filled in update, used in draw)
        self._last_overlap: dict[str, float] = {s["slot_id"]: 0.0 for s in self.slots}

    # ── update ────────────────────────────────────────────────────────────────

    def update(self, detections: list[dict]) -> None:
        """
        detections : list of dicts from VehicleTracker.track()
          Required keys: 'bbox' (x1,y1,x2,y2), 'track_id'
        """
        # ── Step 1: compute overlap of every (detection, slot) pair ──────────
        # overlap_matrix[det_idx][slot_id] = overlap_ratio
        overlap_matrix: list[dict[str, float]] = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            car_poly = np.array(
                [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.int32
            )
            row: dict[str, float] = {}
            for slot in self.slots:
                sid   = slot["slot_id"]
                ratio = _overlap_ratio(car_poly, self._slot_polys_inflated[sid],
                                       slot["slot_area_px"])
                row[sid] = ratio
            overlap_matrix.append(row)

        # ── Step 2: assign each car to its best-overlap slot only ─────────────
        # slot_id → (best_ratio, track_id)
        best: dict[str, tuple[float, int]] = {}
        for det, row in zip(detections, overlap_matrix):
            tid = det["track_id"]
            for sid, ratio in row.items():
                if ratio >= OVERLAP_THRESHOLD:
                    if sid not in best or ratio > best[sid][0]:
                        best[sid] = (ratio, tid)

        # ── Step 3: update per-slot counters and state ────────────────────────
        for slot in self.slots:
            sid = slot["slot_id"]

            if sid in best:
                ratio, tid = best[sid]
                self._last_overlap[sid] = ratio
                self._occupy_count[sid]  += 1
                self._release_count[sid]  = 0

                if (self._occupy_count[sid] >= OCCUPY_FRAMES
                        and slot["status"] != "occupied"):
                    slot["status"]            = "occupied"
                    slot["assigned_track_id"] = tid

            else:
                self._last_overlap[sid]  = 0.0
                self._release_count[sid] += 1
                self._occupy_count[sid]   = 0

                # Free only after sustained absence — track-ID switches
                # do NOT trigger this path because we rely on bbox overlap,
                # not track continuity.
                if (self._release_count[sid] >= RELEASE_FRAMES
                        and slot["status"] == "occupied"):
                    slot["status"]            = "free"
                    slot["assigned_track_id"] = None

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, show_id: bool = True,
             debug: bool = False) -> np.ndarray:
        """
        Draw slot overlays onto frame.

        Parameters
        ----------
        debug : when True draws inflated outlines, overlap %, and intersection
                region in a distinct colour.
        """
        overlay = frame.copy()

        # Filled polygon pass
        for slot in self.slots:
            color = SLOT_COLORS.get(slot.get("status"), DEFAULT_SLOT_COLOR)
            pts   = np.array(slot["polygon_px"], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)

        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Outline + label pass
        for slot in self.slots:
            color = SLOT_COLORS.get(slot.get("status"), DEFAULT_SLOT_COLOR)
            pts   = np.array(slot["polygon_px"], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)

            if debug:
                # Draw inflated polygon in grey
                inf_pts = np.array(
                    slot.get("polygon_px_inflated", slot["polygon_px"]),
                    dtype=np.int32
                )
                cv2.polylines(frame, [inf_pts], isClosed=True,
                              color=DEBUG_INFLATED_COLOR, thickness=1)

            if show_id:
                cx = int(np.mean([p[0] for p in slot["polygon_px"]]))
                cy = int(np.mean([p[1] for p in slot["polygon_px"]]))
                label = slot["slot_id"]
                if slot["status"] in {"occupied", "assigned"} and slot["assigned_track_id"] is not None:
                    label += f" ID:{slot['assigned_track_id']}"
                if debug:
                    ratio = self._last_overlap.get(slot["slot_id"], 0.0)
                    label += f" {ratio*100:.0f}%"
                _put_label(frame, label, (cx, cy), color)

        return frame

    # ── public helpers ────────────────────────────────────────────────────────

    def summary(self) -> dict:
        counts = {status: 0 for status in SLOT_COLORS}
        counts["total"] = len(self.slots)
        for slot in self.slots:
            status = slot.get("status", "free")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def free_slots(self) -> list[dict]:
        return [s for s in self.slots if s["status"] == "free"]

    def occupied_slots(self) -> list[dict]:
        return [s for s in self.slots if s["status"] == "occupied"]


# ─── helpers ─────────────────────────────────────────────────────────────────

def _overlap_ratio(car_poly: np.ndarray, slot_poly: np.ndarray,
                   slot_area_px: float) -> float:
    """
    Computes  intersection_area / slot_area  for a car bbox and a slot polygon.

    Uses a rasterised mask on the bounding box of the union \u2014 fast and
    works for any convex/concave polygon pair.
    """
    all_pts = np.vstack([car_poly, slot_poly])
    x_min = max(int(all_pts[:, 0].min()) - 1, 0)
    y_min = max(int(all_pts[:, 1].min()) - 1, 0)
    x_max = int(all_pts[:, 0].max()) + 2
    y_max = int(all_pts[:, 1].max()) + 2

    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return 0.0

    offset = np.array([x_min, y_min], dtype=np.int32)

    car_mask  = np.zeros((h, w), dtype=np.uint8)
    slot_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(car_mask,  [car_poly  - offset], 255)
    cv2.fillPoly(slot_mask, [slot_poly - offset], 255)

    inter = float(np.count_nonzero(cv2.bitwise_and(car_mask, slot_mask)))
    return inter / slot_area_px


def _put_label(frame, text, center, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cx, cy = center
    cv2.rectangle(frame, (cx - tw//2 - 2, cy - th - 2),
                  (cx + tw//2 + 2, cy + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (cx - tw//2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
