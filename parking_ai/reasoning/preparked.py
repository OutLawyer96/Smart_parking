"""
Stopped-Car Monitor — continuous detection of cars that stop moving.

Unlike the old startup-only ``detect_preparked``, the ``StoppedCarMonitor``
runs *every frame*.  For each tracked vehicle it keeps a centroid history.
When a car's centroid has barely moved for ``PARK_SECONDS`` seconds it is
classified as "parked" and a pinned slot is created around its bbox.  When
it starts moving again (or disappears) the pin is removed.

Any addition / removal of a pinned slot signals the main loop that a
layout rebuild is needed.

Startup-compatible:
   On the very first call the monitor runs normally — if the camera already
   shows stationary cars at startup they will be classified after the
   threshold elapses (no special warm-up logic needed).
"""

import cv2
import time
import numpy as np
from typing import Optional


# ── Tuning knobs ──────────────────────────────────────────────────────────
PARK_SECONDS       = 3.0    # seconds a car must be still → "parked"
MOVE_THRESHOLD_PX  = 8.0    # centroid drift in px within window → still
GONE_SECONDS       = 2.0    # seconds after track disappears → unpin
_SLOT_PAD          = 4      # bbox → polygon padding (px)
_CONFLICT_OVERLAP  = 0.01   # overlap ratio to remove generated slots


class StoppedCarMonitor:
    """
    Frame-by-frame monitor that pins/unpins cars as they stop/start.

    Usage
    -----
    monitor = StoppedCarMonitor(zone_map)

    # each frame:
    rebuild_needed = monitor.update(detections)
    pinned_slots   = monitor.pinned_slots()

    if rebuild_needed:
        occupancy = rebuild_layout(..., pinned_slots=pinned_slots)
    """

    def __init__(self, zone_map, park_seconds=PARK_SECONDS,
                 move_threshold_px=MOVE_THRESHOLD_PX,
                 gone_seconds=GONE_SECONDS):
        self._zone_map      = zone_map
        self._park_secs     = park_seconds
        self._move_thr      = move_threshold_px
        self._gone_secs     = gone_seconds

        # per track_id state
        self._tracks: dict[int, _TrackState] = {}

        # track_ids currently pinned
        self._pinned_ids: set[int] = set()

        # cached pinned slot dicts  {track_id: slot_dict}
        self._pinned_map: dict[int, dict] = {}

    # ── public API ────────────────────────────────────────────────────

    def update(self, detections: list) -> bool:
        """
        Feed tracker detections for the current frame.

        Parameters
        ----------
        detections : list[dict] with 'track_id', 'bbox' (x1,y1,x2,y2)

        Returns
        -------
        True if the set of pinned slots changed since last call
        (signals that a layout rebuild is needed).
        """
        now = time.time()
        seen_ids: set[int] = set()
        changed = False

        for det in detections:
            tid  = det["track_id"]
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            seen_ids.add(tid)

            # ── Check if centroid is in a parking zone ────────────
            zone = self._zone_map.get_zone_at((int(cx), int(cy)))
            in_parking = zone is not None and zone["type"] == "parking"

            if tid not in self._tracks:
                self._tracks[tid] = _TrackState(cx, cy, now)
            else:
                self._tracks[tid].push(cx, cy, now)

            state = self._tracks[tid]

            if in_parking and state.is_still(self._move_thr, self._park_secs, now):
                # Should be pinned
                if tid not in self._pinned_ids:
                    self._pinned_ids.add(tid)
                    self._pinned_map[tid] = _make_pinned_slot(tid, bbox)
                    changed = True
                    print(f"[StoppedCar] Track {tid} pinned "
                          f"(still for >{self._park_secs:.1f}s)")
                else:
                    # Update bbox in case the car shifted slightly
                    self._pinned_map[tid] = _make_pinned_slot(tid, bbox)
            else:
                # Should NOT be pinned (moving or not in parking zone)
                if tid in self._pinned_ids:
                    self._pinned_ids.discard(tid)
                    self._pinned_map.pop(tid, None)
                    changed = True
                    print(f"[StoppedCar] Track {tid} unpinned (moving / left zone)")

        # ── Handle disappeared tracks ─────────────────────────────
        disappeared = set(self._tracks.keys()) - seen_ids
        for tid in disappeared:
            state = self._tracks[tid]
            if state.last_seen is None:
                state.last_seen = now
            elif now - state.last_seen >= self._gone_secs:
                if tid in self._pinned_ids:
                    self._pinned_ids.discard(tid)
                    self._pinned_map.pop(tid, None)
                    changed = True
                    print(f"[StoppedCar] Track {tid} gone → unpinned")
                del self._tracks[tid]

        return changed

    def pinned_slots(self) -> list:
        """Return current list of pinned slot dicts."""
        return list(self._pinned_map.values())


# ─── per-track state ─────────────────────────────────────────────────────

class _TrackState:
    """Centroid history for one tracked vehicle."""

    __slots__ = ("_history", "last_seen")

    def __init__(self, cx: float, cy: float, t: float):
        self._history: list[tuple[float, float, float]] = [(cx, cy, t)]
        self.last_seen: Optional[float] = None   # set when track disappears

    def push(self, cx: float, cy: float, t: float):
        self._history.append((cx, cy, t))
        self.last_seen = None  # still visible
        # Keep only relevant history (last park_seconds * 2 at most)
        cutoff = t - 30.0
        while self._history and self._history[0][2] < cutoff:
            self._history.pop(0)

    def is_still(self, move_thr: float, park_secs: float, now: float) -> bool:
        """
        True if the centroid has stayed within `move_thr` pixels
        for the last `park_secs` seconds.
        """
        if not self._history:
            return False
        cutoff = now - park_secs
        # Find the earliest entry within the window
        window = [h for h in self._history if h[2] >= cutoff]
        if not window:
            return False
        # We need data spanning at least park_secs
        if window[0][2] > cutoff + 0.5:
            # Not enough history yet
            return False
        # Check max drift from the latest position
        latest_cx, latest_cy = window[-1][0], window[-1][1]
        for cx, cy, _ in window:
            dx = cx - latest_cx
            dy = cy - latest_cy
            if (dx * dx + dy * dy) > move_thr * move_thr:
                return False
        return True


# ─── slot-conflict removal (unchanged API) ───────────────────────────────

def remove_conflicting_slots(generated_slots, pinned_slots):
    """
    Remove generated slots that overlap significantly with any pinned slot.
    """
    if not pinned_slots:
        return generated_slots

    pinned_polys = [np.array(s["polygon_px"], dtype=np.int32) for s in pinned_slots]

    kept = []
    for slot in generated_slots:
        if not _overlaps_any_pinned(slot["polygon_px"], pinned_polys):
            kept.append(slot)

    removed = len(generated_slots) - len(kept)
    if removed:
        print(f"[StoppedCar] {removed} generated slot(s) removed "
              f"(overlap with stopped cars)")
    return kept


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_pinned_slot(track_id: int, bbox: tuple) -> dict:
    x1, y1, x2, y2 = bbox
    poly_px = _bbox_to_padded_polygon(x1, y1, x2, y2, _SLOT_PAD)
    area = float((x2 - x1 + 2 * _SLOT_PAD) * (y2 - y1 + 2 * _SLOT_PAD))
    return {
        "slot_id":             f"stopped_T{track_id}",
        "type":                "stopped",
        "status":              "occupied",
        "polygon_world":       [],
        "polygon_px":          poly_px,
        "polygon_px_inflated": poly_px,
        "slot_area_px":        max(area, 1.0),
        "assigned_track_id":   track_id,
        "assigned_user_id":    None,
        "pinned":              True,
    }


def _bbox_to_padded_polygon(x1, y1, x2, y2, pad):
    return [
        (x1 - pad, y1 - pad),
        (x2 + pad, y1 - pad),
        (x2 + pad, y2 + pad),
        (x1 - pad, y2 + pad),
    ]


def _overlaps_any_pinned(slot_poly, pinned_polys):
    slot_pts = np.array(slot_poly, dtype=np.int32)

    x_min = max(int(slot_pts[:, 0].min()) - 1, 0)
    y_min = max(int(slot_pts[:, 1].min()) - 1, 0)
    x_max = int(slot_pts[:, 0].max()) + 2
    y_max = int(slot_pts[:, 1].max()) + 2
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return False

    offset = np.array([x_min, y_min], dtype=np.int32)

    slot_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(slot_mask, [slot_pts - offset], 255)
    slot_area = float(np.count_nonzero(slot_mask))
    if slot_area == 0:
        return False

    for pinned_pts in pinned_polys:
        pinned_shifted = pinned_pts - offset
        pinned_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(pinned_mask, [pinned_shifted], 255)
        inter = float(np.count_nonzero(cv2.bitwise_and(slot_mask, pinned_mask)))
        if (inter / slot_area) >= _CONFLICT_OVERLAP:
            return True

    return False
