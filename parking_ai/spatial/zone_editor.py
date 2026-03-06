"""
Zone Editor — interactive tool to visually define parking zones.

Usage:
    python -m spatial.zone_editor                  # grab frame from camera
    python -m spatial.zone_editor --image path.jpg # use a saved image

Controls:
    Left click      — add vertex to current zone
    C / Enter       — close & save current polygon (min 3 points)
    T               — cycle zone type  (parking → drive → restricted)
    Z / Backspace   — undo last vertex
    D               — delete last completed zone
    R               — rename last completed zone (enter name in terminal)
    S               — save zones to config/zones.json
    Q               — save and quit
"""

import cv2
import json
import numpy as np
import os
import argparse
import sys

# Zone type definitions
ZONE_TYPES = ["parking", "drive", "restricted", "exit"]

ZONE_COLORS = {
    "parking":    (0,   210,  60),   # green
    "drive":      (30,  160, 255),   # orange-ish
    "restricted": (0,    50, 220),   # red
    "exit":       (255, 150,  50),   # blue
}

FILL_ALPHA = 0.30   # how transparent the polygon fill is
CONFIG_PATH = os.path.join("config", "zones.json")


# ─────────────────────────────────────────────
class ZoneEditor:
    def __init__(self, background: np.ndarray):
        self.background = background.copy()
        self.zones: list[dict] = []

        self.current_points: list[tuple] = []
        self.current_type = "parking"
        self.mouse_pos = (0, 0)

        self._zone_counter = 0
        self._win = "Zone Editor  |  T=type  C=close  Z=undo  D=delete  S=save  Q=quit"

    # ── internal helpers ──────────────────────

    def _new_zone_name(self) -> str:
        same_type = [z for z in self.zones if z["type"] == self.current_type]
        return f"{self.current_type}_{len(same_type)}"

    def _close_current(self):
        if len(self.current_points) < 3:
            print("[editor] Need at least 3 points to close a zone.")
            return
        zone = {
            "id":     f"zone_{self._zone_counter}",
            "type":   self.current_type,
            "name":   self._new_zone_name(),
            "points": list(self.current_points),
        }
        self._zone_counter += 1
        self.zones.append(zone)
        self.current_points = []
        print(f"[editor] Zone '{zone['name']}' ({zone['type']}) saved  ({len(self.zones)} total)")

    def _delete_last(self):
        if self.zones:
            removed = self.zones.pop()
            print(f"[editor] Deleted zone '{removed['name']}'")

    def _rename_last(self):
        if not self.zones:
            return
        name = input(f"New name for '{self.zones[-1]['name']}': ").strip()
        if name:
            self.zones[-1]["name"] = name
            print(f"[editor] Renamed to '{name}'")

    def _cycle_type(self):
        idx = ZONE_TYPES.index(self.current_type)
        self.current_type = ZONE_TYPES[(idx + 1) % len(ZONE_TYPES)]
        print(f"[editor] Zone type → {self.current_type}")

    # ── save / load ───────────────────────────

    def save(self, path: str = CONFIG_PATH):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "unlabeled_is_restricted": True,
            "zones": [
                {
                    "id":     z["id"],
                    "type":   z["type"],
                    "name":   z["name"],
                    "points": z["points"],
                }
                for z in self.zones
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[editor] Saved {len(self.zones)} zones → {path}  (unlabeled areas = restricted)")
        self._validate_exits()

    def load(self, path: str = CONFIG_PATH):
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.zones = data.get("zones", [])
        self._zone_counter = len(self.zones)
        print(f"[editor] Loaded {len(self.zones)} zones from {path}")

    def _validate_exits(self):
        """Warn if any parking zone has no exit zone touching it."""
        parking = [z for z in self.zones if z["type"] == "parking"]
        exits   = [z for z in self.zones if z["type"] == "exit"]

        if parking and not exits:
            print("[editor] WARNING: No exit zones (blue) defined!")
            print("[editor]   Draw exit zones touching each parking zone.")
            return

        for pz in parking:
            pz_pts = np.array(pz["points"], dtype=np.float32)
            has_exit = False
            for ez in exits:
                # Check if any exit vertex is inside/near this parking zone
                for pt in ez["points"]:
                    dist = cv2.pointPolygonTest(
                        pz_pts, (float(pt[0]), float(pt[1])), True)
                    if dist >= -15:          # inside or within 15 px
                        has_exit = True
                        break
                if has_exit:
                    break
                # Reverse: parking vertex inside/near exit zone
                ez_pts = np.array(ez["points"], dtype=np.float32)
                for pt in pz["points"]:
                    dist = cv2.pointPolygonTest(
                        ez_pts, (float(pt[0]), float(pt[1])), True)
                    if dist >= -15:
                        has_exit = True
                        break
                if has_exit:
                    break

            if not has_exit:
                print(f"[editor] WARNING: '{pz['name']}' has no exit zone!")
                print(f"[editor]   Draw an exit zone (blue) touching it.")

    # ── drawing ───────────────────────────────

    def _draw(self) -> np.ndarray:
        frame = self.background.copy()
        h, w = frame.shape[:2]

        # ── build a single color overlay:
        #    start entirely red (unlabeled = restricted),
        #    then paint each defined zone with its own color.
        #    result: only truly unlabeled pixels stay red.
        color_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        color_overlay[:] = ZONE_COLORS["restricted"]          # whole canvas = red

        for zone in self.zones:
            color = ZONE_COLORS[zone["type"]]
            pts = np.array(zone["points"], np.int32)
            cv2.fillPoly(color_overlay, [pts], color)          # zone punches out red

        # ── also fill in-progress polygon with current type color
        if len(self.current_points) >= 3:
            color = ZONE_COLORS[self.current_type]
            pts = np.array(self.current_points, np.int32)
            cv2.fillPoly(color_overlay, [pts], color)

        cv2.addWeighted(color_overlay, 0.40, frame, 0.60, 0, frame)

        # ── crisp borders + labels on top of blend
        for zone in self.zones:
            color = ZONE_COLORS[zone["type"]]
            pts = np.array(zone["points"], np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            cx = int(np.mean([p[0] for p in zone["points"]]))
            cy = int(np.mean([p[1] for p in zone["points"]]))
            _put_label(frame, zone["name"], (cx, cy), color)

        # ── in-progress polygon
        if self.current_points:
            color = ZONE_COLORS[self.current_type]
            pts = self.current_points

            # drawn segments
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], color, 2)

            # live preview line from last point to mouse
            cv2.line(frame, pts[-1], self.mouse_pos, color, 1, cv2.LINE_AA)

            # closing preview line (first point back to mouse) when ≥3 pts
            if len(pts) >= 3:
                cv2.line(frame, pts[0], self.mouse_pos, color, 1, cv2.LINE_AA)

            # vertex dots
            for pt in pts:
                cv2.circle(frame, pt, 5, color, -1)
            cv2.circle(frame, pts[0], 7, (255, 255, 255), 2)  # highlight first

        # ── HUD panel
        self._draw_hud(frame)

        return frame

    def _draw_hud(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        panel_w = 260
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        color = ZONE_COLORS[self.current_type]
        lines = [
            ("ZONE EDITOR",      (200, 200, 200), 0.55, 2),
            ("",                 None,            0.4,  1),
            (f"Type: {self.current_type}", color, 0.55, 2),
            (f"Vertices: {len(self.current_points)}", (200, 200, 200), 0.45, 1),
            (f"Zones: {len(self.zones)}",             (200, 200, 200), 0.45, 1),
            ("",                 None,            0.4,  1),
            ("── LEGEND ──",     (160, 160, 160), 0.45, 1),
        ]
        for zt in ZONE_TYPES:
            lines.append((f"  {zt}", ZONE_COLORS[zt], 0.45, 1))

        lines += [
            ("",              None,            0.4, 1),
            ("── CONTROLS ──", (160, 160, 160), 0.45, 1),
            ("[T]  cycle type",     (180, 180, 180), 0.4, 1),
            ("[C]  close zone",     (180, 180, 180), 0.4, 1),
            ("[Z]  undo vertex",    (180, 180, 180), 0.4, 1),
            ("[D]  delete last",    (180, 180, 180), 0.4, 1),
            ("[R]  rename last",    (180, 180, 180), 0.4, 1),
            ("[S]  save",           (180, 180, 180), 0.4, 1),
            ("[Q]  save & quit",    (180, 180, 180), 0.4, 1),
        ]

        y = 24
        for text, col, scale, thickness in lines:
            if text and col:
                cv2.putText(panel, text, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, col, thickness)
            y += 22

        combined = np.hstack([frame, panel])
        frame[:] = combined[:h, :w]
        # paste panel beside frame
        frame_with_hud = np.hstack([frame, panel])
        frame[:, :] = frame_with_hud[:h, :w]

        # actually just write to a wider display — handled in run()
        self._last_panel = panel

    # ── main loop ────────────────────────────

    def run(self):
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._win, self._mouse_cb)

        while True:
            frame = self._draw()
            display = np.hstack([frame, self._last_panel]) if hasattr(self, "_last_panel") else frame
            cv2.imshow(self._win, display)

            key = cv2.waitKey(30) & 0xFF

            if key in (ord("c"), 13):       # C or Enter
                self._close_current()
            elif key == ord("t"):
                self._cycle_type()
            elif key in (ord("z"), 8):      # Z or Backspace
                if self.current_points:
                    self.current_points.pop()
            elif key == ord("d"):
                self._delete_last()
            elif key == ord("r"):
                self._rename_last()
            elif key == ord("s"):
                self.save()
            elif key == ord("q"):
                self.save()
                break

        cv2.destroyAllWindows()

    def _mouse_cb(self, event, x, y, flags, param):
        # x is offset by panel width in display — keep coords relative to frame
        frame_w = self.background.shape[1]
        if x < frame_w:
            self.mouse_pos = (x, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_points.append((x, y))


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


# ─── entry point ──────────────────────────────

def _grab_frame_from_camera() -> np.ndarray:
    from perception.camera import CameraStream
    print("[editor] Connecting to camera to capture background frame...")
    with CameraStream() as cam:
        for _ in range(10):          # skip first few frames (auto-exposure settle)
            frame = cam.read()
        if frame is None:
            raise RuntimeError("Could not read frame from camera.")
    print("[editor] Frame captured.")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Zone Editor for Smart Parking")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a background image. If omitted, grabs a frame from camera.")
    parser.add_argument("--load", action="store_true",
                        help="Load existing zones from config/zones.json on startup.")
    args = parser.parse_args()

    if args.image:
        bg = cv2.imread(args.image)
        if bg is None:
            print(f"[editor] Could not load image: {args.image}")
            sys.exit(1)
    else:
        bg = _grab_frame_from_camera()

    editor = ZoneEditor(bg)

    if args.load:
        editor.load()

    editor.run()
