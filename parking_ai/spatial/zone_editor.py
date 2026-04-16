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
import requests
import time
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

# Zone type definitions
ZONE_TYPES = ["parking", "drive", "restricted", "exit"]

ZONE_COLORS = {
    "parking":    (0,   210,  60),   # green
    "drive":      (30,  160, 255),   # orange-ish
    "restricted": (0,    50, 220),   # red
    "exit":       (255, 150,  50),   # blue
}

FILL_ALPHA = 0.30   # how transparent the polygon fill is
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(_ROOT_DIR, "config", "zones.json")
DEFAULT_CAPTURE_URL = "http://10.54.215.196/capture"


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
            "source_frame_size": {
                "width": int(self.background.shape[1]),
                "height": int(self.background.shape[0]),
            },
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
        panel_w = max(150, min(220, int(w * 0.58)))
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        compact = w <= 360
        head_scale = 0.45 if compact else 0.55
        text_scale = 0.35 if compact else 0.45
        row_h = 17 if compact else 22

        color = ZONE_COLORS[self.current_type]
        lines = [
            ("ZONE EDITOR",      (200, 200, 200), head_scale, 1 if compact else 2),
            ("",                 None,            text_scale, 1),
            (f"Type: {self.current_type}", color, head_scale, 1 if compact else 2),
            (f"Vertices: {len(self.current_points)}", (200, 200, 200), text_scale, 1),
            (f"Zones: {len(self.zones)}",             (200, 200, 200), text_scale, 1),
            ("",                 None,            text_scale, 1),
            ("LEGEND",           (160, 160, 160), text_scale, 1),
        ]
        for zt in ZONE_TYPES:
            lines.append((f"- {zt}", ZONE_COLORS[zt], text_scale, 1))

        lines += [
            ("",              None,            text_scale, 1),
            ("CONTROLS",      (160, 160, 160), text_scale, 1),
            ("[T] type",      (180, 180, 180), text_scale, 1),
            ("[C] close",     (180, 180, 180), text_scale, 1),
            ("[Z] undo",      (180, 180, 180), text_scale, 1),
            ("[D] delete",    (180, 180, 180), text_scale, 1),
            ("[R] rename",    (180, 180, 180), text_scale, 1),
            ("[S] save",      (180, 180, 180), text_scale, 1),
            ("[Q] save+quit", (180, 180, 180), text_scale, 1),
        ]

        y = 20 if compact else 24
        for text, col, scale, thickness in lines:
            if text and col:
                cv2.putText(panel, text, (8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, col, thickness)
            y += row_h

        # Panel is appended in run(); keep frame geometry untouched.
        self._last_panel = panel

    # ── main loop ────────────────────────────

    def run(self):
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._win, self._mouse_cb)

        first = True
        while True:
            frame = self._draw()
            display = np.hstack([frame, self._last_panel]) if hasattr(self, "_last_panel") else frame
            if first:
                cv2.resizeWindow(self._win, display.shape[1], display.shape[0])
                first = False
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
    base_url = os.getenv("ESP32_CAPTURE_URL") or DEFAULT_CAPTURE_URL
    print(f"[editor] Fetching one snapshot from {base_url} for background...")
    require_hq = os.getenv("ZONE_EDITOR_REQUIRE_HQ", "1") == "1"
    min_w = int(os.getenv("ZONE_EDITOR_MIN_HQ_WIDTH", "640"))
    min_h = int(os.getenv("ZONE_EDITOR_MIN_HQ_HEIGHT", "480"))

    def _set_query_param(url: str, key: str, value: str) -> str:
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query[key] = value
        return urlunparse(parsed._replace(query=urlencode(query)))

    def _candidate_urls(url: str) -> list[str]:
        # Prefer HQ snapshots because zones source_frame_size is persisted and
        # used to scale zones onto runtime low-res streams.
        hq_first = [
            _set_query_param(url, "mode", "zone_editor"),
            _set_query_param(url, "mode", "calibration"),
        ]
        if require_hq:
            return hq_first
        return hq_first + [
            _set_query_param(url, "mode", "normal"),
            _set_query_param(url, "mode", "stream"),
            url,
        ]

    candidates = _candidate_urls(base_url)
    last_error = ""
    saw_busy = False
    for snap_url in candidates:
        for attempt in range(3):
            ts = int(time.time() * 1000)
            probe_url = f"{snap_url}{'&' if '?' in snap_url else '?'}_ts={ts}"
            try:
                # Warm-up call helps mode changes settle before actual decode.
                requests.get(probe_url, timeout=3.0, headers={"Cache-Control": "no-cache"})
            except requests.RequestException:
                pass

            time.sleep(0.14)
            try:
                resp = requests.get(
                    probe_url,
                    timeout=6.0,
                    headers={"Cache-Control": "no-cache"},
                )
            except requests.RequestException as e:
                last_error = f"{snap_url} request error: {e}"
                continue

            if resp.status_code != 200:
                last_error = f"{snap_url} HTTP {resp.status_code}"
                if resp.status_code == 503:
                    saw_busy = True
                    time.sleep(0.20)
                continue

            arr = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                last_error = f"{snap_url} decode failed"
                continue

            if require_hq:
                h, w = frame.shape[:2]
                if w < min_w or h < min_h:
                    last_error = (
                        f"{snap_url} returned low-res frame {w}x{h}; "
                        f"expected at least {min_w}x{min_h}"
                    )
                    continue

            print(f"[editor] Frame captured from {snap_url}.")
            return frame

    if saw_busy:
        print(
            "[editor] ESP32 reported camera busy (HTTP 503) for one or more "
            "HQ snapshot attempts. This is expected while /stream is active on "
            "the new firmware."
        )
    raise RuntimeError(f"Snapshot request failed after fallbacks. Last error: {last_error}")


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
