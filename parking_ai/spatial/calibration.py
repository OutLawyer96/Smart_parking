"""
Calibration — maps pixels ↔ real-world coordinates (cm).

Uses a 4-point homography computed from known ground markers.

**Perspective compensation**:
  The homography (3×3 matrix) computed by ``cv2.findHomography`` is a full
  *projective* transform.  This inherently corrects for any camera tilt,
  rotation, or off-axis mounting — it is NOT limited to top-down views.
  As long as the 4 calibration points are accurately placed, all pixel↔world
  conversions will account for perspective foreshortening automatically.

  A reprojection error is reported during calibration so the user can verify
  accuracy.  If the error is large, the points should be re-selected.

Usage (interactive setup):
    python -m spatial.calibration                  # grab frame from camera
    python -m spatial.calibration --image path.jpg

Usage (runtime):
    from spatial.calibration import CalibrationMap
    cal = CalibrationMap()
    cal.load()
    world_pt = cal.to_world((px, py))   # → (x_cm, y_cm)
    pixel_pt = cal.to_pixel((x_cm, y_cm))
"""

import cv2
import json
import numpy as np
import os
import argparse
import sys
import requests
import time
from typing import Optional
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(_ROOT_DIR, "config", "calibration.json")
DEFAULT_CAPTURE_URL = "http://10.54.215.196/capture"

_POINT_COLORS = [
    (0, 255, 255),   # 1 — yellow
    (255, 0, 255),   # 2 — magenta
    (0, 165, 255),   # 3 — orange
    (255, 255, 0),   # 4 — cyan
]

_DISPLAY_MAX_W = int(os.getenv("DISPLAY_MAX_W", "320"))
_DISPLAY_MAX_H = int(os.getenv("DISPLAY_MAX_H", "240"))
_STATUS_BAR_H = int(os.getenv("DISPLAY_STATUS_BAR_H", "42"))


# ─────────────────────────────────────────────
class CalibrationMap:
    """
    Loaded at runtime. Provides pixel ↔ world transforms.
    If no calibration file exists, to_world/to_pixel return the
    input unchanged (1 pixel = 1 cm fallback so the system still runs).
    """

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self._H: Optional[np.ndarray] = None        # runtime pixel → world
        self._H_inv: Optional[np.ndarray] = None    # world → runtime pixel
        self._H_base: Optional[np.ndarray] = None   # source pixel → world
        self._source_frame_size: Optional[tuple[int, int]] = None  # (w, h)
        self._runtime_frame_size: Optional[tuple[int, int]] = None  # (w, h)
        self.is_calibrated = False

    def load(self, path: Optional[str] = None):
        path = path or self.config_path
        if not os.path.exists(path):
            print(f"[Calibration] No calibration file at {path}. "
                  f"Running in 1px=1cm fallback mode.")
            return
        with open(path) as f:
            data = json.load(f)
        H = np.array(data["H"], dtype=np.float64)
        self._H_base = H

        src = data.get("source_frame_size")
        if isinstance(src, dict) and "width" in src and "height" in src:
            self._source_frame_size = (int(src["width"]), int(src["height"]))
        else:
            self._source_frame_size = None
            # Optional backward compatibility inference. Disabled by default
            # because zones can be edited at a different resolution and cause
            # incorrect scaling of older calibration matrices.
            if os.getenv("CALIBRATION_INFER_SOURCE_FROM_ZONES", "0") == "1":
                zones_cfg = os.path.join(_ROOT_DIR, "config", "zones.json")
                try:
                    if os.path.exists(zones_cfg):
                        with open(zones_cfg) as zf:
                            zdata = json.load(zf)
                        zsrc = zdata.get("source_frame_size")
                        if isinstance(zsrc, dict) and "width" in zsrc and "height" in zsrc:
                            self._source_frame_size = (
                                int(zsrc["width"]),
                                int(zsrc["height"]),
                            )
                            print(
                                "[Calibration] source_frame_size missing in calibration; "
                                f"using zones source size {self._source_frame_size}."
                            )
                except Exception:
                    # Keep None if inference fails.
                    pass
            else:
                print(
                    "[Calibration] source_frame_size missing in calibration file; "
                    "not inferring from zones by default."
                )

        # Default runtime matrix equals saved matrix; can be adapted later.
        self._H = self._H_base.copy()
        self._H_inv = np.linalg.inv(self._H)
        self.is_calibrated = True
        print(f"[Calibration] Loaded homography from {path}")

    def save(
        self,
        H: np.ndarray,
        path: Optional[str] = None,
        source_frame_size: Optional[tuple[int, int]] = None,
    ):
        path = path or self.config_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"H": H.tolist()}
        if source_frame_size is not None:
            payload["source_frame_size"] = {
                "width": int(source_frame_size[0]),
                "height": int(source_frame_size[1]),
            }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Calibration] Saved → {path}")

    def set_runtime_frame_size(self, width: int, height: int) -> bool:
        """
        Adapt homography for a different runtime frame size than calibration size.
        Returns True if the active matrix changed.
        """
        if self._H_base is None:
            return False

        runtime = (int(width), int(height))
        if self._runtime_frame_size == runtime:
            return False
        self._runtime_frame_size = runtime

        if self._source_frame_size is None:
            # Backward compatibility for old calibration files without size metadata.
            self._H = self._H_base.copy()
            self._H_inv = np.linalg.inv(self._H)
            return True

        src_w, src_h = self._source_frame_size
        run_w, run_h = runtime
        if run_w <= 0 or run_h <= 0:
            return False

        sx = float(src_w) / float(run_w)
        sy = float(src_h) / float(run_h)
        scale_runtime_to_source = np.array(
            [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        # world = H_base * source_px = H_base * S * runtime_px
        self._H = self._H_base @ scale_runtime_to_source
        self._H_inv = np.linalg.inv(self._H)
        return True

    # ── transforms ───────────────────────────

    def to_world(self, pixel_pt: tuple) -> tuple:
        """Pixel (px, py) → real world (x_cm, y_cm)."""
        if self._H is None:
            return pixel_pt
        src = np.array([[[float(pixel_pt[0]), float(pixel_pt[1])]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self._H)
        return (float(dst[0][0][0]), float(dst[0][0][1]))

    def to_pixel(self, world_pt: tuple) -> tuple:
        """Real world (x_cm, y_cm) → pixel (px, py)."""
        if self._H_inv is None:
            return world_pt
        src = np.array([[[float(world_pt[0]), float(world_pt[1])]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self._H_inv)
        return (int(dst[0][0][0]), int(dst[0][0][1]))

    def polygon_to_world(self, pixel_polygon: list) -> list:
        return [self.to_world(pt) for pt in pixel_polygon]

    def polygon_to_pixel(self, world_polygon: list) -> list:
        return [self.to_pixel(pt) for pt in world_polygon]

    @property
    def source_frame_size(self) -> Optional[tuple[int, int]]:
        return self._source_frame_size

    def adopt_source_frame_size(self, width: int, height: int) -> bool:
        """
        For legacy calibration files missing source_frame_size, adopt a known
        source frame size (typically from zones) so runtime scaling can work.
        Returns True if adopted.
        """
        if self._H_base is None or self._source_frame_size is not None:
            return False
        w, h = int(width), int(height)
        if w <= 0 or h <= 0:
            return False
        self._source_frame_size = (w, h)
        self._H = self._H_base.copy()
        self._H_inv = np.linalg.inv(self._H)
        return True


# ─────────────────────────────────────────────
# Interactive calibration tool
# ─────────────────────────────────────────────

class _Calibrator:
    """
    Flow per point:
      1. User clicks a pixel on the frame.
      2. An on-screen input box appears: "Enter X (cm):"
      3. User types digits / dot / minus, presses Enter.
      4. Box changes to "Enter Y (cm):" — same input method.
      5. Point is confirmed; move to next point.
      6. After 4 points, homography is computed and shown as a verification overlay.
    """

    # States
    _ST_CLICK   = "click"    # waiting for a click
    _ST_TYPE_X  = "type_x"  # typing the X world coord
    _ST_TYPE_Y  = "type_y"  # typing the Y world coord
    _ST_VERIFY  = "verify"  # showing verification result

    def __init__(self, frame: np.ndarray):
        self.frame        = frame.copy()
        self.pixel_points: list[tuple]  = []
        self.world_points: list[tuple]  = []

        self._state       = self._ST_CLICK
        self._typed       = ""          # current text buffer
        self._current_x   = None        # x world coord being built
        self._error_msg   = ""

        self._H:   Optional[np.ndarray] = None
        self._verify_lines: list[str]   = []

        self._win = "Calibration  |  Q = quit"

        fh, fw = self.frame.shape[:2]
        self._display_scale = min(1.0, _DISPLAY_MAX_W / max(fw, 1), _DISPLAY_MAX_H / max(fh, 1))
        self._display_w = max(1, int(round(fw * self._display_scale)))
        self._display_h = max(1, int(round(fh * self._display_scale)))

    # ── state helpers ─────────────────────────

    @property
    def _point_idx(self):
        return len(self.pixel_points)

    @property
    def _color(self):
        if self._state in (self._ST_TYPE_X, self._ST_TYPE_Y) and self._point_idx > 0:
            idx = self._point_idx - 1
        else:
            idx = self._point_idx
        return _POINT_COLORS[min(max(idx, 0), 3)]

    # ── draw ──────────────────────────────────

    def _draw(self) -> np.ndarray:
        disp = self.frame.copy()
        h, w = disp.shape[:2]

        # Scale circle size for low-resolution frames
        circle_radius = 3 if w <= 360 else 8
        circle_outline = 4 if w <= 360 else 11
        text_scale = 0.35 if w <= 360 else 0.5

        # ── completed points
        for i, (px, wd) in enumerate(zip(self.pixel_points, self.world_points)):
            color = _POINT_COLORS[i]
            cv2.circle(disp, px, circle_radius, color, -1)
            cv2.circle(disp, px, circle_outline, (255, 255, 255), 1)
            label = f"{i+1}  ({wd[0]:.1f},{wd[1]:.1f})cm"
            cv2.putText(disp, label, (px[0] + 13, px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, 1, cv2.LINE_AA)

        if self._state == self._ST_VERIFY:
            self._draw_verify(disp)

        msg = ""
        if self._state == self._ST_CLICK:
            if self._point_idx < 4:
                msg = (f"Point {self._point_idx + 1}/4 — LEFT CLICK a known marker")
            else:
                msg = "All 4 points clicked — computing..."
        elif self._state == self._ST_TYPE_X:
            msg = f"Point {self._point_idx}/4 — Type REAL-WORLD X (cm), press Enter"
        else:
            msg = f"Point {self._point_idx}/4 — Type REAL-WORLD Y (cm), press Enter"

        # ── in-progress click dot (ghost)
        if self._state in (self._ST_TYPE_X, self._ST_TYPE_Y):
            px = self.pixel_points[-1]  # the point just clicked
            cv2.circle(disp, px, circle_radius, self._color, -1)
            cv2.circle(disp, px, circle_outline, (255, 255, 255), 1)
            cv2.putText(disp, str(self._point_idx), (px[0] + 13, px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale + 0.05, self._color, 1)

        # ── input box (bottom-centre)
        if self._state in (self._ST_TYPE_X, self._ST_TYPE_Y):
            self._draw_input_box(disp, h, w)

        # ── error message
        if self._error_msg:
            cv2.putText(disp, self._error_msg, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 50, 220), 2, cv2.LINE_AA)

        if self._display_scale < 1.0:
            disp = cv2.resize(disp, (self._display_w, self._display_h), interpolation=cv2.INTER_AREA)

        if self._state == self._ST_VERIFY:
            return disp

        status = np.zeros((_STATUS_BAR_H, self._display_w, 3), dtype=np.uint8)
        status[:, :] = (28, 28, 28)
        msg_scale = 0.4 if w <= 360 else 0.55
        hint_scale = 0.35 if w <= 360 else 0.5
        cv2.putText(status, msg, (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, msg_scale, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(status, "Q = quit", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, hint_scale, (170, 170, 170), 1, cv2.LINE_AA)

        disp = np.vstack([disp, status])

        return disp

    def _draw_input_box(self, disp, h, w):
        if self._state == self._ST_TYPE_X:
            prompt = "X (cm):"
            stored = ""
        else:
            x_val  = f"{self._current_x:.2f}"
            prompt = "Y (cm):"
            stored = f"X = {x_val}  |  "

        box_w, box_h = 420, 60
        bx = (w - box_w) // 2
        by = h - box_h - 20

        # shadow
        cv2.rectangle(disp, (bx - 3, by - 3), (bx + box_w + 3, by + box_h + 3),
                      (0, 0, 0), -1)
        # box bg
        cv2.rectangle(disp, (bx, by), (bx + box_w, by + box_h), (40, 40, 40), -1)
        cv2.rectangle(disp, (bx, by), (bx + box_w, by + box_h), self._color, 2)

        full_text = stored + prompt + "  " + self._typed + "|"
        cv2.putText(disp, full_text, (bx + 10, by + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_verify(self, disp):
        h, w = disp.shape[:2]
        # dark overlay
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, disp, 0.55, 0, disp)

        cv2.putText(disp, "Calibration complete!", (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 210, 60), 2, cv2.LINE_AA)
        cv2.putText(disp, "Press S to save  |  R to redo  |  Q to discard",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        y = 120
        for line in self._verify_lines:
            cv2.putText(disp, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            y += 26

    # ── mouse ─────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self._state == self._ST_CLICK:
            if self._point_idx < 4:
                if y >= self._display_h:
                    return

                if self._display_scale < 1.0:
                    x = int(round(x / self._display_scale))
                    y = int(round(y / self._display_scale))

                h, w = self.frame.shape[:2]
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))

                self.pixel_points.append((x, y))
                self._typed = ""
                self._error_msg = ""
                self._state = self._ST_TYPE_X

    # ── keyboard input handling ───────────────

    def _handle_key(self, key: int) -> str:
        """Returns 'quit', 'save', 'redo', or ''."""
        if self._state == self._ST_VERIFY:
            if key == ord("s"):
                return "save"
            elif key == ord("r"):
                return "redo"
            elif key == ord("q"):
                return "quit"
            return ""

        if key == ord("q"):
            return "quit"

        if self._state not in (self._ST_TYPE_X, self._ST_TYPE_Y):
            return ""

        ch = chr(key) if 32 <= key <= 126 else None

        if key in (13, 10):   # Enter — confirm current field
            try:
                val = float(self._typed.strip())
            except ValueError:
                self._error_msg = "Invalid number. Type digits only (e.g. 30.5)"
                return ""

            self._error_msg = ""

            if self._state == self._ST_TYPE_X:
                self._current_x = val
                self._typed     = ""
                self._state     = self._ST_TYPE_Y
            else:
                # Both X and Y collected — store world point
                self.world_points.append((self._current_x, val))
                self._current_x = None
                self._typed     = ""
                if len(self.world_points) == 4:
                    self._compute()
                else:
                    self._state = self._ST_CLICK

        elif key in (8, 127):  # Backspace / Delete
            self._typed = self._typed[:-1]

        elif ch and (ch.isdigit() or ch in (".", "-")):
            self._typed += ch

        return ""

    # ── compute homography ────────────────────

    def _compute(self):
        src = np.array(self.pixel_points, dtype=np.float64)
        dst = np.array(self.world_points,  dtype=np.float64)
        H, _ = cv2.findHomography(src, dst)

        self._H = H
        self._verify_lines = []

        if H is None:
            self._verify_lines.append("ERROR: Could not compute homography.")
            self._state = self._ST_VERIFY
            return

        cal = CalibrationMap()
        cal._H = H
        cal._H_inv = np.linalg.inv(H)

        self._verify_lines.append("Pixel  →  Expected (cm)  →  Got (cm)")
        self._verify_lines.append("─" * 52)
        total_err = 0.0
        for i, (px, wd) in enumerate(zip(self.pixel_points, self.world_points)):
            got = cal.to_world(px)
            err = ((got[0] - wd[0]) ** 2 + (got[1] - wd[1]) ** 2) ** 0.5
            total_err += err
            self._verify_lines.append(
                f"  Pt {i+1}:  {str(px):<18}  "
                f"({wd[0]:.1f}, {wd[1]:.1f})  →  "
                f"({got[0]:.1f}, {got[1]:.1f})"
            )

        avg_err = total_err / max(len(self.pixel_points), 1)
        self._verify_lines.append("")
        self._verify_lines.append(f"  Mean reprojection error: {avg_err:.3f} cm")
        if avg_err < 0.5:
            self._verify_lines.append("  Perspective compensation: GOOD")
        elif avg_err < 2.0:
            self._verify_lines.append("  Perspective compensation: OK (consider re-calibrating)")
        else:
            self._verify_lines.append("  Perspective compensation: POOR — re-select points!")

        self._state = self._ST_VERIFY

    # ── main loop ─────────────────────────────

    def run(self) -> Optional[np.ndarray]:
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self._win, self._display_w, self._display_h + _STATUS_BAR_H)
        cv2.setMouseCallback(self._win, self._mouse_cb)

        while True:
            cv2.imshow(self._win, self._draw())
            key = cv2.waitKey(30) & 0xFF

            action = self._handle_key(key)

            if action == "save":
                cv2.destroyAllWindows()
                return self._H
            elif action == "redo":
                # Reset everything
                self.pixel_points = []
                self.world_points  = []
                self._typed        = ""
                self._current_x    = None
                self._error_msg    = ""
                self._H            = None
                self._state        = self._ST_CLICK
            elif action == "quit":
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()
        return None


# ─── entry point ──────────────────────────────

def _grab_frame() -> np.ndarray:
    base_url = os.getenv("ESP32_CAPTURE_URL") or DEFAULT_CAPTURE_URL
    print(f"[calibration] Fetching one snapshot from {base_url} ...")
    require_hq = os.getenv("CALIBRATION_REQUIRE_HQ", "1") == "1"
    min_w = int(os.getenv("CALIBRATION_MIN_HQ_WIDTH", "640"))
    min_h = int(os.getenv("CALIBRATION_MIN_HQ_HEIGHT", "480"))

    def _set_query_param(url: str, key: str, value: str) -> str:
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query[key] = value
        return urlunparse(parsed._replace(query=urlencode(query)))

    def _candidate_urls(url: str) -> list[str]:
        # Prefer HQ snapshots for calibration because source_frame_size is
        # persisted and later used for runtime scaling.
        hq_first = [
            _set_query_param(url, "mode", "calibration"),
            _set_query_param(url, "mode", "zone_editor"),
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

            print(f"[calibration] Snapshot received from {snap_url}.")
            return frame

    if saw_busy:
        print(
            "[calibration] ESP32 reported camera busy (HTTP 503) for one or more "
            "HQ snapshot attempts. This is expected while /stream is active on "
            "the new firmware."
        )
    raise RuntimeError(f"Snapshot request failed after fallbacks. Last error: {last_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration tool — pixels to real-world cm")
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()

    if args.image:
        bg = cv2.imread(args.image)
        if bg is None:
            print(f"[calibration] Could not load: {args.image}")
            sys.exit(1)
    else:
        bg = _grab_frame()

    print("[calibration] Window open.")
    print("  1. Click 4 ground markers on the image.")
    print("  2. Type real-world X then Y (cm) directly in the window after each click.")
    print("  3. Press S to save, R to redo, Q to quit.\n")

    calibrator = _Calibrator(bg)
    H = calibrator.run()

    if H is not None:
        cal = CalibrationMap()
        cal.save(H, source_frame_size=(bg.shape[1], bg.shape[0]))
        print("[calibration] Done. Run main.py to use calibrated layout.")
