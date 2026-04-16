"""
Layout Engine — computes parking slot positions for open-ground lots.

Three modes (per session, selected at runtime):
  open      — perpendicular back-to-back rows with shared aisle
  parallel  — nose-to-tail along the longer edge of the zone
  angled    — 45° parallelogram stalls with one-way aisle

Unit handling:
  • If calibration exists:  all tiling is in world-cm; slots converted to px.
  • If no calibration:       tiling is done directly in pixel space.

  Vehicle dimensions can be provided dynamically via the `vehicle_dims` argument
  (from a backend or config).  Falls back to the module-level defaults.

Usage:
    from reasoning.layout_engine import LayoutEngine
    from spatial.calibration import CalibrationMap
    from spatial.zone_map import ZoneMap

    cal = CalibrationMap(); cal.load()
    zones = ZoneMap(); zones.load()

    engine = LayoutEngine(parking_type="open", calibrated=cal.is_calibrated)
    slots = engine.generate_all(zones, cal)      # list of slot dicts
"""

import cv2
import numpy as np
from typing import Optional

# ── Default vehicle dimensions (cm) — override via vehicle_dims arg ──────────
CAR_LENGTH_CM = 8.7
CAR_WIDTH_CM  = 3.6

# ── Pixel-fallback vehicle dimensions (no calibration) ───────────────────────
# Tune these to match your actual car size on screen.
CAR_LENGTH_PX = 80
CAR_WIDTH_PX  = 40

# ── Parking-slot profiles ─────────────────────────────────────────────────────
#
# All dimensions are direct multipliers of the vehicle's own size.
#
#   slot_w_mult  — total slot width     = car_width  × mult  (door clearance)
#   slot_d_mult  — total slot depth     = car_length × mult  (nose/tail buffer)
#   aisle_mult   — aisle between rows   = car_width  × mult  (car must fit
#                  through width-wise to drive out)
#
# Rows are packed flush against the far (restricted) edge of the parking
# zone.  Any leftover space between the last row and the drive zone
# serves as the exit path — no explicit exit buffer is needed.
#
# "open" tries all three orientations and picks the one that yields the
# most slots.
# ──────────────────────────────────────────────────────────────────────────────
_PROFILES = {
    "open":     {"slot_w_mult": 1.24, "slot_d_mult": 1.08, "aisle_mult": 1.30},
    "parallel": {"slot_w_mult": 1.36, "slot_d_mult": 1.18, "aisle_mult": 1.10},
    "angled":   {"slot_w_mult": 1.30, "slot_d_mult": 1.10, "aisle_mult": 1.20},
}

# ── Tiling quality ────────────────────────────────────────────────────────────
# Accept slots whose area mostly lies inside the parking zone.
# Keep the threshold permissive enough to maintain slot count while a
# centroid-in-zone check (below) prevents obvious out-of-zone slots.
MIN_ZONE_OVERLAP = 0.55

# ── Slot inflation margin (px) — expands slot poly for occupancy tolerance ───
SLOT_INFLATION_PX = 6

# ── Colours ───────────────────────────────────────────────────────────────────
SLOT_COLORS = {
    "free":     (0,  210,  60),
    "occupied": (0,   50, 220),
    "reserved": (30, 160, 255),
}


# ─────────────────────────────────────────────
class LayoutEngine:

    def __init__(
        self,
        parking_type: str  = "open",
        calibrated:   bool = False,
        vehicle_dims: Optional[dict] = None,  # {"length_cm": float, "width_cm": float}
    ):
        """
        Parameters
        ----------
        parking_type  : "open" | "parallel" | "angled"
        calibrated    : True if CalibrationMap.is_calibrated
        vehicle_dims  : optional backend-provided dims in cm, e.g.
                        {"length_cm": 8.7, "width_cm": 3.6}
                        Falls back to module defaults when None.
        """
        if parking_type not in ("open", "parallel", "angled"):
            raise ValueError(f"Unknown parking_type: {parking_type!r}. "
                             f"Choose from: open, parallel, angled")
        self.parking_type = parking_type
        self.calibrated   = calibrated

        # ── Single source of truth for vehicle dimensions ────────────────────
        if calibrated:
            vd = vehicle_dims or {}
            self.car_length = float(vd.get("length_cm", CAR_LENGTH_CM))
            self.car_width  = float(vd.get("width_cm",  CAR_WIDTH_CM))
        else:
            if vehicle_dims:
                scale = CAR_LENGTH_PX / CAR_LENGTH_CM
                self.car_length = float(vehicle_dims.get("length_cm", CAR_LENGTH_CM)) * scale
                self.car_width  = float(vehicle_dims.get("width_cm",  CAR_WIDTH_CM))  * scale
            else:
                self.car_length = CAR_LENGTH_PX
                self.car_width  = CAR_WIDTH_PX

        prof = _PROFILES[parking_type]

        # ── Derived slot geometry (direct multipliers) ──────────────────
        self._slot_w = self.car_width  * prof["slot_w_mult"]   # total slot width
        self._slot_d = self.car_length * prof["slot_d_mult"]   # total slot depth / length
        self._aisle  = self.car_width  * prof["aisle_mult"]    # aisle width (car-width-based)

        # Keep raw multipliers for open-mode mixed orientation per-row recalc
        self._slot_w_mult = prof["slot_w_mult"]
        self._slot_d_mult = prof["slot_d_mult"]

    # ── public ───────────────────────────────

    def generate_all(self, zone_map, calibration, pinned_slots=None) -> list[dict]:
        """
        Generate slots for every parking zone in zone_map.

        In open mode the engine tries all orientations across the ENTIRE
        zone and filters out candidates that overlap any pinned (pre-parked)
        car, so freed aisle space can host slots in a different orientation.

        In parallel / angled modes pinned cars act as fixed anchors —
        slots in each row are shifted to maximise count around them.
        """
        slots = []
        parking_zones = zone_map.zones_by_type("parking")

        if not parking_zones:
            print("[LayoutEngine] No parking zones defined. Draw zones first.")
            return slots

        # Use drive/exit orientation hints only for parallel mode.
        # Open mode intentionally searches orientation-agnostically and the
        # downstream exit-path filter enforces route feasibility.
        drive_zones = []
        if self.parking_type == "parallel":
            drive_zones = zone_map.zones_by_type("drive")
            if not drive_zones:
                drive_zones = zone_map.zones_by_type("exit")
        drive_polys_wld = []
        for dz in drive_zones:
            dz_px = [tuple(p) for p in dz["points"]]
            drive_polys_wld.append(calibration.polygon_to_world(dz_px))

        # Build obstacle pixel-polygons from pinned (pre-parked) slots
        obstacle_polys_px = []
        if pinned_slots:
            obstacle_polys_px = [
                np.array(s["polygon_px"], dtype=np.int32)
                for s in pinned_slots
            ]

        for zone in parking_zones:
            zone_px   = [tuple(p) for p in zone["points"]]
            zone_wld  = calibration.polygon_to_world(zone_px)
            new_slots = self._generate(zone_wld, zone["name"], calibration,
                                       drive_polys_wld, obstacle_polys_px)
            slots.extend(new_slots)
            print(f"[LayoutEngine] Zone '{zone['name']}': "
                  f"{len(new_slots)} {self.parking_type} slots")

        print(f"[LayoutEngine] Total slots: {len(slots)}")
        return slots

    # ── routing ──────────────────────────────

    def _generate(self, poly_world, zone_name, calibration, drive_polys,
                  obstacle_polys_px=None):
        dispatch = {
            "open":     self._layout_open,
            "parallel": self._layout_parallel,
            "angled":   self._layout_angled,
        }
        return dispatch[self.parking_type](poly_world, zone_name, calibration,
                                           drive_polys, obstacle_polys_px)

    # ── drive-edge helpers ───────────────────

    def _find_drive_edge(self, parking_poly, drive_polys):
        """
        Identify the parking-zone edge nearest to any drive zone.
        Falls back to the longest edge when no drive zones are defined.
        Returns (p1, p2) as numpy arrays.
        """
        pts = [np.array(p, dtype=np.float64) for p in parking_poly]
        n = len(pts)
        edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]

        if drive_polys:
            best_edge = edges[0]
            best_dist = float('inf')
            for p1, p2 in edges:
                mid = (p1 + p2) / 2.0
                for dpoly in drive_polys:
                    dp = np.array(dpoly, dtype=np.float32)
                    d = abs(cv2.pointPolygonTest(
                        dp, (float(mid[0]), float(mid[1])), True))
                    if d < best_dist:
                        best_dist = d
                        best_edge = (p1, p2)
            return best_edge

        # No drive zones — use the longest edge as fallback
        return max(edges, key=lambda e: float(np.linalg.norm(e[1] - e[0])))

    def _drive_frame(self, parking_poly, drive_polys):
        """
        Build a local (u, v) frame from the drive-adjacent edge.
          u  runs along the drive edge
          v  points into the parking zone (perpendicular to drive edge)
        Returns (origin, u_hat, v_hat).
        """
        drive_edge = self._find_drive_edge(parking_poly, drive_polys)
        p1, p2 = drive_edge
        edge = p2 - p1
        length = float(np.linalg.norm(edge))
        u_hat = edge / max(length, 1e-9)
        v_hat = np.array([-u_hat[1], u_hat[0]])   # 90° CCW

        # Ensure v_hat points INTO the parking zone
        centroid = np.mean([np.array(p, dtype=np.float64)
                            for p in parking_poly], axis=0)
        if np.dot(v_hat, centroid - p1) < 0:
            v_hat = -v_hat

        return p1.copy(), u_hat, v_hat

    def _local_extent(self, poly, origin, u_hat, v_hat):
        """Project all polygon vertices to (u, v) and return bounds."""
        uvs = [self._to_local(p, origin, u_hat, v_hat) for p in poly]
        us = [uv[0] for uv in uvs]
        vs = [uv[1] for uv in uvs]
        return min(us), max(us), min(vs), max(vs)

    @staticmethod
    def _to_local(pt, origin, u_hat, v_hat):
        d = np.array(pt, dtype=np.float64) - origin
        return float(np.dot(d, u_hat)), float(np.dot(d, v_hat))

    @staticmethod
    def _to_world_pt(u, v, origin, u_hat, v_hat):
        p = origin + u * u_hat + v * v_hat
        return (float(p[0]), float(p[1]))

    def _make_rect(self, u, v, w, h, origin, u_hat, v_hat):
        """Create a (u,v)-aligned rect and return its world-coord polygon."""
        return [
            self._to_world_pt(u,     v,     origin, u_hat, v_hat),
            self._to_world_pt(u + w, v,     origin, u_hat, v_hat),
            self._to_world_pt(u + w, v + h, origin, u_hat, v_hat),
            self._to_world_pt(u,     v + h, origin, u_hat, v_hat),
        ]

    # ── helpers: spread slots edge-to-edge ───────────────────────────
    #
    # Instead of packing slots tightly at one end and leaving a big
    # unused gap at the other, we count how many slots fit, then
    # distribute the leftover space evenly BETWEEN slots.

    @staticmethod
    def _spread_positions(start, end, slot_size, min_gap=0.0):
        """
        Return a list of slot-start positions so that `count` slots of
        `slot_size` are evenly spread across [start, end].

        The first slot is flush at `start` and the last slot is flush at
        `end - slot_size`.  Any remainder is distributed as equal gaps
        between adjacent slots.  `min_gap` is the minimum gap that must
        exist (e.g. the parking margin); if there isn't enough room for
        even one gap the slots are packed tight.
        """
        span = end - start
        if span < slot_size - 0.5:
            return []
        count = max(1, int((span + 0.5) / max(slot_size + min_gap, 1e-9)))
        # Clamp count so slots don't overflow the span
        while count > 1 and (count * slot_size + (count - 1) * min_gap) > span + 0.5:
            count -= 1
        if count <= 1:
            return [start]
        total_slot = count * slot_size
        total_gap  = span - total_slot
        gap = total_gap / (count - 1) if count > 1 else 0.0
        gap = max(gap, min_gap)
        return [start + i * (slot_size + gap) for i in range(count)]

    # ── open (maximum-density optimiser) ────────────────────────────
    #
    # Tries perpendicular, parallel AND 45° angled orientations for
    # the whole zone, calculates the total slot count for each, and
    # picks whichever yields the most.  Rows are packed flush against
    # the far (restricted) edge; leftover space near the drive zone
    # becomes the exit aisle.

    def _layout_open(self, poly_world, zone_name, cal, drive_polys,
                     obstacle_polys_px=None):
        zone_np = np.array(poly_world, dtype=np.float32)
        zone_top_y_px = min(cal.to_pixel(tuple(p))[1] for p in poly_world)

        def _top_gap_score(polys_world):
            """Smaller is better: how close slots are to the top parking edge."""
            if not polys_world:
                return float("inf")
            min_slot_y = float("inf")
            for poly_w in polys_world:
                py = min(cal.to_pixel(tuple(pt))[1] for pt in poly_w)
                if py < min_slot_y:
                    min_slot_y = py
            return max(0.0, float(min_slot_y - zone_top_y_px))

        def _candidates_for_frame(origin, u_hat, v_hat):
            u0, u1, v0, v1 = self._local_extent(poly_world, origin, u_hat, v_hat)
            v_span = v1 - v0
            aisle = self._aisle

            # Candidate geometries
            sw_p = self._slot_w
            sd_p = self._slot_d
            sw_r = self.car_length * self._slot_w_mult
            sd_r = self.car_width * self._slot_d_mult

            _a_prof = _PROFILES["angled"]
            theta = np.radians(45)
            sin_t, cos_t = np.sin(theta), np.cos(theta)
            sw_a = self.car_width * _a_prof["slot_w_mult"]
            sd_a = self.car_length * _a_prof["slot_d_mult"]
            step_a = sw_a / sin_t
            shft_a = sd_a * cos_t
            dep_a = sd_a * sin_t

            def _max_rows(depth):
                if depth > v_span + 0.5:
                    return 0
                n = max(1, int((v_span + aisle + 0.5) / (depth + aisle)))
                while n > 1 and n * depth + (n - 1) * aisle > v_span + 0.5:
                    n -= 1
                return n

            def _pack_rows_v(depth, n_rows):
                rows = []
                v_cur = v1
                for _ in range(n_rows):
                    v_start = v_cur - depth
                    if v_start < v0 - 0.5:
                        break
                    rows.append(v_start)
                    v_cur = v_start - aisle
                return rows

            def _row_packings(depth):
                n_rows = _max_rows(depth)
                if n_rows <= 0:
                    return []

                packings = []

                rows_far = _pack_rows_v(depth, n_rows)
                if rows_far:
                    packings.append(rows_far)

                rows_near = []
                v_cur = v0
                for _ in range(n_rows):
                    if v_cur + depth > v1 + 0.5:
                        break
                    rows_near.append(v_cur)
                    v_cur += depth + aisle
                if rows_near:
                    packings.append(rows_near)

                total = n_rows * depth + (n_rows - 1) * aisle
                start = v0 + max((v_span - total) / 2.0, 0.0)
                rows_center = []
                for i in range(n_rows):
                    rv = start + i * (depth + aisle)
                    if rv + depth <= v1 + 0.5:
                        rows_center.append(rv)
                if rows_center:
                    packings.append(rows_center)

                uniq = []
                seen = set()
                for rows in packings:
                    key = tuple(round(r, 2) for r in rows)
                    if key not in seen:
                        uniq.append(rows)
                        seen.add(key)
                return uniq

            def _phase_values(step):
                return [0.0, 0.25 * step, 0.5 * step, 0.75 * step]

            def _tile_row_rect(slot_w, slot_d, v_start, u_shift=0.0, from_left=True):
                if from_left:
                    positions = self._spread_positions(u0 + u_shift, u1, slot_w)
                else:
                    positions = self._spread_positions(u0, u1 - u_shift, slot_w)
                polys = []
                for pu in positions:
                    p = self._make_rect(pu, v_start, slot_w, slot_d,
                                        origin, u_hat, v_hat)
                    if _sufficient_zone_overlap(p, zone_np):
                        polys.append(p)
                return polys

            def _tile_row_angled(v_start, u_shift=0.0, direction=1):
                if direction >= 0:
                    positions = self._spread_positions(u0 + u_shift, u1 - shft_a, step_a)
                else:
                    positions = self._spread_positions(u0 + shft_a + u_shift, u1, step_a)

                polys = []
                for pu in positions:
                    if direction >= 0:
                        p = [
                            self._to_world_pt(pu,                   v_start,         origin, u_hat, v_hat),
                            self._to_world_pt(pu + step_a,          v_start,         origin, u_hat, v_hat),
                            self._to_world_pt(pu + step_a + shft_a, v_start + dep_a, origin, u_hat, v_hat),
                            self._to_world_pt(pu + shft_a,          v_start + dep_a, origin, u_hat, v_hat),
                        ]
                    else:
                        p = [
                            self._to_world_pt(pu,                   v_start,         origin, u_hat, v_hat),
                            self._to_world_pt(pu + step_a,          v_start,         origin, u_hat, v_hat),
                            self._to_world_pt(pu + step_a - shft_a, v_start + dep_a, origin, u_hat, v_hat),
                            self._to_world_pt(pu - shft_a,          v_start + dep_a, origin, u_hat, v_hat),
                        ]
                    if _sufficient_zone_overlap(p, zone_np):
                        polys.append(p)
                return polys

            candidates = []

            for rows in _row_packings(sd_p):
                for phase in _phase_values(sw_p):
                    polys_l = []
                    polys_r = []
                    for rv in rows:
                        polys_l.extend(_tile_row_rect(sw_p, sd_p, rv, phase, True))
                        polys_r.extend(_tile_row_rect(sw_p, sd_p, rv, phase, False))
                    candidates.append(polys_l)
                    candidates.append(polys_r)

            for rows in _row_packings(sd_r):
                for phase in _phase_values(sw_r):
                    polys_l = []
                    polys_r = []
                    for rv in rows:
                        polys_l.extend(_tile_row_rect(sw_r, sd_r, rv, phase, True))
                        polys_r.extend(_tile_row_rect(sw_r, sd_r, rv, phase, False))
                    candidates.append(polys_l)
                    candidates.append(polys_r)

            for rows in _row_packings(dep_a):
                for phase in _phase_values(step_a):
                    for direction in (1, -1):
                        polys = []
                        for rv in rows:
                            polys.extend(_tile_row_angled(rv, phase, direction))
                        candidates.append(polys)

            return candidates

        # Frame A: drive-aligned frame (existing behavior)
        origin_a, u_hat_a, v_hat_a = self._drive_frame(poly_world, drive_polys)
        candidates = _candidates_for_frame(origin_a, u_hat_a, v_hat_a)

        # Frame B: 90°-rotated alternative frame to capture horizontal/vertical
        # transposed packings that drive-aligned frame can miss.
        origin_b = origin_a
        u_hat_b = v_hat_a
        v_hat_b = -u_hat_a
        candidates.extend(_candidates_for_frame(origin_b, u_hat_b, v_hat_b))

        if obstacle_polys_px:
            candidate_sets = []
            for cand_polys in candidates:
                kept = [
                    p for p in cand_polys
                    if not _overlaps_any([cal.to_pixel(pt) for pt in p], obstacle_polys_px)
                ]
                candidate_sets.append(kept)
        else:
            candidate_sets = candidates

        best_polys = []
        best_count = -1
        best_gap = float("inf")
        for cand in candidate_sets:
            ccount = len(cand)
            if ccount < best_count:
                continue
            cgap = _top_gap_score(cand)
            if ccount > best_count or cgap < best_gap:
                best_polys = cand
                best_count = ccount
                best_gap = cgap

        slots = []
        for i, p in enumerate(best_polys):
            slots.append(_slot(i, zone_name, "open", p, cal))
        return slots

    # ── parallel (nose-to-tail, packed toward restricted edge) ─────────
    #
    # Rows are packed flush against v1 (restricted edge) and grow
    # toward v0 (drive zone).  Leftover space near the drive zone
    # becomes the exit lane.  Slots along u are spread edge-to-edge.

    def _layout_parallel(self, poly_world, zone_name, cal, drive_polys,
                         obstacle_polys_px=None):
        origin, u_hat, v_hat = self._drive_frame(poly_world, drive_polys)
        u0, u1, v0, v1 = self._local_extent(poly_world, origin, u_hat, v_hat)
        zone_np = np.array(poly_world, dtype=np.float32)

        slot_len = self._slot_d          # along u
        slot_wid = self._slot_w          # along v
        aisle    = self._aisle
        v_span   = v1 - v0

        # How many rows fit?
        n_rows = max(1, int((v_span + aisle + 0.5) / max(slot_wid + aisle, 1e-9)))
        while n_rows > 1 and n_rows * slot_wid + (n_rows - 1) * aisle > v_span + 0.5:
            n_rows -= 1

        # Pack rows from v1 (restricted edge) toward v0 (drive edge)
        row_vs = []
        v_cur = v1
        for _ in range(n_rows):
            v_start = v_cur - slot_wid
            if v_start < v0 - 0.5:
                break
            row_vs.append(v_start)
            v_cur = v_start - aisle

        # Convert obstacles to local (u,v) bounding boxes
        obs_local = _obstacles_to_local(obstacle_polys_px, cal,
                                        origin, u_hat, v_hat)

        slots, n = [], 0
        for rv in row_vs:
            row_v0, row_v1 = rv, rv + slot_wid

            # Find obstacles that overlap this row's v-range
            row_obs = [(ou0, ou1) for (ou0, ou1, ov0, ov1) in obs_local
                       if ov1 > row_v0 - 0.5 and ov0 < row_v1 + 0.5]

            if not row_obs:
                u_positions = self._spread_positions(u0, u1, slot_len)
            else:
                row_obs.sort()
                segments = _free_segments(u0, u1, row_obs)
                u_positions = []
                for seg_s, seg_e in segments:
                    u_positions.extend(
                        self._spread_positions(seg_s, seg_e, slot_len))

            for pu in u_positions:
                poly = self._make_rect(pu, rv, slot_len, slot_wid,
                                       origin, u_hat, v_hat)
                if _sufficient_zone_overlap(poly, zone_np):
                    slots.append(_slot(n, zone_name, "parallel", poly, cal))
                    n += 1

        return slots

    # ── angled 45° (packed toward restricted edge) ────────────────────
    #
    # Rows packed flush against v1 (restricted edge) toward v0 (drive).
    # Slots along u spread edge-to-edge.

    def _layout_angled(self, poly_world, zone_name, cal, drive_polys,
                       obstacle_polys_px=None):
        origin, u_hat, v_hat = self._drive_frame(poly_world, drive_polys)
        u0, u1, v0, v1 = self._local_extent(poly_world, origin, u_hat, v_hat)
        zone_np = np.array(poly_world, dtype=np.float32)

        theta  = np.radians(45)
        sin_t  = np.sin(theta)
        cos_t  = np.cos(theta)

        sw      = self._slot_w
        sd      = self._slot_d
        step_u  = sw / sin_t
        shift_u = sd * cos_t
        depth_v = sd * sin_t
        aisle   = self._aisle
        v_span  = v1 - v0

        # How many rows fit?
        n_rows = max(1, int((v_span + aisle + 0.5) / max(depth_v + aisle, 1e-9)))
        while n_rows > 1 and n_rows * depth_v + (n_rows - 1) * aisle > v_span + 0.5:
            n_rows -= 1

        # Pack rows from v1 toward v0
        row_vs = []
        v_cur = v1
        for _ in range(n_rows):
            v_start = v_cur - depth_v
            if v_start < v0 - 0.5:
                break
            row_vs.append(v_start)
            v_cur = v_start - aisle

        # Convert obstacles to local (u,v) bounding boxes
        obs_local = _obstacles_to_local(obstacle_polys_px, cal,
                                        origin, u_hat, v_hat)

        slots, n = [], 0
        for rv in row_vs:
            row_v0, row_v1 = rv, rv + depth_v

            # Find obstacles that overlap this row's v-range
            row_obs = [(ou0, ou1) for (ou0, ou1, ov0, ov1) in obs_local
                       if ov1 > row_v0 - 0.5 and ov0 < row_v1 + 0.5]

            if not row_obs:
                u_positions = self._spread_positions(u0, u1 - shift_u, step_u)
            else:
                row_obs.sort()
                segments = _free_segments(u0, u1, row_obs)
                u_positions = []
                for seg_s, seg_e in segments:
                    u_positions.extend(
                        self._spread_positions(seg_s, seg_e - shift_u, step_u))

            for pu in u_positions:
                poly = [
                    self._to_world_pt(pu,                    rv,            origin, u_hat, v_hat),
                    self._to_world_pt(pu + step_u,           rv,            origin, u_hat, v_hat),
                    self._to_world_pt(pu + step_u + shift_u, rv + depth_v,  origin, u_hat, v_hat),
                    self._to_world_pt(pu + shift_u,          rv + depth_v,  origin, u_hat, v_hat),
                ]
                if _sufficient_zone_overlap(poly, zone_np):
                    slots.append(_slot(n, zone_name, "angled", poly, cal))
                    n += 1

        return slots

    # ── drawing ──────────────────────────────

    @staticmethod
    def draw_slots(frame: np.ndarray, slots: list[dict], show_id: bool = True):
        """Draw all slots onto frame using pixel polygons."""
        overlay = frame.copy()

        for slot in slots:
            color  = SLOT_COLORS.get(slot["status"], SLOT_COLORS["free"])
            pts    = np.array(slot["polygon_px"], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)

        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        for slot in slots:
            color  = SLOT_COLORS.get(slot["status"], SLOT_COLORS["free"])
            pts    = np.array(slot["polygon_px"], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)

            if show_id:
                cx = int(np.mean([p[0] for p in slot["polygon_px"]]))
                cy = int(np.mean([p[1] for p in slot["polygon_px"]]))
                _put_label(frame, slot["slot_id"], (cx, cy), color)

        return frame


# ─── helpers ──────────────────────────────────

def _obstacles_to_local(obstacle_polys_px, cal, origin, u_hat, v_hat):
    """Convert pixel obstacle polygons to local (u,v) bounding boxes.
    Returns list of (u_min, u_max, v_min, v_max)."""
    if not obstacle_polys_px:
        return []
    result = []
    for obs_px in obstacle_polys_px:
        obs_wld = [cal.to_world(tuple(p)) for p in obs_px]
        obs_uv  = [LayoutEngine._to_local(pt, origin, u_hat, v_hat)
                   for pt in obs_wld]
        us = [uv[0] for uv in obs_uv]
        vs = [uv[1] for uv in obs_uv]
        result.append((min(us), max(us), min(vs), max(vs)))
    return result


def _free_segments(u0, u1, obstacles):
    """Given sorted obstacle u-ranges [(obs_u0, obs_u1), ...],
    return free segments [(start, end), ...] between them."""
    segments = []
    cursor = u0
    for obs_u0, obs_u1 in obstacles:
        if cursor < obs_u0 - 0.5:
            segments.append((cursor, obs_u0))
        cursor = max(cursor, obs_u1)
    if cursor < u1 - 0.5:
        segments.append((cursor, u1))
    return segments


def _overlaps_any(poly_px, obstacle_polys, threshold=0.15):
    """Return True if poly_px overlaps any obstacle polygon above threshold."""
    pts = np.array(poly_px, dtype=np.int32)
    x_min = max(int(pts[:, 0].min()) - 1, 0)
    y_min = max(int(pts[:, 1].min()) - 1, 0)
    x_max = int(pts[:, 0].max()) + 2
    y_max = int(pts[:, 1].max()) + 2
    w, h = x_max - x_min, y_max - y_min
    if w <= 0 or h <= 0:
        return True
    slot_mask = np.zeros((h, w), dtype=np.uint8)
    shifted = pts - np.array([x_min, y_min], dtype=np.int32)
    cv2.fillPoly(slot_mask, [shifted], 255)
    slot_area = float(np.count_nonzero(slot_mask))
    if slot_area == 0:
        return True
    for obs in obstacle_polys:
        obs_pts = obs.copy()
        obs_shifted = obs_pts - np.array([x_min, y_min], dtype=np.int32)
        obs_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(obs_mask, [obs_shifted], 255)
        inter = float(np.count_nonzero(cv2.bitwise_and(slot_mask, obs_mask)))
        if inter / slot_area > threshold:
            return True
    return False


def _bbox(poly: list) -> tuple:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _rect(x, y, w, h) -> list:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _poly_area_px(poly_px: list) -> float:
    """Shoelace formula — returns area in pixels squared."""
    pts = np.array(poly_px, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _sufficient_zone_overlap(slot_poly: list, zone_np: np.ndarray,
                              threshold: float = MIN_ZONE_OVERLAP) -> bool:
    """
    Returns True when at least `threshold` fraction of the slot's pixel area
    lies inside the zone polygon.

    Uses a rasterised mask approach — works for any convex/concave zone.
    """
    pts = np.array(slot_poly, dtype=np.int32)
    x_min = max(int(pts[:, 0].min()) - 1, 0)
    y_min = max(int(pts[:, 1].min()) - 1, 0)
    x_max = int(pts[:, 0].max()) + 2
    y_max = int(pts[:, 1].max()) + 2

    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return False

    # Mask for slot
    slot_mask = np.zeros((h, w), dtype=np.uint8)
    shifted_slot = pts - np.array([x_min, y_min], dtype=np.int32)
    cv2.fillPoly(slot_mask, [shifted_slot], 255)

    # Mask for zone
    zone_pts = (zone_np - np.array([[x_min, y_min]], dtype=np.float32)).astype(np.int32)
    zone_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(zone_mask, [zone_pts], 255)

    slot_area  = float(np.count_nonzero(slot_mask))
    if slot_area == 0:
        return False
    inter_area = float(np.count_nonzero(cv2.bitwise_and(slot_mask, zone_mask)))

    # Reject slots whose centroid is clearly outside the parking zone.
    # This prevents visually spilled slots even when overlap ratio is marginal.
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    if cv2.pointPolygonTest(zone_np, (cx, cy), False) < 0:
        return False

    return (inter_area / slot_area) >= threshold


def _inflate_polygon_px(poly_px: list, margin: int = SLOT_INFLATION_PX) -> list:
    """
    Expands a convex polygon outward by `margin` pixels using the centroid
    push technique.  Used by occupancy engine for tolerant overlap testing.
    """
    pts = np.array(poly_px, dtype=np.float64)
    cx  = pts[:, 0].mean()
    cy  = pts[:, 1].mean()
    dirs = pts - np.array([cx, cy])
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    inflated = pts + margin * dirs / norms
    return [(int(p[0]), int(p[1])) for p in inflated]


def _slot(num: int, zone_name: str, ptype: str, poly_world: list, cal) -> dict:
    poly_px      = [cal.to_pixel(pt) for pt in poly_world]
    inflated_px  = _inflate_polygon_px(poly_px, SLOT_INFLATION_PX)
    slot_area    = _poly_area_px(poly_px)
    return {
        "slot_id":           f"{zone_name}_S{num:02d}",
        "type":              ptype,
        "status":            "free",
        "polygon_world":     poly_world,
        "polygon_px":        poly_px,
        "polygon_px_inflated": inflated_px,  # for occupancy overlap tests
        "slot_area_px":      max(slot_area, 1.0),
        "assigned_track_id": None,
        "assigned_user_id":  None,
    }


def _put_label(frame, text, center, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cx, cy = center
    cv2.rectangle(frame, (cx - tw//2 - 2, cy - th - 2),
                  (cx + tw//2 + 2, cy + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (cx - tw//2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    cv2.putText(frame, text, (cx - tw//2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
