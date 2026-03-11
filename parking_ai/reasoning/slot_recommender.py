"""
Slot Recommender — picks the best free slot for an incoming vehicle.

Strategy: fill the **farthest row** from the exit first, then move to the
next row.  Within a row, slots are filled in spatial order (top-to-bottom,
left-to-right).  This keeps the exit path clear and packs cars efficiently
from the back of the lot.

Rows are identified by grouping free slots that share the same BFS grid
distance from the exit zones.

Usage:
    from reasoning.slot_recommender import SlotRecommender

    recommender = SlotRecommender(zone_map, calibration)
    result = recommender.recommend(occupancy_engine, vehicle_dims)
    # result = {"slot": <slot_dict>, "distance_from_exit_m": float} or None
"""

import numpy as np
import cv2
from collections import deque


class SlotRecommender:

    def __init__(self, zone_map, calibration):
        self._zone_map = zone_map
        self._calibration = calibration
        # Cache exit distance grid (rebuilt when slots change)
        self._dist_grid = None
        self._grid_res = 10
        self._grid_shape = None

    def recommend(self, occupancy_engine, vehicle_dims=None):
        """
        Pick the best free slot from the occupancy engine.

        Strategy: fill the farthest row first, then move to the next.
        Within a row, slots are filled top-to-bottom / left-to-right.

        Parameters
        ----------
        occupancy_engine : OccupancyEngine with current slot states
        vehicle_dims     : dict  {"length_m": float, "width_m": float}
                           (optional — reserved for future use)

        Returns
        -------
        dict with keys: slot, distance_from_exit_m
        or None if no free slot is available.
        """
        free = occupancy_engine.free_slots()
        if not free:
            return None

        exit_distances = self._compute_exit_distances(free)

        # Group free slots by exit distance — same distance ≈ same row
        rows: dict[float, list] = {}
        for slot, dist in zip(free, exit_distances):
            rows.setdefault(dist, []).append(slot)

        # Iterate rows from farthest to nearest
        for dist in sorted(rows, reverse=True):
            row_slots = rows[dist]
            # Sort within row by spatial position for consistent fill order
            row_slots.sort(key=lambda s: (
                int(np.mean([p[1] for p in s["polygon_px"]])),  # y
                int(np.mean([p[0] for p in s["polygon_px"]])),  # x
            ))
            best_slot = row_slots[0]
            dist_m = self._distance_to_metres(best_slot, dist)
            return {
                "slot": best_slot,
                "distance_from_exit_m": round(dist_m, 2),
            }

        return None

    def _distance_to_metres(self, slot, grid_dist):
        """Convert BFS grid distance to real-world metres."""
        dist_m = grid_dist * self._grid_res  # grid cells → pixels (rough)
        if self._calibration.is_calibrated:
            cx = int(np.mean([p[0] for p in slot["polygon_px"]]))
            cy = int(np.mean([p[1] for p in slot["polygon_px"]]))
            world_slot = self._calibration.to_world((cx, cy))
            exit_zones = self._zone_map.zones_by_type("exit")
            if exit_zones:
                min_real_dist = float("inf")
                for ez in exit_zones:
                    ecx = int(np.mean([p[0] for p in ez["points"]]))
                    ecy = int(np.mean([p[1] for p in ez["points"]]))
                    world_exit = self._calibration.to_world((ecx, ecy))
                    d = np.hypot(world_slot[0] - world_exit[0],
                                 world_slot[1] - world_exit[1])
                    min_real_dist = min(min_real_dist, d)
                dist_m = min_real_dist / 100.0  # cm → m
            else:
                dist_m = dist_m / 100.0
        else:
            dist_m = float(dist_m)
        return dist_m

    def _compute_exit_distances(self, slots):
        """
        BFS distance from exit zones to each slot centre in grid cells.
        """
        # Determine pixel bounds
        all_pts = []
        for z in self._zone_map.all_zones():
            all_pts.extend(z["points"])
        for s in slots:
            all_pts.extend(s["polygon_px"])
        if not all_pts:
            return [0.0] * len(slots)

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        w_px = max(xs) + self._grid_res * 2
        h_px = max(ys) + self._grid_res * 2

        # Build type map
        type_map = np.zeros((h_px, w_px), dtype=np.uint8)
        for zone in self._zone_map.all_zones():
            pts = np.array(zone["points"], dtype=np.int32)
            ztype = zone["type"]
            if ztype == "exit":
                val = 3
            elif ztype == "drive":
                val = 2
            elif ztype == "parking":
                val = 1
            else:
                val = 0
            cv2.fillPoly(type_map, [pts], int(val))

        grid_h = h_px // self._grid_res
        grid_w = w_px // self._grid_res
        if grid_h == 0 or grid_w == 0:
            return [0.0] * len(slots)

        ys_idx = np.arange(grid_h) * self._grid_res + self._grid_res // 2
        xs_idx = np.arange(grid_w) * self._grid_res + self._grid_res // 2
        ys_idx = np.clip(ys_idx, 0, h_px - 1)
        xs_idx = np.clip(xs_idx, 0, w_px - 1)
        grid = type_map[np.ix_(ys_idx, xs_idx)]

        walkable = {1, 2, 3}  # parking, drive, exit

        # BFS from exit cells
        dist = np.full((grid_h, grid_w), -1, dtype=np.int32)
        queue = deque()
        for r in range(grid_h):
            for c in range(grid_w):
                if grid[r, c] == 3:  # exit
                    dist[r, c] = 0
                    queue.append((r, c))

        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w and dist[nr, nc] == -1:
                    if grid[nr, nc] in walkable:
                        dist[nr, nc] = dist[r, c] + 1
                        queue.append((nr, nc))

        # Lookup distance for each slot centre
        results = []
        for slot in slots:
            cx = int(np.mean([p[0] for p in slot["polygon_px"]]))
            cy = int(np.mean([p[1] for p in slot["polygon_px"]]))
            gr = min(cy // self._grid_res, grid_h - 1)
            gc = min(cx // self._grid_res, grid_w - 1)
            d = dist[gr, gc]
            results.append(float(d) if d >= 0 else 0.0)

        return results
