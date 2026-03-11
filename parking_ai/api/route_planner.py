"""
Route Planner — A* pathfinding from a car's position to a target slot.

Builds a walkable grid from the zone map (drive + parking + exit cells),
marks slots as obstacles (except the target), and runs A* with 8-directional
movement.  Returns a list of pixel-space waypoints with maneuver hints.

Usage:
    from api.route_planner import RoutePlanner
    planner = RoutePlanner(zone_map)
    route = planner.plan(start_px=(100, 50), target_slot=slot_dict)
"""

import cv2
import heapq
import numpy as np

GRID_RES = 10  # px per cell — matches exit_path.py


class RoutePlanner:

    def __init__(self, zone_map):
        self._zone_map = zone_map
        self._grid = None
        self._grid_h = 0
        self._grid_w = 0
        self._build_grid()

    def _build_grid(self):
        all_pts = []
        for z in self._zone_map.all_zones():
            all_pts.extend(z["points"])
        if not all_pts:
            self._grid = np.zeros((1, 1), dtype=np.uint8)
            self._grid_h = 1
            self._grid_w = 1
            return

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        w_px = max(xs) + GRID_RES * 2
        h_px = max(ys) + GRID_RES * 2

        type_map = np.zeros((h_px, w_px), dtype=np.uint8)
        for zone in self._zone_map.all_zones():
            pts = np.array(zone["points"], dtype=np.int32)
            ztype = zone["type"]
            if ztype in ("drive", "parking", "exit"):
                cv2.fillPoly(type_map, [pts], 1)

        self._grid_h = h_px // GRID_RES
        self._grid_w = w_px // GRID_RES
        if self._grid_h == 0 or self._grid_w == 0:
            self._grid = np.zeros((1, 1), dtype=np.uint8)
            self._grid_h = 1
            self._grid_w = 1
            return

        ys_idx = np.arange(self._grid_h) * GRID_RES + GRID_RES // 2
        xs_idx = np.arange(self._grid_w) * GRID_RES + GRID_RES // 2
        ys_idx = np.clip(ys_idx, 0, h_px - 1)
        xs_idx = np.clip(xs_idx, 0, w_px - 1)
        self._grid = type_map[np.ix_(ys_idx, xs_idx)]

    def plan(self, start_px, target_slot, occupied_slots=None):
        """
        A* path from start_px to the centre of target_slot.

        Parameters
        ----------
        start_px       : (x, y) pixel position of the car
        target_slot    : slot dict with 'polygon_px'
        occupied_slots : list of slot dicts to mark as obstacles

        Returns
        -------
        list of {"x": int, "y": int, "maneuver": str}
        """
        grid = self._grid.copy()

        # Mark occupied slots as blocked
        if occupied_slots:
            for s in occupied_slots:
                pts = np.array(s["polygon_px"], dtype=np.int32)
                scaled = (pts.astype(np.float64) / GRID_RES).astype(np.int32)
                mask = np.zeros((self._grid_h, self._grid_w), dtype=np.uint8)
                cv2.fillPoly(mask, [scaled], 1)
                grid[mask > 0] = 0

        # Target slot centre
        tcx = int(np.mean([p[0] for p in target_slot["polygon_px"]]))
        tcy = int(np.mean([p[1] for p in target_slot["polygon_px"]]))

        sr = min(int(start_px[1]) // GRID_RES, self._grid_h - 1)
        sc = min(int(start_px[0]) // GRID_RES, self._grid_w - 1)
        er = min(tcy // GRID_RES, self._grid_h - 1)
        ec = min(tcx // GRID_RES, self._grid_w - 1)

        # Make sure start and end are walkable
        grid[sr, sc] = 1
        grid[er, ec] = 1

        path_cells = self._astar(grid, (sr, sc), (er, ec))
        if not path_cells:
            # Fallback: direct line
            return [
                {"x": int(start_px[0]), "y": int(start_px[1]), "maneuver": "straight"},
                {"x": tcx, "y": tcy, "maneuver": "park"},
            ]

        # Convert cells back to pixel waypoints and simplify
        raw_points = [
            (c * GRID_RES + GRID_RES // 2, r * GRID_RES + GRID_RES // 2)
            for r, c in path_cells
        ]
        simplified = self._simplify(raw_points)

        # Add maneuver hints
        route = []
        for i, (x, y) in enumerate(simplified):
            if i == len(simplified) - 1:
                maneuver = "park"
            elif i == 0:
                maneuver = "straight"
            else:
                maneuver = self._detect_maneuver(
                    simplified[i - 1], (x, y), simplified[i + 1]
                )
            route.append({"x": x, "y": y, "maneuver": maneuver})

        return route

    def _astar(self, grid, start, end):
        h, w = grid.shape
        sr, sc = start
        er, ec = end

        open_set = [(0, sr, sc)]
        g_score = {(sr, sc): 0}
        came_from = {}

        # 8-directional movement
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
        costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]

        while open_set:
            _, cr, cc = heapq.heappop(open_set)

            if (cr, cc) == (er, ec):
                # Reconstruct
                path = [(cr, cc)]
                while (cr, cc) in came_from:
                    cr, cc = came_from[(cr, cc)]
                    path.append((cr, cc))
                path.reverse()
                return path

            for (dr, dc), cost in zip(dirs, costs):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] > 0:
                    ng = g_score[(cr, cc)] + cost
                    if (nr, nc) not in g_score or ng < g_score[(nr, nc)]:
                        g_score[(nr, nc)] = ng
                        f = ng + abs(nr - er) + abs(nc - ec)
                        heapq.heappush(open_set, (f, nr, nc))
                        came_from[(nr, nc)] = (cr, cc)

        return []  # no path

    def _simplify(self, points, tolerance=15.0):
        """Douglas-Peucker line simplification."""
        if len(points) <= 2:
            return points
        pts = np.array(points, dtype=np.float64)
        # Use OpenCV's approxPolyDP
        epsilon = tolerance
        approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2).astype(np.float32),
                                  epsilon, closed=False)
        result = [(int(p[0][0]), int(p[0][1])) for p in approx]
        # Always include start and end
        if result[0] != (int(points[0][0]), int(points[0][1])):
            result.insert(0, (int(points[0][0]), int(points[0][1])))
        if result[-1] != (int(points[-1][0]), int(points[-1][1])):
            result.append((int(points[-1][0]), int(points[-1][1])))
        return result

    def _detect_maneuver(self, prev, curr, nxt):
        dx1 = curr[0] - prev[0]
        dy1 = curr[1] - prev[1]
        dx2 = nxt[0] - curr[0]
        dy2 = nxt[1] - curr[1]
        # Cross product for turn direction
        cross = dx1 * dy2 - dy1 * dx2
        if abs(cross) < 50:
            return "straight"
        return "turn_right" if cross > 0 else "turn_left"
