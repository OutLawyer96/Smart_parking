"""
Exit-Path Filter — BFS reachability with corridor-aware sacrifice.

After the LayoutEngine generates candidate slots, this module filters
them to the maximum subset where every slot has a clear walkable path
wide enough for a car through free space (parking aisles, drive zones)
to a designated exit zone.

Algorithm overview:
  1. Build a cell grid from the zone map (zone-priority: later zones win).
  2. Pre-filter slots whose centre is no longer in a parking zone after overlay.
  3. Mark pinned slots (pre-parked / stopped cars) — these can NEVER be removed.
  4. Erode the walkable space by half the car width so only paths
     actually wide enough for a vehicle to drive through count as
     reachable.
  5. Corridor-aware greedy loop:
       a. Wide-BFS from exit cells.
       b. Identify reachable vs unreachable slots.
       c. Find cheapest corridor whose removal opens a car-wide path.
       d. Sacrifice if net_gain > 0; else drop unreachable.

Backward-compatible: if no exit zones exist, all slots pass through.

Pinned slots:
  Any slot whose 'pinned' key is True cannot be sacrificed.
"""

import cv2
import numpy as np
from collections import deque

# ── Grid resolution (pixels per cell) ─────────────────────────────────────
GRID_RES = 10

# ── Cell types ─────────────────────────────────────────────────────────────
CELL_BLOCKED  = 0   # restricted / outside all zones
CELL_PARKING  = 1   # parking zone free space (walkable)
CELL_DRIVE    = 2   # drive zone (walkable road)
CELL_EXIT     = 3   # exit zone (walkable + BFS source)
CELL_SLOT     = 4   # occupied by a parking slot (not walkable)

_WALKABLE = (CELL_PARKING, CELL_DRIVE, CELL_EXIT)
_WALKABLE_SET = {CELL_PARKING, CELL_DRIVE, CELL_EXIT}

# ── Proximity tolerance (px) for exit–parking adjacency validation ────────
_TOUCH_TOL = 12

# ── Maximum corridor search depth (slots in a single corridor) ───────────
_MAX_CORRIDOR = 8

# ── Default car width in pixels (used if not supplied) ────────────────────
_DEFAULT_CAR_WIDTH_PX = 40


# ─── public API ──────────────────────────────────────────────────────────────

def filter_slots_by_exit_path(slots, zone_map, car_width_px=None):
    """
    Filter *slots* to the maximum subset where every slot can reach an
    exit zone through a path at least car_width_px wide.

    Parameters
    ----------
    slots        : list[dict]  — slot dicts (need 'polygon_px')
    zone_map     : ZoneMap
    car_width_px : float — minimum path width in pixels.  Falls back to
                   _DEFAULT_CAR_WIDTH_PX if None.

    Returns
    -------
    (filtered_slots, warnings)
    """
    warnings = []
    if not slots:
        return slots, warnings

    car_w = car_width_px or _DEFAULT_CAR_WIDTH_PX
    # Car width in grid cells, then erosion radius so that passages
    # of exactly car_width survive: 2*R + 1 <= car_width_cells.
    car_width_cells = max(1, int(car_w / GRID_RES))
    erode_cells = max(1, (car_width_cells - 1) // 2)

    # ── No exit zones → pass-through ──────────────────────────────────
    exit_zones = zone_map.zones_by_type("exit")
    if not exit_zones:
        warnings.append(
            "[ExitPath] WARNING: No exit zones defined. All slots shown. "
            "Draw exit zones (blue) in the Zone Editor for path validation."
        )
        return slots, warnings

    warnings.extend(_validate_parking_exits(zone_map))

    # ── Pre-filter: centre must be in a parking zone ─────────────────
    valid_indices = []
    for i, slot in enumerate(slots):
        cx = int(np.mean([p[0] for p in slot["polygon_px"]]))
        cy = int(np.mean([p[1] for p in slot["polygon_px"]]))
        zone = zone_map.get_zone_at((cx, cy))
        if zone and zone["type"] == "parking":
            valid_indices.append(i)

    if not valid_indices:
        warnings.append("[ExitPath] No slots remain after overlay pre-filter.")
        return [], warnings

    working_slots = [slots[i] for i in valid_indices]

    # ── Build zone grid ──────────────────────────────────────────────
    grid_h, grid_w, base_grid = _build_zone_grid(zone_map, working_slots)

    # ── Map each slot → its grid cells ───────────────────────────────
    slot_cells = [
        _polygon_to_cells(s["polygon_px"], grid_h, grid_w)
        for s in working_slots
    ]

    # ── Identify pinned (immovable) slots ────────────────────────────
    pinned = {i for i, s in enumerate(working_slots) if s.get("pinned", False)}

    # ── Exit-distance for each slot (on empty grid) ──────────────────
    exit_dist = _compute_exit_distances(base_grid, slot_cells)

    # ── Slot adjacency ───────────────────────────────────────────────
    adjacency = _build_slot_adjacency(slot_cells)

    # ── Greedy corridor sacrifice loop ───────────────────────────────
    active = set(range(len(working_slots)))

    for _ in range(len(working_slots)):
        reachable, unreachable = _eval_reachability_wide(
            active, slot_cells, base_grid, grid_h, grid_w, erode_cells
        )

        if not unreachable:
            break

        # ── Find best corridor to sacrifice ──────────────────────────
        best_corridor = None
        best_net      = 0
        best_avg_dist = float("inf")

        removable_reachable = reachable - pinned

        # 1) Single slot sacrifices
        for candidate in removable_reachable:
            trial = active - {candidate}
            _, trial_unreach = _eval_reachability_wide(
                trial, slot_cells, base_grid, grid_h, grid_w, erode_cells
            )
            newly_reached = len(unreachable) - len(trial_unreach)
            net = newly_reached - 1
            avg_d = exit_dist[candidate]

            if (net > best_net
                    or (net == best_net and net > 0 and avg_d < best_avg_dist)):
                best_net      = net
                best_corridor = {candidate}
                best_avg_dist = avg_d

        # 2) Multi-slot corridors
        for start in removable_reachable:
            corridors = _find_corridors(
                start, removable_reachable, adjacency, _MAX_CORRIDOR
            )
            for corridor in corridors:
                if len(corridor) <= 1:
                    continue
                trial = active - corridor
                _, trial_unreach = _eval_reachability_wide(
                    trial, slot_cells, base_grid, grid_h, grid_w, erode_cells
                )
                newly_reached = len(unreachable) - len(trial_unreach)
                net = newly_reached - len(corridor)
                avg_d = sum(exit_dist[i] for i in corridor) / len(corridor)

                if (net > best_net
                        or (net == best_net and net > 0
                            and avg_d < best_avg_dist)):
                    best_net      = net
                    best_corridor = corridor
                    best_avg_dist = avg_d

        if best_net > 0 and best_corridor:
            active -= best_corridor
        else:
            active -= unreachable
            break

    filtered = [working_slots[i] for i in sorted(active)]

    removed = len(slots) - len(filtered)
    if removed:
        warnings.append(
            f"[ExitPath] {removed} slot(s) removed — no car-wide exit path."
        )

    return filtered, warnings


# ─── slot adjacency ──────────────────────────────────────────────────────

def _build_slot_adjacency(slot_cells):
    cell_to_slot = {}
    for idx, cells in enumerate(slot_cells):
        for cell in cells:
            cell_to_slot.setdefault(cell, set()).add(idx)

    adj = {}
    for idx, cells in enumerate(slot_cells):
        neighbours = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                for nbr in cell_to_slot.get((nr, nc), set()):
                    if nbr != idx:
                        neighbours.add(nbr)
        adj[idx] = neighbours
    return adj


def _find_corridors(start, removable, adjacency, max_depth):
    corridors = []
    queue = deque()
    queue.append((frozenset({start}), start))
    visited_sets = {frozenset({start})}

    while queue:
        path, last = queue.popleft()
        corridors.append(path)

        if len(path) >= max_depth:
            continue

        for nbr in adjacency.get(last, set()):
            if nbr in removable and nbr not in path:
                new_path = path | {nbr}
                if new_path not in visited_sets:
                    visited_sets.add(new_path)
                    queue.append((new_path, nbr))

    return corridors


# ─── wide-path reachability ──────────────────────────────────────────────

def _eval_reachability_wide(active, slot_cells, base_grid, grid_h, grid_w,
                            erode_cells):
    """
    Like _eval_reachability but the BFS only floods through corridors
    at least `erode_cells * 2` cells wide (≈ car width).

    Approach: build the slot-blocked grid, create a binary walkable mask,
    erode it by `erode_cells`, then BFS on the eroded mask.  Slots are
    reachable if any of their edge cells touches a reached cell in the
    *un-eroded* grid that is connected to the eroded BFS.
    """
    grid = base_grid.copy()
    for idx in active:
        for (r, c) in slot_cells[idx]:
            if grid[r, c] != CELL_EXIT:
                grid[r, c] = CELL_SLOT

    # Build binary walkable mask (1 = walkable, 0 = blocked)
    walkable_mask = np.isin(grid, list(_WALKABLE)).astype(np.uint8)

    # Remember exit cells — they must survive erosion because the exit
    # zone is the *destination*, not a passage that needs to be car-wide.
    exit_mask = (grid == CELL_EXIT).astype(np.uint8)

    # Erode: removes narrow passages < car width
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (erode_cells * 2 + 1, erode_cells * 2 + 1)
    )
    eroded = cv2.erode(walkable_mask, kernel)

    # Re-inject exit cells so the BFS can always start from them
    eroded = np.maximum(eroded, exit_mask)

    # BFS on eroded mask from exit cells
    reached_core = _bfs_on_mask(grid, eroded)

    # Dilate the reached region back so slots at edges can connect
    reached_wide = cv2.dilate(
        reached_core.astype(np.uint8), kernel
    ).astype(bool)

    # Combine: a cell is reached if it was reached by the dilated core
    # AND is walkable in the original grid (or is an exit cell)
    reached = reached_wide & ((walkable_mask > 0) | (exit_mask > 0))

    reachable = set()
    unreachable = set()
    for idx in active:
        if _slot_reachable(slot_cells[idx], grid, reached):
            reachable.add(idx)
        else:
            unreachable.add(idx)

    return reachable, unreachable


def _bfs_on_mask(grid, eroded_mask):
    """
    BFS from exit cells that survive in the eroded mask.
    Returns a bool array of reached cells.
    """
    h, w = grid.shape
    reached = np.zeros((h, w), dtype=bool)
    queue = deque()

    # Seed from ALL exit cells — exit zone need not be car-wide
    exit_cells = np.argwhere(grid == CELL_EXIT)
    for r, c in exit_cells:
        reached[r, c] = True
        queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not reached[nr, nc]:
                if eroded_mask[nr, nc]:
                    reached[nr, nc] = True
                    queue.append((nr, nc))

    return reached


# ─── exit distance computation ────────────────────────────────────────────

def _compute_exit_distances(base_grid, slot_cells):
    h, w = base_grid.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    queue = deque()

    for r in range(h):
        for c in range(w):
            if base_grid[r, c] == CELL_EXIT:
                dist[r, c] = 0
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] == -1:
                if base_grid[nr, nc] in _WALKABLE_SET:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))

    result = []
    for cells in slot_cells:
        if not cells:
            result.append(float("inf"))
            continue
        min_d = float("inf")
        for r, c in cells:
            if dist[r, c] >= 0 and dist[r, c] < min_d:
                min_d = dist[r, c]
        result.append(min_d)

    return result


# ─── grid construction ────────────────────────────────────────────────────

def _build_zone_grid(zone_map, slots):
    """
    Build a cell grid covering all zones and slots.
    Zones are painted in list order so later zones override earlier ones
    (zone-priority / overlay behaviour).

    Returns (grid_h, grid_w, grid).
    """
    # Determine pixel bounds
    all_pts = []
    for z in zone_map.all_zones():
        all_pts.extend(z["points"])
    for s in slots:
        all_pts.extend(s["polygon_px"])
    if not all_pts:
        return 1, 1, np.zeros((1, 1), dtype=np.uint8)

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    w_px = max(xs) + GRID_RES * 2
    h_px = max(ys) + GRID_RES * 2

    # Pixel-level type map — later zones overwrite earlier ones
    type_map = np.full((h_px, w_px), CELL_BLOCKED, dtype=np.uint8)
    for zone in zone_map.all_zones():
        pts = np.array(zone["points"], dtype=np.int32)
        ztype = zone["type"]
        if ztype == "exit":
            val = CELL_EXIT
        elif ztype == "drive":
            val = CELL_DRIVE
        elif ztype == "parking":
            val = CELL_PARKING
        else:
            val = CELL_BLOCKED
        cv2.fillPoly(type_map, [pts], int(val))

    # Down-sample: sample the centre of each grid cell
    grid_h = h_px // GRID_RES
    grid_w = w_px // GRID_RES
    if grid_h == 0 or grid_w == 0:
        return 1, 1, np.zeros((1, 1), dtype=np.uint8)

    ys_idx = np.arange(grid_h) * GRID_RES + GRID_RES // 2
    xs_idx = np.arange(grid_w) * GRID_RES + GRID_RES // 2
    ys_idx = np.clip(ys_idx, 0, h_px - 1)
    xs_idx = np.clip(xs_idx, 0, w_px - 1)
    grid = type_map[np.ix_(ys_idx, xs_idx)]

    return grid_h, grid_w, grid


def _polygon_to_cells(polygon_px, grid_h, grid_w):
    """Rasterise a pixel-space polygon to a set of (row, col) grid cells."""
    pts = np.array(polygon_px, dtype=np.int32)
    scaled = (pts.astype(np.float64) / GRID_RES).astype(np.int32)
    mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
    cv2.fillPoly(mask, [scaled], 1)
    return set(map(tuple, np.argwhere(mask > 0)))


def _slot_reachable(cells, grid, reached):
    """
    A slot is reachable when any of its cells is 4-adjacent to a
    reached cell.
    """
    h, w = grid.shape
    for r, c in cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and reached[nr, nc]:
                return True
    return False


# ─── parking-zone exit validation ─────────────────────────────────────────

def _validate_parking_exits(zone_map):
    """Warn about parking zones that have no exit zone touching/overlapping."""
    warnings = []
    exit_zones = zone_map.zones_by_type("exit")
    parking_zones = zone_map.zones_by_type("parking")

    for pz in parking_zones:
        has_exit = False
        for ez in exit_zones:
            if _zones_touch(pz, ez):
                has_exit = True
                break
        if not has_exit:
            warnings.append(
                f"[ExitPath] WARNING: '{pz['name']}' has no exit zone "
                f"touching it. Draw an exit zone (blue) in contact with it."
            )
    return warnings


def _zones_touch(zone_a, zone_b):
    """Check if two zones touch or overlap (within _TOUCH_TOL pixels)."""
    pts_a = np.array(zone_a["points"], dtype=np.int32)
    pts_b = np.array(zone_b["points"], dtype=np.int32)

    all_pts = np.vstack([pts_a, pts_b])
    x_min = int(all_pts[:, 0].min()) - _TOUCH_TOL
    y_min = int(all_pts[:, 1].min()) - _TOUCH_TOL
    x_max = int(all_pts[:, 0].max()) + _TOUCH_TOL + 1
    y_max = int(all_pts[:, 1].max()) + _TOUCH_TOL + 1

    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return False

    offset = np.array([[x_min, y_min]], dtype=np.int32)

    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_a, [pts_a - offset], 255)
    cv2.fillPoly(mask_b, [pts_b - offset], 255)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_TOUCH_TOL * 2 + 1, _TOUCH_TOL * 2 + 1)
    )
    mask_b_dilated = cv2.dilate(mask_b, kernel)

    return bool(np.any(cv2.bitwise_and(mask_a, mask_b_dilated)))
