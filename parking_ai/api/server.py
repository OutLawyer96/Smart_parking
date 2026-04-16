"""
Parking API — REST compatibility + WebSocket-first live API.

REST endpoints are kept for backward compatibility, but WebSocket is the
recommended integration path for continuously rendered frontends.

WebSocket:
    ws://host:8000/ws/live

    Server push (continuous):
        world_state, slots_state, assignments_state, parking_event, slot_update,
        map_state (on layout/full-sync)

    Client actions (request/reply over same socket):
        gate_match, get_map, get_slots, get_assignments
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from api.state import ParkingState


# ── Request / Response models ────────────────────────────────────────────────

class VehicleDims(BaseModel):
    length_m: float
    width_m: float


class GateMatchRequest(BaseModel):
    tracking_id: int
    vehicle: VehicleDims


# ── WebSocket manager ───────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        gone = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                gone.append(ws)
        for ws in gone:
            self.disconnect(ws)


ws_manager = ConnectionManager()


def _compute_gate_match(state: ParkingState, req: GateMatchRequest) -> dict:
    if not state.occupancy_engine:
        raise HTTPException(status_code=503, detail="System not ready — no occupancy data yet.")

    vehicle_dims = {
        "length_m": req.vehicle.length_m,
        "width_m": req.vehicle.width_m,
    }

    # Find best slot (farthest from exit)
    result = state.slot_recommender.recommend(
        state.occupancy_engine, vehicle_dims
    )

    if result is None:
        raise HTTPException(status_code=409, detail="No free slots available.")

    slot = result["slot"]
    distance_m = result["distance_from_exit_m"]

    # Plan route from car's current position to the slot
    car_pos = None
    for det in state.get_detections():
        if det["track_id"] == req.tracking_id:
            x1, y1, x2, y2 = det["bbox"]
            car_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
            break

    if car_pos is None:
        # Car not yet visible — use gate/entry zone centre as start
        entry_zones = state.zone_map.zones_by_type("exit")
        if entry_zones:
            pts = entry_zones[0]["points"]
            car_pos = (
                int(sum(p[0] for p in pts) / len(pts)),
                int(sum(p[1] for p in pts) / len(pts)),
            )
        else:
            car_pos = (0, 0)

    route = state.route_planner.plan(
        start_px=car_pos,
        target_slot=slot,
        occupied_slots=state.occupancy_engine.occupied_slots(),
    )

    # Mark slot as assigned
    slot["status"] = "assigned"
    slot["assigned_track_id"] = req.tracking_id

    # Store assignment
    state.add_assignment(req.tracking_id, slot, route, vehicle_dims)

    # Build slot polygon for response
    cx = int(sum(p[0] for p in slot["polygon_px"]) / len(slot["polygon_px"]))
    cy = int(sum(p[1] for p in slot["polygon_px"]) / len(slot["polygon_px"]))

    return {
        "event": "gate_match_result",
        "tracking_id": req.tracking_id,
        "assigned_slot": {
            "slot_id": slot["slot_id"],
            "cx": cx,
            "cy": cy,
            "polygon": [[int(p[0]), int(p[1])] for p in slot["polygon_px"]],
            "zone": slot["slot_id"].rsplit("_S", 1)[0],
            "distance_from_exit_m": distance_m,
        },
        "route": route,
    }


# ── Background broadcast loop ───────────────────────────────────────────────

async def _broadcast_loop():
    """Push continuous snapshots + delta events to all WS clients."""
    state = ParkingState()
    last_slots_push = 0.0
    last_assignments_push = 0.0
    while True:
        await asyncio.sleep(0.2)

        if not ws_manager.active:
            continue

        # World state (all car positions)
        world = state.build_world_state()
        if world.get("cars") is not None:
            await ws_manager.broadcast(world)

        now = time.time()
        # Push slots/assignments continuously for frontend rendering.
        if now - last_slots_push >= 0.4:
            await ws_manager.broadcast(state.build_slots_summary())
            last_slots_push = now

        if now - last_assignments_push >= 0.4:
            await ws_manager.broadcast(state.build_assignments_summary())
            last_assignments_push = now

        # Full map + slot refresh when layout changed.
        if state.drain_full_sync_needed():
            await ws_manager.broadcast(state.build_map_snapshot())
            await ws_manager.broadcast(state.build_slots_summary())
            await ws_manager.broadcast(state.build_assignments_summary())

        # Parking events (state transitions)
        for ev in state.drain_events():
            await ws_manager.broadcast(ev)

        # Slot updates (status changes)
        for su in state.drain_slot_updates():
            await ws_manager.broadcast(su)


# ── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_broadcast_loop())
    yield
    task.cancel()


app = FastAPI(title="Smart Parking AI", version="1.0.0", lifespan=lifespan)


# ── REST: POST /api/v1/gate/match ────────────────────────────────────────────

@app.post("/api/v1/gate/match")
def gate_match(req: GateMatchRequest):
    state = ParkingState()
    # Compatibility endpoint. WebSocket action 'gate_match' is preferred.
    return _compute_gate_match(state, req)


# ── REST: GET /api/v1/map ────────────────────────────────────────────────────

@app.get("/api/v1/map")
def get_map():
    state = ParkingState()
    # Compatibility endpoint. WebSocket action 'get_map' is preferred.
    return state.build_map_snapshot()


# ── REST: GET /api/v1/slots ──────────────────────────────────────────────────

@app.get("/api/v1/slots")
def get_slots():
    state = ParkingState()
    # Compatibility endpoint. WebSocket action 'get_slots' is preferred.
    return state.build_slots_summary()


# ── WebSocket: /ws/live ──────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws_manager.connect(ws)
    state = ParkingState()
    try:
        # Send initial full snapshots for immediate frontend render.
        await ws.send_text(json.dumps(state.build_map_snapshot()))
        await ws.send_text(json.dumps(state.build_slots_summary()))
        await ws.send_text(json.dumps(state.build_assignments_summary()))

        while True:
            msg = await ws.receive_text()
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({
                    "event": "error",
                    "message": "Invalid JSON payload",
                }))
                continue

            action = payload.get("action")
            if action == "ping":
                await ws.send_text(json.dumps({"event": "pong", "ts": time.time()}))
            elif action == "get_map":
                await ws.send_text(json.dumps(state.build_map_snapshot()))
            elif action == "get_slots":
                await ws.send_text(json.dumps(state.build_slots_summary()))
            elif action == "get_assignments":
                await ws.send_text(json.dumps(state.build_assignments_summary()))
            elif action == "gate_match":
                try:
                    req = GateMatchRequest(**payload)
                    result = _compute_gate_match(state, req)
                    await ws.send_text(json.dumps(result))
                except HTTPException as e:
                    await ws.send_text(json.dumps({
                        "event": "gate_match_result",
                        "ok": False,
                        "status": e.status_code,
                        "message": str(e.detail),
                    }))
                except Exception as e:
                    await ws.send_text(json.dumps({
                        "event": "gate_match_result",
                        "ok": False,
                        "status": 500,
                        "message": str(e),
                    }))
            else:
                await ws.send_text(json.dumps({
                    "event": "error",
                    "message": f"Unknown action: {action}",
                }))
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
