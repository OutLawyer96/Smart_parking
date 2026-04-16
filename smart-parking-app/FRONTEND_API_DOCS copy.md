# Smart Parking System — Frontend API Documentation

**Base URL:** `http://localhost:8080`  
**Frontend WebSocket:** `ws://localhost:8080/ws/parking`  
**ML ↔ Backend WebSocket (internal):** `ws://127.0.0.1:8000/ws/live` *(backend-managed, frontend never connects here)*

---

## Quick-start Flow

```
── Gate screen (operator) ──────────────────────────────────────────────────────
1.  POST /api/parking/register        → plate number in → PIN + assigned slot out

── Driver's device ─────────────────────────────────────────────────────────────
2.  POST /api/parking/pin/verify      → enter 4-digit PIN → map + slot + route
3.  WS   ws://localhost:8080/ws/parking
         send {"action":"subscribe","pin":"XXXX"}
         → subscription_ack (map + slot + route again, plus live stream starts)
4.  WS keeps sending world_state (car position) + parking_event + slot_update

── Dashboard / admin (optional) ────────────────────────────────────────────────
5.  GET  /api/parking/map             → static lot layout
6.  GET  /api/parking/slots           → live slot occupancy
7.  WS   ws://localhost:8080/ws/parking  (no subscribe → receives all cars)
```

---

## 1. Vehicle Registration (Gate Entry)

### `POST /api/parking/register`

Submit a vehicle registration plate when a car arrives at the gate.  
The backend will:
1. look up the vehicle's dimensions in the database
2. generate / reuse a tracking ID for this plate
3. call the AI to assign the nearest free slot and compute a navigation route
4. generate a **4-digit PIN** tied to this session
5. return everything in one response

**Request Body**
```json
{
  "plate_number": "DL8CAF5031"
}
```

**Response — 200 Slot Assigned**
```json
{
  "plate_number": "DL8CAF5031",
  "tracking_id": 1,
  "status": "assigned",
  "pin": "4823",
  "pin_expires_in_minutes": 30,
  "vehicle": {
    "plate_number": "DL8CAF5031",
    "make": "Toyota",
    "model": "Camry",
    "length_m": 4.7,
    "width_m": 1.8
  },
  "assigned_slot": {
    "slot_id": "parking_0_S05",
    "cx": 118,
    "cy": 226,
    "polygon": [[73,250],[166,248],[164,203],[72,206]],
    "zone": "parking_0",
    "distance_from_exit_m": 0.3
  },
  "route": [
    { "x": 365, "y": 65,  "maneuver": "straight"    },
    { "x": 205, "y": 225, "maneuver": "turn_right"  },
    { "x": 115, "y": 225, "maneuver": "park"        }
  ]
}
```

> **The operator shows the `pin` on the gate display. The driver enters it on their device.**

**Response — 404 Vehicle Not in Database**
```json
{
  "plate_number": "XX99YY0000",
  "status": "vehicle_not_found",
  "message": "Vehicle with plate number 'XX99YY0000' is not registered in the system. Please add it via POST /api/parking/vehicles first."
}
```

**Response — 409 No Free Slots**
```json
{
  "plate_number": "DL8CAF5031",
  "tracking_id": 1,
  "status": "no_slots_available",
  "message": "No free parking slots available at this time.",
  "vehicle": { ... }
}
```

**Response — 503 AI Still Warming Up**
```json
{
  "plate_number": "DL8CAF5031",
  "status": "ai_not_ready",
  "message": "Parking AI system is still initializing. Please try again shortly."
}
```

**Status Reference**

| `status` value         | HTTP | Meaning                                      |
|------------------------|------|----------------------------------------------|
| `assigned`             | 200  | Slot assigned; `pin`, `assigned_slot` + `route` set |
| `vehicle_not_found`    | 404  | Plate not in DB — register vehicle first     |
| `no_slots_available`   | 409  | Lot is full                                  |
| `ai_not_ready`         | 503  | AI warming up — retry in ~1 s                |
| `error`                | 500  | Unexpected internal error                    |

---

## 2. PIN Verification (Driver's Device)

### `POST /api/parking/pin/verify`

The driver enters the 4-digit PIN shown on the gate display.  
Returns **everything the device needs** to render the full parking experience:
map geometry, assigned slot, navigation route, and WebSocket instructions.

**Request Body**
```json
{
  "pin": "4823"
}
```

**Response — 200 Valid PIN**
```json
{
  "pin": "4823",
  "tracking_id": 1,
  "plate_number": "DL8CAF5031",
  "expires_at": "2026-03-13T06:22:00Z",
  "vehicle": {
    "plate_number": "DL8CAF5031",
    "make": "Toyota",
    "model": "Camry",
    "length_m": 4.7,
    "width_m": 1.8
  },
  "assigned_slot": {
    "slot_id": "parking_0_S05",
    "cx": 118,
    "cy": 226,
    "polygon": [[73,250],[166,248],[164,203],[72,206]],
    "zone": "parking_0",
    "distance_from_exit_m": 0.3
  },
  "route": [
    { "x": 365, "y": 65,  "maneuver": "straight"   },
    { "x": 205, "y": 225, "maneuver": "turn_right" },
    { "x": 115, "y": 225, "maneuver": "park"       }
  ],
  "map": {
    "width_px": 440,
    "height_px": 499,
    "scale_m_per_px": 0.001128,
    "layers": { "parking_slots": [...], "entry_exit": [...], "driveways": [...] }
  },
  "websocket": {
    "url": "ws://localhost:8080/ws/parking",
    "subscribe_message": "{\"action\":\"subscribe\",\"pin\":\"4823\"}"
  }
}
```

**Response — 400 Missing PIN**
```json
{ "error": "MISSING_PIN", "message": "pin is required." }
```

**Response — 404 Invalid / Expired PIN**
```json
{ "error": "INVALID_PIN", "message": "Invalid or expired PIN. Please request a new one at the gate." }
```

> **After calling this endpoint, the device should open the WebSocket and send the `subscribe_message` to start receiving live car-position updates.**

---

## 2. Parking Map

### `GET /api/parking/map`

Returns the full parking lot geometry — draw this **once** on startup to render the canvas background.

**Response — 200**
```json
{
  "map": {
    "width_px": 440,
    "height_px": 499,
    "scale_m_per_px": 0.001128,
    "layers": {
      "parking_slots": [
        {
          "slot_id": "parking_0_S00",
          "cx": 265,
          "cy": 421,
          "polygon": [[287,470],[290,376],[243,373],[240,466]],
          "status": "free"
        }
      ],
      "restricted_zones": [],
      "entry_exit": [
        {
          "id": "zone_2",
          "type": "exit",
          "polygon": [[381,51],[364,476],[374,476],[391,53]]
        }
      ],
      "driveways": [
        {
          "id": "zone_1",
          "polygon": [[387,52],[370,479],[399,479],[420,51]]
        }
      ]
    }
  }
}
```

**Response — 503** (AI still loading)
```json
{ "map": null, "message": "AI map is still loading. Retry in a moment." }
```

---

## 3. Live Slot Status

### `GET /api/parking/slots`

Returns the real-time free / occupied status of all slots. Poll this every few seconds **or** rely on `slot_update` WebSocket events instead.

**Response — 200**
```json
{
  "slots": [
    { "slot_id": "parking_0_S00", "status": "free",     "tracking_id": null },
    { "slot_id": "parking_0_S03", "status": "free",     "tracking_id": null },
    { "slot_id": "stopped_T1",    "status": "occupied",  "tracking_id": 1   },
    { "slot_id": "stopped_T2",    "status": "occupied",  "tracking_id": 2   }
  ]
}
```

**`status` values:** `free` · `occupied` · `assigned`

---

## 4. WebSocket — Live Events

### Connection

```
ws://localhost:8080/ws/parking
```

This is the **only** WebSocket your frontend needs to open.

The backend maintains a separate, permanent upstream connection to the ML model at  
`ws://127.0.0.1:8000/ws/live` and relays the relevant events to every connected browser client.

```
ML model ──► ws://127.0.0.1:8000/ws/live ──► Backend ──► ws://localhost:8080/ws/parking ──► Browser
```

---

### Connecting (JavaScript)

```js
const ws = new WebSocket("ws://localhost:8080/ws/parking");

ws.onopen = () => {
  console.log("WebSocket open");
  // If this is the driver's device, subscribe with the PIN immediately
  ws.send(JSON.stringify({ action: "subscribe", pin: "4823" }));
};

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  switch (msg.event) {
    case "connection_ack":    handleAck(msg);          break; // fires before subscribe
    case "subscription_ack":  handleSubscriptionAck(msg); break; // map + slot + route
    case "world_state":       handleWorldState(msg);   break; // car position update
    case "parking_event":     handleParkingEvent(msg); break; // parked / unparked
    case "slot_update":       handleSlotUpdate(msg);   break; // slot status change
    case "upstream_status":   handleUpstream(msg);     break; // AI connection status
    case "error":             handleError(msg);        break; // invalid PIN etc.
  }
};

ws.onclose = () => console.log("WebSocket closed");
ws.onerror = (err) => console.error("WebSocket error", err);

// keep-alive
setInterval(() => ws.readyState === WebSocket.OPEN && ws.send("ping"), 30_000);
```

---

### Event: `connection_ack`
Sent **immediately** when the browser opens the WebSocket (before any subscribe).

```json
{
  "event": "connection_ack",
  "status": "connected",
  "upstream_connected": true,
  "message": "Connected to Smart Parking backend. Send {\"action\":\"subscribe\",\"pin\":\"XXXX\"} to receive updates for your vehicle."
}
```

---

### Event: `subscription_ack`
Sent after a valid `{"action":"subscribe","pin":"XXXX"}` message.  
Contains everything the driver's screen needs: **map + slot + route + tracking_id**.  
After this the session is registered and `world_state` will be filtered to this car only.

```json
{
  "event": "subscription_ack",
  "tracking_id": 1,
  "plate_number": "DL8CAF5031",
  "vehicle": {
    "plate_number": "DL8CAF5031",
    "make": "Toyota",
    "model": "Camry",
    "length_m": 4.7,
    "width_m": 1.8
  },
  "assigned_slot": {
    "slot_id": "parking_0_S05",
    "cx": 118,
    "cy": 226,
    "polygon": [[73,250],[166,248],[164,203],[72,206]],
    "zone": "parking_0",
    "distance_from_exit_m": 0.3
  },
  "route": [
    { "x": 365, "y": 65,  "maneuver": "straight"   },
    { "x": 205, "y": 225, "maneuver": "turn_right" },
    { "x": 115, "y": 225, "maneuver": "park"       }
  ],
  "map": { "width_px": 440, "height_px": 499, "scale_m_per_px": 0.001128, "layers": { ... } },
  "message": "Subscribed. You will now receive live updates for your vehicle."
}
```

---

### Event: `world_state`
Fired at high frequency by the ML model.

**Two shapes depending on session type:**

#### Unsubscribed session (admin / dashboard) — full car list
```json
{
  "event": "world_state",
  "timestamp": "2026-03-13T05:22:19.681986+00:00",
  "cars": [
    {
      "tracking_id": 6,
      "cx": 133,
      "cy": 257,
      "polygon": [[92,236],[175,236],[175,278],[92,278]],
      "heading_deg": 0.0,
      "status": "moving",
      "assigned_slot": null,
      "parked_duration_seconds": null
    },
    {
      "tracking_id": 1,
      "cx": 107,
      "cy": 329,
      "polygon": [[85,289],[129,289],[129,370],[85,370]],
      "heading_deg": 0.0,
      "status": "parked_correct",
      "assigned_slot": "parking_0_S03",
      "parked_duration_seconds": 116.8
    }
  ]
}
```

#### PIN-subscribed session (driver's device) — single car object
```json
{
  "event": "world_state",
  "timestamp": "2026-03-13T05:22:19.681986+00:00",
  "tracking_id": 1,
  "car": {
    "tracking_id": 1,
    "cx": 107,
    "cy": 329,
    "polygon": [[85,289],[129,289],[129,370],[85,370]],
    "heading_deg": 0.0,
    "status": "parked_correct",
    "assigned_slot": "parking_0_S03",
    "parked_duration_seconds": 116.8
  }
}
```

> `car` is `null` if the car is temporarily not visible in the camera frame.

**Car `status` values**

| Value               | Meaning                                      |
|---------------------|----------------------------------------------|
| `moving`            | Car is actively driving through the lot      |
| `parked_correct`    | Car is inside its assigned slot ✓            |
| `parked_incorrect`  | Car stopped outside its assigned slot ✗      |

---

### Event: `parking_event`
Fired once whenever a car **transitions** into or out of a parked state.

```json
{
  "event": "parking_event",
  "tracking_id": 1,
  "type": "parked_correct",
  "slot_id": "parking_0_S03",
  "timestamp": "2026-03-13T05:22:30.000000+00:00"
}
```

| `type` value        | Meaning                                     |
|---------------------|---------------------------------------------|
| `parked_correct`    | Car entered its assigned slot               |
| `parked_incorrect`  | Car stopped outside its assigned slot       |
| `unparked`          | Car left a slot (if emitted by the AI)      |

---

### Event: `slot_update`
Fired whenever a slot changes status.

```json
{
  "event": "slot_update",
  "slot_id": "parking_0_S03",
  "status": "occupied",
  "tracking_id": 1
}
```

| `status` value | Meaning                        |
|----------------|--------------------------------|
| `free`         | Slot is empty                  |
| `occupied`     | A car is parked in this slot   |
| `assigned`     | Slot is reserved but not yet occupied |

---

### Event: `upstream_status`
Emitted by the backend when **its own connection** to the ML model changes.  
Frontend should show a banner / warning when status is `"disconnected"`.

```json
{ "event": "upstream_status", "status": "connected" }
```
```json
{ "event": "upstream_status", "status": "disconnected" }
```

> **Auto-reconnect:** the backend reconnects to the ML model automatically with  
> exponential back-off (1 s → 2 s → 4 s … up to 15 s max).  
> You will receive `{ "event": "upstream_status", "status": "connected" }` as soon as it recovers.

---

### Event: `error`
Sent when a client message (e.g. a subscribe with bad PIN) cannot be processed.

```json
{ "event": "error", "code": "INVALID_PIN", "message": "Invalid or expired PIN. Please try again." }
```
```json
{ "event": "error", "code": "MISSING_PIN", "message": "pin is required." }
```

---

### Client → Server commands

| Send text                                         | Response / effect                          |
|---------------------------------------------------|--------------------------------------------|
| `ping`                                            | `pong`                                     |
| `{"action":"subscribe","pin":"4823"}`             | `subscription_ack` + filtered live stream  |

---

## 5. Vehicle Database Management

Manage the vehicle registry used for dimension lookups.

---

### `GET /api/parking/vehicles`
List all registered vehicles.

**Response — 200**
```json
[
  { "plate_number": "DL8CAF5031", "make": "Toyota",   "model": "Camry",     "length_m": 4.7,  "width_m": 1.8  },
  { "plate_number": "MH12AB1234", "make": "Honda",    "model": "City",      "length_m": 4.2,  "width_m": 1.75 },
  { "plate_number": "KA01ZZ9999", "make": "Ford",     "model": "Endeavour", "length_m": 4.9,  "width_m": 2.0  }
]
```

---

### `GET /api/parking/vehicles/{plateNumber}`
Look up a single vehicle by plate.

```
GET /api/parking/vehicles/DL8CAF5031
```

**Response — 200**
```json
{ "plate_number": "DL8CAF5031", "make": "Toyota", "model": "Camry", "length_m": 4.7, "width_m": 1.8 }
```

**Response — 404**
```json
{ "message": "Vehicle not found: DL8CAF5031" }
```

---

### `POST /api/parking/vehicles`
Register a new vehicle (required before it can enter via `POST /api/parking/register`).

**Request Body**
```json
{
  "plate_number": "MH14ZZ0099",
  "make": "Hyundai",
  "model": "Verna",
  "length_m": 4.44,
  "width_m": 1.73
}
```

**Response — 201 Created** / **200 Updated** (if plate already exists)
```json
{ "plate_number": "MH14ZZ0099", "make": "Hyundai", "model": "Verna", "length_m": 4.44, "width_m": 1.73 }
```

---

### `PUT /api/parking/vehicles/{plateNumber}`
Update an existing vehicle's details.

**Request Body** — same as POST  
**Response — 200** on success, **404** if not found.

---

### `DELETE /api/parking/vehicles/{plateNumber}`

```
DELETE /api/parking/vehicles/MH14ZZ0099
```

**Response — 200**
```json
{ "message": "Vehicle MH14ZZ0099 removed." }
```

---

## 6. System Health

### `GET /api/parking/status`
```json
{
  "connectedClients": 2,
  "isStreaming": true,
  "mapReady": true,
  "websocketUrl": "ws://localhost:8080/ws/parking"
}
```

### `GET /api/parking/health`
```json
{ "status": "UP", "message": "Smart Parking System Backend is running" }
```

---

## Pre-seeded Vehicle Database

| Plate Number | Make        | Model      | Length (m) | Width (m) |
|--------------|-------------|------------|------------|-----------|
| DL8CAF5031   | Toyota      | Camry      | 4.70       | 1.80      |
| MH12AB1234   | Honda       | City       | 4.20       | 1.75      |
| KA01ZZ9999   | Ford        | Endeavour  | 4.90       | 2.00      |
| TN09BC5678   | Maruti      | Swift      | 3.84       | 1.69      |
| GJ05DE2345   | Hyundai     | Creta      | 4.31       | 1.80      |
| UP32XY7890   | Tata        | Nexon      | 3.99       | 1.81      |
| RJ14MN3456   | Volkswagen  | Polo       | 4.05       | 1.75      |
| HR26PQ1122   | BMW         | 3 Series   | 4.71       | 1.83      |
| PB10RS4321   | Mercedes    | C-Class    | 4.69       | 1.81      |
| WB20TU8888   | Kia         | Seltos     | 4.31       | 1.80      |
| MH01AA0001   | Maruti      | Baleno     | 3.99       | 1.74      |
| DL01CG1234   | Hyundai     | i20        | 3.99       | 1.78      |

> If a plate is not found the backend returns **404 `vehicle_not_found`**.  
> Add the vehicle first via `POST /api/parking/vehicles`, then retry registration.

---

## Coordinate System

All `x`, `y`, `cx`, `cy`, and `polygon` values are in **pixels** relative to the top-left corner of the parking map canvas (`width_px` × `height_px`).  
To convert to metres: `metres = pixels × scale_m_per_px`.

---

## Error Response Shape

All non-2xx responses share this envelope:

```json
{
  "status": "<status_string>",
  "message": "<human-readable reason>"
}
```



