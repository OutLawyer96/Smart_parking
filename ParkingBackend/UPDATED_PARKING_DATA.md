# Updated Parking Lot Simulation

## Overview
The parking backend has been enhanced with a larger, more complex parking lot scenario featuring multiple obstacles and different car positioning.

---

## Canvas Size
**Increased from 400×700 to 800×900 pixels** for a larger, more realistic parking lot layout.

---

## New Layout Features

### Target Parking Slot
- **Location**: x=320, y=500 (relocated from x=160, y=400)
- **Size**: 100 wide × 180 long (increased from 90×160)
- **Center Point**: (370, 590) — where the car parks

### Starting Position
- **X**: 150.0 (left lane entry)
- **Y**: 30.0 (top of lot)
- The car navigates from entry to the parking slot with smooth pathfinding

---

## Obstacles (18 total)

### Structural Pillars (4)
- `pillar-1`: (50, 150) 50×50
- `pillar-2`: (700, 150) 50×50
- `pillar-3`: (50, 400) 50×50
- `pillar-4`: (700, 400) 50×50

### Bollards / Concrete Blocks (4)
- `bollard-1`: (150, 250) 20×20
- `bollard-2`: (630, 250) 20×20
- `bollard-3`: (300, 300) 20×20
- `bollard-4`: (500, 300) 20×20

### Trash Bins / Containers (4)
- `trash-bin-1`: (80, 600) 35×35
- `trash-bin-2`: (680, 600) 35×35
- `trash-bin-3`: (250, 750) 35×35
- `trash-bin-4`: (515, 750) 35×35

### EV Charging Stations (2)
- `charging-station-1`: (150, 800) 60×50
- `charging-station-2`: (590, 800) 60×50

### Parked Vehicles (6) — marked as `isDynamic=true`
**Left Lane:**
- `parked-car-1`: (80, 500) 90×180
- `parked-car-2`: (80, 720) 90×180

**Right Lane:**
- `parked-car-3`: (630, 500) 90×180
- `parked-car-4`: (630, 720) 90×180

**Middle Section:**
- `parked-car-5`: (355, 350) 90×180
- `parked-car-6`: (355, 720) 90×180

---

## Car Navigation

### Movement Algorithm
The car now uses **smooth pathfinding** rather than straight-line movement:

1. **Phase 1**: Move toward target Y (drive forward down the lot)
2. **Phase 2**: Adjust X position (lane change) while moving forward
3. **Phase 3**: Snap to final parking position when close

### Waypoint Generation
Each frame emits a list of intermediate waypoints from the car's current position to the target parking slot, allowing the frontend to render a smooth animated path.

---

## JSON Response Structure

```json
{
  "path": [
    { "x": 150.0, "y": 30.0 },    // Current car position (path[0])
    { "x": 152.5, "y": 38.2 },    // Next waypoint
    { "x": 155.1, "y": 46.4 },    // Next waypoint
    // ... more waypoints ...
    { "x": 370.0, "y": 590.0 }    // Final destination
  ],
  "obstacles": [
    { "id": "pillar-1", "rect": { "x": 50, "y": 150, "width": 50, "height": 50 }, "isDynamic": false },
    { "id": "parked-car-1", "rect": { "x": 80, "y": 500, "width": 90, "height": 180 }, "isDynamic": true },
    // ... more obstacles ...
  ],
  "targetSlot": {
    "x": 320,
    "y": 500,
    "width": 100,
    "length": 180
  }
}
```

---

## Frontend Integration

No changes needed! The existing React Native hook automatically handles:
- ✅ Extracting `currentLocation` from `path[0]`
- ✅ Computing remaining `path` from `path.slice(1)`
- ✅ Rendering all 18 obstacles
- ✅ Displaying the relocated target slot

---

## Testing Commands

### Start the backend
```bash
cd /Users/hitendrasingh/Desktop/ParkingBackend
./mvnw spring-boot:run
```

### Connect from Expo app
```javascript
const WS_URL = 'ws://YOUR_LOCAL_IP:8080/ws/parking';
// Use hook: const { currentLocation, path, obstacles, ... } = useParkingWebSocket();
```

### Test with curl (WebSocket)
```bash
# Terminal 1: Listen for messages
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  http://localhost:8080/ws/parking

# Terminal 2: Send commands
# (WebSocket commands: restart, stop, start, ping)
```

---

## What Changed in Code

### File: `ParkingDataService.java`

**Old canvas**: 400×700 → **New canvas**: 800×900

**Old obstacles**: 4 total → **New obstacles**: 18 total

**Old movement**: Fixed lane straight-down → **New movement**: Dynamic pathfinding to (150, 30) → (370, 590)

**Old slot**: (160, 400) 90×160 → **New slot**: (320, 500) 100×180

---

## Future Enhancements

Consider adding:
1. **Multiple simultaneous vehicles** — each with independent paths
2. **Animated obstacles** (e.g., moving vehicles, rotating barriers)
3. **Different parking scenarios** — API to select layout
4. **Real-time obstacle avoidance** — ML-based pathfinding
5. **Difficulty levels** — tight parking vs. spacious lots

