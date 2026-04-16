# Smart Parking System (AI + Backend + Mobile App)

This repository contains a complete smart parking stack with three main parts:

1. `parking_ai` - Python AI perception, slot logic, and live map/state APIs
2. `ParkingBackend` - Spring Boot backend for registration, PIN verification, and client WebSocket streaming
3. `smart-parking-app` - Expo React Native app for driver and gate/operator flows

## Project Working Demo

YouTube: https://www.youtube.com/watch?v=z91Y8kbZax0

## Repository Structure

```text
smart_parking_backend_frontend/
├── parking_ai/          # Computer vision + parking intelligence (FastAPI + WS)
├── ParkingBackend/      # Java Spring Boot integration and business backend
└── smart-parking-app/   # Expo mobile frontend
```

## How the 3 Folders Work Together

- `parking_ai` detects/tracks vehicles and computes slot occupancy and routing data.
- `ParkingBackend` consumes AI outputs and exposes app-ready APIs:
  - vehicle registration
  - PIN verification
  - WebSocket stream for live updates
- `smart-parking-app` calls backend APIs and renders the parking guidance experience on mobile.

## Prerequisites

- Python 3.10+
- Java 17+
- Node.js 18+
- npm 9+
- Maven wrapper (already included in `ParkingBackend`)

## Quick Start (Run Full System)

Open 3 terminals from repository root (`smart_parking_backend_frontend`).

### 1) Start AI service (`parking_ai`)

```bash
cd parking_ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Notes:
- AI API base is `http://127.0.0.1:8000`
- AI WebSocket is `ws://127.0.0.1:8000/ws/live`
- For ESP32 camera, set one of:
  - `ESP32_CAPTURE_URL`
  - `ESP32_CAM_URL`
  - `CAMERA_SOURCE`

Example:

```bash
export ESP32_CAPTURE_URL="http://<esp32-ip>/capture"
```

### 2) Start Backend (`ParkingBackend`)

```bash
cd ParkingBackend
./mvnw spring-boot:run
```

Default backend endpoints:
- Base: `http://localhost:8080`
- Frontend WebSocket: `ws://localhost:8080/ws/parking`

Backend is configured to reach AI at:
- `ai.base-url=http://127.0.0.1:8000`
- `ai.websocket-url=ws://127.0.0.1:8000/ws/live`

### 3) Start Mobile App (`smart-parking-app`)

```bash
cd smart-parking-app
npm install
export EXPO_PUBLIC_BACKEND_URL="http://<your-laptop-ip>:8080"
export EXPO_PUBLIC_ML_MAP_URL="http://<your-laptop-ip>:8000/api/v1/map"
# Optional gate controller
export EXPO_PUBLIC_ESP32_GATE_URL="http://<esp32-ip>"
npx expo start
```

Replace `<your-laptop-ip>` with the machine IP reachable by your phone/emulator.

## Core APIs (Backend)

- `POST /api/parking/register`
- `POST /api/parking/pin/verify`
- `GET /api/parking/map`
- `GET /api/parking/slots`
- WebSocket: `/ws/parking`

For full payload examples, see:
- `ParkingBackend/FRONTEND_API_DOCS.md`

## AI Calibration and Zones (Important)

Before reliable slot assignment, set zones and calibration in `parking_ai`:

```bash
cd parking_ai
python -m spatial.zone_editor
python -m spatial.calibration
```

Then run:

```bash
python main.py
```

## Folder-wise Documentation

- AI docs: `parking_ai/Readme.md`
- Backend docs: `ParkingBackend/README_COMPLETE.md`
- Mobile docs: `smart-parking-app/README.md`

## Troubleshooting

- If backend cannot fetch AI data, confirm AI is running on port `8000`.
- If app cannot connect, ensure phone and laptop are on the same network.
- If WebSocket fails, verify backend is up at port `8080`.
- If camera feed fails, verify ESP32 URL and local network access.

## Suggested Startup Order

1. Start `parking_ai`
2. Start `ParkingBackend`
3. Start `smart-parking-app`

This order ensures backend and app find AI services immediately.
