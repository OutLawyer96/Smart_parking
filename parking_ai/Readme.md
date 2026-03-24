# Smart Parking System — AI Perception Layer

Real-time vehicle detection, tracking, and zone-based occupancy for a top-down parking camera.

---

## Initial Setup

1. Create virtual environment:

   ```
   python -m venv venv
   ```

2. Activate:

   ```
   venv\Scripts\activate

   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Model weights are hosted on GitHub Releases and loaded by default:
   ```
   https://github.com/OutLawyer96/Smart_parking/releases/download/v1.0/best.pt
   ```
   No manual model copy is required for normal use.

---

## Running the System

### Main detection + tracking loop

```
python main.py
```

- Connects to the IP camera stream defined in `perception/camera.py`
- Runs YOLOv8 detection + ByteTrack tracking on every frame
- Draws bounding boxes showing: label, confidence %, and track ID
- HUD shows live FPS and car count
- Press `Q` to quit

---

## Zone Setup (do this before using occupancy features)

Zones define where cars are allowed to park, where they can drive, and where they cannot go at all. Every area that is NOT explicitly defined as a zone is automatically treated as **restricted**.

### Zone types

| Type         | Color  | Meaning                                  |
| ------------ | ------ | ---------------------------------------- |
| `parking`    | Green  | Cars are allowed to park here            |
| `drive`      | Orange | Moving lane / entry / exit path          |
| `restricted` | Red    | No-go area (trees, walls, out-of-bounds) |

### Step 1 — Open the zone editor

Grab a live frame from the camera as background:

```
python -m spatial.zone_editor
```

Use a saved image instead:

```
python -m spatial.zone_editor --image path/to/image.jpg
```

Load and continue editing previously saved zones:

```
python -m spatial.zone_editor --load
```

### Step 2 — Draw your zones

| Key              | Action                                                     |
| ---------------- | ---------------------------------------------------------- |
| Left click       | Add a vertex to the current polygon                        |
| `T`              | Cycle zone type (parking → drive → restricted)             |
| `C` or Enter     | Close and save the current polygon (min 3 points)          |
| `Z` or Backspace | Undo last vertex                                           |
| `D`              | Delete the last completed zone                             |
| `R`              | Rename the last completed zone (type new name in terminal) |
| `S`              | Save zones to `config/zones.json`                          |
| `Q`              | Save and quit                                              |

Tips:

- The entire background starts with a faint red tint — this represents unlabeled (restricted) areas
- As you draw and close zones, they override the background with their own color
- You can draw as many zones of any type as you need
- Zone names are auto-generated (e.g. `parking_0`, `drive_1`) but can be renamed with `R`

### Step 3 — Zones are saved to `config/zones.json`

This file is loaded automatically at runtime. Re-run the editor any time to adjust zones.

---

## Calibration (do this once after fixing the camera position)

Calibration maps pixels to real-world centimetres so the layout engine knows
how large a parking slot should be on screen. Skip this step and the system
still runs, but slot sizes will be in pixels instead of cm.

### Step 1 — Place 4 ground markers

Put tape crosses or small cones on the ground at positions whose real-world
distances you have measured. A simple rectangle works best, e.g.:

```
(0, 0) ────── (30, 0)
  │                │
(0, 20) ───── (30, 20)
```

### Step 2 — Run the calibration tool

```
python -m spatial.calibration
```

Use a saved image instead of the live camera:

```
python -m spatial.calibration --image path/to/image.jpg
```

### Step 3 — Mark the 4 points on screen

1. **Click** each marker on the image in any order — a coloured dot appears
2. An input box appears at the bottom — type the **X coordinate (cm)** and press **Enter**
3. Type the **Y coordinate (cm)** and press **Enter**
4. Repeat for all 4 points
5. A verification screen shows expected vs computed coords for each point
6. Press **S** to save → `config/calibration.json`
7. Press **R** to redo from scratch if something looks wrong
8. Press **Q** to discard

Calibration is saved to `config/calibration.json` and loaded automatically by `main.py`.

---

## Parking Modes

The layout engine supports three open-ground parking arrangements. Change the
mode at the top of `main.py`:

```python
# main.py — line 13
PARKING_TYPE = "open"      # change to "parallel" or "angled"
```

| Mode       | Description                                                     | Best for                    |
| ---------- | --------------------------------------------------------------- | --------------------------- |
| `open`     | Perpendicular back-to-back rows, shared aisle between each pair | Square / wide parking areas |
| `parallel` | Nose-to-tail slots along the longer edge, drive lane alongside  | Narrow rectangular strips   |
| `angled`   | 45° parallelogram stalls, one-way aisle                         | Medium lots, higher density |

Slot sizes are set in `reasoning/layout_engine.py`:

```python
CAR_LENGTH_CM = 8.7   # toy car length
CAR_WIDTH_CM  = 3.6   # toy car width
```

Update these values when switching to real cars (or when the backend sends
dimensions — the engine already accepts `car_length` and `car_width` as
constructor arguments).

---

## Recommended First-Run Order

```
1.  python -m spatial.zone_editor          # draw parking / drive / restricted zones
2.  python -m spatial.calibration          # calibrate pixel → cm mapping
3.  python main.py                          # run the full system
```

---

## Project Structure

```
parking_ai/
├── main.py                      # entry point — detection + tracking + occupancy
├── perception/
│   ├── camera.py                # IP camera stream wrapper
│   ├── detector.py              # VehicleDetector (YOLOv8, single inference)
│   └── tracker.py               # VehicleTracker (YOLOv8 + ByteTrack)
├── spatial/
│   ├── calibration.py           # pixel ↔ real-world cm homography tool
│   ├── zone_editor.py           # interactive visual zone drawing tool
│   └── zone_map.py              # runtime zone loader and point-in-zone queries
├── reasoning/
│   ├── layout_engine.py         # computes slot polygons for open/parallel/angled
│   └── occupancy.py             # temporal occupancy state per slot
├── config/
│   ├── zones.json               # saved zone definitions
│   └── calibration.json         # saved homography matrix
├── dataset/                     # training data (YOLO format)
└── model weights                # hosted on GitHub Releases (v1.0/best.pt)
```

---

## Updating the Camera URL

If your camera IP changes, edit the `DEFAULT_STREAM_URL` in `perception/camera.py`:

```python
DEFAULT_STREAM_URL = "http://<your-ip>:8080/video"
```

---

## Retraining the Model

If detection is missing cars or picking up false positives:

1. Collect new images from the exact camera angle using:
   ```python
   python -m utils.collect_frames   # (or manually save frames from main.py)
   ```
2. Label images on [Roboflow](https://roboflow.com) — export in YOLOv8 format
3. Retrain from the existing weights:
   ```python
   from ultralytics import YOLO
   model = YOLO("runs/detect/train2/weights/best.pt")
   model.train(data="new_data/data.yaml", epochs=30, imgsz=640, batch=8, name="train3")
   ```
4. Update `DEFAULT_MODEL_PATH` in `perception/detector.py` and `perception/tracker.py` to point to `train3/weights/best.pt`
