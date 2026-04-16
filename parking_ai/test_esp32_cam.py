"""
Quick ESP32-CAM connectivity test.

Usage examples:
  python test_esp32_cam.py
    python test_esp32_cam.py --url http://10.54.215.196/capture
  python test_esp32_cam.py --url 0
"""

import argparse
import time
import cv2

from perception.camera import CameraStream


def parse_source(value: str):
    if value.isdigit():
        return int(value)
    return value


def main():
    parser = argparse.ArgumentParser(description="Test ESP32-CAM source")
    parser.add_argument(
        "--url",
        default=None,
        help="ESP32 capture URL (or webcam index like 0). Defaults to env var or camera.py default.",
    )
    args = parser.parse_args()

    source = parse_source(args.url) if args.url is not None else None

    print("[ESP32 Test] Starting camera source...")
    with CameraStream(source=source) as cam:
        print(f"[ESP32 Test] Connected to source: {cam.source}")

        frame_count = 0
        miss_count = 0
        max_misses = 20
        started_at = time.time()

        cv2.namedWindow("ESP32-CAM Test", cv2.WINDOW_NORMAL)
        first = True

        while True:
            frame = cam.read()
            if frame is None:
                miss_count += 1
                if miss_count >= max_misses:
                    print("[ESP32 Test] Stream stalled (too many missed frames).")
                    break
                continue

            miss_count = 0

            frame_count += 1
            elapsed = max(time.time() - started_at, 1e-6)
            fps = frame_count / elapsed

            h, w = frame.shape[:2]
            if first:
                cv2.resizeWindow("ESP32-CAM Test", w, h)
                first = False
            scale = 0.5 if w <= 360 else 0.7
            thick = 1 if w <= 360 else 2

            cv2.putText(
                frame,
                f"Source: {cam.source}",
                (8, 20 if w <= 360 else 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 255, 255),
                thick,
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (8, 40 if w <= 360 else 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 255, 0),
                thick,
            )
            cv2.putText(
                frame,
                "Press Q to quit",
                (8, 60 if w <= 360 else 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                thick,
            )

            cv2.imshow("ESP32-CAM Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("[ESP32 Test] Done.")


if __name__ == "__main__":
    main()
