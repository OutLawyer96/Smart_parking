"""
Quick ESP32-CAM connectivity test.

Usage examples:
  python test_esp32_cam.py
  python test_esp32_cam.py --url http://192.168.4.1:81/stream
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
    parser = argparse.ArgumentParser(description="Test ESP32-CAM stream")
    parser.add_argument(
        "--url",
        default=None,
        help="ESP32 stream URL (or webcam index like 0). Defaults to env var or camera.py default.",
    )
    args = parser.parse_args()

    source = parse_source(args.url) if args.url is not None else None

    print("[ESP32 Test] Starting camera stream...")
    with CameraStream(source=source) as cam:
        print(f"[ESP32 Test] Connected to source: {cam.source}")

        frame_count = 0
        started_at = time.time()

        cv2.namedWindow("ESP32-CAM Test", cv2.WINDOW_NORMAL)

        while True:
            frame = cam.read()
            if frame is None:
                print("[ESP32 Test] Failed to read frame. Stream may be unstable or disconnected.")
                break

            frame_count += 1
            elapsed = max(time.time() - started_at, 1e-6)
            fps = frame_count / elapsed

            cv2.putText(
                frame,
                f"Source: {cam.source}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Press Q to quit",
                (10, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("ESP32-CAM Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("[ESP32 Test] Done.")


if __name__ == "__main__":
    main()
