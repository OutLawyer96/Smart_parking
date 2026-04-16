import cv2
import time
import argparse
from ultralytics import YOLO
from perception.camera import CameraStream
from perception.tracker import DEFAULT_MODEL_PATH


class VehicleDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Single class: car (class 0)
            if cls == 0 and conf >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf
                })

        return detections


def main():
    parser = argparse.ArgumentParser(description="Video detection test")
    parser.add_argument(
        "--url",
        default=None,
        help="Camera source URL (or webcam index like 0). Defaults to env var/camera.py default.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Run detection every Nth frame (default: 2).",
    )
    args = parser.parse_args()

    source = int(args.url) if args.url is not None and args.url.isdigit() else args.url
    detection_stride = max(1, args.stride)

    model_path = DEFAULT_MODEL_PATH

    detector = VehicleDetector(model_path, conf_threshold=0.6)

    with CameraStream(source=source) as cam:
        print(f"Reading from: {cam.source}")
        print(f"Detection stride: {detection_stride}")
        miss_count = 0
        max_misses = 20
        frame_idx = 0
        fps_ema = 0.0
        fps_alpha = 0.2
        cached_detections = []
        while True:
            loop_start = time.time()
            frame = cam.read()
            if frame is None:
                miss_count += 1
                if miss_count >= max_misses:
                    print("[Video Detection Test] Stream stalled (too many missed frames).")
                    break
                continue

            miss_count = 0

            # Resize for consistent inference speed
            frame_resized = cv2.resize(frame, (640, 640))

            frame_idx += 1
            should_infer = (frame_idx % detection_stride == 0) or not cached_detections
            if should_infer:
                detections = detector.detect(frame_resized)
                cached_detections = detections
            else:
                detections = cached_detections

            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]

                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_resized,
                    f"Car {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Display FPS
            loop_fps = 1 / max(time.time() - loop_start, 1e-6)
            if fps_ema == 0.0:
                fps_ema = loop_fps
            else:
                fps_ema = (fps_alpha * loop_fps) + ((1 - fps_alpha) * fps_ema)

            cv2.putText(
                frame_resized,
                f"FPS: {fps_ema:.2f} | stride: {detection_stride}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.imshow("Smart Parking Detection", frame_resized)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()