import cv2
import time
import argparse
from ultralytics import YOLO
from perception.camera import CameraStream
from perception.tracker import DEFAULT_MODEL_PATH

def main():
    parser = argparse.ArgumentParser(description="Tracking test")
    parser.add_argument(
        "--url",
        default=None,
        help="Camera source URL (or webcam index like 0). Defaults to env var/camera.py default.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Run tracking inference every Nth frame (default: 2).",
    )
    args = parser.parse_args()

    source = int(args.url) if args.url is not None and args.url.isdigit() else args.url
    detection_stride = max(1, args.stride)

    model = YOLO(DEFAULT_MODEL_PATH)
    miss_count = 0
    max_misses = 20

    with CameraStream(source=source) as cam:
        print(f"Reading from: {cam.source}")
        print(f"Tracking stride: {detection_stride}")
        frame_idx = 0
        fps_ema = 0.0
        fps_alpha = 0.2
        cached_tracks = []

        while True:
            loop_start = time.time()

            frame = cam.read()
            if frame is None:
                miss_count += 1
                if miss_count >= max_misses:
                    print("[Tracking Test] Stream stalled (too many missed frames).")
                    break
                continue

            miss_count = 0

            frame_idx += 1
            should_infer = (frame_idx % detection_stride == 0) or not cached_tracks
            if should_infer:
                results = model.track(frame, persist=True, conf=0.5)
                cached_tracks = []

                if results and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()

                    for box, track_id, conf in zip(boxes, ids, confs):
                        x1, y1, x2, y2 = map(int, box)
                        cached_tracks.append((x1, y1, x2, y2, int(track_id), float(conf)))

            for x1, y1, x2, y2, track_id, conf in cached_tracks:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id} | {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

            loop_fps = 1 / max(time.time() - loop_start, 1e-6)
            if fps_ema == 0.0:
                fps_ema = loop_fps
            else:
                fps_ema = (fps_alpha * loop_fps) + ((1 - fps_alpha) * fps_ema)

            cv2.putText(frame, f"FPS: {fps_ema:.2f} | stride: {detection_stride}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

            cv2.imshow("Tracking Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()