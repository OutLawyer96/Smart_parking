import cv2
import time
from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train2/weights/best.pt")
    cap = cv2.VideoCapture("http://100.98.138.181:8080/video")

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking
        results = model.track(frame, persist=True, conf=0.5)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {int(track_id)} | {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        cv2.imshow("Tracking Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()