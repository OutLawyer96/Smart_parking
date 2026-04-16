from ultralytics import YOLO
import os

# Path to the fine-tuned model (trained on the latest local dataset split)
DEFAULT_MODEL_PATH = "weights/best.pt"


class VehicleTracker:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, confidence: float = 0.60,
                 imgsz: int = 320, min_box_area: int = 900):
        self.model = YOLO(model_path)
        self.confidence = float(os.getenv("TRACKER_CONF", str(confidence)))
        self.imgsz = int(os.getenv("TRACKER_IMGSZ", str(imgsz)))
        self.min_box_area = int(os.getenv("TRACKER_MIN_BOX_AREA", str(min_box_area)))
        self.max_det = int(os.getenv("TRACKER_MAX_DET", "12"))

        # Custom dataset: single class (0 = car)
        self.vehicle_classes = [0]
        self.class_names = {0: "car"}

    def track(self, frame):
        """
        Run ByteTrack-based tracking on a single frame using the fine-tuned toy car model.

        Returns:
            List of dicts: {track_id, bbox, confidence, class_id, label}
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            imgsz=self.imgsz,
            iou=0.4,
            classes=self.vehicle_classes,
            max_det=self.max_det,
            verbose=False
        )

        detections = []

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, track_id, conf, class_id in zip(boxes, ids, confs, classes):
                class_id = int(class_id)

                # Only keep detections from the toy car class
                if class_id not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(int, box)

                # Drop detections smaller than min_box_area (text, noise, etc.)
                if (x2 - x1) * (y2 - y1) < self.min_box_area:
                    continue

                detections.append({
                    "track_id": int(track_id),
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(conf),
                    "class_id": class_id,
                    "label": self.class_names.get(class_id, "car")
                })

        return detections