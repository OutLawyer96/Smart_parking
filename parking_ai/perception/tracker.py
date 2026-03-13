from ultralytics import YOLO

# Path to the fine-tuned model (trained on real toy car images)
DEFAULT_MODEL_PATH = "https://github.com/OutLawyer96/Smart_parking/releases/download/v1.0/best.pt"


class VehicleTracker:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, confidence: float = 0.60,
                 imgsz: int = 640, min_box_area: int = 900):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.imgsz = imgsz
        self.min_box_area = min_box_area  # px² — kills tiny false positives (e.g. text labels)

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