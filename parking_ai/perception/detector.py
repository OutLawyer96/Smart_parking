from ultralytics import YOLO
import numpy as np


class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        
        self.model = YOLO(model_path)
        self.confidence = confidence

        # COCO class IDs for vehicles
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def detect(self, frame: np.ndarray):
        """
        Run detection on a single frame.

        Returns:
            List of dictionaries with bounding box info.
        """
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0])

                if class_id not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence,
                    "class_id": class_id
                })

        return detections