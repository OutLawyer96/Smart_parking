from ultralytics import YOLO
import numpy as np

# Path to the fine-tuned model (trained on the latest local dataset split)
DEFAULT_MODEL_PATH = "weights/best.pt"


class VehicleDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, confidence: float = 0.60):
        self.model = YOLO(model_path)
        self.confidence = confidence

        # Custom dataset has a single class: car (class 0)
        self.vehicle_classes = [0]  # toy car
        self.class_names = {0: "car"}

    def detect(self, frame: np.ndarray):
        """
        Run detection on a single frame using the fine-tuned toy car model.

        Returns:
            List of dicts: {bbox, confidence, class_id, label}
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
                    "class_id": class_id,
                    "label": self.class_names.get(class_id, "car")
                })

        return detections