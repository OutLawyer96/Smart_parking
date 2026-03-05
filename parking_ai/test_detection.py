import cv2
from perception.detector import VehicleDetector

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

detector = VehicleDetector()  # uses fine-tuned model by default

detections = detector.detect(frame)
print(detections)