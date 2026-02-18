import cv2
from perception.detector import VehicleDetector

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

detector = VehicleDetector("yolov8n.pt")

detections = detector.detect(frame)
print(detections)