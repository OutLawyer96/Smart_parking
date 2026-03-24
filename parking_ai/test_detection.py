import cv2
import argparse
from perception.detector import VehicleDetector
from perception.camera import CameraStream


def main():
	parser = argparse.ArgumentParser(description="Single-frame detection test")
	parser.add_argument(
		"--url",
		default=None,
		help="Camera source URL (or webcam index like 0). Defaults to env var/camera.py default.",
	)
	args = parser.parse_args()

	source = int(args.url) if args.url is not None and args.url.isdigit() else args.url

	detector = VehicleDetector()  # uses fine-tuned model by default

	with CameraStream(source=source) as cam:
		frame = cam.read()
		if frame is None:
			raise RuntimeError(f"Could not read frame from source: {cam.source}")

	detections = detector.detect(frame)
	print(detections)


if __name__ == "__main__":
	main()