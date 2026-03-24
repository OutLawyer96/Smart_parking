import os
import cv2

# ESP32-CAM streams are commonly exposed as MJPEG URLs on port 81.
# Override via ESP32_CAM_URL or CAMERA_SOURCE env vars.
DEFAULT_STREAM_URL = "http://192.168.4.1:81/stream"


def _resolve_source(source: str | int | None = None) -> str | int:
    if source is not None:
        return source

    env_source = os.getenv("CAMERA_SOURCE") or os.getenv("ESP32_CAM_URL")
    if not env_source:
        return DEFAULT_STREAM_URL

    # Allow webcam index via env var (e.g. CAMERA_SOURCE=0)
    if env_source.isdigit():
        return int(env_source)
    return env_source


class CameraStream:
    def __init__(self, source: str | int | None = None):
        self.source = _resolve_source(source)
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera stream: {self.source}")

    def read(self):
        """
        Read a single frame from the stream.

        Returns:
            frame (np.ndarray) or None if read failed
        """
        if self.cap is None:
            raise RuntimeError("Stream not started. Call start() first.")

        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
