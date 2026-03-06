import cv2

# IP camera stream (phone/Raspberry Pi via DroidCam or similar)
DEFAULT_STREAM_URL = "http://100.111.201.140:8080/video"


class CameraStream:
    def __init__(self, source: str = DEFAULT_STREAM_URL):
        self.source = source
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
