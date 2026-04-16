import os
import cv2
import time
import threading
import requests
import numpy as np
from urllib.parse import urlencode, urlparse, parse_qsl, urlunparse

# ESP32-CAM realtime MJPEG endpoint.
# Override via CAMERA_SOURCE / ESP32_STREAM_URL / ESP32_CAPTURE_URL env vars.
DEFAULT_STREAM_URL = "http://10.54.215.196/stream"


def _to_stream_url(url: str) -> str:
    if "/capture" in url:
        return url.replace("/capture", "/stream")
    return url


def _resolve_source(source: str | int | None = None) -> str | int:
    if source is not None:
        return source

    env_source = (
        os.getenv("CAMERA_SOURCE")
        or os.getenv("ESP32_STREAM_URL")
        or os.getenv("ESP32_CAPTURE_URL")
        or os.getenv("ESP32_CAM_URL")
    )
    if not env_source:
        return DEFAULT_STREAM_URL

    # Allow webcam index via env var (e.g. CAMERA_SOURCE=0)
    if env_source.isdigit():
        return int(env_source)
    return _to_stream_url(env_source)


class CameraStream:
    def __init__(self, source: str | int | None = None, esp32_mode: str | None = None):
        self.source = _resolve_source(source)
        self.esp32_mode = esp32_mode  # "calibration", "zone_editor", or None (default fast)
        self.cap = None
        self.session = None
        self._reader_thread = None
        self._running = False
        self._latest_frame = None
        self._latest_frame_id = 0
        self._latest_frame_ts = 0.0
        self._last_returned_frame_id = 0
        self._frame_lock = threading.Lock()
        self._init_timeout_s = float(os.getenv("CAMERA_INIT_TIMEOUT_S", "5.0"))
        # Keep read timeout short in snapshot mode to avoid UI stalls.
        self._read_timeout_s = float(os.getenv("CAMERA_READ_TIMEOUT_S", "0.25"))
        # Keep timeout tight so a single slow ESP32 response does not stall
        # snapshot cadence for several seconds.
        self._http_timeout_s = float(os.getenv("CAMERA_HTTP_TIMEOUT_S", "0.9"))
        self._snapshot_interval_s = float(os.getenv("CAMERA_SNAPSHOT_INTERVAL_S", "0.12"))
        self._stream_open_retry_s = float(os.getenv("CAMERA_STREAM_OPEN_RETRY_S", "0.35"))
        # Decode stream at a controlled cadence while continuously grabbing
        # packets, so stale buffered frames are discarded quickly.
        self._stream_decode_interval_s = float(os.getenv("CAMERA_STREAM_DECODE_INTERVAL_S", "0.03"))
        # Use direct HTTP MJPEG parsing for network streams to avoid hidden
        # buffering in OpenCV backends.
        self._use_http_mjpeg = os.getenv("CAMERA_USE_HTTP_MJPEG", "1") == "1"
        self._mjpeg_chunk_size = int(os.getenv("CAMERA_MJPEG_CHUNK_SIZE", "4096"))

        self._mode = self._detect_mode(self.source)

    def _snapshot_url(self) -> str:
        """Build snapshot URL with optional mode query param safely."""
        if not self.esp32_mode or not isinstance(self.source, str):
            return self.source  # type: ignore[return-value]

        parsed = urlparse(self.source)
        q = dict(parse_qsl(parsed.query, keep_blank_values=True))
        q["mode"] = self.esp32_mode
        return urlunparse(parsed._replace(query=urlencode(q)))

    @staticmethod
    def _detect_mode(source: str | int) -> str:
        if isinstance(source, int):
            return "stream"
        if source.startswith("http://") or source.startswith("https://"):
            if "/capture" in source:
                return "snapshot"
        return "stream"

    def _configure_capture(self):
        if self.cap is None:
            return

        # Keep the internal queue tiny so we always process fresh frames.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _open_stream_capture_with_retry(self) -> bool:
        """Retry opening MJPEG stream up to init timeout before failing."""
        deadline = time.time() + self._init_timeout_s
        while self._running and time.time() < deadline:
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                self.cap = cap
                return True
            cap.release()
            time.sleep(max(self._stream_open_retry_s, 0.05))
        return False

    def _stream_connect_diagnostic(self) -> str:
        if not isinstance(self.source, str):
            return ""
        if not (self.source.startswith("http://") or self.source.startswith("https://")):
            return ""
        try:
            resp = requests.get(self.source, timeout=2.0, stream=True)
            return f" HTTP probe status={resp.status_code}."
        except requests.RequestException as e:
            return f" HTTP probe failed: {e}."

    @staticmethod
    def _is_http_source(source: str | int) -> bool:
        return isinstance(source, str) and (
            source.startswith("http://") or source.startswith("https://")
        )

    def _mjpeg_loop(self):
        if self.session is None:
            return

        data = bytearray()
        while self._running and self.session is not None:
            try:
                with self.session.get(
                    self.source,
                    stream=True,
                    timeout=(2.0, self._http_timeout_s),
                    headers={"Accept": "multipart/x-mixed-replace"},
                ) as resp:
                    if resp.status_code != 200:
                        time.sleep(0.15)
                        continue

                    for chunk in resp.iter_content(chunk_size=max(self._mjpeg_chunk_size, 1024)):
                        if not self._running:
                            return
                        if not chunk:
                            continue
                        data.extend(chunk)

                        # Keep only the latest complete JPEG in buffer.
                        while True:
                            soi = data.find(b"\xff\xd8")
                            if soi < 0:
                                # Prevent unbounded growth from malformed data.
                                if len(data) > 1_000_000:
                                    data = data[-200_000:]
                                break

                            eoi = data.find(b"\xff\xd9", soi + 2)
                            if eoi < 0:
                                if soi > 0:
                                    del data[:soi]
                                break

                            jpg = bytes(data[soi:eoi + 2])
                            del data[:eoi + 2]

                            arr = np.frombuffer(jpg, dtype=np.uint8)
                            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame is None:
                                continue

                            with self._frame_lock:
                                self._latest_frame = frame
                                self._latest_frame_id += 1
                                self._latest_frame_ts = time.perf_counter()
            except requests.RequestException:
                # Brief backoff before reconnecting.
                time.sleep(0.15)

    def _read_snapshot(self):
        if self.session is None:
            return None
        try:
            url = self._snapshot_url()
            resp = self.session.get(
                url,
                timeout=self._http_timeout_s,
                headers={"Accept": "image/jpeg"},
            )
            if resp.status_code != 200:
                return None
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            if arr.size == 0:
                return None
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except requests.RequestException:
            return None

    def _bootstrap_snapshot_frame(self) -> bool:
        """Try to fetch the first snapshot synchronously before starting loops."""
        deadline = time.time() + self._init_timeout_s
        while self._running and time.time() < deadline:
            frame = self._read_snapshot()
            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_frame_id += 1
                    self._latest_frame_ts = time.perf_counter()
                return True
            time.sleep(0.12)
        return False

    def _reader_loop(self):
        next_decode_at = time.perf_counter()
        while self._running and self.cap is not None:
            # grab() advances to the newest packet with less decode overhead.
            if not self.cap.grab():
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            if now < next_decode_at:
                continue

            ok, frame = self.cap.retrieve()
            if not ok:
                time.sleep(0.005)
                continue

            next_decode_at = now + max(self._stream_decode_interval_s, 0.005)

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_id += 1
                self._latest_frame_ts = time.perf_counter()

    def _snapshot_loop(self):
        next_tick = time.perf_counter()
        while self._running and self.session is not None:
            frame = self._read_snapshot()
            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_frame_id += 1
                    self._latest_frame_ts = time.perf_counter()

            # Keep polling cadence stable even if one request is slow.
            interval = max(self._snapshot_interval_s, 0.05)
            next_tick += interval
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()

    def start(self):
        self._running = True

        if self._mode == "snapshot":
            self.session = requests.Session()
            # Bootstrap first frame synchronously to avoid issuing parallel
            # requests to the ESP32 during startup.
            if not self._bootstrap_snapshot_frame():
                print(
                    "[CameraStream] Warning: no initial snapshot received yet; "
                    "continuing and retrying in background."
                )

            self._reader_thread = threading.Thread(target=self._snapshot_loop, daemon=True)
            self._reader_thread.start()
            return
        else:
            if self._use_http_mjpeg and self._is_http_source(self.source):
                self.session = requests.Session()
                self._reader_thread = threading.Thread(target=self._mjpeg_loop, daemon=True)
                self._reader_thread.start()
            else:
                if not self._open_stream_capture_with_retry():
                    self._running = False
                    diag = self._stream_connect_diagnostic()
                    raise RuntimeError(
                        f"Failed to open camera stream after {self._init_timeout_s:.1f}s: "
                        f"{self.source}.{diag}"
                    )

                self._configure_capture()
                self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
                self._reader_thread.start()

        # Wait briefly for the first frame so callers can fail fast on bad streams.
        start_time = time.time()
        while time.time() - start_time < self._init_timeout_s:
            with self._frame_lock:
                if self._latest_frame is not None:
                    return
            time.sleep(0.01)

        self.release()
        raise RuntimeError(f"Camera opened but no frames received: {self.source}")

    def read(self):
        """
        Read a single frame from the stream.

        Returns:
            frame (np.ndarray) or None if read failed
        """
        frame, _ = self.read_with_meta()
        return frame

    def read_with_meta(self):
        """
        Read a single frame and basic source-latency metadata.

        Returns:
            (frame, meta) where meta includes source_age_ms for diagnostics.
        """
        if not self._running:
            raise RuntimeError("Stream not started. Call start() first.")

        start_time = time.time()
        while time.time() - start_time < self._read_timeout_s:
            with self._frame_lock:
                frame = self._latest_frame
                frame_id = self._latest_frame_id
                frame_ts = self._latest_frame_ts
            if frame is not None and frame_id != self._last_returned_frame_id:
                self._last_returned_frame_id = frame_id
                # Return a copy so overlay drawing in the main loop never
                # mutates the shared latest-frame buffer.
                source_age_ms = max(0.0, (time.perf_counter() - frame_ts) * 1000.0)
                return frame.copy(), {
                    "source_age_ms": source_age_ms,
                    "frame_id": frame_id,
                }
            time.sleep(0.005)

        # In snapshot mode the ESP32 can jitter; returning the most recent frame
        # keeps downstream rendering/layout alive even when a fresh snapshot is late.
        if self._mode == "snapshot":
            with self._frame_lock:
                if self._latest_frame is not None:
                    # Same reason as above: avoid accumulating overlays when
                    # a stale frame is reused during ESP32 jitter.
                    source_age_ms = max(0.0, (time.perf_counter() - self._latest_frame_ts) * 1000.0)
                    return self._latest_frame.copy(), {
                        "source_age_ms": source_age_ms,
                        "frame_id": self._latest_frame_id,
                    }
        return None, {"source_age_ms": -1.0, "frame_id": -1}

    def release(self):
        self._running = False
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.session is not None:
            self.session.close()
            self.session = None

        with self._frame_lock:
            self._latest_frame = None
            self._latest_frame_id = 0
            self._latest_frame_ts = 0.0
            self._last_returned_frame_id = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def mode(self) -> str:
        return self._mode
