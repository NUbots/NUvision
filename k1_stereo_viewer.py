import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

LEFT_TOPIC = "/boostercamera/head/raw/rgb"
RIGHT_TOPIC = "/boostercamera/head/raw/right/rgb"
PANEL_WIDTH = 640
PANEL_HEIGHT = 400


def image_to_bgr(msg: Image) -> np.ndarray:
    """Decode ROS Image using row stride, including K1's NV12 image format."""
    enc = msg.encoding.strip().lower()
    width, height, step = int(msg.width), int(msg.height), int(msg.step)
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if width <= 0 or height <= 0 or step <= 0:
        raise ValueError(f"Invalid image dimensions or step: {width}x{height}, step={step}")

    if enc in ("nv12", "nv21"):
        # NV12/NV21 is 8-bit YUV 4:2:0. First H rows contain luminance;
        # the next H/2 rows contain interleaved UV/VU samples.
        if width % 2 or height % 2:
            raise ValueError(f"{enc} requires even width/height, got {width}x{height}")
        if step < width:
            raise ValueError(f"{enc} stride {step} < width {width}")
        total_rows = height * 3 // 2
        needed = total_rows * step
        if raw.size < needed:
            raise ValueError(
                f"{enc} buffer too small ({raw.size} bytes, expected >= {needed}; "
                f"width={width}, height={height}, step={step})"
            )
        # OpenCV expects a contiguous (H*3/2) x W NV12/NV21 matrix.
        yuv = np.ascontiguousarray(raw[:needed].reshape(total_rows, step)[:, :width])
        conversion = cv2.COLOR_YUV2BGR_NV12 if enc == "nv12" else cv2.COLOR_YUV2BGR_NV21
        return cv2.cvtColor(yuv, conversion)

    channels = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
        "8uc1": 1,
        "8uc3": 3,
    }.get(enc)
    if channels is None:
        raise ValueError(f"Unsupported image encoding {msg.encoding!r}")

    visible_row_bytes = width * channels
    required = height * step
    if step < visible_row_bytes or raw.size < required:
        raise ValueError(
            f"Invalid {enc} image buffer: {raw.size} bytes, "
            f"size={width}x{height}, step={step}"
        )
    pixels = np.ascontiguousarray(
        raw[:required].reshape(height, step)[:, :visible_row_bytes]
    ).reshape(height, width, channels)

    if enc == "rgb8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    return pixels


class StereoViewer(Node):
    def __init__(self, left_topic: str, right_topic: str):
        super().__init__("k1_stereo_browser_viewer")
        self.lock = Lock()
        self.latest = {"LEFT": None, "RIGHT": None}
        self.timestamps = {"LEFT": None, "RIGHT": None}
        self.received_at = {"LEFT": None, "RIGHT": None}
        self.announced = set()

        self.create_subscription(
            Image, left_topic,
            lambda message: self.receive("LEFT", message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, right_topic,
            lambda message: self.receive("RIGHT", message),
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"LEFT  topic: {left_topic}")
        self.get_logger().info(f"RIGHT topic: {right_topic}")

    def receive(self, side: str, msg: Image):
        try:
            bgr = image_to_bgr(msg)
            height, width = bgr.shape[:2]
            # Keep browser preview inexpensive while maintaining aspect ratio.
            ratio = min(PANEL_WIDTH / width, PANEL_HEIGHT / height, 1.0)
            if ratio < 1.0:
                bgr = cv2.resize(
                    bgr, (max(1, round(width * ratio)), max(1, round(height * ratio))),
                    interpolation=cv2.INTER_AREA,
                )
            stamp = float(msg.header.stamp.sec) + msg.header.stamp.nanosec / 1e9
            with self.lock:
                self.latest[side] = bgr
                self.timestamps[side] = stamp
                self.received_at[side] = time.monotonic()
            if side not in self.announced:
                self.announced.add(side)
                self.get_logger().info(
                    f"First {side} frame: {width}x{height}, "
                    f"encoding={msg.encoding}, frame_id={msg.header.frame_id}"
                )
        except Exception as exc:
            self.get_logger().error(
                f"Cannot decode {side} frame: {exc}", throttle_duration_sec=5.0
            )

    def render(self) -> np.ndarray:
        with self.lock:
            frames = dict(self.latest)
            times = dict(self.timestamps)
            arrival_times = dict(self.received_at)

        output = []
        now = time.monotonic()
        for side in ("LEFT", "RIGHT"):
            panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 25, dtype=np.uint8)
            frame = frames[side]
            if frame is not None:
                h, w = frame.shape[:2]
                x, y = (PANEL_WIDTH - w) // 2, (PANEL_HEIGHT - h) // 2
                panel[y:y + h, x:x + w] = frame
                age = now - arrival_times[side]
                status = "LIVE" if age <= 2.0 else f"STALE {age:.1f}s"
                title = f"{side}  {status}  timestamp={times[side]:.3f}"
            else:
                title = f"{side}  Waiting for camera frames..."
            cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 36), (0, 0, 0), -1)
            cv2.putText(panel, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.57, (255, 255, 255), 1, cv2.LINE_AA)
            output.append(panel)
        return np.hstack(output)


def main():
    p = argparse.ArgumentParser(description="Live left/right camera viewer for Booster K1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--fps", type=float, default=15, help="Browser refresh FPS (default 15)")
    p.add_argument("--left-topic", default=LEFT_TOPIC)
    p.add_argument("--right-topic", default=RIGHT_TOPIC)
    args = p.parse_args()
    if not (0 < args.fps <= 60):
        p.error("--fps must be > 0 and <= 60")
    if not (1 <= args.port <= 65535):
        p.error("--port must be 1..65535")

    rclpy.init()
    node = StereoViewer(args.left_topic, args.right_topic)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    ros_thread = Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = ("<!doctype html><html><head>"
                        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                        "<title>Booster K1 Stereo Cameras</title></head>"
                        "<body style='font-family:system-ui;background:#111;color:#eee;"
                        "text-align:center;margin:24px'>"
                        "<h2>Booster K1 - Left and Right Cameras</h2>"
                        "<img src='/stream.mjpg' alt='Stereo camera video' "
                        "style='max-width:100%;height:auto'>"
                        "<p>Live preview of latest images (not timestamp-synchronised).</p>"
                        "</body></html>").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    ok, jpeg_array = cv2.imencode(
                        ".jpg", node.render(), [cv2.IMWRITE_JPEG_QUALITY, 72]
                    )
                    if ok:
                        jpeg = jpeg_array.tobytes()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg + b"\r\n")
                    time.sleep(1 / args.fps)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, fmt, *values):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ViewerHandler)
    server.daemon_threads = True
    print(f"\nK1 stereo viewer listening on 127.0.0.1:{args.port}")
    print(f"On Mac: ssh -N -L {args.port}:127.0.0.1:{args.port} booster@10.1.2.2")
    print(f"Browser: http://localhost:{args.port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer...")
    finally:
        server.server_close()
        executor.shutdown()
        ros_thread.join(timeout=3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
