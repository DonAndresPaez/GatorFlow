'''tracker.py: the live tracker. Camera in, plate pose out over OSC.

Run: python -m tracking.tracker              (from tools/)
     python -m tracking.tracker --no-view    (no preview window, a little faster)

Reads the camera named in setup/camera.yml, finds the marker plate in every
frame, and sends its pose to TouchDesigner (port 9000) and the simulation
(port 9001) as OSC /car/transform [tx, ty, tz, rx, ry, rz]: mm and degrees,
TouchDesigner world, measured from the plate's home spot.

Needs: calibration/<camera>/intrinsics.yml   (python -m calibration.camera_lens)
       calibration/<camera>/world_origin.yml (python -m calibration.set_origin)

Speed: searching a whole 4K frame for markers is slow, so the search runs on a
copy shrunk to --detect-width pixels, and the corners it finds are then
refined on the full-resolution frame. A webcam is read on its own thread and
only the newest frame is used, so a slow frame never makes the pose fall
behind the camera.

When the plate is lost nothing is sent (TouchDesigner keeps the last pose),
and the smoothing starts fresh so the model doesn't slide across the gap.
'''

import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from scipy.spatial.transform import Rotation

from common import config
from common.camera import PROFILE, check_frame_size, load_intrinsics, open_camera
from common.plate import LAYOUT, MarkerTracker, load_camera_to_world, pose_to_td

SMOOTHING_ALPHA = 0.5   # 1.0 = raw readings, lower = steadier but laggier
DETECT_WIDTH = 1920     # marker search runs on a copy this wide; corners are refined at full size
VIEW_WIDTH = 1280       # preview window width
HISTORY = 60            # frames used for the lock rate (~2 s at 30 fps)


# ---------------------------------------------------------------- smoothing

class PoseSmoother:
    """Exponential smoothing of position and rotation.

    Rotations are blended as quaternions (a blend of Euler angles goes wrong
    near +-180). q and -q are the same rotation, so the sign closest to the
    previous reading is used, otherwise the blend takes the long way round."""

    def __init__(self, alpha=SMOOTHING_ALPHA):
        self.alpha = float(alpha)
        self.reset()

    def reset(self):
        self.t = None
        self.q = None

    def apply(self, R, t):
        q = Rotation.from_matrix(R).as_quat()
        t = np.ravel(t).astype(float)
        if self.t is None:
            self.t, self.q = t, q
        else:
            if np.dot(self.q, q) < 0.0:
                q = -q
            a = self.alpha
            self.t = (1.0 - a) * self.t + a * t
            self.q = (1.0 - a) * self.q + a * q
            self.q /= np.linalg.norm(self.q)   # blending shortens it; only unit quaternions are rotations
        return Rotation.from_quat(self.q).as_matrix(), self.t.copy()


# ---------------------------------------------------------------- detection

def detect_markers(tracker, frame, detect_width=DETECT_WIDTH):
    """ArUco detections as (corners, ids), corners in FULL-resolution pixels.

    Fast path: find the markers in a shrunk copy, then run the very same ArUco
    detector again at full resolution but only in a small window around each
    marker found. The corners therefore come from exactly the same full-size
    refinement as a whole-frame search (what set_origin uses), so the two agree,
    at a fraction of the cost. A marker the second look misses keeps its
    scaled-up corners from the first."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    scale = min(1.0, detect_width / gray.shape[1]) if detect_width else 1.0
    if scale == 1.0:
        corners, ids, _ = tracker.detector.detectMarkers(gray)
        return corners, ids

    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    corners, ids, _ = tracker.detector.detectMarkers(small)
    if ids is None or len(ids) == 0:
        return corners, ids

    height, width = gray.shape
    refined = []
    for marker, marker_id in zip(corners, np.ravel(ids)):
        pts = marker.reshape(4, 2) / scale
        side = float(np.max(np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)))
        margin = 0.35 * side + 12          # room for the white border ArUco needs around a marker
        x0, y0 = np.maximum(np.floor(pts.min(axis=0) - margin).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(pts.max(axis=0) + margin).astype(int), [width, height])
        found, found_ids, _ = tracker.detector.detectMarkers(gray[y0:y1, x0:x1])
        best = pts
        if found_ids is not None:
            for c, i in zip(found, np.ravel(found_ids)):
                if int(i) == int(marker_id):
                    best = c.reshape(4, 2) + [x0, y0]
                    break
        refined.append(best.reshape(1, 4, 2).astype(np.float32))
    return tuple(refined), ids


# ---------------------------------------------------------------- camera

class LatestFrame:
    """Reads the camera on a background thread and keeps only the newest
    frame, so processing always works on what the camera sees right now."""

    def __init__(self, camera):
        self.camera = camera
        self.lock = threading.Lock()
        self.frame, self.count = None, 0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.002)
                continue
            with self.lock:
                self.frame, self.count = frame, self.count + 1

    def newest(self, seen):
        """The newest frame if it is newer than frame number `seen`, else None."""
        with self.lock:
            if self.count == seen:
                return None, seen
            return self.frame, self.count

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


# ---------------------------------------------------------------- preview

def draw_view(frame, tracker, corners, ids, R, t, lines):
    scale = min(1.0, VIEW_WIDTH / frame.shape[1])
    view = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
    if ids is not None and len(ids):
        cv2.aruco.drawDetectedMarkers(view, [c * scale for c in corners], ids)
    if R is not None:
        K_view = tracker.K.copy()
        K_view[:2] *= scale
        rvec, _ = cv2.Rodrigues(R)
        cv2.drawFrameAxes(view, K_view, tracker.dist, rvec, t.reshape(3, 1), 150, 2)
    for row, (text, color) in enumerate(lines):
        y = 30 + row * 30
        cv2.putText(view, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return view


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Live plate tracker: camera -> OSC /car/transform")
    ap.add_argument("--no-view", action="store_true", help="no preview window")
    ap.add_argument("--alpha", type=float, default=SMOOTHING_ALPHA,
                    help=f"smoothing, 1.0 = raw, lower = steadier but laggier (default {SMOOTHING_ALPHA})")
    ap.add_argument("--detect-width", type=int, default=DETECT_WIDTH,
                    help="width of the copy markers are searched in; 0 = full resolution")
    args = ap.parse_args()

    lens = load_intrinsics()
    tracker = MarkerTracker(lens.K, lens.dist)
    camera_to_world = load_camera_to_world()
    if np.allclose(camera_to_world, np.eye(4)):
        print("WARNING: no world_origin.yml: poses are measured from the camera. "
              "Run python -m calibration.set_origin")

    outputs = [SimpleUDPClient(config.OSC_HOST, port) for port in (config.TD_PORT, config.SIM_PORT)]
    camera = open_camera()
    reader = LatestFrame(camera)
    smoother = PoseSmoother(args.alpha)

    print(f"Camera '{PROFILE.name}', plate of {len(LAYOUT.ids)} x {LAYOUT.size_mm:g} mm markers")
    print(f"Sending OSC {config.OSC_ADDRESS} [tx ty tz rx ry rz] to {config.OSC_HOST} "
          f"ports {config.TD_PORT} (TouchDesigner) and {config.SIM_PORT} (simulation)")
    print(f"Smoothing alpha {args.alpha}, marker search at {args.detect_width or 'full'} px wide. q or ESC to stop.")

    window = "GatorFlow tracker"
    if not args.no_view:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, VIEW_WIDTH, VIEW_WIDTH * 9 // 16)

    hits = deque(maxlen=HISTORY)
    frame_times = deque(maxlen=HISTORY)
    work_ms = deque(maxlen=HISTORY)
    seen, size_checked, last_print, sent = 0, False, time.time(), 0
    try:
        while True:
            frame, seen = reader.newest(seen)
            if frame is None:
                if not args.no_view and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                time.sleep(0.001)
                continue
            if not size_checked:
                size_checked = True
                warning = check_frame_size(frame, lens)
                if warning:
                    print("WARNING:", warning)

            started = time.perf_counter()
            corners, ids = detect_markers(tracker, frame, args.detect_width)
            pose = tracker.solve(corners, ids)
            R = t = None
            if pose:
                R, t = smoother.apply(pose.R, pose.t)
                td = [float(v) for v in pose_to_td(R, t, camera_to_world)]
                for out in outputs:
                    out.send_message(config.OSC_ADDRESS, td)
                sent += 1
            else:
                smoother.reset()
            work_ms.append((time.perf_counter() - started) * 1000)
            hits.append(pose is not None)
            frame_times.append(time.time())

            rate = 100.0 * sum(hits) / len(hits)
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0
            timing = f"{fps:4.1f} fps, {np.mean(work_ms):4.1f} ms/frame"
            if pose:
                status = [(f"LOCK {rate:3.0f}%   markers {pose.marker_ids}   fit {pose.error_px:.1f} px   {timing}", (0, 255, 0)),
                          (f"pos {td[0]:8.1f} {td[1]:8.1f} {td[2]:8.1f} mm", (255, 255, 255)),
                          (f"rot {td[3]:8.1f} {td[4]:8.1f} {td[5]:8.1f} deg", (255, 255, 255))]
            elif tracker.last_error_px:
                status = [(f"LOST {rate:3.0f}%   rejected: fit {tracker.last_error_px:.1f} px > "
                           f"{tracker.max_error_px:.1f} (check setup/plate.yml)   {timing}", (0, 0, 255))]
            else:
                status = [(f"LOST {rate:3.0f}%   plate not seen   {timing}", (0, 0, 255))]

            now = time.time()
            if now - last_print >= 1.0:
                print("   ".join(text for text, _ in status), flush=True)
                last_print = now

            if not args.no_view:
                cv2.imshow(window, draw_view(frame, tracker, corners, ids, R, t, status))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        camera.release()
        cv2.destroyAllWindows()
        print(f"Stopped. {sent} poses sent.")


if __name__ == "__main__":
    main()
