'''
This script watches the ArUco marker stuck on the 3d model and works out where the car is
and how it is oriented. Then it sends that to TouchDesigner and the simulation.

1. grab a frame
2. find the marker
3. work out its 3d pose
4. convert to TouchDesigner axes
5. smooth it and send it
'''

import json
from pathlib import Path

import cv2 #OpenCV: camera and marker detection
import numpy as np #matrices
from pythonosc.udp_client import SimpleUDPClient #sets up UDP to send the OSC messages
from scipy.spatial.transform import Rotation #deals with rotation formats

import config #brings the shared ports/addresses
from tracking.camera import open_camera #webcam or Kinect, chosen in camera.py

CAMERA_FILE = Path(__file__).parent / "camera.json"
ORIGIN_FILE = Path(__file__).parent / "world_origin.json"

# Settings I need to tune in the lab.
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
MARKER_SIZE_MM = 80.0             # measure the black square, not the paper
SMOOTHING = 0.5                   # 1 = raw, lower = smoother but laggier

# OpenCV camera axes are X right, Y down, Z forward.
# TouchDesigner is X right, Y up, Z toward the viewer. This flips Y and Z.
OPENCV_TO_TD = np.diag([1.0, -1.0, -1.0, 1.0])


# Camera position in the TouchDesigner world (4x4, mm), written by set_origin.py.
# Until I run that, it stays identity, which puts the world origin at the camera.
def load_camera_to_world():
    if ORIGIN_FILE.exists():
        return np.array(json.loads(ORIGIN_FILE.read_text())["camera_to_world"], float)
    return np.eye(4)


CAMERA_TO_WORLD = load_camera_to_world()


def pose_to_td(R, t_mm, camera_to_world=CAMERA_TO_WORLD):
    """OpenCV marker pose -> [tx, ty, tz, rx, ry, rz] in TouchDesigner world."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t_mm)
    W = camera_to_world @ OPENCV_TO_TD @ T
    rx, ry, rz = Rotation.from_matrix(W[:3, :3]).as_euler(config.EULER_ORDER, degrees=True)
    return [*W[:3, 3], rx, ry, rz]


class Smoother:
    """Averages position linearly and rotation with quaternions (averaging
    Euler angles jumps when an angle wraps past 180°)."""

    def __init__(self, alpha=SMOOTHING):
        self.alpha, self.p, self.q = alpha, None, None

    def reset(self):
        self.p = self.q = None

    def __call__(self, pose):
        p = np.array(pose[:3])
        q = Rotation.from_euler(config.EULER_ORDER, pose[3:], degrees=True).as_quat()
        if self.p is None:
            self.p, self.q = p, q
        else:
            if np.dot(self.q, q) < 0:
                q = -q
            self.p = (1 - self.alpha) * self.p + self.alpha * p
            self.q = (1 - self.alpha) * self.q + self.alpha * q
            self.q /= np.linalg.norm(self.q)
        return [*self.p, *Rotation.from_quat(self.q).as_euler(config.EULER_ORDER, degrees=True)]


class MarkerTracker:
    def __init__(self, camera_matrix, dist_coeffs):
        self.K = np.array(camera_matrix, float)
        self.dist = np.array(dist_coeffs, float)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)
        h = MARKER_SIZE_MM / 2
        self.corners_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)

    def find(self, frame):
        """Return (R, t_mm, rvec, tvec) for our marker, or None."""
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None:
            return None
        for c, i in zip(corners, ids.ravel()):
            if i == MARKER_ID:
                ok, rvec, tvec = cv2.solvePnP(self.corners_3d, c.reshape(4, 2), self.K, self.dist,
                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    return cv2.Rodrigues(rvec)[0], tvec.ravel(), rvec, tvec
        return None


def main():
    cam = json.loads(CAMERA_FILE.read_text())
    tracker = MarkerTracker(cam["camera_matrix"], cam["dist_coeffs"])
    smooth = Smoother()
    outputs = [SimpleUDPClient(config.OSC_HOST, p) for p in (config.TD_PORT, config.SIM_PORT)]
    camera = open_camera()
    print(f"Sending {config.OSC_ADDRESS} to ports {config.TD_PORT} (TD) and {config.SIM_PORT} (simulation)")

    while True:
        frame = camera.read()
        if frame is None:
            continue
        found = tracker.find(frame)
        if found is None:
            smooth.reset()  # don't blend across a gap
        else:
            R, t, rvec, tvec = found
            pose = smooth(pose_to_td(R, t))
            for out in outputs:
                out.send_message(config.OSC_ADDRESS, [float(v) for v in pose])
            cv2.drawFrameAxes(frame, tracker.K, tracker.dist, rvec, tvec, MARKER_SIZE_MM)
        cv2.putText(frame, "LOCK" if found else "lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("GatorFlow tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
