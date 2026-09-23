'''marker.py: the marker description and the pose math, as a library.

Imported by: calibrate.py, set_origin.py, check_marker.py, make_marker.py
Run directly: no

The live tracking loop is the C++ KinectReader. What stayed here is what
the calibration tools still need: which ArUco marker we look for, how big it
is, and pose_to_td(), the transform from an OpenCV camera pose to a
TouchDesigner world pose. KinectReader does the same transform in
poseToWorld(); set_origin uses this one to work out the matrix it feeds there.

MARKER_SIZE_MM must match ArucoMarkerSize in main.cpp.
'''

import json
from pathlib import Path

import cv2 #OpenCV: marker detection
import numpy as np #matrices
from scipy.spatial.transform import Rotation #deals with rotation formats

import config #brings the shared ports/addresses

CAMERA_FILE = Path(__file__).parent / "camera.json"
ORIGIN_FILE = Path(__file__).parent / "world_origin.json"

# Files both languages read: Python writes them here, KinectReader (C++) reads them.
SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"

# Must match ArucoConfiguration in KinectReader/src/main.cpp.
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
MARKER_SIZE_MM = 80.0             # measure the black square, not the paper

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
