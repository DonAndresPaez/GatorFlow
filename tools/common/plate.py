'''plate.py: the marker plate description and the pose math, as a library.

Imported by: calibration/set_origin.py, calibration/make_markers.py,
             checks/live_view.py, checks/check_marker.py, tests
Run directly: no

ArUco markers sit on a base plate under the model. Their size, dictionary and
positions are in setup/plate.yml, which KinectReader (C++) reads too. Every
visible marker's four corners go into ONE solvePnP, so the pose comes from up
to 16 points spread across the plate instead of 4 points on one small square:
  - rotation is much steadier (the corners are far apart, a longer lever)
  - the flip a single flat marker suffers when seen head-on mostly disappears
  - tracking survives a hand covering one or two markers

The pose reported is the PLATE's centre (see setup/plate.yml), not any one
marker. pose_to_td() then turns it into a TouchDesigner world pose; KinectReader
does the same in poseToWorld().

Must match main.cpp: MAX_REPROJECTION_ERROR_PX.
'''

from dataclasses import dataclass, field

import cv2 #OpenCV: marker detection
import numpy as np #matrices
from scipy.spatial.transform import Rotation #deals with rotation formats

from common import config, paths

# A solve whose corners land further than this (RMS, pixels) from where the
# layout says they should be is thrown away: wrong layout numbers, a marker
# peeling off the plate, or a misdetection.
MAX_REPROJECTION_ERROR_PX = 3.0

# OpenCV camera axes are X right, Y down, Z forward.
# TouchDesigner is X right, Y up, Z toward the viewer. This flips Y and Z.
OPENCV_TO_TD = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class MarkerLayout:
    dictionary_name: str              # e.g. "DICT_4X4_50"
    size_mm: float
    ids: list
    centers_mm: np.ndarray            # (N, 2), plate frame
    rotations_deg: np.ndarray         # (N,)

    def corners(self, index):
        """The 4 corners of marker `index` in the plate frame, in the order
        ArUco reports them: top-left, top-right, bottom-right, bottom-left."""
        h = self.size_mm / 2
        local = np.array([[-h, h], [h, h], [h, -h], [-h, -h]])
        a = np.radians(self.rotations_deg[index])
        turn = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        xy = local @ turn.T + self.centers_mm[index]
        return np.column_stack([xy, np.zeros(4)]).astype(np.float32)

    @property
    def dictionary(self):
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary_name))


def load_layout(path=paths.PLATE_FILE):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open {path}")
    dictionary = fs.getNode("dictionary").string() or "DICT_4X4_50"
    size = fs.getNode("markerSizeMm").real()
    node = fs.getNode("markers")
    ids, centers, rotations = [], [], []
    for i in range(node.size()):
        m = node.at(i)
        ids.append(int(m.getNode("id").real()))
        centers.append([m.getNode("xMm").real(), m.getNode("yMm").real()])
        r = m.getNode("rotationDeg")
        rotations.append(0.0 if r.empty() else r.real())
    fs.release()
    if not ids or size <= 0:
        raise ValueError(f"{path} needs markerSizeMm and at least one entry under markers")
    if not hasattr(cv2.aruco, dictionary):
        raise ValueError(f"{path}: unknown dictionary {dictionary} (use a cv2.aruco DICT_* name)")
    return MarkerLayout(dictionary, size, ids, np.array(centers, float), np.array(rotations, float))


LAYOUT = load_layout()
MARKER_IDS = LAYOUT.ids
MARKER_SIZE_MM = LAYOUT.size_mm


# Camera position in the TouchDesigner world (4x4, mm), from
# calibration/<camera>/world_origin.yml (written by calibration/set_origin.py).
# Until that has been run it stays identity, which puts the world origin at the camera.
def load_camera_to_world(path=None):
    path = path or paths.origin_file()
    if not path.exists():
        return np.eye(4)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    matrix = fs.getNode("cameraToWorld").mat()
    fs.release()
    return np.eye(4) if matrix is None or matrix.shape != (4, 4) else matrix.astype(float)


CAMERA_TO_WORLD = load_camera_to_world()


def pose_to_td(R, t_mm, camera_to_world=CAMERA_TO_WORLD):
    """OpenCV plate pose -> [tx, ty, tz, rx, ry, rz] in TouchDesigner world."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t_mm)
    W = camera_to_world @ OPENCV_TO_TD @ T
    rx, ry, rz = Rotation.from_matrix(W[:3, :3]).as_euler(config.EULER_ORDER, degrees=True)
    return [*W[:3, 3], rx, ry, rz]


@dataclass
class PlatePose:
    R: np.ndarray                     # 3x3, plate -> camera
    t: np.ndarray                     # (3,), plate centre in the camera, mm
    rvec: np.ndarray
    tvec: np.ndarray
    marker_ids: list = field(default_factory=list)   # which layout markers were used
    error_px: float = 0.0             # RMS reprojection error of the solve


class MarkerTracker:
    def __init__(self, camera_matrix, dist_coeffs, layout=None):
        self.layout = layout or LAYOUT
        self.K = np.array(camera_matrix, float)
        self.dist = np.array(dist_coeffs, float)
        dictionary = self.layout.dictionary
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        # A Board is just "these ids have their corners at these 3D points";
        # matchImagePoints pairs each detected corner with its 3D point.
        self.board = cv2.aruco.Board([self.layout.corners(i) for i in range(len(self.layout.ids))],
                                     dictionary, np.array(self.layout.ids, np.int32))
        self.last_error_px = None     # error of the last rejected solve, for display

    def detect(self, frame):
        """All ArUco detections in the frame: (corners, ids), ids may be None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def solve(self, corners, ids):
        """Plate pose from the detections, or None if no layout marker is seen
        or the fit is too poor to trust."""
        self.last_error_px = None
        if ids is None or len(ids) == 0:
            return None
        obj, img = self.board.matchImagePoints(corners, ids)
        if obj is None or len(obj) < 4:
            return None
        obj, img = obj.reshape(-1, 3).astype(np.float64), img.reshape(-1, 2).astype(np.float64)

        # IPPE: the exact solution for points on a plane. Then polish it with a
        # Levenberg-Marquardt pass over all the corners at once.
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.K, self.dist, rvec, tvec)

        projected, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img) ** 2, axis=1))))
        if error > MAX_REPROJECTION_ERROR_PX:
            self.last_error_px = error
            return None

        used = sorted({int(i) for i in np.ravel(ids) if int(i) in self.layout.ids})
        return PlatePose(cv2.Rodrigues(rvec)[0], tvec.ravel(), rvec, tvec, used, error)

    def find(self, frame):
        return self.solve(*self.detect(frame))