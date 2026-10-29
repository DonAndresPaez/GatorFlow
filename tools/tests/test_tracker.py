'''test_tracker.py - checks the live tracker's own pieces.

Run: pytest

1. The smoother behaves as expected (first reading passes through,
   steps are halved at alpha 0.5, q and -q count as the same rotation).
2. Searching for markers in a shrunk copy of a 4K frame, then refining the
   corners at full size, gives the same corners and the same plate pose as
   searching the whole 4K frame. Uses a rendered picture of the real plate
   layout, no camera needed.
'''

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common.plate import LAYOUT, MarkerTracker
from tracking.tracker import PoseSmoother, detect_markers

W, H = 3840, 2160
K = np.array([[3430.0, 0, 1996], [0, 3434.0, 999], [0, 0, 1]])
DIST = np.zeros(5)


# ---------------------------------------------------------------- smoother

def test_first_reading_passes_through():
    R = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    R_out, t_out = PoseSmoother(0.5).apply(R, [1.0, 2.0, 3.0])
    assert np.allclose(R_out, R) and np.allclose(t_out, [1, 2, 3])


def test_step_is_halved_at_alpha_half():
    s = PoseSmoother(0.5)
    s.apply(np.eye(3), [0.0, 0.0, 0.0])
    R_step = Rotation.from_euler("z", 40, degrees=True).as_matrix()
    R_out, t_out = s.apply(R_step, [100.0, 0.0, 0.0])
    assert t_out == pytest.approx([50.0, 0.0, 0.0])
    assert np.degrees(Rotation.from_matrix(R_out).magnitude()) == pytest.approx(20.0, abs=0.5)


def test_opposite_quaternion_sign_is_the_same_rotation():
    """Two readings of the same rotation must not be blended into a different one."""
    s = PoseSmoother(0.5)
    R = Rotation.from_euler("z", 179, degrees=True).as_matrix()
    s.apply(R, [0, 0, 0])
    s.q = -s.q                      # same rotation, other sign
    R_out, _ = s.apply(R, [0, 0, 0])
    assert np.allclose(R_out, R, atol=1e-9)


def test_reset_starts_fresh():
    s = PoseSmoother(0.5)
    s.apply(np.eye(3), [0.0, 0.0, 0.0])
    s.reset()
    _, t_out = s.apply(np.eye(3), [200.0, 0.0, 0.0])
    assert t_out == pytest.approx([200.0, 0.0, 0.0])


# ---------------------------------------------------------------- detection

def render_plate(R, t, blur=0.8, noise=2.0, seed=0):
    """A 4K grey picture of the plate layout seen through K at pose (R, t)."""
    canvas = np.full((H, W), 235, np.uint8)
    rvec, _ = cv2.Rodrigues(R)
    for index, marker_id in enumerate(LAYOUT.ids):
        art = cv2.aruco.generateImageMarker(LAYOUT.dictionary, marker_id, 600, borderBits=1)
        corners3d = LAYOUT.corners(index).astype(np.float64)
        # white quiet zone around the marker, like the printed sheet
        margin = LAYOUT.size_mm * 0.15
        h = LAYOUT.size_mm / 2 + margin
        centre = corners3d.mean(axis=0)
        sheet = centre + np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        sheet_px, _ = cv2.projectPoints(sheet, rvec, t, K, DIST)
        cv2.fillConvexPoly(canvas, sheet_px.reshape(-1, 2).astype(np.int32), 255)
        img_px, _ = cv2.projectPoints(corners3d, rvec, t, K, DIST)
        src = np.float32([[0, 0], [600, 0], [600, 600], [0, 600]])
        M = cv2.getPerspectiveTransform(src, img_px.reshape(4, 2).astype(np.float32))
        warped = cv2.warpPerspective(art, M, (W, H), flags=cv2.INTER_AREA, borderValue=255)
        mask = cv2.warpPerspective(np.full((600, 600), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
        canvas[mask > 0] = warped[mask > 0]
    img = cv2.GaussianBlur(canvas, (0, 0), blur)
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


def plate_pose(tilt=(4, -3, 25), offset=(60.0, -40.0)):
    R = (Rotation.from_euler("x", 180, degrees=True)
         * Rotation.from_euler("xyz", tilt, degrees=True)).as_matrix()
    return R, np.array([offset[0], offset[1], 1800.0])


def as_dict(corners, ids):
    return {int(i): c.reshape(4, 2) for i, c in zip(np.ravel(ids), corners)}


def test_shrunk_search_finds_the_same_corners_as_full_search():
    R, t = plate_pose()
    img = render_plate(R, t)
    tracker = MarkerTracker(K, DIST)
    full = as_dict(*detect_markers(tracker, img, detect_width=0))
    fast = as_dict(*detect_markers(tracker, img, detect_width=1920))
    assert sorted(full) == sorted(fast) == sorted(LAYOUT.ids)
    worst = max(np.abs(full[i] - fast[i]).max() for i in full)
    assert worst < 0.01, f"corners differ by {worst:.3f} px"   # same refinement as a full search


@pytest.mark.parametrize("detect_width", [1920, 1280])
def test_shrunk_search_gives_the_true_plate_pose(detect_width):
    R, t = plate_pose()
    img = render_plate(R, t)
    tracker = MarkerTracker(K, DIST)
    pose = tracker.solve(*detect_markers(tracker, img, detect_width))
    assert pose is not None, f"rejected, fit {tracker.last_error_px}"
    assert pose.marker_ids == LAYOUT.ids
    assert np.abs(pose.t - t).max() < 1.0                                       # mm
    # 0.3 deg: the rendered image's blur + noise limits rotation to ~0.2 deg with
    # 40 mm markers even when searching the full 4K frame (it was < 0.1 deg with
    # the old 119 mm ones). At the model's edge, 150 mm out, 0.3 deg is ~0.8 mm.
    assert np.degrees(Rotation.from_matrix(pose.R.T @ R).magnitude()) < 0.3     # deg
    full = tracker.solve(*detect_markers(tracker, img, detect_width=0))
    assert np.degrees(Rotation.from_matrix(pose.R.T @ full.R).magnitude()) < 0.01  # same as a full search
