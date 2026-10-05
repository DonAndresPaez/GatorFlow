'''test_plate.py - checks the four-marker plate pose.

Run: pytest

Makes fake detections by projecting the layout's corners through a known
camera and a known plate pose, then checks MarkerTracker.solve() gets that
pose back. No Kinect needed. KinectReader does the same solve in C++, so if
these pass and the lab numbers are still wrong, look at setup/plate.yml or
the calibration, not the maths.
'''

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common.plate import LAYOUT, MarkerTracker

K = np.array([[1060.0, 0, 960], [0, 1060.0, 540], [0, 0, 1]])
DIST = np.zeros(5)


def plate_seen_from_above(tilt_deg=(5, -3, 20), height_mm=1900.0):
    """Plate lying on the table, Kinect looking down at it, slightly off-square."""
    R = (Rotation.from_euler("x", 180, degrees=True)
         * Rotation.from_euler("xyz", tilt_deg, degrees=True)).as_matrix()
    return R, np.array([30.0, -40.0, height_mm])


def fake_detections(R, t, visible=None, noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rvec, _ = cv2.Rodrigues(R)
    corners, ids = [], []
    for index, marker_id in enumerate(LAYOUT.ids):
        if visible is not None and marker_id not in visible:
            continue
        pts, _ = cv2.projectPoints(LAYOUT.corners(index).astype(np.float64), rvec, t, K, DIST)
        pts = pts.reshape(1, 4, 2) + rng.normal(0, noise_px, (1, 4, 2))
        corners.append(pts.astype(np.float32))
        ids.append([marker_id])
    return tuple(corners), np.array(ids, np.int32)


def rotation_error_deg(R_a, R_b):
    return np.degrees(Rotation.from_matrix(R_a.T @ R_b).magnitude())


def test_all_four_markers_recover_the_plate_pose():
    R, t = plate_seen_from_above()
    pose = MarkerTracker(K, DIST).solve(*fake_detections(R, t))
    assert pose is not None
    assert pose.marker_ids == LAYOUT.ids
    assert pose.t == pytest.approx(t, abs=1e-3)
    assert rotation_error_deg(pose.R, R) < 1e-4
    assert pose.error_px < 1e-3


def test_one_marker_alone_still_gives_the_plate_centre():
    """With three markers covered, the pose is still the PLATE's, not the marker's."""
    R, t = plate_seen_from_above()
    pose = MarkerTracker(K, DIST).solve(*fake_detections(R, t, visible=[LAYOUT.ids[0]]))
    assert pose is not None
    assert pose.marker_ids == [LAYOUT.ids[0]]
    assert pose.t == pytest.approx(t, abs=1e-2)


def test_four_markers_beat_one_under_pixel_noise():
    """The reason for the plate: same corner noise, much steadier rotation."""
    R, t = plate_seen_from_above()
    tracker = MarkerTracker(K, DIST)
    errors_one, errors_four = [], []
    for seed in range(40):
        one = tracker.solve(*fake_detections(R, t, visible=[LAYOUT.ids[0]], noise_px=0.3, seed=seed))
        four = tracker.solve(*fake_detections(R, t, noise_px=0.3, seed=seed))
        errors_one.append(rotation_error_deg(one.R, R))
        errors_four.append(rotation_error_deg(four.R, R))
    assert np.mean(errors_four) < np.mean(errors_one) / 3


def test_wrong_layout_is_rejected():
    """Corners that don't fit the layout (a marker moved on the plate) must not
    produce a pose: better lost than confidently wrong."""
    R, t = plate_seen_from_above()
    corners, ids = fake_detections(R, t)
    shifted = list(corners)
    shifted[1] = shifted[1] + np.float32(25.0)   # marker 1 is 25 px off where the layout says
    tracker = MarkerTracker(K, DIST)
    assert tracker.solve(tuple(shifted), ids) is None
    assert tracker.last_error_px > 3.0


def test_unknown_ids_are_ignored():
    R, t = plate_seen_from_above()
    corners, ids = fake_detections(R, t)
    stray = (np.array([[[100, 100], [140, 100], [140, 140], [100, 140]]], np.float32),)
    pose = MarkerTracker(K, DIST).solve(corners + stray, np.vstack([ids, [[42]]]).astype(np.int32))
    assert pose is not None and 42 not in pose.marker_ids