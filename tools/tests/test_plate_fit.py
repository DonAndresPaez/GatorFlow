'''test_plate_fit.py - checks checks/plate_fit.py measures the real marker layout.

Run: pytest

Renders 4K pictures of a plate whose markers are NOT where plate.yml says
(a few mm off, one slightly turned), runs the fit, and checks it recovers the
true positions and that the fit error drops. No camera needed.
'''

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from checks.plate_fit import fit_layout, keep_plate_frame, mean_residuals
from common.plate import LAYOUT, MarkerLayout
from tracking.tracker import detect_markers
from common.plate import MarkerTracker

W, H = 3840, 2160
K = np.array([[3430.0, 0, 1996], [0, 3434.0, 999], [0, 0, 1]])
DIST = np.zeros(5)


def true_layout():
    shifts = np.array([[1.8, -1.2], [-0.9, 2.1], [2.4, 0.6], [-1.5, -2.0]])[:len(LAYOUT.ids)]
    turns = np.array([0.0, 1.5, -0.8, 0.0])[:len(LAYOUT.ids)]
    return MarkerLayout(LAYOUT.dictionary_name, LAYOUT.size_mm, list(LAYOUT.ids),
                        LAYOUT.centers_mm + shifts, LAYOUT.rotations_deg + turns)


def render(layout, R, t):
    canvas = np.full((H, W), 235, np.uint8)
    rvec, _ = cv2.Rodrigues(R)
    for index, marker_id in enumerate(layout.ids):
        art = cv2.aruco.generateImageMarker(layout.dictionary, marker_id, 600)
        c3 = layout.corners(index).astype(np.float64)
        h = layout.size_mm * 0.8
        centre = c3.mean(axis=0)
        sheet = centre + np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        px, _ = cv2.projectPoints(sheet, rvec, t, K, DIST)
        cv2.fillConvexPoly(canvas, px.reshape(-1, 2).astype(np.int32), 255)
        img_px, _ = cv2.projectPoints(c3, rvec, t, K, DIST)
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [600, 0], [600, 600], [0, 600]]),
                                        img_px.reshape(4, 2).astype(np.float32))
        warped = cv2.warpPerspective(art, M, (W, H), flags=cv2.INTER_AREA, borderValue=255)
        mask = cv2.warpPerspective(np.full((600, 600), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
        canvas[mask > 0] = warped[mask > 0]
    return cv2.GaussianBlur(canvas, (0, 0), 0.8)


def frames_of(layout):
    tracker = MarkerTracker(K, DIST)
    frames = []
    for tilt, offset in [((3, -2, 10), (40, -30)), ((-4, 3, -25), (-60, 20)), ((1, 5, 40), (10, 50))]:
        R = (Rotation.from_euler("x", 180, degrees=True) * Rotation.from_euler("xyz", tilt, degrees=True)).as_matrix()
        corners, ids = detect_markers(tracker, render(layout, R, np.array([*offset, 1800.0])))
        frames.append({int(i): c.reshape(4, 2).astype(np.float64) for c, i in zip(corners, np.ravel(ids))})
    return frames


def test_fit_recovers_the_real_layout():
    truth = true_layout()
    frames = frames_of(truth)
    assert all(len(f) == len(LAYOUT.ids) for f in frames)

    before = np.mean(list(mean_residuals(frames, LAYOUT, K, DIST).values()))
    fitted, sides = fit_layout(frames, LAYOUT, K, DIST)
    after = np.mean(list(mean_residuals(frames, fitted, K, DIST).values()))
    assert before > 2.0                      # the wrong layout really is visible
    assert after < 0.5                       # and the measured one fits

    # Same shape as the truth. Where the plate frame sits and its overall scale
    # come from plate.yml (the hand measurements), so compare after removing those.
    centred = fitted.centers_mm - fitted.centers_mm.mean(0)
    target = truth.centers_mm - truth.centers_mm.mean(0)
    scale = np.sqrt((target ** 2).sum() / (centred ** 2).sum())
    aligned, turns = keep_plate_frame(centred * scale, fitted.rotations_deg, truth.centers_mm)
    assert abs(scale - 1) < 0.01
    assert np.abs(aligned - truth.centers_mm).max() < 0.3                       # mm
    assert np.abs(turns - truth.rotations_deg).max() < 0.2                      # deg
    assert np.ptp(sides) < 0.3                                                  # all markers the same size

def test_plate_frame_is_kept():
    """Only the markers' relative positions change: the centroid of the
    centres stays where plate.yml put it."""
    fitted, _ = fit_layout(frames_of(true_layout()), LAYOUT, K, DIST)
    assert np.abs(fitted.centers_mm.mean(0) - LAYOUT.centers_mm.mean(0)).max() < 1e-6
