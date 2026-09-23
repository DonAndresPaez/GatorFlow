"""Measure the camera's lens numbers and write them into camera.json.

    python -m tracking.calibrate

Print a checkerboard, hold it in front of the camera, and press SPACE whenever
the colored corners appear. Take 15-20 shots: close, far, tilted, and in the
corners of the frame. Press q when done.

Why: solvePnP needs to know how the lens maps the world onto pixels. With the
placeholder numbers in camera.json, every distance is off by a few percent.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from tracking.camera import open_camera
from tracking.track import CAMERA_FILE

COLUMNS, ROWS = 9, 6      # inner corners, not squares: a 10x7 board has 9x6
SQUARE_MM = 20.0          # measure one square after printing; this is the ruler for everything
MINIMUM_SHOTS = 10


def main():
    board = np.zeros((ROWS * COLUMNS, 3), np.float32)
    board[:, :2] = np.mgrid[0:COLUMNS, 0:ROWS].T.reshape(-1, 2) * SQUARE_MM

    world_points, image_points = [], []
    camera = open_camera()
    size = None
    print("SPACE = keep this shot, q = finish")

    while True:
        frame = camera.read()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (COLUMNS, ROWS))
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, (COLUMNS, ROWS), corners, found)
        cv2.putText(view, f"{len(world_points)} shots" + ("  BOARD FOUND" if found else ""),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("calibration", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            world_points.append(board)
            image_points.append(corners)
            print(f"kept shot {len(world_points)}")
        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    if len(world_points) < MINIMUM_SHOTS:
        print(f"Only {len(world_points)} shots, need {MINIMUM_SHOTS}. Nothing saved.")
        return

    error, K, dist, _, _ = cv2.calibrateCamera(world_points, image_points, size, None, None)
    print(f"Reprojection error: {error:.3f} pixels (under 0.5 is good, over 1.0 means retake the shots)")

    if CAMERA_FILE.exists():
        shutil.copy(CAMERA_FILE, CAMERA_FILE.with_suffix(".json.backup"))
    Path(CAMERA_FILE).write_text(json.dumps({
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "image_size": list(size),
        "reprojection_error_px": float(error),
    }, indent=2))
    print(f"Wrote {CAMERA_FILE}")


if __name__ == "__main__":
    main()
