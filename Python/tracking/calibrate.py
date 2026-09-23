'''calibrate.py: measure the camera lens.

Run: python -m tracking.calibrate
Makes: tracking/camera.json and shared/calibration.yml
Needs: the printed checkerboard (python -m tracking.make_checkerboard)

How to use it: hold the board in front of the camera, press SPACE whenever the
corners light up, q when done. 15-20 shots, varied: close, far, tilted, and in
each corner of the frame. Hold still for each one. Under 0.5 px reprojection
error is good, over 1.0 means retake them. Think about it as a registering a new finger
into a fingerprint scanner: the more varied shots, the better it will work.

solvePnP can only turn marker corners into a distance if it knows the lens:
how zoomed in it is, and how it bends straight lines. This measures both by
photographing a board whose real geometry is known exactly.

Do it once per camera. It survives the camera being moved, so it does not
need redoing between sessions. shared/calibration.yml is the file
KinectReader reads at startup.
'''

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from tracking.camera import open_camera
from tracking.marker import CAMERA_FILE, SHARED_DIR

YAML_FILE = SHARED_DIR / "calibration.yml"   # what KinectReader (C++) reads

COLUMNS, ROWS = 9, 6      # inner corners, not squares: a 10x7 board has 9x6
SQUARE_MM = 20.0          # measure one square after printing; this is the ruler for everything
MINIMUM_SHOTS = 10


def write_opencv_yaml(camera_matrix, dist_coeffs, size, error):
    """cv2.FileStorage writes the format cv::FileStorage reads, so the C++ side
    gets the same calibration without anyone converting anything by hand."""
    YAML_FILE.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(YAML_FILE), cv2.FILE_STORAGE_WRITE)
    fs.write("cameraMatrix", camera_matrix)
    fs.write("distortionCoefficients", dist_coeffs)
    fs.write("imageWidth", int(size[0]))
    fs.write("imageHeight", int(size[1]))
    fs.write("reprojectionErrorPx", float(error))
    fs.release()


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

    # Same numbers again in OpenCV's YAML format, which is what the C++ side reads.
    write_opencv_yaml(K, dist, size, error)
    print(f"Wrote {YAML_FILE}  (used by KinectReader)")


if __name__ == "__main__":
    main()
