'''camera_lens.py: measure the camera lens.

Run: python -m calibration.camera_lens        (from tools/)
Makes: calibration/<camera name>/intrinsics.yml  (the old one is kept as intrinsics.backup.yml)
Needs: the printed checkerboard (python -m calibration.make_checkerboard)

How to use it: hold the board in front of the camera, press SPACE whenever the
corners light up, q when done. 15-20 shots, varied: close, far, tilted, and in
each corner of the frame. Hold still for each one. Under 0.5 px reprojection
error is good, over 1.0 means retake them. Think about it as a registering a new finger
into a fingerprint scanner: the more varied shots, the better it will work.

solvePnP can only turn marker corners into a distance if it knows the lens:
how zoomed in it is, and how it bends straight lines. This measures both by
photographing a board whose real geometry is known exactly.

Do it once per camera AND resolution (setup/camera.yml). It survives the
camera being moved, so it does not need redoing between sessions. Changing
the resolution, zoom or focus does need it. KinectReader reads the same file.
'''

import shutil

import cv2
import numpy as np

from common import paths
from common.camera import PROFILE, open_camera

COLUMNS, ROWS = 9, 6      # inner corners, not squares: a 10x7 board has 9x6
SQUARE_MM = 22.0          # measure one square after printing; this is the ruler for everything
MINIMUM_SHOTS = 10


def write_intrinsics(path, camera_matrix, dist_coeffs, size, error):
    """cv2.FileStorage writes the format cv::FileStorage reads, so the C++ side
    gets the same calibration without anyone converting anything by hand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
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
    print(f"Calibrating camera '{PROFILE.name}' (setup/camera.yml)")
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

    path = paths.intrinsics_file()
    if path.exists():
        shutil.copy(path, path.with_name("intrinsics.backup.yml"))
    write_intrinsics(path, K, dist, size, error)
    print(f"Wrote {path.relative_to(paths.REPO)}  (read by the Python tools and KinectReader)")
    print("If the camera has also moved, run: python -m calibration.set_origin")

if __name__ == "__main__":
    main()
