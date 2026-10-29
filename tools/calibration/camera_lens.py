'''camera_lens.py: measure the camera lens.

Run: python -m calibration.camera_lens        (from tools/)
Makes: calibration/<camera name>/intrinsics.yml  (the old one is kept as intrinsics.backup.yml)
Needs: the printed checkerboard (python -m calibration.make_checkerboard)

How to use it: hold the board in front of the camera, press SPACE whenever the
corners light up, q when done. 20-30 shots, varied: close, far, tilted, and in
each corner of the frame. Hold still for each one. Under 0.5 px reprojection
error is good, over 1.0 means retake them. Think about it as a registering a new finger
into a fingerprint scanner: the more varied shots, the better it will work.
The green grid on the preview shows which parts of the image your kept shots
cover; fill it, especially the corners and edges.

Keys: SPACE keep shot | s open the webcam's driver settings (focus, exposure) | q finish

solvePnP can only turn marker corners into a distance if it knows the lens:
how zoomed in it is, and how it bends straight lines. This measures both by
photographing a board whose real geometry is known exactly.

Do it once per camera AND resolution (setup/camera.yml). It survives the
camera being moved, so it does not need redoing between sessions. Changing
the resolution, zoom or focus does need it.

Webcams: lock the focus at the TABLE distance first (autofocus: 0 and a fixed
focus value in setup/camera.yml), then calibrate. Press s to find the focus
value; the current one is shown on the preview. If the board looks soft, hold
it farther from the camera rather than refocusing.
'''

import shutil

import cv2
import numpy as np

from common import paths
from common.camera import PROFILE, open_camera

COLUMNS, ROWS = 9, 6      # inner corners, not squares: a 10x7 board has 9x6
SQUARE_MM = 22.0          # measure one square after printing; this is the ruler for everything
MINIMUM_SHOTS = 10

# A full-resolution corner search on a 4K frame takes seconds when the board is
# not in view, which freezes the preview. Search a downscaled copy, then refine
# the corners on the full-resolution image when a shot is kept.
DETECT_WIDTH = 1920
VIEW_WIDTH = 1280
GRID = (6, 4)             # coverage grid drawn on the preview


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


def find_board(gray):
    """Corners in FULL-resolution pixel coordinates (unrefined), or None."""
    scale = min(1.0, DETECT_WIDTH / gray.shape[1])
    small = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale,
                                                 interpolation=cv2.INTER_AREA)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(small, (COLUMNS, ROWS), flags=flags)
    if not found:
        return None
    return (corners / scale).astype(np.float32)


def refine(gray, corners):
    """Sub-pixel corners on the full image. The search window is a quarter of a
    square so it never reaches the neighbouring corner, whatever the distance."""
    grid = corners.reshape(ROWS, COLUMNS, 2)
    square_px = float(np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2)))
    half = int(np.clip(square_px / 4, 3, 15))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    return cv2.cornerSubPix(gray, corners.copy(), (half, half), (-1, -1), criteria)


def coverage(image_points, size):
    """Which cells of a GRID over the image contain at least one kept corner."""
    hit = np.zeros(GRID[::-1], bool)
    for corners in image_points:
        for x, y in corners.reshape(-1, 2):
            hit[min(int(y / size[1] * GRID[1]), GRID[1] - 1),
                min(int(x / size[0] * GRID[0]), GRID[0] - 1)] = True
    return hit


def draw_preview(frame, corners, image_points, size, text, ok):
    scale = min(1.0, VIEW_WIDTH / frame.shape[1])
    view = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
    h, w = view.shape[:2]
    hit = coverage(image_points, size)
    shade = view.copy()
    for gy in range(GRID[1]):
        for gx in range(GRID[0]):
            if hit[gy, gx]:
                cv2.rectangle(shade, (gx * w // GRID[0], gy * h // GRID[1]),
                              ((gx + 1) * w // GRID[0], (gy + 1) * h // GRID[1]), (0, 160, 0), -1)
    view = cv2.addWeighted(shade, 0.2, view, 0.8, 0)
    for gx in range(1, GRID[0]):
        cv2.line(view, (gx * w // GRID[0], 0), (gx * w // GRID[0], h), (90, 90, 90), 1)
    for gy in range(1, GRID[1]):
        cv2.line(view, (0, gy * h // GRID[1]), (w, gy * h // GRID[1]), (90, 90, 90), 1)
    for kept in image_points:
        for x, y in kept.reshape(-1, 2) * scale:
            cv2.circle(view, (int(x), int(y)), 1, (0, 200, 255), -1)
    if corners is not None:
        cv2.drawChessboardCorners(view, (COLUMNS, ROWS), corners * scale, True)
    cv2.putText(view, text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if ok else (0, 0, 255), 2)
    return view


def main():
    board = np.zeros((ROWS * COLUMNS, 3), np.float32)
    board[:, :2] = np.mgrid[0:COLUMNS, 0:ROWS].T.reshape(-1, 2) * SQUARE_MM

    world_points, image_points = [], []
    camera = open_camera()
    capture = getattr(camera, "capture", None)       # only webcams have one
    size = None
    print(f"Calibrating camera '{PROFILE.name}' (setup/camera.yml)")
    print("SPACE = keep this shot, s = webcam settings, q = finish")

    while True:
        frame = camera.read()
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners = find_board(gray)

        cells = coverage(image_points, size).mean() * 100 if image_points else 0
        text = f"{len(world_points)} shots | coverage {cells:.0f}%"
        if capture is not None:
            text += f" | focus {capture.get(cv2.CAP_PROP_FOCUS):.0f}"
        if corners is not None:
            text += "  BOARD FOUND"
        cv2.imshow("calibration", draw_preview(frame, corners, image_points, size, text, corners is not None))

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and corners is not None:
            world_points.append(board)
            image_points.append(refine(gray, corners))
            print(f"kept shot {len(world_points)}")
        elif key == ord("s") and capture is not None:
            capture.set(cv2.CAP_PROP_SETTINGS, 1)
        elif key == ord("q"):
            break

    focus = capture.get(cv2.CAP_PROP_FOCUS) if capture is not None else None
    camera.release()
    cv2.destroyAllWindows()

    if len(world_points) < MINIMUM_SHOTS:
        print(f"Only {len(world_points)} shots, need {MINIMUM_SHOTS}. Nothing saved.")
        return

    error, K, dist, _, _ = cv2.calibrateCamera(world_points, image_points, size, None, None)
    print(f"Reprojection error: {error:.3f} pixels (under 0.5 is good, over 1.0 means retake the shots)")

    fx, fy = K[0, 0], K[1, 1]
    print(f"fx, fy: {fx:.1f}, {fy:.1f}   cx, cy: {K[0, 2]:.1f}, {K[1, 2]:.1f}  (image centre {size[0] / 2:.0f}, {size[1] / 2:.0f})")
    print(f"Field of view: {np.degrees(2 * np.arctan(size[0] / (2 * fx))):.1f} deg horizontal, "
          f"{np.degrees(2 * np.arctan(np.hypot(*size) / (2 * np.sqrt(fx * fy)))):.1f} deg diagonal")
    cells = coverage(image_points, size).mean()
    if cells < 0.85:
        print(f"Only {cells * 100:.0f}% of the image is covered: the edges are guessed, not measured. "
              f"Consider redoing it with more shots near the corners.")
    if focus is not None:
        print(f"Focus during calibration: {focus:.0f}. Put focus: {focus:.0f} in setup/camera.yml "
              f"so the tracker opens the camera at the same focus.")

    path = paths.intrinsics_file()
    if path.exists():
        shutil.copy(path, path.with_name("intrinsics.backup.yml"))
    write_intrinsics(path, K, dist, size, error)
    print(f"Wrote {path.relative_to(paths.REPO)}")
    print("If the camera has also moved, run: python -m calibration.set_origin")

if __name__ == "__main__":
    main()
